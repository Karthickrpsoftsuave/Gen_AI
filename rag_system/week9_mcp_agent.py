"""Week 9 - Module 5: MCP-aware Recipe Adaptation Agent
========================================================
This agent discovers ALL tools from MCP servers listed in mcp_config.json.
It NEVER hard-codes tool names or schemas — discovery is automatic.

Adding a second MCP server = edit mcp_config.json only.
Zero lines change in this file. That is the whole point.

Architecture
------------
  HOST (this file)        CLIENT (mcp library)      SERVER (subprocess)
  ─────────────────       ─────────────────────     ──────────────────────
  Gemini LLM runs here    JSON-RPC over stdio       recipe / ingredient DB
  Decides which tools     initialize handshake      exposes tools only
  to call                 tools/list discovery      no LLM inside server
                          tools/call dispatch

Usage
-----
  # List tools (no API key needed):
  python week9_mcp_agent.py --list-tools

  # Run a recipe query:
  python week9_mcp_agent.py
  python week9_mcp_agent.py "Adapt the kimchi recipe for a soy-free household."

  # Demo ingredient DB lookup:
  python week9_mcp_agent.py --demo-ingredient

  # Capture raw JSON-RPC wire exchange:
  python week9_mcp_agent.py --wire-capture
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("week9_mcp_agent")

# ── Config ──────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "mcp_config.json"
PROJECT_ROOT = Path(__file__).parent.resolve()

PRICE_INPUT_PER_M  = 0.30
PRICE_OUTPUT_PER_M = 2.50
MAX_ITERATIONS = 10
MAX_TOKENS     = 8000
MAX_COST_USD   = 0.10
MAX_WALL_SECS  = 90.0

SYSTEM_INSTRUCTION = (
    "You are a recipe adaptation assistant. "
    "You have access to tools discovered from MCP servers. "
    "Always use tools to look up facts — never invent ingredient data. "
    "When adapting a recipe: search for it, scale if needed, check allergens. "
    "If a substitute ingredient is itself an allergen, follow the cascade. "
    "Be concise and cite tool results."
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_mcp_config() -> list[dict]:
    """Load server definitions from mcp_config.json."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"MCP config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    return cfg.get("servers", [])


def resolve_command(cmd: str) -> str:
    """Resolve a relative command (e.g. .venv/Scripts/python) to an absolute path."""
    p = PROJECT_ROOT / cmd
    return str(p) if p.exists() else cmd


def mcp_tools_to_gemini(tools: list[dict]) -> list[gtypes.FunctionDeclaration]:
    """Convert MCP tool schemas to Gemini FunctionDeclarations."""
    decls = []
    for t in tools:
        schema = t.get("inputSchema") or {}
        props_raw = schema.get("properties") or {}
        required = schema.get("required") or []

        props = {}
        for pname, pschema in props_raw.items():
            ptype = pschema.get("type", "string").upper()
            gemini_type = getattr(gtypes.Type, ptype, gtypes.Type.STRING)
            props[pname] = gtypes.Schema(
                type=gemini_type,
                description=pschema.get("description", ""),
            )

        decls.append(gtypes.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties=props,
                required=required,
            ) if props else None,
        ))
    return decls


# ── Wire-capture ─────────────────────────────────────────────────────────────

class WireCapture:
    """Records JSON-RPC messages for the wire.json deliverable."""
    def __init__(self):
        self.messages: list[dict] = []

    def record(self, direction: str, msg: dict):
        self.messages.append({"direction": direction, "msg": msg})

    def dump(self, path: Path):
        path.write_text(json.dumps(self.messages, indent=2), encoding="utf-8")
        log.info("Wire capture saved to %s (%d messages)", path, len(self.messages))


# ── Core: multi-server tool discovery ─────────────────────────────────────────

async def _run_with_all_servers(request: str, api_key: str, model: str,
                                 verbose: bool = True, wire: WireCapture | None = None) -> dict:
    """
    Nested async-with for all servers so anyio task groups stay in scope.
    The LLM (Gemini) runs HERE, in the host. Servers run in subprocesses.
    """
    servers = load_mcp_config()
    log.info("MCP config: %d server(s) — %s", len(servers), [s["name"] for s in servers])

    async def _open_and_connect(server_cfg):
        """Open stdio transport and return (tools_list, session) within the caller's task group."""
        cmd = resolve_command(server_cfg["command"])
        params = StdioServerParameters(
            command=cmd,
            args=server_cfg.get("args", []),
            env=dict(os.environ),
            cwd=str(PROJECT_ROOT),
        )
        return params, server_cfg

    # We need all sessions open simultaneously during the agent loop.
    # Use a recursive-nesting helper to open N sessions with proper async-with.
    all_tools: list[dict] = []
    sessions: dict[str, ClientSession] = {}

    async def _open_servers_and_run(idx: int):
        nonlocal all_tools, sessions
        if idx >= len(servers):
            return await _agent_loop(all_tools, sessions, request, api_key, model, verbose, wire)

        server_cfg = servers[idx]
        cmd = resolve_command(server_cfg["command"])
        params = StdioServerParameters(
            command=cmd,
            args=server_cfg.get("args", []),
            env=dict(os.environ),
            cwd=str(PROJECT_ROOT),
        )
        log.info("Connecting to MCP server: %s", server_cfg["name"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                server_tools = [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "inputSchema": t.input_schema,
                        "server": server_cfg["name"],
                    }
                    for t in tools_result.tools
                ]
                log.info("  Discovered %d tool(s): %s", len(server_tools),
                         [t["name"] for t in server_tools])

                if wire:
                    wire.record("meta/discovery", {
                        "server": server_cfg["name"],
                        "tools": [t["name"] for t in server_tools],
                    })

                for t in server_tools:
                    all_tools.append(t)
                    sessions[t["name"]] = session

                result = await _open_servers_and_run(idx + 1)
                return result

    return await _open_servers_and_run(0)


async def _agent_loop(all_tools, sessions, request, api_key, model, verbose, wire):
    """Run the Gemini agent loop with all discovered tools."""
    log.info("Total tools discovered: %d — %s", len(all_tools), [t["name"] for t in all_tools])

    if wire:
        wire.record("meta/all_tools", {
            "count": len(all_tools),
            "names": [t["name"] for t in all_tools],
        })

    gemini_decls = mcp_tools_to_gemini(all_tools)
    gemini_tools = gtypes.Tool(function_declarations=gemini_decls)

    client = genai.Client(api_key=api_key)
    messages: list[gtypes.Content] = [
        gtypes.Content(role="user", parts=[gtypes.Part(text=request)])
    ]

    total_in = total_out = 0
    total_cost = 0.0
    iteration = 0
    tool_calls_log = []
    start = time.monotonic()
    final_answer = ""

    while True:
        elapsed = time.monotonic() - start
        if iteration >= MAX_ITERATIONS:
            log.warning("Budget: max iterations"); break
        if total_in + total_out >= MAX_TOKENS:
            log.warning("Budget: max tokens"); break
        if total_cost >= MAX_COST_USD:
            log.warning("Budget: max cost"); break
        if elapsed >= MAX_WALL_SECS:
            log.warning("Budget: max wall time"); break

        if verbose:
            log.info("[lap %d] model call — tok=%d cost=$%.5f elapsed=%.1fs",
                     iteration + 1, total_in + total_out, total_cost, elapsed)

        if wire:
            wire.record("host->model", {"lap": iteration + 1})

        response = client.models.generate_content(
            model=model,
            contents=messages,
            config=gtypes.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[gemini_tools],
                temperature=0,
                max_output_tokens=1000,
            ),
        )

        iteration += 1
        usage = response.usage_metadata
        if usage:
            total_in  += getattr(usage, "prompt_token_count", 0) or 0
            total_out += getattr(usage, "candidates_token_count", 0) or 0
        total_cost = (total_in * PRICE_INPUT_PER_M + total_out * PRICE_OUTPUT_PER_M) / 1_000_000

        messages.append(response.candidates[0].content)

        tool_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call
        ]

        if not tool_calls:
            for part in response.candidates[0].content.parts:
                if part.text:
                    final_answer = part.text.strip()
                    break
            if verbose:
                log.info("[lap %d] Final answer: %s", iteration, final_answer[:200])
            break

        tool_response_parts = []
        for fc in tool_calls:
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}
            if verbose:
                log.info("[lap %d] TOOL CALL: %s  args=%s", iteration, tool_name, json.dumps(tool_args))

            session = sessions.get(tool_name)
            if session is None:
                result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})
            else:
                if wire:
                    wire.record("host->server", {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {"name": tool_name, "arguments": tool_args},
                    })
                mcp_result = await session.call_tool(tool_name, tool_args)
                content = mcp_result.content
                result_str = content[0].text if (content and hasattr(content[0], "text")) else json.dumps({"result": str(content)})
                if wire:
                    wire.record("server->host", {
                        "jsonrpc": "2.0",
                        "result": {"content": [{"type": "text", "text": result_str[:500]}]},
                    })

            if verbose:
                log.info("[lap %d] result=%s", iteration, result_str[:200])

            tool_calls_log.append({"lap": iteration, "tool": tool_name, "args": tool_args})
            tool_response_parts.append(
                gtypes.Part(
                    function_response=gtypes.FunctionResponse(
                        name=tool_name,
                        response={"result": result_str},
                    )
                )
            )
        messages.append(gtypes.Content(role="user", parts=tool_response_parts))

    return {
        "answer": final_answer,
        "iterations": iteration,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": total_cost,
        "latency_secs": time.monotonic() - start,
        "tool_calls": tool_calls_log,
        "tools_discovered": [t["name"] for t in all_tools],
    }


# ── Public API ───────────────────────────────────────────────────────────────

async def run_agent(request: str, api_key: str, model: str,
                    verbose: bool = True, wire: WireCapture | None = None) -> dict:
    """Discover tools from all MCP servers in config, then run the agent loop."""
    return await _run_with_all_servers(request, api_key, model, verbose, wire)


async def list_tools_only() -> list[str]:
    """Print discovered tools from all configured servers — no LLM call."""
    servers = load_mcp_config()
    print(f"\nMCP config: {len(servers)} server(s)")
    all_names = []
    for server_cfg in servers:
        cmd = resolve_command(server_cfg["command"])
        params = StdioServerParameters(
            command=cmd,
            args=server_cfg.get("args", []),
            env=dict(os.environ),
            cwd=str(PROJECT_ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                print(f"\nServer: {server_cfg['name']}")
                print(f"  Tools ({len(result.tools)}):")
                for t in result.tools:
                    print(f"    [{t.name}]  {(t.description or '')[:80]}")
                all_names.extend([t.name for t in result.tools])
    print(f"\nTotal tools discovered: {len(all_names)}")
    print("Names:", all_names)
    return all_names


async def capture_wire_exchange(api_key: str, model: str) -> None:
    """Capture raw JSON-RPC initialize -> tools/list -> tools/call exchange."""
    wire = WireCapture()
    servers = load_mcp_config()

    # Find ingredient-database-server for the wire capture
    ing_server = next((s for s in servers if "ingredient" in s.get("name", "")), None)
    if ing_server:
        cmd = resolve_command(ing_server["command"])
        params = StdioServerParameters(
            command=cmd,
            args=ing_server.get("args", []),
            env=dict(os.environ),
            cwd=str(PROJECT_ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                # 1. initialize
                wire.record("client->server", {
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"roots": {"listChanged": True}},
                        "clientInfo": {"name": "week9_mcp_agent", "version": "1.0.0"},
                    },
                })
                await session.initialize()
                wire.record("server->client", {
                    "jsonrpc": "2.0", "id": 1, "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": ing_server["name"], "version": "1.0.0"},
                    },
                })

                # 2. tools/list
                wire.record("client->server", {
                    "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
                })
                tools_result = await session.list_tools()
                wire.record("server->client", {
                    "jsonrpc": "2.0", "id": 2, "result": {
                        "tools": [
                            {
                                "name": t.name,
                                "description": (t.description or "")[:120],
                                "inputSchema": t.input_schema,
                            }
                            for t in tools_result.tools
                        ],
                    },
                })

                # 3. tools/call
                wire.record("client->server", {
                    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "lookup_ingredient", "arguments": {"ingredient_name": "soy sauce"}},
                })
                call_result = await session.call_tool("lookup_ingredient", {"ingredient_name": "soy sauce"})
                result_text = call_result.content[0].text if call_result.content else "{}"
                wire.record("server->client", {
                    "jsonrpc": "2.0", "id": 3,
                    "result": {
                        "content": [{"type": "text", "text": json.loads(result_text)}],
                    },
                })

    wire.dump(PROJECT_ROOT / "wire.json")
    print("\nwire.json written.")

    # Run a real agent query to append tool-call messages
    print("\nRunning agent query for ingredient demo...")
    result = await run_agent(
        "What allergens does soy sauce contain, and what is its sodium per 100g?",
        api_key, model, wire=wire,
    )
    print("\nAnswer:", result["answer"])
    wire.dump(PROJECT_ROOT / "wire.json")
    print(f"Tool calls made: {[c['tool'] for c in result['tool_calls']]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 9 MCP Recipe Agent")
    parser.add_argument("request", nargs="?", default=None)
    parser.add_argument("--list-tools",      action="store_true", help="List discovered tools and exit (no API key needed)")
    parser.add_argument("--demo-ingredient", action="store_true", help="Demo query using ingredient DB")
    parser.add_argument("--wire-capture",    action="store_true", help="Capture raw JSON-RPC wire exchange to wire.json")
    args = parser.parse_args()

    # --list-tools needs no API key
    if args.list_tools:
        asyncio.run(list_tools_only())
        return

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        sys.exit("Missing GEMINI_API_KEY in .env")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if args.wire_capture:
        asyncio.run(capture_wire_exchange(api_key, model))
        return

    if args.demo_ingredient:
        req = "What allergens are in soy sauce, and what is its sodium content per 100 grams?"
    else:
        req = args.request or "Adapt the kimchi recipe for a soy-free vegan household and scale to 4 servings."

    print("\n" + "=" * 60)
    print("WEEK 9 MCP RECIPE AGENT")
    print("=" * 60)
    print(f"Request: {req}\n")

    result = asyncio.run(run_agent(req, api_key, model))

    print("\n" + "-" * 60)
    print("FINAL ANSWER:")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)
    print(f"iter={result['iterations']}  tok={result['input_tokens']+result['output_tokens']}  "
          f"cost=${result['cost_usd']:.5f}  latency={result['latency_secs']:.2f}s")
    print(f"Tools discovered: {result['tools_discovered']}")
    print(f"Tool calls made:  {[c['tool'] for c in result['tool_calls']]}")


if __name__ == "__main__":
    main()

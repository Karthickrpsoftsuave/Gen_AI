# Verdict — Keep or Kill the Orchestrator?

**Verdict: KILL**

## Sunk-cost bias acknowledgement

We spent a week building the orchestrator. The temptation is to declare it the winner or find reasons to keep it regardless of the numbers. **That is the sunk-cost bias.** We name it here and then ignore it when reading the numbers below.

## The numbers

| Metric | Single Agent | Orchestrator |
|--------|-------------|---------------|
| Pass rate | 70% | 70% |
| p50 latency | 1.54s | 4.00s |
| p99 latency | 32.43s | 33.92s |
| Cost per question | $0.00020 | $0.00087 |
| Token multiplier | 1.0x | 3.0x |

## Reasoning (max 10 lines)

1. **Pass rate** did not improve (70% → 70%) — the orchestrator is not more accurate.
2. **Cost per question** increased 4.4x ($0.00020 → $0.00087).
3. **p50 latency** rose from 1.54s to 4.00s; p99 from 32.43s to 33.92s — noticeably slower for users.
4. The context re-send multiplier is **3.0x** — every hand-off resends the full recipe context.
5. The dominant cost driver is `workers->orchestrator synthesis resend`, where context is re-sent to both workers.
6. The worker failure experiment showed the orchestrator **degraded gracefully** — good hygiene, but insufficient reason to keep a system that doesn't beat the baseline on quality or cost.
7. **Conclusion:** Kill. The single agent achieves the same pass rate at a fraction of the cost and latency.
   When to revisit: if tasks become truly parallelisable (e.g., 10 independent recipe checks), or if a worker can access a specialised database unavailable to the single agent.

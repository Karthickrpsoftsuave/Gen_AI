# Week 10 Race Table — Single Agent vs Orchestrator
_Same 10 Week-6 eval cases, same judge (judge_v2), same model._

## Arm Definitions
| Arm | Description |
|-----|-------------|
| **Single Agent** | Week-6 RAG-based single agent: retrieve 3 chunks, generate substitution answer |
| **Orchestrator** | Manager + substitution worker + allergen/nutrition worker; synthesises both outputs |

## 10 Eval Cases
| # | Case ID | Mode | Human Label |
|---|---------|------|-------------|
| 1 | S01 | ALLERGEN_SAFETY | UNACCEPTABLE |
| 2 | S02 | ALLERGEN_SAFETY | UNACCEPTABLE |
| 3 | S06 | FLAVOUR_PLAUSIBILITY | ACCEPTABLE |
| 4 | S07 | FLAVOUR_PLAUSIBILITY | ACCEPTABLE |
| 5 | S11 | METHOD_COMPATIBILITY | UNACCEPTABLE |
| 6 | S12 | METHOD_COMPATIBILITY | ACCEPTABLE |
| 7 | S16 | QUANTITY_SCALING | ACCEPTABLE |
| 8 | S18 | QUANTITY_SCALING | ACCEPTABLE |
| 9 | S21 | OOC_REFUSAL | ACCEPTABLE |
| 10 | S22 | OOC_REFUSAL | ACCEPTABLE |

## Metric Comparison
| Metric | Single Agent | Orchestrator |
|--------|-------------|---------------|
| **Pass rate** | 70% (7/10) | 70% (7/10) |
| **p50 latency** | 1.54s | 4.00s |
| **p99 latency** | 32.43s | 33.92s |
| **Total tokens (10 cases)** | 5,131 | 15,364 |
| **Cost per question** | $0.00020 | $0.00087 |

## Per-Case Results
| Case | Mode | Single Pass | Single Tok | Single Cost | Multi Pass | Multi Tok | Multi Cost | Multi Latency |
|------|------|-------------|------------|-------------|------------|-----------|------------|---------------|
| S01 | ALLERGEN_SAFETY | ✗ | 533 | $0.00018 | ✗ ⚠️500 | 1,152 | $0.00072 | 2.34s |
| S02 | ALLERGEN_SAFETY | ✗ | 451 | $0.00016 | ✗ | 1,517 | $0.00090 | 3.80s |
| S06 | FLAVOUR_PLAUSIBILITY | ✓ | 580 | $0.00020 | ✓ | 1,869 | $0.00117 | 14.04s |
| S07 | FLAVOUR_PLAUSIBILITY | ✓ | 451 | $0.00016 | ✓ | 1,448 | $0.00081 | 6.86s |
| S11 | METHOD_COMPATIBILITY | ✗ | 472 | $0.00017 | ✗ | 1,568 | $0.00099 | 3.35s |
| S12 | METHOD_COMPATIBILITY | ✓ | 599 | $0.00020 | ✓ | 1,660 | $0.00080 | 33.92s |
| S16 | QUANTITY_SCALING | ✓ | 525 | $0.00018 | ✓ | 1,737 | $0.00110 | 3.21s |
| S18 | QUANTITY_SCALING | ✓ | 564 | $0.00039 | ✓ | 1,550 | $0.00094 | 4.21s |
| S21 | OOC_REFUSAL | ✓ | 549 | $0.00019 | ✓ | 1,501 | $0.00059 | 3.42s |
| S22 | OOC_REFUSAL | ✓ | 407 | $0.00015 | ✓ | 1,362 | $0.00073 | 4.21s |

## Context Re-send Multiplier
```
multi_tokens / single_tokens = 15,364 / 5,131 = 3.0x

Dominant hand-off: workers->orchestrator synthesis resend
Share of all hand-off tokens: 56%
```

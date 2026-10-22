# agent_v1 (agent layer)

| run | n | EM | F1 | evidence R | evidence all | evidence sent | tool calls | kb first % | seq ok % | bridge 1st % | bridge later % | step limit % | bad cite % | tokens in | tokens out | latency p50/p95 s | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 300 | 60.7 [55.3, 66.0] | 77.3 [73.4, 81.0] | 95.2 | 90.7 | 88.7 | 2.24 | 100.0 | 88.7 | 11.9 | 35.7 | 0.0 | 0.0 | 5355 | 369 | 3.2 / 5.5 | 0 |
| dense_bge | 300 | 60.0 [54.3, 65.3] | 76.2 [72.2, 80.2] | 94.7 | 89.3 | 87.3 | 2.18 | 100.0 | 87.7 | 10.1 | 31.0 | 0.0 | 0.0 | 5458 | 372 | 15.6 / 30.5 | 0 |
| hybrid | 300 | 60.3 [54.7, 65.7] | 76.0 [72.0, 79.9] | 94.5 | 89.0 | 87.0 | 2.18 | 100.0 | 87.7 | 11.3 | 32.1 | 0.0 | 0.0 | 5193 | 372 | 3.3 / 5.6 | 0 |
| hybrid_bge | 300 | 60.3 [55.0, 65.7] | 76.5 [72.6, 80.5] | 95.8 | 91.7 | 88.7 | 2.13 | 100.0 | 87.7 | 10.7 | 30.4 | 0.0 | 0.0 | 5261 | 370 | 15.2 / 29.1 | 0 |
| hybrid_minilm | 300 | 61.3 [55.7, 67.0] | 77.9 [74.0, 81.8] | 94.7 | 89.3 | 86.7 | 2.19 | 100.0 | 89.0 | 9.5 | 29.8 | 0.0 | 0.0 | 5240 | 374 | 5.1 / 9.1 | 0 |
| no_retrieval | 300 | 42.7 [37.0, 48.3] | 55.1 [49.7, 60.0] | 0.0 | 0.0 | 0.0 | 0.00 | 0.0 | - | - | - | 0.0 | 0.0 | 841 | 260 | 2.4 / 4.0 | 0 |

EM/F1 in %, with 95% bootstrap confidence intervals. evidence all: every supporting paragraph retrieved (by title); evidence sent: every supporting sentence appears in a retrieved chunk.

## By question type

| run | type | n | EM | F1 |
|---|---|---|---|---|
| dense | bridge | 254 | 57.1 | 75.5 |
| dense | comparison | 46 | 80.4 | 87.3 |
| dense_bge | bridge | 254 | 55.9 | 73.8 |
| dense_bge | comparison | 46 | 82.6 | 89.5 |
| hybrid | bridge | 254 | 56.7 | 74.0 |
| hybrid | comparison | 46 | 80.4 | 87.3 |
| hybrid_bge | bridge | 254 | 56.3 | 74.2 |
| hybrid_bge | comparison | 46 | 82.6 | 89.5 |
| hybrid_minilm | bridge | 254 | 57.5 | 75.8 |
| hybrid_minilm | comparison | 46 | 82.6 | 89.5 |
| no_retrieval | bridge | 254 | 35.8 | 49.1 |
| no_retrieval | comparison | 46 | 80.4 | 88.2 |

## Served models

| run | model/fingerprint (questions) |
|---|---|
| dense | - |
| dense_bge | - |
| hybrid | - |
| hybrid_bge | - |
| hybrid_minilm | - |
| no_retrieval | deepseek-flash/e72d3c7f663d3e1383b00b3884cb4501 (300) |

As reported by the API for each LLM call. More than one entry, or entries that differ between runs, means the backend changed mid-evaluation.

## Per-question EM vs `dense`

| run | n shared | dense | run | wins | losses | ties |
|---|---|---|---|---|---|---|
| dense_bge | 300 | 60.7 | 60.0 | 8 | 10 | 282 |
| hybrid | 300 | 60.7 | 60.3 | 5 | 6 | 289 |
| hybrid_bge | 300 | 60.7 | 60.3 | 9 | 10 | 281 |
| hybrid_minilm | 300 | 60.7 | 61.3 | 8 | 6 | 286 |
| no_retrieval | 300 | 60.7 | 42.7 | 10 | 64 | 226 |

Only qids both runs answered without error. wins: questions where the run's EM beats `dense`.

# qwen3_l

| model version | gpu token/s | cpu token/s | model-parallel token/s |
|---|---|---|---|
| Qwen3-4B-Thinking-2507-FP8 (2× RTX 5060 Ti, parallel) | 13.25 (6.91 / 6.34 each) | 4.01 | 7.28 |
| Qwen3-4B-Thinking-2507-FP8 (RTX 4070 Ti) | 8.11 | — | — |
| Qwen3-30B-A3B-Thinking-2507-FP8 (2× RTX 5060 Ti, expert offload slice-LRU) | **2.31** (was 0.24 layer-LRU) | — | — |

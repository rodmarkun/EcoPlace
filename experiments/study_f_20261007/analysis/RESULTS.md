**Study F (Sol engine with validated selection): registered results, 9 October 2026**

Engine gpt-6.1-sol via Codex on Study E's 90 registered roots; GP and DeepSeek arms re-read from E's archives; E-side outcomes are Study E's sealed final grids. Statistics module sha256 `b42bd4fd…` (imported from Study E).

**Execution.** 90 registered roots. Evaluated on the final grid: sol_champion 90 (47 in F, 43 byte-copied from E), sol_validated 90 (46 in F, 44 byte-copied from E), gp_validated 90 (20 in F, 70 byte-copied from E), deepseek_validated 90 (15 in F, 75 byte-copied from E). Missing: none. Missing E observations: none. Re-read GP/DeepSeek search champions differing from E's: none.

## Confirmatory (registered, fixed sequence)

| | S1: Sol validated vs DeepSeek validated | P: Sol validated vs GP validated |
|---|---|---|
| Wins / losses / ties | **47 / 19 / 24** | **46 / 24 / 20** |
| Exact two-sided p | **0.00076** | **0.012** |
| Directional claim | **supported** (Sol over DeepSeek) | **supported** (Sol over GP) |
| Energy difference (right − left) | +0.153 [+0.091, +0.215], n = 79, mean relative saving +11.1% | +0.150 [+0.093, +0.208], n = 80, mean relative saving +12.4% |

Excluding the 12 E-01 rerun roots (78 roots): S1 40–16–22 (0.0018); P 39–20–19 (0.018). E-style comparison (descriptive), Sol champion vs E's GP champion: 46–19–25 (0.0011).

## By family and pooled (descriptive, unadjusted)

| | T (historical) | W (wide) | R (drift) | Pooled |
|---|---|---|---|---|
| Sol validated vs DeepSeek validated | 12–8–10 (0.50) | 14–5–11 (0.064) | **21–6–3 (0.0059)** | **47–19–24 (0.00076)** |
| Sol validated vs GP validated | 11–6–13 (0.33) | 11–14–5 (0.69) | **24–4–2 (0.00018)** | **46–24–20 (0.012)** |
| Sol champion vs E GP champion | **10–1–19 (0.012)** | 13–14–3 (1.00) | **23–4–3 (0.00031)** | **46–19–25 (0.0011)** |
| Sol champion vs E DeepSeek champion | 11–6–13 (0.33) | **19–7–4 (0.029)** | **22–5–3 (0.0015)** | **52–18–20 (5.8e-05)** |
| Sol validated vs Sol champion | 8–3–19 (0.23) | 5–2–23 (0.45) | 2–0–28 (0.50) | **15–5–70 (0.041)** |
| GP validated vs E GP champion | 9–3–18 (0.15) | 8–2–20 (0.11) | 3–6–21 (0.51) | 20–11–59 (0.15) |
| DeepSeek validated vs E DeepSeek champion | 8–4–18 (0.39) | **16–2–12 (0.0013)** | 4–1–25 (0.38) | **28–7–55 (0.00051)** |
| Sol validated vs oracle-best reference | 11–5–14 (0.21) | **11–1–18 (0.0063)** | **23–1–6 (3e-06)** | **45–7–38 (7e-08)** |
| GP validated vs oracle-best reference | 6–4–20 (0.75) | **20–2–8 (0.00012)** | **18–6–6 (0.023)** | **44–12–34 (2.1e-05)** |
| DeepSeek validated vs oracle-best reference | 12–6–12 (0.24) | 9–5–16 (0.42) | **12–3–15 (0.035)** | **33–14–43 (0.0079)** |
| Sol validated vs initialization best | **17–3–10 (0.0026)** | **11–1–18 (0.0063)** | **23–0–7 (2.4e-07)** | **51–4–35 (2e-11)** |
| GP validated vs initialization best | 9–2–19 (0.065) | **20–2–8 (0.00012)** | 17–7–6 (0.064) | **46–11–33 (3.3e-06)** |
| DeepSeek validated vs initialization best | **15–4–11 (0.019)** | 9–5–16 (0.42) | **10–2–18 (0.039)** | **34–11–45 (0.00082)** |
| Feasible: Sol validated / GP validated / DeepSeek validated / Sol champion / E GP champion / E DeepSeek champion / initialization best / oracle-best reference | 0.83 / 0.90 / 0.83 / 0.70 / 0.63 / 0.60 / 0.63 / 1.00 | 0.97 / 0.93 / 0.93 / 0.90 / 0.90 / 0.70 / 1.00 / 1.00 | 1.00 / 0.93 / 0.97 / 1.00 / 0.93 / 0.93 / 1.00 / 1.00 | 0.93 / 0.92 / 0.91 / 0.87 / 0.82 / 0.74 / 0.88 / 1.00 |
| Identical program: Sol validated = DeepSeek validated | 8 | 11 | 3 | 22 |
| Identical program: Sol validated = GP validated | 12 | 5 | 2 | 19 |
| Identical program: Sol champion = E GP champion | 18 | 2 | 3 | 23 |
| Identical program: Sol champion = E DeepSeek champion | 12 | 4 | 3 | 19 |
| Identical program: Sol validated = Sol champion | 18 | 22 | 28 | 68 |
| Identical program: GP validated = E GP champion | 17 | 20 | 21 | 58 |
| Identical program: DeepSeek validated = E DeepSeek champion | 15 | 12 | 25 | 52 |

Jointly-feasible energy differences (right − left normalized energy; positive favours the left program; mean, descriptive bootstrap 95% interval with E's analysis seeds):

- Sol validated vs DeepSeek validated: T +0.045 [-0.005, +0.094], n = 23, mean relative saving +4.6%; W +0.134 [+0.047, +0.226], n = 27, mean relative saving +9.3%; R +0.256 [+0.122, +0.384], n = 29, mean relative saving +17.8%; pooled +0.153 [+0.091, +0.215], n = 79, mean relative saving +11.1%.
- Sol validated vs GP validated: T +0.045 [-0.003, +0.094], n = 25, mean relative saving +4.8%; W +0.031 [-0.054, +0.120], n = 27, mean relative saving +1.9%; R +0.358 [+0.261, +0.453], n = 28, mean relative saving +29.4%; pooled +0.150 [+0.093, +0.208], n = 80, mean relative saving +12.4%.
- Sol champion vs E GP champion: T +0.075 [+0.022, +0.126], n = 19, mean relative saving +8.6%; W +0.045 [-0.048, +0.140], n = 25, mean relative saving +3.1%; R +0.344 [+0.244, +0.439], n = 28, mean relative saving +28.2%; pooled +0.169 [+0.108, +0.230], n = 72, mean relative saving +14.3%.
- Sol champion vs E DeepSeek champion: T +0.029 [-0.029, +0.091], n = 17, mean relative saving +2.6%; W +0.148 [+0.025, +0.275], n = 18, mean relative saving +9.7%; R +0.288 [+0.146, +0.420], n = 28, mean relative saving +20.3%; pooled +0.178 [+0.101, +0.255], n = 63, mean relative saving +12.5%.
- Sol validated vs Sol champion: T +0.000 [+0.000, +0.000], n = 18, mean relative saving +0.0%; W -0.013 [-0.041, +0.003], n = 27, mean relative saving -1.3%; R +0.000 [+0.000, +0.000], n = 30, mean relative saving +0.0%; pooled -0.005 [-0.015, +0.001], n = 75, mean relative saving -0.5%.
- GP validated vs E GP champion: T -0.012 [-0.034, +0.000], n = 18, mean relative saving -1.8%; W +0.005 [-0.010, +0.025], n = 26, mean relative saving +0.3%; R -0.008 [-0.016, -0.001], n = 26, mean relative saving -0.6%; pooled -0.004 [-0.013, +0.005], n = 70, mean relative saving -0.6%.
- DeepSeek validated vs E DeepSeek champion: T -0.011 [-0.033, +0.000], n = 17, mean relative saving -1.7%; W +0.008 [-0.019, +0.031], n = 20, mean relative saving +0.4%; R +0.027 [+0.000, +0.080], n = 27, mean relative saving +2.1%; pooled +0.011 [-0.008, +0.038], n = 64, mean relative saving +0.6%.
- Sol validated vs oracle-best reference: T +0.088 [+0.049, +0.128], n = 25, mean relative saving +10.5%; W +0.162 [+0.088, +0.243], n = 29, mean relative saving +12.3%; R +0.400 [+0.299, +0.496], n = 30, mean relative saving +31.3%; pooled +0.225 [+0.172, +0.279], n = 84, mean relative saving +18.6%.
- GP validated vs oracle-best reference: T +0.039 [+0.013, +0.069], n = 27, mean relative saving +4.7%; W +0.149 [+0.097, +0.210], n = 28, mean relative saving +11.3%; R +0.070 [+0.032, +0.116], n = 28, mean relative saving +5.5%; pooled +0.087 [+0.061, +0.116], n = 83, mean relative saving +7.2%.
- DeepSeek validated vs oracle-best reference: T +0.056 [+0.018, +0.096], n = 25, mean relative saving +6.5%; W +0.040 [+0.005, +0.094], n = 28, mean relative saving +3.0%; R +0.135 [+0.054, +0.224], n = 29, mean relative saving +10.5%; pooled +0.079 [+0.044, +0.118], n = 82, mean relative saving +6.7%.
- Sol validated vs initialization best: T +0.100 [+0.049, +0.150], n = 16, mean relative saving +12.0%; W +0.162 [+0.088, +0.243], n = 29, mean relative saving +12.3%; R +0.400 [+0.300, +0.496], n = 30, mean relative saving +31.3%; pooled +0.244 [+0.186, +0.303], n = 75, mean relative saving +19.9%.
- GP validated vs initialization best: T -0.001 [-0.004, +0.000], n = 18, mean relative saving -0.2%; W +0.149 [+0.097, +0.210], n = 28, mean relative saving +11.3%; R +0.070 [+0.032, +0.116], n = 28, mean relative saving +5.5%; pooled +0.083 [+0.055, +0.113], n = 74, mean relative saving +6.3%.
- DeepSeek validated vs initialization best: T +0.045 [-0.004, +0.097], n = 16, mean relative saving +5.0%; W +0.040 [+0.005, +0.094], n = 28, mean relative saving +3.0%; R +0.135 [+0.055, +0.224], n = 29, mean relative saving +10.5%; pooled +0.079 [+0.040, +0.123], n = 73, mean relative saving +6.5%.

Families are related variants; per-family and descriptive contrasts are unadjusted. Non-significance does not establish equivalence.

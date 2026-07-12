# Qwen3 MoE — что именно в этой модели

> Конкретика MoE-блока в **Qwen/Qwen3-30B-A3B-Thinking-2507-FP8** на фоне
> классической formulation из `MoE.md`. Числа взяты из `config.json` этой
> модели. Общая философия та же (Shazeer → GShard), но есть три ключевых
> отличия, разобранных ниже.

$$
\begin{gather*}
d_{model} = 2048 \text{ (hidden size)} \\
L = 48 \text{ (num\_hidden\_layers)} \\
E = 128 \text{ (num\_experts)} \\
k = 8 \text{ (num\_experts\_per\_tok — активных на токен)} \\
d_{ff}^{(\text{expert})} = 768 \text{ (moe\_intermediate\_size — у каждого эксперта МАЛЕНЬКИЙ)} \\
d_{ff}^{(\text{shared})} = 1408 \text{ (intermediate\_size — общий shared expert)} \\
\text{decoder\_sparse\_step} = 1 \quad (\text{каждый слой — MoE}) \\
A = 3\text{B активных параметров} \quad (\text{отсюда «A3B» в имени: 3B активных из 30B})
\end{gather*}
$$

---

# ОТЛИЧИЕ 1: FINE-GRAINED EXPERTS (много маленьких)

В ванильном MoE (Shazeer) — **мало больших** экспертов: $E{=}8{-}64$, каждый
размером с обычную FFN ($d_{ff}\sim 4 d_{model}$).

В Qwen3 — **много маленьких**: $E=128$, но каждый эксперт крошечный
($d_{ff}^{(\text{expert})}=768 \ll 2048 = d_{model}$).

$$
\text{FFN}_e(\mathbf{x}) = W_2^{(e)} \cdot \sigma\!\big(W_1^{(e)}\,\mathbf{x}\big)
$$

$$
W_1^{(e)} \in \mathbb{R}^{768 \times 2048}, \quad W_2^{(e)} \in \mathbb{R}^{2048 \times 768}
$$

$$
\text{параметров одного эксперта} \approx 2 \cdot 768 \cdot 2048 \approx 3.1\text{M}
$$

$$
\text{всего экспертных параметров} = 48 \text{ слоёв} \times 128 \times 3.1\text{M} \approx 19\text{B}
$$

**Зачем:** при фиксированном числе активных параметров $A = k \cdot d_{ff}^{(\text{expert})}$
бóльшее $E$ даёт больше **комбинаторной гибкости** — из $\binom{128}{8}$
возможных комбинаций роутер подбирает специализированный «ансамбль» под токен,
а не зависит от $k$ крупных блоков. Это закон GShard / DeepSeek-V2 (fine-grained
segmentation).

---

# ОТЛИЧИЕ 2: SHARED EXPERT (общий всегда-активный эксперт)

Помимо $E$ роутируемых экспертов, в каждом слое есть **один общий expert**,
который активен для **всех** токенов без маршрутизации.

$$
\text{MoE}_{\text{Qwen3}}(\mathbf{x}) = \underbrace{\text{FFN}_{\text{shared}}(\mathbf{x})}_{\text{shared expert, всегда активен}} + \underbrace{\sum_{e \in \text{top-}k} G_e(\mathbf{x}) \cdot \text{FFN}_e(\mathbf{x})}_{\text{роутируемые эксперты}}
$$

$$
\text{FFN}_{\text{shared}}(\mathbf{x}) = W_2^{(\text{shared})} \cdot \sigma\!\big(W_1^{(\text{shared})}\,\mathbf{x}\big), \quad W_1^{(\text{shared})} \in \mathbb{R}^{1408 \times 2048}
$$

$$
\text{параметров shared (на слой)} \approx 2 \cdot 1408 \cdot 2048 \approx 5.8\text{M}, \quad \text{на все слои} \approx 0.3\text{B}
$$

**Зачем:**(shared expert усваивает «общее» знание, нужное почти каждому токену
(синтаксис, пунктуация, базовая семантика), а роутируемые эксперты
специализируются на специфическом. Так устраняется дублирование: в ванильном
MoE каждый из $k$ активных экспертов вынужден переучивать это «общее» знание
независимо. Идея пришла из DeepSeekMoE (2024).

---

# ОТЛИЧИЕ 3: СТЕКИРОВАННЫЕ ТЕНЗОРЫ (не ModuleList)

Это **реализационное** отличие, важное для offload'а. В ванильном PyTorch MoE
эксперты — это `nn.ModuleList([FFN_1, FFN_2, ..., FFN_E])`, и каждого можно
переносить на CPU поодиночке.

В Qwen3-MoE все 128 экспертов **упакованы в два больших тензора** внутри одного
модуля `Qwen3MoeExperts`:

$$
W_{\text{gate\_up}}^{(\text{all})} \in \mathbb{R}^{E \times 2\,d_{ff}^{(\text{expert})} \times d_{model}} = \mathbb{R}^{128 \times 1536 \times 2048}
$$

$$
W_{\text{down}}^{(\text{all})} \in \mathbb{R}^{E \times d_{model} \times d_{ff}^{(\text{expert})}} = \mathbb{R}^{128 \times 2048 \times 768}
$$

$$
W_{\text{gate\_up}}^{(\text{all})} = \begin{bmatrix} [W_1^{(1)}\ \|\ \text{gate}^{(1)}] \\ [W_1^{(2)}\ \|\ \text{gate}^{(2)}] \\ \vdots \\ [W_1^{(128)}\ \|\ \text{gate}^{(128)}] \end{bmatrix}
$$

(Конкатенация gate- и up-проекций как у SwiGLU, и стэк по первой оси.)

**Следствие для инференса:** матрица выбранных $k$ экспертов вычисляется через
один **batched matmul** (`w8a8_fp8_matmul_batched`), а не $k$ отдельных умножений.
Это эффективно на GPU (особенно в FP8), но **нельзя выгрузить отдельных
экспертов** — они в одном тензоре. Поэтому offload у нас **послойный**: весь
`mlp.experts` модуль слоя переезжает на CPU целиком.

---

# ПОЛНЫЙ ПРОХОД ОДНОГО MoE-СЛОЯ Qwen3

$$
\mathbf{h}_0 = \mathbf{x}
$$

$$
\mathbf{h}_1 = \text{RMSNorm}\big(\mathbf{h}_0 + \text{Attention}(\mathbf{h}_0)\big)
$$

$$
\text{logits} = \mathbf{h}_1\, W_g, \qquad W_g \in \mathbb{R}^{2048 \times 128}
$$

$$
\text{top-}k\text{ experts},\ G = \text{softmax}\big(\text{top-}k(\text{logits})\big) \in \mathbb{R}^{8}
$$

$$
\mathbf{h}_{\text{routed}} = \sum_{e \in \text{top-}8} G_e \cdot \text{FFN}_e(\mathbf{h}_1)
$$

$$
\mathbf{h}_{\text{shared}} = \text{FFN}_{\text{shared}}(\mathbf{h}_1)
$$

$$
\mathbf{h}_2 = \text{RMSNorm}\big(\mathbf{h}_1 + \mathbf{h}_{\text{routed}} + \mathbf{h}_{\text{shared}}\big)
$$

$$
\text{router logits}\ \mathbf{r} = \text{softmax}(\text{logits}) \quad (\text{для auxiliary load-balancing loss при обучении})
$$

$$
\text{routed\_scaling}: \quad G \leftarrow G \cdot \gamma \quad (\text{routed\_scaling\_factor, ре-нормировка перед смешиванием})
$$

---

# АКТИВНЫЕ ПАРАМЕТРЫ (почему «30B / A3B»)

Имя `Qwen3-30B-A3B` читается как «30B всего, 3B активных».

$$
\text{active per token} \approx \underbrace{\text{attention} + \text{embed}}_{\text{общие}} + \underbrace{k \cdot d_{ff}^{(\text{expert})}}_{8 \times 768} + \underbrace{d_{ff}^{(\text{shared})}}_{1408}
$$

$$
\text{total params} \approx 30\text{B}, \qquad \text{active params} \approx 3\text{B}
$$

То есть на каждом токене «работает» лишь ~10% весов модели — отсюда
декодинг-пропускная способность как у 3B-модели при качестве ближе к 30B.

---

# KV-CACHE: НЕ ЗАВИСИТ ОТ MoE

Важно: **KV-кэш в MoE такой же, как в dense-модели** — его выделяет только
attention, а не эксперты. Эксперты stateless (только веса, без состояния).

$$
\text{KV per token} = 2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot \text{dtype\_bytes}
$$

$$
= 2 \cdot 48 \cdot 4 \cdot 128 \cdot 2 = 98\,304 \text{ байт/токен} \approx 96\text{ KiB/токен (в FP8-модели — bf16 KV)}
$$

Поэтому memory-бюджет демки считает контекст одинаково для 4B и 30B
(отличается только числом слоёв / kv-heads). Напротив, **веса** MoE — основная
статья расхода VRAM.

---

# СВОДКА: VANILLA MoE vs Qwen3 MoE

| аспект | vanilla (Shazeer 2017) | Qwen3-30B-A3B |
|---|---|---|
| эксперты | мало больших ($E{=}8{-}64$, $d_{ff}{=}4d$) | много мелких ($E{=}128$, $d_{ff}^{(e)}{=}768$) |
| shared expert | нет | есть, всегда активен ($d_{ff}^{(sh)}{=}1408$) |
| активных/токен | $k \cdot d_{ff}$ | $8\cdot768 + 1408 \approx 7.5$k нейронов |
| роутер | noisy top-k | top-k + `routed_scaling_factor` |
| хранение весов | `ModuleList[FFN]` | два стекированных тензора |
| вычисление экспертов | $k$ отдельных matmul | 1 batched matmul (эффективно в FP8) |
| load balancing | $L_{\text{aux}} = \alpha\,\text{CV}^2$ | тот же принцип, `output_router_logits` |
| цель fine-grained | — | больше $\binom{E}{k}$ комбинаций при том же активном бюджете |
| цель shared | — | убрать дублирование «общего» знания между экспертами |

---

# ИСТОЧНИКИ

- DeepSeekMoE (Bi et al., 2024) — fine-grained segmentation + shared expert:
  <https://arxiv.org/abs/2401.06066>
- Qwen3 technical report (2025): <https://arxiv.org/abs/2505.09388>
- config модели: `config.json` в `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`.

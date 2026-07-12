# Mixture of Experts (MoE)

> Математический конспект по классическим работам: **Shazeer et al., 2017,
> "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
> Layer"** (ICLR 2017, Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz,
> Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean) — работе, которая ввела
> sparsely-gated MoE-слой для глубоких сеток. Стиль конспекта повторяет
> `AttentionIsAllYouNeed.md`: сначала обозначения, потом формулы по блокам.

$$
\begin{gather*}
N\text{: batch} \\
S\text{: длина входной последовательности} \\
d_{model}\text{: размер скрытого представления (hidden size)} \\
E\text{: число экспертов (number of experts)} \\
k\text{: число активных экспертов на токен (top-k)} \\
\text{FFN}_e(\cdot)\text{: e-й эксперт — полносвязная сеть (feed-forward)} \\
W_g\text{: матрица маршрутизации (gate / router), } W_g \in \mathbb{R}^{E \times d_{model}} \\
\mathbf{x}\text{: входной вектор токена, } \mathbf{x} \in \mathbb{R}^{d_{model}}
\end{gather*}
$$

---

# Идея MoE: conditional computation

В обычном трансформере каждый токен проходит через одну и ту же FFN:
$$
\text{FFN}(\mathbf{x}) = W_2 \cdot \sigma(W_1 \mathbf{x})
$$
Все параметры активны на каждом токене — стоимость растёт линейно с шириной.

MoE заменяет единственную FFN на **набор из $E$ экспертных FFN** и для каждого
токена активирует только $k \ll E$ из них. Параметров много (как у огромной
модели), но FLOPs на токен — как у маленькой (только $k$ экспертов).

$$
\text{MoE}(\mathbf{x}) = \sum_{e=1}^{E} G_e(\mathbf{x}) \cdot \text{FFN}_e(\mathbf{x})
$$

$$
\text{FFN}_e(\mathbf{x}) = W_2^{(e)} \cdot \sigma\!\big(W_1^{(e)}\,\mathbf{x}\big), \qquad W_1^{(e)} \in \mathbb{R}^{d_{ff} \times d_{model}},\ W_2^{(e)} \in \mathbb{R}^{d_{model} \times d_{ff}}
$$

$$
G(\mathbf{x}) \in \mathbb{R}^{E}, \qquad \|\,G(\mathbf{x})\,\|_0 = k \quad (\text{только } k \text{ ненулевых компонент})
$$

---

# SPARSELY-GATED ROUTER (Shazeer 2017)

Ключевой вклад Shazeer et al. — **гейт (маршрутизатор) с top-k маской** и
**шумом** для exploration'а.

## Логиты маршрутизации

$$
\text{logits}(\mathbf{x}) = \mathbf{x}\, W_g \in \mathbb{R}^{E}, \qquad W_g \in \mathbb{R}^{d_{model} \times E}
$$

$$
\text{logits}_e(\mathbf{x}) = \mathbf{x} \cdot W_g[:,\,e]
$$

## Noisy top-k гейтинг

Шум поощряет роутер исследовать разных экспертов (без него он быстро
сходится к нескольким «любимым» и остальные отмирают):

$$
H_e(\mathbf{x}) \sim \mathcal{N}(0,\,1), \qquad \text{Softplus}(\mathbf{x}\,W_{noise})_e
$$

$$
\text{noisy\_logits}(\mathbf{x}) = \mathbf{x}\,W_g + \text{Softplus}(\mathbf{x}\,W_{noise}) \odot H
$$

$$
\text{KeepTopK}(\mathbf{v},\,k)_i =
\begin{cases}
v_i & \text{если } v_i \in \text{top-}k(\mathbf{v}) \\
-\infty & \text{иначе}
\end{cases}
$$

$$
G(\mathbf{x}) = \text{softmax}\!\big(\text{KeepTopK}\big(\text{noisy\_logits}(\mathbf{x}),\,k\big)\big)
$$

$$
\begin{gather*}
G(\mathbf{x}) \in \mathbb{R}^{E}, \quad \sum_{e} G_e(\mathbf{x}) = 1, \quad G_e(\mathbf{x}) > 0 \text{ только для } k \text{ экспертов}
\end{gather*}
$$

## Итоговый MoE-слой

$$
\text{MoE}(\mathbf{x}) = \sum_{e=1}^{E} G_e(\mathbf{x}) \cdot \text{FFN}_e(\mathbf{x})
$$

На практике считается только для $k$ активных (остальные $G_e = 0$):
$$
\text{MoE}(\mathbf{x}) = \sum_{e \in \text{top-}k} G_e(\mathbf{x}) \cdot \text{FFN}_e(\mathbf{x})
$$

---

# LOAD BALANCING (балансировка нагрузки)

**Проблема collapse:** роутер быстро сводится к 1–2 «любимым» экспертам →
остальные не получают градиента → отмирают → модель фактически не MoE.

## Важность (importance) и загрузка (load)

$$
\text{Importance}(\text{batch})_e = \sum_{\mathbf{x} \in \text{batch}} G_e(\mathbf{x}) \qquad \in \mathbb{R}^{E}
$$

$$
\text{Load}(\text{batch})_e = \sum_{\mathbf{x} \in \text{batch}} \mathbb{1}\!\big[e \in \text{top-}k(\mathbf{x})\big] \qquad \in \mathbb{R}^{E}
$$

(Importance — сумма весов; Load — сколько раз эксперт попал в top-k.)

## Loss баланса (вариационный коэффициент)

$$
L_{\text{aux}} = \alpha \cdot \Big(\text{CV}\big(\text{Importance}\big)^2 + \text{CV}\big(\text{Load}\big)^2\Big)
$$

$$
\text{CV}(\mathbf{v}) = \frac{\text{std}(\mathbf{v})}{\text{mean}(\mathbf{v})} \quad (\text{коэффициент вариации})
$$

$$
L_{\text{total}} = L_{\text{task}} + L_{\text{aux}}
$$

Чем равномернее Importance/Load по экспертам, тем меньше CV → меньше штраф.
Это выталкивает роутер из collapse в равномерное распределение.

---

# BATCHED DISPATCH (как это считается эффективно)

Для батча токенов маршрутизация одинакова по всем экспертам — нужен **dispatch**:
разложить токены по ведёркам экспертов, посчитать, собрать обратно.

$$
X \in \mathbb{R}^{B \times d_{model}}, \qquad B = N \cdot S \quad (\text{все токены батча})
$$

$$
\text{Dispatch}(X) \to \big\{(X_e,\, w_e)\big\}_{e=1}^{E}, \qquad X_e \in \mathbb{R}^{B_e \times d_{model}}
$$

где $B_e = |\{\mathbf{x} : e \in \text{top-}k(\mathbf{x})\}|$ — сколько токенов досталось
эксперту $e$, $w_e \in \mathbb{R}^{B_e}$ — их гейт-веса.

$$
Y_e = \text{FFN}_e(X_e) \in \mathbb{R}^{B_e \times d_{model}}
$$

$$
\text{Combine}: \quad Y[\mathbf{x}] = \sum_{e \in \text{top-}k(\mathbf{x})} G_e(\mathbf{x}) \cdot Y_e[\mathbf{x}]
$$

$$
\text{FLOPs}_{\text{MoE}} \approx \sum_{e=1}^{E} B_e \cdot \text{FLOPs}(\text{FFN}_e) \quad \approx \quad B \cdot k \cdot \text{FLOPs}(\text{FFN})
$$

Параметров $E\times$ больше, а FLOPs — лишь в $k$ раз (если нагрузка сбалансирована).

---

# MoE ВНУТРИ ТРАНСФОРМЕР-СЛОЯ

MoE-слой ставится **вместо FFN** в (обычно не во всех) слоях энкодера/декодера.

$$
\text{dense FFN layer}: \quad A_2 = \text{layer\_norm} \circ (\text{FFN} + I)
$$

$$
\text{MoE layer}: \quad A_2 = \text{layer\_norm} \circ (\text{MoE} + I)
$$

Параметр `decoder_sparse_step` / `moe_layer_freq` управляет, какие слои MoE
(например каждый 2-й). Attention-блок остаётся без изменений.

$$
\text{encoder\_layer}^{\text{MoE}} = A_2^{\text{MoE}} \circ A_1, \qquad A_1 = \text{layer\_norm} \circ (\text{self\_attention} + I)
$$

---

# СВОДКА ОТЛИЧИЙ ОТ ВАНИЛЬНОГО FFN

| | dense FFN | MoE (sparsely-gated) |
|---|---|---|
| активных параметров на токен | все | $k$ из $E$ экспертов |
| FLOPs/token | $\propto d_{ff}$ | $\propto k \cdot d_{ff}^{(\text{expert})}$ |
| доп. параметры | — | $W_g$, $W_{noise}$ (роутер) |
| доп. loss | — | $L_{\text{aux}}$ (load balancing) |
| доп. операция | — | top-k выбор + dispatch/combine |
| риск collapse | — | роутер сводится к 1–2 экспертам |

---

# КЛАССИЧЕСКИЕ РАБОТЫ (для углубления)

- **Shazeer et al., 2017** — "Outrageously Large Neural Networks":
  sparsently-gated MoE-слой, noisy top-k, load-balancing loss.
  <https://arxiv.org/abs/1701.06538>
- **GShard (Lepikhin et al., 2020)** — масштабирование MoE до 600B, top-2
  роутинг, auxiliary loss, распределённое dispatch.
  <https://arxiv.org/abs/2006.16668>
- **Switch Transformer (Fedus et al., 2021)** — top-1 (один эксперт на токен),
  упрощённый load balancing, масштабирование на триллион параметров.
  <https://arxiv.org/abs/2101.03961>

Конкретные отличия MoE **в Qwen3** (shared expert, тонкий роутинг, FP8-формат)
разобраны в соседнем файле: **`Qwen3MoE.md`**.

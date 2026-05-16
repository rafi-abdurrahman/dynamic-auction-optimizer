# Regret Bound Analysis for the Bandit-Style Auction Systems

## 1. Context from the Project

The project studies an inventory-constrained dynamic bidding problem for one-shot auctions.

At each time step

$$
t=1,2,\dots,T,
$$

there are products indexed by

$$
i=1,2,\dots,N.
$$

The decision variable is

$$
x_i(t)\in\{0,1\},
$$

where

$$
x_i(t)=
\begin{cases}
1, & \text{we bid for product } i \text{ at time } t,\\
0, & \text{we do not bid for product } i \text{ at time } t.
\end{cases}
$$

The inventory dynamics are

$$
s_i(t+1)=s_i(t)+q_i(t)x_i(t)-d_i(t),
$$

where:

- $s_i(t)$ is the inventory of product $i$,
- $q_i(t)$ is the amount of product $i$ offered in the auction,
- $d_i(t)$ is the depletion or demand of product $i$,
- $b_i(t)$ is the winning bid cost for product $i$.

The hard inventory constraint is

$$
s_i(t)\ge \alpha_i,
\qquad \forall i,t.
$$

The global objective is

$$
\min_{\{x_i(t)\}}
\sum_{t=1}^{T}\sum_{i=1}^{N} b_i(t)x_i(t)
$$

subject to

$$
s_i(t)\ge \alpha_i,
\qquad \forall i,t.
$$

The proposal contains two bandit-style settings:

1. **All-Seeing Bandit-Style Problem**
2. **Partially Blind Bandit-Style Problem**

The main difference is whether the winning bid cost $b_i(t)$ is known before the decision.

---

## 2. Lyapunov Deficit State

For each product $i$, define the deficit state

$$
H_i(t)=\beta_i-s_i(t),
$$

where

$$
\beta_i>\alpha_i.
$$

The value $\beta_i$ is a soft inventory target. It is set above the hard safety threshold $\alpha_i$ to create a safety buffer.

Using the inventory dynamics,

$$
s_i(t+1)=s_i(t)+q_i(t)x_i(t)-d_i(t),
$$

we get

$$
\begin{aligned}
H_i(t+1)
&=\beta_i-s_i(t+1)\\
&=\beta_i-\left(s_i(t)+q_i(t)x_i(t)-d_i(t)\right)\\
&=H_i(t)-q_i(t)x_i(t)+d_i(t).
\end{aligned}
$$

Thus,

$$
\boxed{
H_i(t+1)=H_i(t)-q_i(t)x_i(t)+d_i(t).
}
$$

Large $H_i(t)$ means inventory is below the soft target $\beta_i$, so product $i$ is urgent.

---

## 3. Lyapunov Drift-Plus-Penalty Rule

Define the Lyapunov function

$$
L(H(t))=\frac{1}{2}\sum_{i=1}^{N}H_i(t)^2.
$$

The one-step conditional drift is

$$
\Delta(H(t))=
\mathbb{E}\left[L(H(t+1))-L(H(t))\mid H(t)\right].
$$

From

$$
H_i(t+1)=H_i(t)-q_i(t)x_i(t)+d_i(t),
$$

we obtain

$$
H_i(t+1)^2-H_i(t)^2
=
\left(d_i(t)-q_i(t)x_i(t)\right)^2
+2H_i(t)\left(d_i(t)-q_i(t)x_i(t)\right).
$$

Assume bounded supply and demand:

$$
0\le q_i(t)\le q_i^{\max},
$$

$$
0\le d_i(t)\le d_i^{\max}.
$$

Then there exists a finite constant $B>0$ such that

$$
\frac{1}{2}
\mathbb{E}
\left[
\sum_{i=1}^{N}
\left(d_i(t)-q_i(t)x_i(t)\right)^2
\mid H(t)
\right]
\le B.
$$

Therefore,

$$
\Delta(H(t))
\le
B+
\sum_{i=1}^{N}H_i(t)\mathbb{E}[d_i(t)\mid H(t)]
-
\mathbb{E}
\left[
\sum_{i=1}^{N}H_i(t)q_i(t)x_i(t)
\mid H(t)
\right].
$$

The expected cost penalty is

$$
J(x(t))=
\mathbb{E}
\left[
\sum_{i=1}^{N}b_i(t)x_i(t)
\mid H(t)
\right].
$$

The drift-plus-penalty objective is

$$
\Delta(H(t))+VJ(x(t)),
$$

where

$$
V>0
$$

controls the cost-safety tradeoff.

Adding the penalty gives

$$
\begin{aligned}
\Delta(H(t))+VJ(x(t))
&\le
B+
\sum_{i=1}^{N}H_i(t)\mathbb{E}[d_i(t)\mid H(t)]\\
&\quad+
\mathbb{E}
\left[
\sum_{i=1}^{N}
\left(Vb_i(t)-H_i(t)q_i(t)\right)x_i(t)
\mid H(t)
\right].
\end{aligned}
$$

The first two terms are independent of the decision. Therefore, the online rule minimizes

$$
\sum_{i=1}^{N}\left(Vb_i(t)-H_i(t)q_i(t)\right)x_i(t).
$$

Since $x_i(t)\in\{0,1\}$, the rule is

$$
\boxed{
x_i(t)=1
\iff
H_i(t)q_i(t)>Vb_i(t).
}
$$

Equivalently, for $q_i(t)>0$,

$$
\boxed{
x_i(t)=1
\iff
\frac{b_i(t)}{q_i(t)}<\frac{H_i(t)}{V}.
}
$$

This means we buy product $i$ if its unit price is low relative to its inventory urgency.

---

## 4. Regret Definition

Let $\pi$ be the online policy.

Its cumulative cost is

$$
C_T^{\pi}
=
\sum_{t=1}^{T}\sum_{i=1}^{N}b_i(t)x_i^{\pi}(t).
$$

For a comparator policy $\pi^*$, regret is

$$
\boxed{
\operatorname{Reg}_T(\pi)=C_T^{\pi}-C_T^{\pi^*}.
}
$$

The comparator can be chosen in different ways:

1. **Offline clairvoyant benchmark:** knows all future prices, supplies, and demands.
2. **Best stationary feasible policy:** knows the distribution but not future samples.
3. **Current-information oracle:** observes current $b_i(t)$, $q_i(t)$, and $H_i(t)$ before deciding.

For Lyapunov systems, the cleanest theoretical benchmark is usually the best stationary feasible policy.

Let its expected per-round cost be

$$
J^*.
$$

Then

$$
C_T^{\pi^*}=TJ^*.
$$

So stationary-policy regret is

$$
\boxed{
\operatorname{Reg}_T^{\mathrm{stat}}(\pi)
=
\sum_{t=1}^{T}\sum_{i=1}^{N}b_i(t)x_i^{\pi}(t)-TJ^*.
}
$$

---

# 5. All-Seeing Bandit-Style Problem

## 5.1 Setting

In the All-Seeing setting, the other buyers' bids are known at each auction. Therefore, the winning bid cost

$$
b_i(t)
$$

is known before we decide.

The Lyapunov rule is

$$
x_i(t)=1
\iff
H_i(t)q_i(t)>Vb_i(t).
$$

Equivalently,

$$
x_i(t)=1
\iff
\frac{b_i(t)}{q_i(t)}<\frac{H_i(t)}{V}.
$$

Although this is called bandit-style, it is not a pure statistical bandit problem if all current costs are already known before action. There is no bid-prediction regret. The main regret comes from online control.

---

## 5.2 Assumptions

Assume bounded variables:

$$
0\le b_i(t)\le b_i^{\max},
$$

$$
0\le q_i(t)\le q_i^{\max},
$$

$$
0\le d_i(t)\le d_i^{\max}.
$$

Assume the environment is stationary over the horizon $T$.

Assume there exists a feasible stationary policy with slack. That is, for some $\epsilon_i>0$,

$$
\mathbb{E}[q_i(t)x_i(t)-d_i(t)]\ge \epsilon_i.
$$

This means inventory can be replenished faster than it is depleted, at least in expectation.

---

## 5.3 Lyapunov Regret Bound

The standard Lyapunov drift-plus-penalty result gives

$$
\boxed{
\frac{1}{T}
\mathbb{E}
\left[
\sum_{t=1}^{T}\sum_{i=1}^{N}b_i(t)x_i(t)
\right]
\le
J^*+O\left(\frac{1}{V}\right).
}
$$

Multiplying by $T$,

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T^{\mathrm{Lyap}}]
=
O\left(\frac{T}{V}\right).
}
$$

For finite-horizon analysis, it is common to include a transient queue term:

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T^{\mathrm{Lyap}}]
=
O\left(\frac{T}{V}+V\right).
}
$$

Choosing

$$
V=\Theta(\sqrt{T})
$$

gives

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T^{\mathrm{Lyap}}]
=O(\sqrt{T}).
}
$$

Therefore,

$$
\boxed{
\frac{\mathbb{E}[\operatorname{Reg}_T^{\mathrm{Lyap}}]}{T}
=O\left(\frac{1}{\sqrt{T}}\right)
\to 0.
}
$$

So the All-Seeing Lyapunov system has vanishing average regret under stationarity.

---

## 5.4 Queue Bound

The corresponding deficit size satisfies

$$
\boxed{
\mathbb{E}[H_i(t)]=O(V).
}
$$

Thus, the Lyapunov tradeoff is

$$
\boxed{
\text{cost gap}=O\left(\frac{1}{V}\right),
\qquad
\text{inventory-deficit size}=O(V).
}
$$

With

$$
V=\Theta(\sqrt{T}),
$$

we get

$$
\boxed{
\text{regret}=O(\sqrt{T}),
\qquad
\text{deficit size}=O(\sqrt{T}).
}
$$

---

# 6. All-Seeing Case with Explicit MAB Learning

If the system does not know the expected unit cost of each product arm, then a true MAB learning term appears.

Define the unit cost of arm $i$ as

$$
c_i(t)=\frac{b_i(t)}{q_i(t)}.
$$

Let

$$
\mu_i=\mathbb{E}[c_i(t)]
$$

be the expected unit cost of arm $i$.

The best arm is

$$
i^*=\arg\min_i \mu_i.
$$

The suboptimality gap of arm $i$ is

$$
\Delta_i=\mu_i-\mu_{i^*}.
$$

For UCB-style learning, the gap-dependent regret is

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T^{\mathrm{MAB}}]
=
O\left(
\sum_{i:\Delta_i>0}\frac{\log T}{\Delta_i}
\right).
}
$$

A gap-free form is

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T^{\mathrm{MAB}}]
=
O\left(\sqrt{NT\log T}\right).
}
$$

Combining MAB learning with Lyapunov control gives

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T]
=
O\left(\frac{T}{V}+V\right)
+O\left(\sqrt{NT\log T}\right).
}
$$

With

$$
V=\Theta(\sqrt{T}),
$$

this becomes

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T]
=
O(\sqrt{T})+O\left(\sqrt{NT\log T}\right).
}
$$

Usually, for $N\ge 2$, the MAB exploration term dominates:

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T]
=
O\left(\sqrt{NT\log T}\right).
}
$$

However, this term should only be included if the algorithm actually has to learn arm values. In the original All-Seeing formulation, current bids are observed, so this term is not necessary.

---

# 7. Partially Blind Bandit-Style Problem

## 7.1 Setting

In the Partially Blind setting, competitors' bid amounts are not known directly. However, competitors' inventory conditions are known.

For product $i$ and competitor $o$, let

$$
K_o(t)
$$

be the context vector representing competitor $o$'s inventory condition.

The proposal assumes a linear model:

$$
\mathbb{E}[C_{i,o}(t)\mid K_o(t)]
=K_o(t)^\top\theta_{i,o}^*.
$$

The estimated competitor bid is

$$
\widehat C_{i,o}(t)=K_o(t)^\top\widehat\theta_{i,o}(t).
$$

The predicted market price for product $i$ is the maximum predicted competitor bid:

$$
\widehat P_i(t)=\max_o K_o(t)^\top\widehat\theta_{i,o}(t).
$$

The actual bid submitted by our system is

$$
\widehat b_i(t)=\widehat P_i(t)+\varepsilon,
$$

where $\varepsilon>0$ is a small amount added to guarantee winning.

The Lyapunov rule becomes

$$
\boxed{
x_i(t)=1
\iff
H_i(t)q_i(t)>V\widehat b_i(t).
}
$$

Equivalently,

$$
\boxed{
x_i(t)=1
\iff
\frac{\widehat b_i(t)}{q_i(t)}<\frac{H_i(t)}{V}.
}
$$

---

## 7.2 Regret Decomposition

Let the true winning bid be

$$
b_i(t)=P_i(t)+\varepsilon.
$$

Let the predicted winning bid be

$$
\widehat b_i(t)=\widehat P_i(t)+\varepsilon.
$$

Define the prediction error

$$
e_i(t)=b_i(t)-\widehat b_i(t).
$$

Since $\varepsilon$ cancels,

$$
e_i(t)=P_i(t)-\widehat P_i(t).
$$

The total regret decomposes as

$$
\boxed{
\operatorname{Reg}_T
\le
\operatorname{Reg}_T^{\mathrm{Lyap}}
+
\operatorname{Reg}_T^{\mathrm{OLS}}
+
\operatorname{Reg}_T^{\mathrm{explore}}.
}
$$

Here:

- $\operatorname{Reg}_T^{\mathrm{Lyap}}$ is the drift-plus-penalty online-control regret.
- $\operatorname{Reg}_T^{\mathrm{OLS}}$ is regret caused by bid-prediction error.
- $\operatorname{Reg}_T^{\mathrm{explore}}$ is optional exploration regret if the system deliberately explores to learn competitor behavior.

If OLS is updated passively from observed bids and no deliberate exploration is needed, then

$$
\operatorname{Reg}_T^{\mathrm{explore}}=0.
$$

---

# 8. OLS Prediction Error Bound

Assume the competitor bid model is well-specified:

$$
C_{i,o}(t)=K_o(t)^\top\theta_{i,o}^*+\eta_{i,o}(t),
$$

where

$$
\mathbb{E}[\eta_{i,o}(t)\mid K_o(t)]=0.
$$

Assume bounded contexts:

$$
\|K_o(t)\|_2\le K_{\max}.
$$

Let $d$ be the context dimension and let $n_{i,o}(t)$ be the number of samples used to estimate $\theta_{i,o}^*$.

Under standard OLS assumptions,

$$
\|\widehat\theta_{i,o}(t)-\theta_{i,o}^*\|_2
=O_p\left(\sqrt{\frac{d}{n_{i,o}(t)}}\right).
$$

Therefore,

$$
\left|
K_o(t)^\top\left(\widehat\theta_{i,o}(t)-\theta_{i,o}^*\right)
\right|
\le
K_{\max}
\|\widehat\theta_{i,o}(t)-\theta_{i,o}^*\|_2.
$$

So

$$
\boxed{
\left|
\widehat C_{i,o}(t)-\mathbb{E}[C_{i,o}(t)\mid K_o(t)]
\right|
=
O_p\left(K_{\max}\sqrt{\frac{d}{n_{i,o}(t)}}\right).
}
$$

---

# 9. Error Bound for the Maximum Over Competitors

The predicted market price is

$$
\widehat P_i(t)=\max_o \widehat C_{i,o}(t).
$$

The true expected market price is

$$
P_i^*(t)=\max_o K_o(t)^\top\theta_{i,o}^*.
$$

Since the maximum operator is Lipschitz,

$$
\left|
\max_o a_o-
\max_o b_o
\right|
\le
\max_o |a_o-b_o|.
$$

Therefore,

$$
\begin{aligned}
|\widehat P_i(t)-P_i^*(t)|
&=
\left|
\max_o K_o(t)^\top\widehat\theta_{i,o}
-
\max_o K_o(t)^\top\theta_{i,o}^*
\right|\\
&\le
\max_o
\left|
K_o(t)^\top(\widehat\theta_{i,o}-\theta_{i,o}^*)
\right|.
\end{aligned}
$$

If there are $M$ competitors, a uniform high-probability bound gives

$$
\boxed{
|\widehat P_i(t)-P_i^*(t)|
=
O_p\left(
K_{\max}
\sqrt{\frac{d+\log M}{n_{\min}(t)}}
\right),
}
$$

where

$$
n_{\min}(t)=\min_o n_{i,o}(t).
$$

Uniformly over all $N$ products and $M$ competitors,

$$
\boxed{
|\widehat P_i(t)-P_i^*(t)|
=
O_p\left(
K_{\max}
\sqrt{\frac{d+\log(NM)}{n_{\min}(t)}}
\right).
}
$$

---

# 10. OLS-Induced Regret

The OLS prediction error contributes

$$
\sum_{t=1}^{T}\sum_{i=1}^{N}|e_i(t)|x_i(t)
$$

inside the regret bound.

Using the previous bound,

$$
|e_i(t)|
=
O_p\left(
K_{\max}
\sqrt{\frac{d+\log(NM)}{n_{\min}(t)}}
\right).
$$

---

## 10.1 Fixed Historical OLS Dataset

If OLS is trained on a fixed historical dataset of size $n$, then

$$
n_{\min}(t)=n.
$$

If all $N$ products may be bought at each round, then

$$
\boxed{
\operatorname{Reg}_T^{\mathrm{OLS}}
=
O_p\left(
TNK_{\max}
\sqrt{\frac{d+\log(NM)}{n}}
\right).
}
$$

If at most $m$ products are bought each round, replace $N$ by $m$:

$$
\boxed{
\operatorname{Reg}_T^{\mathrm{OLS}}
=
O_p\left(
TmK_{\max}
\sqrt{\frac{d+\log(NM)}{n}}
\right).
}
$$

---

## 10.2 Online OLS

If OLS is updated online and

$$
n_{\min}(t)\approx t,
$$

then

$$
\sum_{t=1}^{T}\sqrt{\frac{1}{t}}=O(\sqrt{T}).
$$

Therefore,

$$
\boxed{
\operatorname{Reg}_T^{\mathrm{OLS}}
=
O_p\left(
NK_{\max}
\sqrt{T(d+\log(NM))}
\right).
}
$$

If at most $m$ products are bought each round,

$$
\boxed{
\operatorname{Reg}_T^{\mathrm{OLS}}
=
O_p\left(
mK_{\max}
\sqrt{T(d+\log(NM))}
\right).
}
$$

---

# 11. Partially Blind Regret Bound: Fixed OLS Training Set

Combining the Lyapunov and OLS terms:

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T]
=
O\left(\frac{T}{V}+V\right)
+
O_p\left(
TNK_{\max}
\sqrt{\frac{d+\log(NM)}{n}}
\right).
}
$$

Choosing

$$
V=\Theta(\sqrt{T})
$$

gives

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T]
=
O(\sqrt{T})
+
O_p\left(
TNK_{\max}
\sqrt{\frac{d+\log(NM)}{n}}
\right).
}
$$

The average regret is

$$
\boxed{
\frac{\mathbb{E}[\operatorname{Reg}_T]}{T}
=
O\left(\frac{1}{\sqrt{T}}\right)
+
O_p\left(
NK_{\max}
\sqrt{\frac{d+\log(NM)}{n}}
\right).
}
$$

Thus, fixed OLS training data creates a prediction-error floor:

$$
\boxed{
O_p\left(
NK_{\max}
\sqrt{\frac{d+\log(NM)}{n}}
\right).
}
$$

This means average regret does not necessarily vanish as $T\to\infty$ unless $n$ also grows.

---

# 12. Partially Blind Regret Bound: Online OLS

With online OLS,

$$
n_{\min}(t)\approx t.
$$

The total regret is

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T]
=
O\left(\frac{T}{V}+V\right)
+
O_p\left(
NK_{\max}
\sqrt{T(d+\log(NM))}
\right).
}
$$

Choosing

$$
V=\Theta(\sqrt{T})
$$

gives

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T]
=
O(\sqrt{T})
+
O_p\left(
NK_{\max}
\sqrt{T(d+\log(NM))}
\right).
}
$$

Usually the OLS term dominates, so

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T]
=
O_p\left(
NK_{\max}
\sqrt{T(d+\log(NM))}
\right).
}
$$

The average regret is

$$
\boxed{
\frac{\mathbb{E}[\operatorname{Reg}_T]}{T}
=
O_p\left(
NK_{\max}
\sqrt{\frac{d+\log(NM)}{T}}
\right).
}
$$

Therefore, the average regret vanishes if

$$
\boxed{
d+\log(NM)=o(T).
}
$$

---

# 13. Adding Explicit Contextual Bandit Exploration

If the system must deliberately explore bids to learn competitor behavior, then the Partially Blind case becomes a contextual bandit problem.

The context is

$$
K_o(t),
$$

and the unknown parameter is

$$
\theta_{i,o}^*.
$$

For linear contextual bandits such as LinUCB, a standard gap-free regret rate for one model is

$$
\boxed{
\operatorname{Reg}_T^{\mathrm{LinUCB}}
=
O\left(d\sqrt{T\log T}\right).
}
$$

For $N$ products and $M$ competitors, a safe product-competitor-wise bound is

$$
\boxed{
\operatorname{Reg}_T^{\mathrm{LinUCB}}
=
O\left(NMd\sqrt{T\log T}\right).
}
$$

Then the total regret is

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T]
=
O\left(\frac{T}{V}+V\right)
+
O\left(NMd\sqrt{T\log T}\right).
}
$$

With

$$
V=\Theta(\sqrt{T}),
$$

$$
\boxed{
\mathbb{E}[\operatorname{Reg}_T]
=
O(\sqrt{T})
+
O\left(NMd\sqrt{T\log T}\right).
}
$$

The average regret is

$$
\boxed{
\frac{\mathbb{E}[\operatorname{Reg}_T]}{T}
=
O\left(\frac{1}{\sqrt{T}}\right)
+
O\left(NMd\sqrt{\frac{\log T}{T}}\right).
}
$$

For fixed $N,M,d$,

$$
\boxed{
\frac{\mathbb{E}[\operatorname{Reg}_T]}{T}\to 0.
}
$$

---

# 14. Margin-Based Sharper Bound

The OLS regret bound above is conservative because it counts all prediction errors.

However, prediction errors only matter if they change the decision.

The predicted decision is

$$
x_i(t)=1
\iff
\widehat b_i(t)<\frac{H_i(t)q_i(t)}{V}.
$$

The ideal expected-bid decision is

$$
x_i^*(t)=1
\iff
b_i^*(t)<\frac{H_i(t)q_i(t)}{V}.
$$

Define the decision threshold

$$
\tau_i(t)=\frac{H_i(t)q_i(t)}{V}.
$$

Define the margin

$$
m_i(t)=\left|b_i^*(t)-\tau_i(t)\right|.
$$

A prediction error changes the decision only if

$$
|e_i(t)|\ge m_i(t).
$$

Assume a margin condition:

$$
\mathbb{P}
\left(
\left|b_i^*(t)-\tau_i(t)\right|\le \delta
\right)
\le C\delta^\alpha.
$$

If

$$
|e_i(t)|=O_p\left(\sqrt{\frac{d+\log(NM)}{n}}\right),
$$

then a sharper OLS-induced regret rate is

$$
\boxed{
\operatorname{Reg}_T^{\mathrm{OLS,margin}}
=
O_p\left(
T
\left(
\sqrt{\frac{d+\log(NM)}{n}}
\right)^{1+\alpha}
\right).
}
$$

The interpretation is:

$$
\boxed{
\text{OLS errors are harmful mainly when the predicted bid is close to the Lyapunov threshold.}
}
$$

---

# 15. Regret Comparison Table

| Setting | What is known? | Main regret terms | With $V=\Theta(\sqrt{T})$ |
|---|---|---|---|
| All-Seeing Lyapunov | Current $b_i(t),q_i(t)$ known | $O(T/V+V)$ | $O(\sqrt{T})$ |
| All-Seeing + UCB | Arm means unknown | $O(T/V+V)+O(\sqrt{NT\log T})$ | $O(\sqrt{T})+O(\sqrt{NT\log T})$ |
| Partially Blind, fixed OLS | Bids predicted from fixed $n$ samples | $O(T/V+V)+O_p(TN\sqrt{(d+\log(NM))/n})$ | $O(\sqrt{T})+O_p(TN\sqrt{(d+\log(NM))/n})$ |
| Partially Blind, online OLS | OLS updated over time | $O(T/V+V)+O_p(N\sqrt{T(d+\log(NM))})$ | $O_p(N\sqrt{T(d+\log(NM))})$ |
| Partially Blind + LinUCB | Exploration needed | $O(T/V+V)+O(NMd\sqrt{T\log T})$ | $O(\sqrt{T})+O(NMd\sqrt{T\log T})$ |

The table suppresses constants such as $K_{\max}$.

---

# 16. Final Practical Interpretation

## All-Seeing Case

If current winning bid costs are fully known, there is no bid-prediction regret.

The regret is mainly Lyapunov control regret:

$$
\boxed{
\operatorname{Reg}_T=O\left(\frac{T}{V}+V\right).
}
$$

With

$$
V=\Theta(\sqrt{T}),
$$

we get

$$
\boxed{
\operatorname{Reg}_T=O(\sqrt{T}).
}
$$

Therefore,

$$
\boxed{
\frac{\operatorname{Reg}_T}{T}=O\left(\frac{1}{\sqrt{T}}\right)\to 0.
}
$$

---

## Partially Blind Case

If the winning bid is predicted using OLS from competitors' inventory contexts, then regret decomposes into Lyapunov control regret and OLS prediction regret:

$$
\boxed{
\operatorname{Reg}_T
\le
\operatorname{Reg}_T^{\mathrm{Lyap}}
+
\operatorname{Reg}_T^{\mathrm{OLS}}.
}
$$

With online OLS,

$$
\boxed{
\operatorname{Reg}_T
=
O_p\left(
NK_{\max}\sqrt{T(d+\log(NM))}
\right).
}
$$

The average regret is

$$
\boxed{
\frac{\operatorname{Reg}_T}{T}
=
O_p\left(
NK_{\max}\sqrt{\frac{d+\log(NM)}{T}}
\right).
}
$$

So the average regret vanishes if

$$
\boxed{
d+\log(NM)=o(T).
}
$$

---

# 17. Recommended Report Statement

In the All-Seeing case, since the current winning bid costs are observed before the decision, the problem has no bid-prediction regret. The Lyapunov drift-plus-penalty policy achieves the standard cost-backlog tradeoff. Against the best stationary feasible policy, its expected regret is

$$
O(T/V+V),
$$

and choosing

$$
V=\Theta(\sqrt{T})
$$

gives

$$
O(\sqrt{T})
$$

regret with

$$
O(\sqrt{T})
$$

inventory-deficit backlog.

In the Partially Blind case, the algorithm uses OLS to estimate the winning bid from competitors' inventory contexts. The total regret decomposes into Lyapunov control regret and OLS prediction regret. If the context dimension is $d$, there are $N$ products and $M$ competitors, and OLS is updated online, then the regret is approximately

$$
O_p\left(
\sqrt{T}
+
NK_{\max}\sqrt{T(d+\log(NM))}
\right).
$$

Hence the average regret is

$$
O_p\left(
NK_{\max}\sqrt{\frac{d+\log(NM)}{T}}
\right),
$$

which vanishes as $T\to\infty$ when

$$
d+\log(NM)=o(T).
$$

---

# 18. Main Conclusion

The bandit-style auction system has different regret behavior depending on the information structure.

For the All-Seeing problem:

$$
\boxed{
\operatorname{Reg}_T=O(\sqrt{T})
}
$$

with

$$
V=\Theta(\sqrt{T}).
$$

For the Partially Blind OLS-based problem:

$$
\boxed{
\operatorname{Reg}_T
=
O_p\left(
NK_{\max}\sqrt{T(d+\log(NM))}
\right)
}
$$

under online OLS.

Therefore, both systems can achieve vanishing average regret under stationarity:

$$
\boxed{
\frac{\operatorname{Reg}_T}{T}\to 0.
}
$$

The All-Seeing case is easier because there is no prediction-error term. The Partially Blind case is harder because regret depends on how accurately OLS predicts the competitors' winning bid.

# Proof Sketch: Convergence of Tabular Q-Learning
## Setting
Consider a finite Markov Decision Process (MDP) with:
- Finite state space $\mathcal{S}$, finite action space $\mathcal{A}$
- Transition probabilities $P(s' \mid s, a)$
- Bounded reward function $R(s, a, s')$ with $|R| \le R_{\max}$
- Discount factor $\gamma \in [0, 1)$
The **Q-learning update** (Watkins, 1989) at time $t$ is:
$$Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \alpha_t(s_t, a_t)\bigl[r_t + \gamma \max_{a'} Q_t(s_{t+1}, a') - Q_t(s_t, a_t)\bigr]$$
where $(s_t, a_t, r_t, s_{t+1})$ is the transition observed at time $t$, and all other entries remain unchanged: $Q_{t+1}(s, a) = Q_t(s, a)$ for $(s, a) \neq (s_t, a_t)$.
**Key point (off-policy):** The data $(s_t, a_t)$ can come from *any* behavior policy (even adversarial), as long as all state-action pairs are visited infinitely often. Q-learning does not need the data to come from the greedy policy — this is what makes it off-policy.
## Theorem (Watkins & Dayan, 1992; Tsitsiklis, 1994)
Under the following conditions:
1. **Finite MDP**: $|\mathcal{S}|$ and $|\mathcal{A}|$ are finite.
2. **Exploration**: Every state-action pair $(s, a)$ is visited infinitely often.
3. **Learning rates (Robbins–Monro)**: For each $(s, a)$:
   $$\sum_{t=0}^{\infty} \alpha_t(s,a) = \infty, \qquad \sum_{t=0}^{\infty} \alpha_t(s,a)^2 < \infty$$
4. **Bounded rewards**: $|R(s,a,s')| \le R_{\max}$ for all $s, a, s'$.
Then $Q_t(s,a) \to Q^*(s,a)$ almost surely for all $(s,a)$, where $Q^*$ is the unique fixed point of the Bellman optimality operator.
## Proof Sketch
The proof proceeds in three main steps.
---
### Step 1: The Bellman Optimality Operator is a $\gamma$-Contraction
Define the **Bellman optimality operator** $\mathcal{T}: \mathbb{R}^{|\mathcal{S}||\mathcal{A}|} \to \mathbb{R}^{|\mathcal{S}||\mathcal{A}|}$ by:
$$(\mathcal{T}Q)(s,a) = \sum_{s'} P(s' \mid s, a)\bigl[R(s,a,s') + \gamma \max_{a'} Q(s', a')\bigr]$$
**Claim:** $\mathcal{T}$ is a $\gamma$-contraction in the sup-norm $\|\cdot\|_\infty$:
$$\|\mathcal{T}Q_1 - \mathcal{T}Q_2\|_\infty \le \gamma \|Q_1 - Q_2\|_\infty$$
*Proof of claim:* For any $(s,a)$:
$$|(\mathcal{T}Q_1)(s,a) - (\mathcal{T}Q_2)(s,a)| = \gamma \left|\sum_{s'} P(s' \mid s,a)\bigl[\max_{a'} Q_1(s',a') - \max_{a'} Q_2(s',a')\bigr]\right|$$
Using $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$ and the fact that $P(\cdot \mid s,a)$ is a probability distribution:
$$\le \gamma \sum_{s'} P(s' \mid s,a) \max_{a'} |Q_1(s',a') - Q_2(s',a')| \le \gamma \|Q_1 - Q_2\|_\infty$$
By the **Banach fixed-point theorem**, $\mathcal{T}$ has a unique fixed point $Q^*$, and $\mathcal{T}^n Q \to Q^*$ for any initial $Q$.
The fixed point $Q^*$ satisfies the **Bellman optimality equation**:
$$Q^*(s,a) = \sum_{s'} P(s' \mid s,a)\bigl[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')\bigr]$$
---
### Step 2: Reformulate Q-Learning as Stochastic Approximation
Define the **error** $\Delta_t(s,a) = Q_t(s,a) - Q^*(s,a)$. We show $\Delta_t \to 0$.
Rewrite the Q-learning update for the visited pair $(s_t, a_t)$:
$$\Delta_{t+1}(s_t, a_t) = (1 - \alpha_t)\,\Delta_t(s_t, a_t) + \alpha_t\,F_t(s_t, a_t)$$
where:
$$F_t(s,a) = r_t + \gamma \max_{a'} Q_t(s_{t+1}, a') - Q^*(s,a)$$
**Key property of $F_t$:** Its conditional expectation (given the history $\mathcal{F}_t$, i.e., all of $Q_t$, $s_t$, $a_t$) satisfies:
$$\mathbb{E}[F_t(s_t, a_t) \mid \mathcal{F}_t] = (\mathcal{T}Q_t)(s_t, a_t) - Q^*(s_t, a_t) = (\mathcal{T}Q_t - \mathcal{T}Q^*)(s_t, a_t)$$
(using the Bellman equation for $Q^*$). By the contraction property from Step 1:
$$|\mathbb{E}[F_t(s_t, a_t) \mid \mathcal{F}_t]| \le \gamma \|\Delta_t\|_\infty$$
Thus the "driving signal" $F_t$ has:
- **Conditional mean bounded by** $\gamma \|\Delta_t\|_\infty$ (contraction)
- **Bounded conditional variance** (since rewards are bounded and Q-values remain bounded under the iteration — this can be shown by induction)
---
### Step 3: Apply the Stochastic Approximation Convergence Theorem
The update $\Delta_{t+1}(s,a) = (1-\alpha_t)\Delta_t(s,a) + \alpha_t F_t(s,a)$ is an instance of the general **asynchronous stochastic approximation** scheme studied by Jaakkola, Jordan, and Singh (1994) and Tsitsiklis (1994).
**Theorem (Jaakkola et al., 1994; Theorem 1):** Consider a stochastic process $(\Delta_t, F_t)$ adapted to a filtration $\{\mathcal{F}_t\}$ where:
$$\Delta_{t+1}(x) = (1 - \alpha_t(x))\,\Delta_t(x) + \alpha_t(x)\,F_t(x)$$
If:
1. $0 \le \alpha_t(x) \le 1$, $\sum_t \alpha_t(x) = \infty$, $\sum_t \alpha_t(x)^2 < \infty$ a.s.
2. $|\mathbb{E}[F_t(x) \mid \mathcal{F}_t]| \le \gamma \|\Delta_t\|_\infty$ with $\gamma < 1$
3. $\text{Var}[F_t(x) \mid \mathcal{F}_t] \le C(1 + \|\Delta_t\|_\infty^2)$ for some constant $C$
Then $\Delta_t(x) \to 0$ almost surely for all $x$.
**Applying this to Q-learning:**
- Condition 1 holds by assumption (Robbins–Monro conditions on learning rates, with infinite visitation ensuring the sums diverge for each $(s,a)$).
- Condition 2 holds by the contraction property established in Step 2.
- Condition 3 holds because rewards are bounded, and Q-values remain bounded (by induction: the max Q-value at each step grows by at most $\alpha_t R_{\max}/(1-\gamma)$, and the learning rates are summable-square, so the variance is controlled).
Therefore $\Delta_t(s,a) = Q_t(s,a) - Q^*(s,a) \to 0$ almost surely for all $(s,a)$. $\square$
---
## Why Off-Policy Data is Fine
The crucial observation is that **none of the three conditions above depend on how $(s_t, a_t)$ is chosen**. The behavior policy determines *which* state-action pair gets updated at each step, but:
- The contraction property (Condition 2) is a property of the Bellman operator, not the behavior policy.
- The variance bound (Condition 3) depends only on reward boundedness.
- The Robbins–Monro conditions (Condition 1) require only that every $(s,a)$ is visited infinitely often with appropriate learning rates — the order and frequency of visits don't matter.
This is fundamentally different from SARSA or other on-policy methods, where the target includes $Q_t(s_{t+1}, a_{t+1})$ with $a_{t+1}$ drawn from the current policy. In Q-learning, the target uses $\max_{a'} Q_t(s_{t+1}, a')$, which is independent of the behavior policy.
---
## References
- Watkins, C.J.C.H. (1989). *Learning from Delayed Rewards*. PhD thesis, Cambridge.
- Watkins, C.J.C.H. & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279–292.
- Tsitsiklis, J.N. (1994). Asynchronous stochastic approximation and Q-learning. *Machine Learning*, 16(3), 185–202.
- Jaakkola, T., Jordan, M.I., & Singh, S.P. (1994). On the convergence of stochastic iterative dynamic programming algorithms. *Neural Computation*, 6(6), 1185–1201.
# Implicit Q-Learning (IQL)

Offline RL aims to minimize deviation from the behaviour policy used to collect the dataset to avoid distributional shift, while simultaneously imporving the learned policy over the behaviour policy. Concretely, this can manifest as 'stitching' suboptimal trajectories
together into an optimal one. 

Implicit Q-Learning (IQL) aims to fit the upper expectile of the value function and compute the Q-value backup. Afterwards, it extracts the policy via advantage weighted behavioural cloning, without querying out-of-sample actions, aiming
to determine how the Q-value function can vary with different actions while averaging future outcomes together as a result of stochastic dynamics. 

## Algorithm

### Expectile Regression
Expectile regression is performed on the state value function with respect to random actions, and the **expectile** $\tau$ of a random variable $X$ is the solution to the asymmetric least squares problem:

$$\underset{m_{\tau}}{\arg\min} \ \mathbb{E}_{x \sim X} \left[ L_2^{\tau}(x - m_{\tau}) \right]$$

where $L_2^{\tau}(u) = |\tau - \mathbf{1}(u < 0)| \cdot u^2$ This can be easily extended to expectiles of a conditional distribution. The main purpose of using an asymmetric loss is to weight the positive and negative residuals asymmetrically. In the case of IQL, $u = Q(s, a) - V(s)$ and the expectile loss can be interpreted as follows:
- when Q > V (positive residual), loss is weighed by $\tau$
- when Q < V (negative residual), loss is weighed by $(1 - \tau)$

If: 
- $\tau = 0.5$, $V(s)$ is symmetric, learns the mean of $Q(s, a)$ over actions in the dataset.
- $\tau \to 1.0$, heavily penalizes underestimation, $V(s)$ learns the maximum of $Q(s, a)$.
- $\tau \to 0.0$, heavily penalizes overestimation, $V(s)$ learns the minimum of $Q(s, a)$.

$V(s)$ approximates the maximum of $Q(s, a)$ over only actions present in the dataset, without having to sample actions or query new actions to compute the max, avoiding the out of sample problem. The policy evaluation objective can be modified to predict the upper expectile of the TD targets in such a way that the approximated maximum of $r(s, a) + \gamma Q_{\hat{\theta}}(s', a')$ over actions a' constrained to the dataset actions.

$$L(\theta) = \mathbb{E}_{(s, a, s', a') \sim \mathcal{D}} \left[ L_2^{\tau} \left( r(s, a) + \gamma Q_{\hat{\theta}}(s', a') - Q_{\theta}(s, a) \right) \right]$$

One major issue is that it can also incorporate stochasticity due to the environment dynamics, so a large target value could have occurred due to a lucky sample that happened to transition to a good state, rather than reflecting the existence of a single action that happens to achieve that value.
This can be resolved by introducing a separate value function that approximates the expectile with respect to the action distribution. The value loss function is as follows:

$$L_V(\psi) = \mathbb{E}_{(s, a) \sim \mathcal{D}} \left[ L_2^{\tau} \left( Q_{\hat{\theta}}(s, a) - V_{\psi}(s) \right) \right]$$

For the Q-networks, the Q-loss is:

$$L_Q(\theta) = \mathbb{E}_{(s, a, s') \sim \mathcal{D}} \left[ \left( r(s, a) + \gamma V_{\psi}(s') - Q_{\theta}(s, a) \right)^2 \right]$$

Note that clipped double Q-learning is used to compute targets. 

### Advantage-Weighted Regression
The policy is extracted through advantage-weighted regression, with $\beta$ controlling how sharply high-advantage actions are weighetd. This allows weighted behaviour cloning where actions are weighted relative to the average. As a consequence, the policy learns to reproduce high-advantage actions, while avoiding low-advantage actions. The actions $a$ come from the dataset, so there is no need to query new actions either.

$$L_{\pi}(\phi) = \mathbb{E}_{(s, a) \sim \mathcal{D}} \left[ \exp\left(\beta \left(Q_{\hat{\theta}}(s, a) - V_{\psi}(s)\right)\right) \log \pi_{\phi}(a \mid s) \right]$$

Note that $\beta \in [0, \infty)$ is the inverse temperature.
- Small $\beta \approx$ behaviour cloning
- Large $\beta \Rightarrow$ attempts to recover the maximum of the Q-function
## Sources

Here is a list of sources used for this README.md and for learning about Implicit Q-Learning (IQL):

1. **Original IQL Paper** – *Offline Reinforcement Learning with Implicit Q-Learning*
   [Link to paper](https://arxiv.org/abs/2110.06169) by Ilya Kostrikov, Ashvin Nair, and Sergey Levine.

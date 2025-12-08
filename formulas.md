# COMPREHENSIVE DEEP LEARNING FORMULAS CHEAT SHEET

**Source: Understanding Deep Learning by Simon J.D. Prince**

---

## 1. SUPERVISED LEARNING

### 1.1 Basic Framework

**Model prediction:**
\[y = f[x, \omega]\]

**Loss minimization:**
\[\hat{\omega} = \arg\min_{\omega} \left\{L[\omega]\right\}\]

---

## 2. LINEAR REGRESSION

### 2.1 1D Linear Regression Model

**Model equation:**
\[y = f[x, \omega] = \omega_0 + \omega_1 x\]

### 2.2 Loss Function (Least Squares)

**Sum of squared errors:**
\[L[\omega] = \sum_{i=1}^{I} (f[x_i, \omega] - y_i)^2 = \sum_{i=1}^{I} (\omega_0 + \omega_1 x_i - y_i)^2\]

---

## 3. SHALLOW NEURAL NETWORKS

### 3.1 Single Input, Single Output with ReLU

**Hidden units (with ReLU activation):**
\[h_d = a[\theta_{d0} + \theta_{d1} x]\]

where \(a[z] = \max(0, z)\) is the ReLU activation function.

**Output (linear combination):**
\[y = \omega_0 + \sum_{d=1}^{D} \omega_d h_d\]

### 3.2 General Shallow Network (Multivariate Case)

**Hidden units:**
\[h_d = a\left(\theta_{d0} + \sum_{i=1}^{D_i} \theta_{di} x_i\right)\]

**Output:**
\[y_j = \omega_{j0} + \sum_{d=1}^{D} \omega_{jd} h_d\]

where:
- \(D_i\) = input dimension
- \(D\) = number of hidden units
- \(D_o\) = output dimension

### 3.3 Universal Approximation Property

A shallow neural network with enough hidden units can approximate any continuous function on a compact domain to arbitrary precision.

---

## 4. DEEP NEURAL NETWORKS

### 4.1 Sequential Composition (General Form)

**Layer-by-layer computation:**
\[h^{(1)} = a[b^{(1)} + W^{(1)} x]\]
\[h^{(2)} = a[b^{(2)} + W^{(2)} h^{(1)}]\]
\[\vdots\]
\[y = b^{(K)} + W^{(K)} h^{(K-1)}\]

where:
- \(K\) = number of layers
- \(W^{(k)}\) = weight matrix at layer \(k\)
- \(b^{(k)}\) = bias vector at layer \(k\)
- \(a[\cdot]\) = activation function

### 4.2 Matrix Notation (Batch Processing)

**For a batch of \(N\) samples:**
\[H^{(1)} = a[B^{(1)} + W^{(1)} X]\]
\[H^{(2)} = a[B^{(2)} + W^{(2)} H^{(1)}]\]
\[\vdots\]
\[Y = B^{(K)} + W^{(K)} H^{(K-1)}\]

where capital letters denote matrices.

---

## 5. ACTIVATION FUNCTIONS

### 5.1 ReLU (Rectified Linear Unit)

\[a[z] = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}\]

### 5.2 Sigmoid Function

\[a[z] = \frac{1}{1 + \exp(-z)}\]

### 5.3 Tanh Function

\[a[z] = \tanh(z) = \frac{\exp(z) - \exp(-z)}{\exp(z) + \exp(-z)}\]

### 5.4 Softmax Function (for multiclass)

\[a[z_j] = \frac{\exp(z_j)}{\sum_{k=1}^{K} \exp(z_k)}\]

---

## 6. LOSS FUNCTIONS

### 6.1 Maximum Likelihood Principle

**General loss:**
\[L[\omega] = -\sum_{i=1}^{I} \log p(y_i | x_i, \omega)\]

### 6.2 Univariate Regression (Squared Error)

**Model:** \(y = f[x, \omega] + \epsilon\) where \(\epsilon \sim \mathcal{N}(0, \sigma^2)\)

**Loss:**
\[L[\omega] = \sum_{i=1}^{I} (f[x_i, \omega] - y_i)^2\]

### 6.3 Binary Classification (Bernoulli)

**Probability:**
\[p(y | x, \omega) = \begin{cases} f[x, \omega] & \text{if } y = 1 \\ 1 - f[x, \omega] & \text{if } y = 0 \end{cases}\]

**Loss (Binary Cross-Entropy):**
\[L[\omega] = -\sum_{i=1}^{I} [y_i \log(f[x_i, \omega]) + (1 - y_i) \log(1 - f[x_i, \omega])]\]

### 6.4 Multiclass Classification (Categorical)

**Probability:**
\[p(y = c | x, \omega) = \frac{\exp(z_c)}{\sum_{k=1}^{K} \exp(z_k)}\]

where \(z_k = \text{logit}_k[x, \omega]\)

**Loss (Categorical Cross-Entropy):**
\[L[\omega] = -\sum_{i=1}^{I} \sum_{c=1}^{C} y_{ic} \log(p_c[x_i, \omega])\]

### 6.5 Cross-Entropy Loss (General Form)

\[L[\omega] = -\sum_{i=1}^{I} \log(p(y_i | x_i, \omega))\]

---

## 7. FITTING MODELS / OPTIMIZATION

### 7.1 Gradient Descent

**Parameter update:**
\[\omega_{t+1} = \omega_t - \alpha \frac{\partial L[\omega_t]}{\partial \omega}\]

where \(\alpha\) is the learning rate and \(t\) is the iteration counter.

### 7.2 Stochastic Gradient Descent (SGD)

**Update with mini-batch:**
\[\omega_{t+1} = \omega_t - \alpha \frac{\partial L_{\text{batch}}[\omega_t]}{\partial \omega}\]

where \(L_{\text{batch}}\) is loss computed on a small batch of data.

### 7.3 Momentum

**Velocity vector:**
\[v_t = \beta v_{t-1} + (1 - \beta) \frac{\partial L[\omega_t]}{\partial \omega}\]

**Parameter update:**
\[\omega_{t+1} = \omega_t - \alpha v_t\]

where \(\beta\) is the momentum coefficient (typically 0.9).

### 7.4 Adam Optimizer

**First moment estimate (mean):**
\[m_t = \beta_1 m_{t-1} + (1 - \beta_1) \frac{\partial L[\omega_t]}{\partial \omega}\]

**Second moment estimate (variance):**
\[v_t = \beta_2 v_{t-1} + (1 - \beta_2) \left(\frac{\partial L[\omega_t]}{\partial \omega}\right)^2\]

**Bias correction:**
\[\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}\]

**Parameter update:**
\[\omega_{t+1} = \omega_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\]

where typically \(\beta_1 = 0.9\), \(\beta_2 = 0.999\), \(\epsilon = 10^{-8}\).

---

## 8. BACKPROPAGATION & GRADIENTS

### 8.1 Chain Rule (Backpropagation)

**Gradient with respect to parameters in layer \(k\):**
\[\frac{\partial L}{\partial W^{(k)}} = \frac{\partial L}{\partial h^{(k)}} \frac{\partial h^{(k)}}{\partial z^{(k)}} \frac{\partial z^{(k)}}{\partial W^{(k)}}\]

**Gradient propagation backwards:**
\[\frac{\partial L}{\partial h^{(k-1)}} = \frac{\partial L}{\partial h^{(k)}} \frac{\partial h^{(k)}}{\partial z^{(k)}} \frac{\partial z^{(k)}}{\partial h^{(k-1)}}\]

### 8.2 Gradient of Loss w.r.t. Output

**For regression:**
\[\frac{\partial L}{\partial y_i} = 2(f[x_i, \omega] - y_i)\]

**For binary classification:**
\[\frac{\partial L}{\partial y_i} = \frac{f[x_i, \omega] - y_i}{f[x_i, \omega](1 - f[x_i, \omega])}\]

**For multiclass classification:**
\[\frac{\partial L}{\partial z_c} = p_c[x, \omega] - y_c\]

### 8.3 Gradient of Activation Functions

**ReLU:**
\[\frac{\partial a[z]}{\partial z} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}\]

**Sigmoid:**
\[\frac{\partial a[z]}{\partial z} = a[z](1 - a[z])\]

**Tanh:**
\[\frac{\partial a[z]}{\partial z} = 1 - a[z]^2\]

---

## 9. PARAMETER INITIALIZATION

### 9.1 Xavier (Glorot) Initialization

**Initialize weights from:**
\[W^{(k)} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)\]

or for normal distribution:
\[W^{(k)} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)\]

### 9.2 He Initialization (for ReLU)

**Initialize weights from:**
\[W^{(k)} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)\]

### 9.3 Biases

**Initialize to zero:**
\[b^{(k)} = 0\]

---

## 10. MEASURING PERFORMANCE

### 10.1 Bias-Variance Decomposition

**Expected test error:**
\[E[\text{Error}_{\text{test}}] = (\text{Bias})^2 + \text{Variance} + \text{Noise}\]

### 10.2 Bias (Underfitting)

\[\text{Bias} = E_{\text{train}}[f[x, \hat{\omega}]] - E[y | x]\]

### 10.3 Variance (Overfitting)

\[\text{Variance} = E[(f[x, \hat{\omega}] - E_{\text{train}}[f[x, \hat{\omega}]])^2]\]

### 10.4 Classification Metrics

**Accuracy:**
\[\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}\]

**Precision:**
\[\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}\]

**Recall (Sensitivity):**
\[\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}\]

**F1-Score:**
\[F_1 = 2 \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\]

---

## 11. REGULARIZATION

### 11.1 L2 Regularization (Weight Decay)

**Regularized loss:**
\[L[\omega] = L_0[\omega] + \lambda \sum_{k} \|\Omega^{(k)}\|_F^2\]

where \(\|\cdot\|_F\) is the Frobenius norm and \(\lambda\) is regularization strength.

### 11.2 L1 Regularization

**Regularized loss:**
\[L[\omega] = L_0[\omega] + \lambda \sum_{k} \|\Omega^{(k)}\|_1\]

### 11.3 Dropout

**During training, each unit is retained with probability \(p\):**
\[h_d^{\text{dropped}} \sim \begin{cases} h_d / p & \text{with probability } p \\ 0 & \text{with probability } 1 - p \end{cases}\]

**At test time, use full network (or scale by \(p\)).**

### 11.4 Early Stopping

Stop training when validation loss stops decreasing for \(N\) iterations.

---

## 12. BATCH NORMALIZATION

### 12.1 Normalization Formula

**Normalize inputs in a batch:**
\[\hat{z}_{nd} = \frac{z_{nd} - \mu_d}{\sqrt{\sigma_d^2 + \epsilon}}\]

where:
- \(\mu_d = \frac{1}{N} \sum_{n=1}^{N} z_{nd}\) (batch mean)
- \(\sigma_d^2 = \frac{1}{N} \sum_{n=1}^{N} (z_{nd} - \mu_d)^2\) (batch variance)
- \(\epsilon\) is a small constant (e.g., \(10^{-5}\))

### 12.2 Scaling and Shifting

**Apply learnable scale and shift:**
\[z_d^{\text{BN}} = \gamma_d \hat{z}_d + \beta_d\]

---

## 13. CONVOLUTIONAL NETWORKS

### 13.1 1D Convolution

**Output at position \(j\):**
\[y_j = b + \sum_{i=0}^{K-1} w_i x_{j+i}\]

where \(K\) is the kernel size.

### 13.2 2D Convolution

**Output at position (j,k):**
\[y_{jk} = b + \sum_{i=0}^{K_1-1} \sum_{i'=0}^{K_2-1} w_{ii'} x_{j+i, k+i'}\]

where \(K_1 \times K_2\) is the kernel size.

### 13.3 Pooling

**Max pooling over region of size \(P \times P\):**
\[y_{jk} = \max_{i=0}^{P-1} \max_{i'=0}^{P-1} x_{j \cdot s + i, k \cdot s + i'}\]

where \(s\) is the stride.

**Average pooling:**
\[y_{jk} = \frac{1}{P^2} \sum_{i=0}^{P-1} \sum_{i'=0}^{P-1} x_{j \cdot s + i, k \cdot s + i'}\]

### 13.4 Output Size After Convolution

\[N_{\text{out}} = \left\lfloor \frac{N_{\text{in}} + 2P - K}{S} \right\rfloor + 1\]

where:
- \(N_{\text{in}}\) = input size
- \(P\) = padding
- \(K\) = kernel size
- \(S\) = stride

---

## 14. RESIDUAL NETWORKS

### 14.1 Residual Block

**Standard:**
\[h = a[f[x] + x]\]

**With skip connection (identity mapping):**
\[h = f[x] + x\]

### 14.2 Bottleneck Block

**Three-layer residual block:**
\[h = a[f_3[a[f_2[a[f_1[x]]]]]] + x\]

where typically \(f_1\) reduces dimension, \(f_2\) applies main computation, \(f_3\) restores dimension.

---

## 15. TRANSFORMERS & ATTENTION

### 15.1 Dot-Product Self-Attention

**Attention weights:**
\[\alpha_{jk} = \frac{\exp(q_j^T k_k / \sqrt{D})}{\sum_{k'=1}^{N} \exp(q_j^T k_{k'} / \sqrt{D})}\]

where:
- \(q_j\) = query vector
- \(k_k\) = key vector
- \(D\) = dimension (for scaling)

**Attention output:**
\[y_j = \sum_{k=1}^{N} \alpha_{jk} v_k\]

where \(v_k\) are value vectors.

### 15.2 Multi-Head Attention

**For \(H\) heads:**
\[\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O\]

where each head computes:
\[\text{head}_h = \text{Attention}(Q W_h^Q, K W_h^K, V W_h^V)\]

### 15.3 Transformer Block

**With residual connections and layer norm:**
\[h_1 = \text{LayerNorm}(x + \text{MultiHead}(x, x, x))\]
\[h_2 = \text{LayerNorm}(h_1 + \text{FFN}(h_1))\]

where FFN is a two-layer feed-forward network.

---

## 16. GENERATIVE ADVERSARIAL NETWORKS (GANs)

### 16.1 Generator Loss

\[L_G = -\mathbb{E}_{z \sim p(z)}[\log D[G[z]]]\]

or (non-saturating form):
\[L_G = -\mathbb{E}_{z \sim p(z)}[\log(1 - D[G[z]])]\]

### 16.2 Discriminator Loss

\[L_D = -\mathbb{E}_{x \sim p_{\text{data}}}[\log D[x]] - \mathbb{E}_{z \sim p(z)}[\log(1 - D[G[z]])]\]

### 16.3 Wasserstein GAN Loss

**Discriminator (critic):**
\[L_D = -\mathbb{E}_{x}[D[x]] + \mathbb{E}_{z}[D[G[z]]]\]

**Generator:**
\[L_G = -\mathbb{E}_{z}[D[G[z]]]\]

---

## 17. VARIATIONAL AUTOENCODERS (VAEs)

### 17.1 ELBO (Evidence Lower Bound)

\[\mathcal{L} = \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - \text{KL}(q_{\phi}(z|x) \| p(z))\]

where:
- \(q_{\phi}(z|x)\) = encoder (recognition model)
- \(p_{\theta}(x|z)\) = decoder (generative model)
- \(p(z)\) = prior (usually standard normal)

### 17.2 KL Divergence (Gaussian Case)

**For \(\mathcal{N}(\mu, \sigma^2)\) vs. \(\mathcal{N}(0, 1)\):**
\[\text{KL}(q||p) = -\frac{1}{2} \sum_d (1 + \log(\sigma_d^2) - \mu_d^2 - \sigma_d^2)\]

### 17.3 Reparameterization Trick

**Sample from \(\mathcal{N}(\mu, \sigma^2)\):**
\[z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)\]

where \(\odot\) denotes element-wise multiplication.

---

## 18. NORMALIZING FLOWS

### 18.1 Flow Transformation

**Sequence of invertible transformations:**
\[z = T_L \circ T_{L-1} \circ \cdots \circ T_1(z_0)\]

### 18.2 Change of Variables Formula

\[\log p(z) = \log p(z_0) - \sum_{l=1}^{L} \log \left|\det \frac{\partial T_l}{\partial z_{l-1}}\right|\]

---

## 19. DIFFUSION MODELS

### 19.1 Forward Process (Encoder)

**Add noise step by step:**
\[z_t = \sqrt{\alpha_t} z_0 + \sqrt{1 - \alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)\]

where \(\alpha_t\) controls noise schedule.

### 19.2 Reverse Process (Decoder)

**Learn to denoise:**
\[z_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(z_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_{\theta}(z_t, t)\right) + \sigma_t \epsilon\]

where \(\epsilon_{\theta}\) is a learned noise prediction model and \(\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s\).

### 19.3 Training Objective

\[L = \mathbb{E}_{t, z_0, \epsilon}\left[\|\epsilon - \epsilon_{\theta}(z_t, t)\|^2\right]\]

---

## 20. REINFORCEMENT LEARNING

### 20.1 Markov Decision Process

**Return (cumulative discounted reward):**
\[R_t = \sum_{t'=t}^{\infty} \gamma^{t'-t} r_{t'}\]

where \(\gamma \in [0, 1]\) is discount factor.

### 20.2 Expected Return

**Value function (expected return from state):**
\[V^{\pi}(s) = \mathbb{E}_{a \sim \pi(s)}[r(s, a) + \gamma V^{\pi}(s')]\]

**Q-function (expected return from state-action):**
\[Q^{\pi}(s, a) = \mathbb{E}[r(s, a) + \gamma V^{\pi}(s')]\]

### 20.3 Bellman Equation

**For optimal policy:**
\[V^*(s) = \max_a [r(s, a) + \gamma \mathbb{E}[V^*(s')]]\]

\[Q^*(s, a) = r(s, a) + \gamma \mathbb{E}[\max_{a'} Q^*(s', a')]\]

### 20.4 Q-Learning Update

\[Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]\]

### 20.5 Policy Gradient

**Gradient for policy \(\pi_{\theta}(a|s)\):**
\[\nabla_{\theta} J(\theta) = \mathbb{E}[\nabla_{\theta} \log \pi_{\theta}(a|s) Q^{\pi}(s, a)]\]

### 20.6 Actor-Critic

**Actor (policy) update:**
\[\theta \leftarrow \theta + \alpha_{\theta} \nabla_{\theta} \log \pi_{\theta}(a|s) A(s, a)\]

**Critic (value) update:**
\[\phi \leftarrow \phi + \alpha_{\phi} (r + \gamma V_{\phi}(s') - V_{\phi}(s)) \nabla_{\phi} V_{\phi}(s)\]

where \(A(s, a) = Q(s, a) - V(s)\) is the advantage.

---

## 21. GRAPH NEURAL NETWORKS

### 21.1 Graph Convolution

**Node update:**
\[h_n^{(k+1)} = a\left(b^{(k)} + \sum_{m \in \text{Ne}(n)} W^{(k)} h_m^{(k)}\right)\]

where \(\text{Ne}(n)\) is the neighborhood of node \(n\).

### 21.2 Aggregation

**Mean aggregation:**
\[h_n^{(k+1)} = a\left(b^{(k)} + W^{(k)} \frac{1}{|\text{Ne}(n)|} \sum_{m \in \text{Ne}(n)} h_m^{(k)}\right)\]

**Sum aggregation:**
\[h_n^{(k+1)} = a\left(b^{(k)} + W^{(k)} \sum_{m \in \text{Ne}(n)} h_m^{(k)}\right)\]

### 21.3 Graph Pooling

**Max pooling:**
\[p = \max_n h_n\]

**Mean pooling:**
\[p = \frac{1}{N} \sum_n h_n\]

---

## 22. PROBABILITY & STATISTICS

### 22.1 Likelihood

\[p(y | x, \omega) = \text{probability of output given input and parameters}\]

**Log-likelihood:**
\[\log p(y | x, \omega)\]

### 22.2 Expectation

\[\mathbb{E}[f(x)] = \int f(x) p(x) dx\]

**For discrete:**
\[\mathbb{E}[f(x)] = \sum_x f(x) p(x)\]

### 22.3 Normal Distribution

\[\mathcal{N}(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)\]

**Multivariate:**
\[\mathcal{N}(x | \mu, \Sigma) = \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)\]

### 22.4 KL Divergence

\[\text{KL}(p \| q) = \mathbb{E}_p\left[\log \frac{p(x)}{q(x)}\right] = \mathbb{E}_p[\log p(x)] - \mathbb{E}_p[\log q(x)]\]

### 22.5 Jensen-Shannon Divergence

\[\text{JS}(p \| q) = \frac{1}{2} \text{KL}(p \| m) + \frac{1}{2} \text{KL}(q \| m)\]

where \(m = \frac{1}{2}(p + q)\).

---

## 23. MATRIX CALCULUS

### 23.1 Derivative w.r.t. Vector

**Vector-to-scalar:**
\[\frac{\partial f}{\partial x} = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_D} \end{bmatrix}\]

### 23.2 Derivative w.r.t. Matrix

**Matrix-to-scalar:**
\[\frac{\partial f}{\partial W} = \begin{bmatrix} \frac{\partial f}{\partial W_{11}} & \cdots & \frac{\partial f}{\partial W_{1C}} \\ \vdots & \ddots & \vdots \\ \frac{\partial f}{\partial W_{R1}} & \cdots & \frac{\partial f}{\partial W_{RC}} \end{bmatrix}\]

### 23.3 Chain Rule (Multivariable)

\[\frac{\partial f}{\partial x} = \frac{\partial f}{\partial y} \frac{\partial y}{\partial x}\]

### 23.4 Common Derivatives

**Linear transformation:** \(\frac{\partial (Wx)}{\partial x} = W^T\)

**Quadratic form:** \(\frac{\partial (x^T A x)}{\partial x} = (A + A^T) x\)

**Trace:** \(\frac{\partial \text{Tr}(AX)}{\partial X} = A^T\)

**Determinant:** \(\frac{\partial \log |W|}{\partial W} = (W^{-1})^T = (W^T)^{-1}\)

---

## 24. KEY NOTATION SUMMARY

| Symbol | Meaning |
|--------|---------|
| \(x\) | Input data |
| \(y\) | Output/target |
| \(\omega, \phi, \theta\) | Parameters |
| \(W\) | Weight matrix |
| \(b\) | Bias vector |
| \(L\) | Loss function |
| \(a[\cdot]\) | Activation function |
| \(\alpha\) | Learning rate |
| \(\beta\) | Momentum coefficient |
| \(\lambda\) | Regularization strength |
| \(\gamma\) | Discount factor (RL) |
| \(\epsilon\) | Random noise |
| \(\mathcal{N}\) | Normal distribution |
| \(\mathbb{E}\) | Expectation |
| \(\text{KL}\) | Kullback-Leibler divergence |
| \(\text{softmax}\) | Softmax function |
| \(\nabla\) | Gradient operator |
| \(\odot\) | Element-wise multiplication |
| \(|\cdot|\) | Determinant |
| \(\|\cdot\|\) | Norm |

---

## IMPORTANT CONSTANTS & VALUES

- **ReLU activation:** \(a[z] = \max(0, z)\)
- **Sigmoid range:** \(a[z] \in (0, 1)\)
- **Softmax:** Outputs sum to 1
- **Bias-Variance tradeoff:** Underfitting vs. overfitting
- **Learning rate:** Typically \(10^{-4}\) to \(10^{-2}\)
- **Batch size:** Typically 32, 64, 128, 256
- **Momentum:** Typically \(\beta = 0.9\)
- **Adam parameters:** \(\beta_1 = 0.9, \beta_2 = 0.999\)
- **Dropout rate:** Typically 0.5
- **Temperature scaling:** Often \(\tau \in [0.1, 10]\)

---

**Last Updated:** December 2025
**Source:** Understanding Deep Learning by Simon J.D. Prince (May 29, 2025)

This cheat sheet covers the core mathematical formulations essential for deep learning exams. Print and review regularly!

# Multi-Agent Reinforcement Learning: Comparison of VDN and QMIX in BenchMARL

## 📌 Project Description

The goal of this project is to tune hyperparameters and compare two popular multi-agent reinforcement learning algorithms—**VDN** (Value Decomposition Networks) and **QMIX**—using the **BenchMARL** framework. Experiments were conducted in an environment that includes both **cooperative** and **competitive** scenarios.

The project includes:
- training agents in various scenarios,
- tuning hyperparameters (learning rate, discount factor),
- result analysis.

---

## 📦 Libraries Used

- `BenchMARL`
- `NumPy`, `Pandas`
- `Matplotlib`
- `PyTorch`

---

## 🔬 Environments

### 🟢 Multi-Agent Particle Environment (MPE)
A lightweight 2D platform designed for MARL testing.

**Scenarios:**
- `Simple Spread`: agents spread out evenly.
- `Simple Adversary`: cooperation and avoiding an opponent.
- `Simple Push`: a competitive scenario where agents compete to push an object to a specified position.

**Features:**
- discrete environment,
- compatible with OpenAI Gym,
- ability to create custom scenarios.

---

## 🧠 Algorithm Description

### ✅ Value Decomposition Networks (VDN)

VDN assumes that the global value of a team of agents $\( Q_{\text{tot}} \)$ can be expressed as the sum of the local values $\( Q_i \)$ for each agent:

$$
\[
Q_{\text{tot}}(\tau, u) = \sum_{i=1}^{n} Q_i(\tau_i, u_i)
\]
$$

**Features:**
- simple structure,
- fast training,
- limitations with complex dependencies.

---

### ✅ QMIX

QMIX extends VDN by using a **non-linear, monotonic mixing function**:

$$
\[
\frac{\partial Q_{\text{tot}}}{\partial Q_i} \geq 0
\]
$$

**Features:**
- better representation of complex dependencies,
- centralized training with decentralized execution,
- support for dynamic environments.

---

## ⚙️ Experiment Plan

- Comparison of QMIX and VDN on MPE tasks.
- Cooperative tests: `Simple Spread`.
- Competitive tests: `Simple Push`, `Simple Adversary`.
- Various hyperparameter configurations.

## ⚗️ Hyperparameter Experiments

To find the best configuration for VDN and QMIX algorithms, a series of experiments were conducted with different hyperparameter values. The starting point was the default configuration of BenchMARL (example below):

```python
default_config = {
    'sampling_device': 'cuda',
    'train_device': 'cuda',
    'buffer_device': 'cuda',
    'share_policy_params': True,
    'prefer_continuous_actions': False,
    'collect_with_grad': False,
    'parallel_collection': False,
    'gamma': 0.99,
    'lr': 0.01,
    'adam_eps': 1.0e-8,
    'clip_grad_norm': 0.5,
    'clip_grad_val': None,
    'soft_target_update': True,
    'polyak_tau': 0.005,
    'hard_target_update_frequency': 100,
    'exploration_eps_init': 1.0,
    'exploration_eps_end': 0.05,
    'exploration_anneal_frames': 1000000,
    'max_n_frames': 1000000,
    'on_policy_collected_frames_per_batch': 2048,
    'on_policy_n_envs_per_worker': 1,
    'on_policy_n_minibatch_iters': 4,
    'on_policy_minibatch_size': 64,
    'off_policy_collected_frames_per_batch': 100,
    'off_policy_n_envs_per_worker': 1,
    'off_policy_n_optimizer_steps': 100,
    'off_policy_train_batch_size': 512,
    'off_policy_memory_size': 1000000,
    'off_policy_init_random_frames': 50000,
    'off_policy_use_prioritized_replay_buffer': False,
    'off_policy_prb_alpha': 0.6,
    'off_policy_prb_beta': 0.4,
    'evaluation': True,
    'render': False,
    'evaluation_interval': 10000,
    'evaluation_episodes': 10,
    'evaluation_deterministic_actions': True,
    'project_name': 'benchmarl',
    'create_json': True,
    'save_folder': 'results',
    'restore_file': None,
    'restore_map_location': None,
    'checkpoint_interval': 100,
    'checkpoint_at_end': True,
    'keep_checkpoints_num': 1,
    'max_n_iters': 5000,
    'loggers': [],
}
```

---

## 🧩 Impact of Hyperparameters on MARL Algorithms

### 1. `lr` (Learning Rate)
- **Description:** Determines the speed of model weight updates during training.
- **Impact:** A value that is too high can lead to model instability, resulting in significant fluctuations in rewards. A value that is too low may result in very slow learning and minimal changes in the agent's policy.

### 2. `clip_grad_norm`
- **Description:** Limits the length of the gradient.
- **Impact:** Prevents gradient explosion, which can be particularly problematic in complex environments. Disabling this parameter (`None`) can lead to unstable results.

### 3. `polyak_tau`
- **Description:** Coefficient for softly updating target weights in the neural network.
- **Impact:** Higher values speed up weight updates, which may result in faster convergence but also greater fluctuations in results. Too low values may slow down adaptation to new data.

### 4. `off_policy_memory_size`
- **Description:** Size of the memory buffer to store the agent's experiences.
- **Impact:** A larger buffer allows for better diversification of training data, which can improve policy quality. However, an excessively large buffer may lead to memory and processing time issues.

### 5. `off_policy_train_batch_size`
- **Description:** The number of samples drawn from the buffer for a single optimization step.
- **Impact:** Smaller values can introduce more noise in the gradients, hindering convergence. Conversely, too large values may lead to longer training times and increased memory load.

### 6. `prefer_continuous_actions`
- **Description:** Indicates whether the model should support continuous actions.
- **Impact:** Enabling this option affects compatibility with different environments. In environments with discrete actions, this may lead to suboptimal strategies.

### 7. `exploration_eps_init`
- **Description:** Initial value of the exploration parameter in the ε-greedy strategy.
- **Impact:** The higher the value, the more exploratory actions are taken at the beginning of training, which can help in discovering better strategies. A too-low parameter can lead to local minima.

### 8. `exploration_anneal_frames`
- **Description:** The number of frames after which ε is reduced to its final value (`exploration_eps_end`).
- **Impact:** A shorter period leads to faster transition from exploration to exploitation, which may be beneficial in simple environments, but in more complex ones, it may result in lost opportunities for discovering better strategies.

---

## 📝 Conclusions

- **Algorithm Stability:** VDN generally shows greater result variability (wider value ranges), while QMIX appears to be more stable, evident in narrower value ranges. This indicates that QMIX may perform better in more complex scenarios.

- **Training Time:** The average training time for the QMIX algorithm was about 30% longer than for VDN. This suggests that QMIX may require more computational resources, which is worth considering when selecting the algorithm.

- **Sensitivity to Hyperparameters:** Both algorithms showed significant sensitivity to the learning rate and batch size. Proper tuning of these parameters is crucial for achieving high-quality results.

- **Exploration:** Increasing the exploration range contributed to better strategy discovery but also introduced greater result variability. This suggests that there is an optimal level of exploration that should be determined depending on the task specifics.

- **Architecture Optimization:** The experiments focused on hyperparameter optimization without interfering with the neural network architecture sizes. This approach allows for maximizing the potential of the model, which is critical for efficiency at lower computational costs. Smaller architectures also lead to faster training and reduced resource consumption.



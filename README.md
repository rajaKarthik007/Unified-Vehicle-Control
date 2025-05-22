# 🚗 Robust Reinforcement Learning for Mixed Autonomy Traffic Systems

This project explores reinforcement learning-based control for mixed-autonomy traffic systems, focusing on policy robustness, safety, and generalization. We develop and benchmark PPO and TRPO agents in multi-agent environments, simulating both autonomous and human-driven vehicles using the SUMO traffic simulator.

---

## 🧠 Objective

To design and evaluate scalable reinforcement learning (RL) policies for traffic systems where human-driven and autonomous vehicles coexist. The goal is to improve traffic throughput, safety, and fuel efficiency by using decentralized, policy-gradient-based RL agents.

---

## ⚙️ Methods & Techniques

- **RL Algorithms:** PPO (Proximal Policy Optimization), TRPO (Trust Region Policy Optimization)
- **Stabilization Techniques:**  
  - Manual KL divergence decay  
  - Entropy regularization  
  - Return normalization  
- **Parallel Training:** Python `multiprocessing` for 40+ parallel SUMO rollout environments
- **Policy Architecture:** Two-layer feedforward neural networks with Tanh activations

---

## 🧪 Simulation Details

- **Simulator:** [SUMO]. (Simulation of Urban MObility)
- **Scenarios:**  
  - Single-lane ring road  
  - Highway ramp merging  
- **Human-driven vehicles:** Modeled using IDM (Intelligent Driver Model)
- **Autonomous vehicles:** Controlled by trained RL policies

---

## 📈 Results

| Metric             | PPO Agent      | TRPO Agent     | IDM Baseline  |
|--------------------|----------------|----------------|----------------|
| Avg Speed          | ✅ +20%         | ✅ +18%         | Baseline       |
| Collision Rate     | ✅ 0 collisions | ✅ 0 collisions | Moderate       |
| Policy Stability   | ✅ High         | ✅ High         | —              |

- **20% increase** in average vehicle speed compared to rule-based IDM controller
- **Zero collision rate** with both PPO and TRPO agents
- **Smooth traffic flow** observed via time-space diagrams (reduced stop-and-go waves)

---

## 🧩 POMDP Formulation

- **State Space:** AV's positions, velocities, headways
- **Action Space:** Discrete or continuous acceleration/lane-change actions
- **Observations:** Local ego-vehicle sensor views (partial observability)
- **Reward Function:** Combines average speed, accleration penalty

---
## 🔁 Reproducibility

- Fixed seeds across agents and environments
- Randomized initial conditions to improve policy generalization
- Evaluation in SUMO GUI mode for visual inspection

---

## 🛠️ Tech Stack

- Python, PyTorch, SUMO, Traci API, Matplotlib, NumPy, Multiprocessing
- Tested on macOS with MPS acceleration and Anaconda environment

---

## 📌 Future Work

- Add support for domain randomization (weather, road conditions)
- Integrate interpretable policy architectures (attention-based)
- Extend to multi-lane, city-level traffic networks
- Train with curriculum learning for more complex merge behaviors

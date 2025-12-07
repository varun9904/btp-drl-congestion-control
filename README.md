# Adaptive TCP Congestion Control using Deep Reinforcement Learning

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NS-3](https://img.shields.io/badge/ns--3-3.38-orange)
![Status](https://img.shields.io/badge/status-active-success)

> **B.Tech Major Project (BTP-I) | 2025-26 Academic Year**
> **Netaji Subhas University of Technology, New Delhi**

**Authors:** Deepanshu Sharma, Varun Sharma, Madhur Bakshi
**Supervisor:** Dr. Vivek Mehta

---

## Abstract

Transmission Control Protocol (TCP) congestion control determines how aggressively a sender injects traffic into the network, balancing throughput and delay. Although **TCP CUBIC** is the default algorithm in modern Linux deployments, its static configuration makes it difficult to cope with highly variable network conditions such as wireless loss, long-propagation paths, and persistent queues.

This project introduces an **adaptive variant of CUBIC** in which a Deep Reinforcement Learning (DRL) agent, trained using **Proximal Policy Optimization (PPO)**, continuously adjusts two internal parameters:
1.  The multiplicative window reduction factor ($\beta$)
2.  The cubic growth coefficient ($C$)

By restricting parameter updates to safe operating ranges, we preserve CUBIC's inherent stability while achieving significant throughput gains in lossy and high-delay environments.

---

## Key Features

* **Hybrid Architecture:** Retains the core logic of TCP CUBIC while using an RL agent for dynamic parameter tuning, ensuring the protocol remains deployable and interpretable.
* **Safety Bounds:** The agent's actions are mapped to empirically safe ranges to prevent network instability or collapse.
* **Scenario-Specific Optimization:**
    * **Wireless:** Recovers faster from random packet loss (non-congestion events).
    * **Satellite:** Adapts to high Bandwidth-Delay Products (BDP).
    * **Bufferbloat:** Mitigates queue buildup in cable modem scenarios.
* **Fairness:** Demonstrated safe coexistence with standard CUBIC flows, maintaining a bandwidth share of 51-64%.

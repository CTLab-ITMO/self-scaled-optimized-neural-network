# SSONN: Self-Scaled Optimized Neural Network

## Overview

This work introduces a new way to build neural networks that grow as they learn, rather than starting large and trimming down. The core idea is to begin with a very small network and intelligently add connections where they are most needed during training – essentially letting the network design itself for optimal efficiency. It focuses on building sparse networks, meaning many of the potential connections between neurons aren't present, reducing computational cost.

The system automatically identifies which connections to add based on how much each one contributes to learning. This dynamic expansion aims to achieve similar accuracy to traditional networks but with significantly fewer parameters, making it more sustainable and suitable for devices with limited resources. Experiments show promising results on image recognition tasks, demonstrating the ability to match dense network performance while being far more compact. Future development will broaden layer support and address potential overfitting issues.

---

## Used algorithms

The codebase implements several algorithms centered around dynamically growing a neural network during training.

**Sparse Network Initialization:** The process begins with creating a minimal, sparse network structure instead of a fully connected one. This sets the stage for efficient expansion.

**Edge Importance Scoring:** A key algorithm calculates the importance of each potential connection (edge) in the network.  This is done using the mean absolute gradient magnitude – essentially measuring how much impact adding that connection would have on learning.

**Dynamic Edge Expansion:** Based on the edge importance scores, new connections are added to the network. A threshold determines which edges are considered important enough to be included, effectively growing the network's capacity where it’s most needed.

**Weight Initialization for New Connections:** When a new connection is added (through layer expansion with intermediate sublayers), specific weight initialization techniques are used to ensure that adding this connection doesn't disrupt the existing learned behavior of the network. The goal is seamless integration of the new capacity.

**Edge Pruning:**  To maintain efficiency and prevent overfitting, unimportant connections are removed from the network. This pruning process also uses a threshold-based approach, identifying edges with low importance scores for removal.

**Iterative Training & Expansion/Pruning Cycles:** The core of the system involves repeatedly cycling through these steps: training the current network, evaluating edge importance, expanding by adding important connections, and pruning unimportant ones. This iterative process allows the network to adapt its structure during learning.

**Sensitivity Analysis (for Pruning):** While not explicitly detailed as a primary algorithm, sensitivity analysis is mentioned in relation to pruning, suggesting it's used to assess how much removing a connection would impact performance before actually deleting it.

---

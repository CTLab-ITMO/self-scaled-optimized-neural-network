# self-expanding-nets

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

This work introduces a new way to build neural networks that grow as they learn, rather than starting large and trimming down. The core idea is to begin with a very small network and intelligently add connections where they are most needed during training – essentially letting the network design itself for optimal efficiency. It focuses on building sparse networks, meaning many of the potential connections between neurons aren't present, reducing computational cost.

The system automatically identifies which connections to add based on how much each one contributes to learning. This dynamic expansion aims to achieve similar accuracy to traditional networks but with significantly fewer parameters, making it more sustainable and suitable for devices with limited resources. Experiments show promising results on image recognition tasks, demonstrating the ability to match dense network performance while being far more compact. Future development will broaden layer support and address potential overfitting issues.

---

## Repository content

The self-expanding-nets repository implements a novel approach to neural network training called Self-Scaled Optimized Neural Networks (SSONN). The core idea is to start with a minimal network and dynamically expand it during training, rather than pre-training a large model and then compressing it. Here's a breakdown of the key components and their relationships:

1. **Model (`senmodel/model`)**: This component defines the building blocks of the expanding neural networks. It includes `SparseModule`, an abstract base class for sparse network modules, and concrete implementations like `EmbedLinear` and `ExpandingLinear`. These classes manage the addition of connections (edges) and weights in a sparse manner, allowing the network to grow efficiently. The `ExpandingLinear` class is central as it handles the iterative expansion process.

2. **Training (`senmodel/train`)**: This component contains the training loop and related functions. It includes `train_one_epoch` and `eval_one_epoch` for standard training and evaluation, but crucially also incorporates functions like `edge_replacement_func_new_layer` and `edge_deletion_func_new_layer`. These functions implement the core SSONN logic of adding and removing connections based on gradient information.

3. **Metrics (`senmodel/metrics`)**: This component provides tools for evaluating edge importance and guiding the expansion process. The `EdgeFinder` class uses a `NonlinearityMetric` (like `AbsGradientEdgeMetric`) to calculate an importance score for each connection in the network, determining which edges should be added or removed.

4. **Experiments (`experiments`)**: This directory contains Jupyter notebooks used for experimentation and demonstration. The `experiment.ipynb` notebook showcases a complete training pipeline using SSONN on a simple dataset (likely MNIST or Fashion-MNIST). It ties together the model, training loop, and metrics to demonstrate the functionality.

5. **Pruning (`experiments/speed_eval`)**: This component provides functions for evaluating sensitivity of weights and performing pruning operations based on different strategies like L1, Random, and SensitivityBased pruning.

The components work together as follows: The training loop uses the model to make predictions and calculate a loss.  The metrics component then analyzes the gradients to identify important edges. Based on this analysis, the training process dynamically expands or prunes the network using functions in the `senmodel/train` module. This iterative cycle continues until the desired performance is achieved. The experiments directory provides examples of how these components are used together.

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

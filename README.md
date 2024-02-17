![Qiskit](https://img.shields.io/badge/Qiskit-%236929C4.svg?style=for-the-badge&logo=Qiskit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)

<h3 align="center">Quantum Graph Neural Netork for solving TSP</h3>
<p align="center">
    Bachelor's degree final computer engineering project on Quantum Machine Learning.
    <br />
</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#methodology">Methodology</a></li>
    <li><a href="#project-files">Project Files</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## Introduction
This project focuses on Quantum Computing and explores solutions to the Travelling Salesperson Problem (TSP). The main objective of this project is to learn and research Quantum Computing concepts, rather than implementing something entirely new. One of the key components of the project is a Reinforcement Learning model utilizing Equivariant Quantum Circuits.

The primary objectives of this project are focused on learning and researching Quantum Computing concepts rather than implementing entirely new solutions. The exploration of Quantum Computing solutions for the Travelling Salesperson Problem (TSP) serves as a practical application to deepen understanding in this field.

The project entails the development of a Reinforcement Learning model using Equivariant Quantum Circuits to address the TSP. Equivariant Quantum Circuits play a crucial role in preserving symmetries within the quantum system, making them ideal for modeling physical systems accurately.

## Methodology

### Introduction to Quantum Computing
Quantum computing is a revolutionary paradigm that leverages the principles of quantum mechanics to perform computations and solve complex problems. Unlike classical computing, which utilizes bits as basic units of information, quantum computing employs quantum bits or qubits, which can exist in states of superposition and entanglement, enabling them to process vast amounts of information simultaneously. 

Quantum computing has the potential to significantly enhance computational power, particularly in the realms of cryptography, optimization, and simulation of quantum systems. The development of quantum algorithms and quantum machine learning techniques holds promise for tackling computationally intensive tasks that are beyond the capabilities of classical computers. As the field of quantum computing continues to advance, it is expected to unlock new opportunities for tackling challenges across various scientific and technological domains, driving innovation and breakthroughs in diverse fields.

### Equivariant Quantum Circuit
Equivariance under node permutation refers to the property or characteristic of a system, such as a graph neural network, to maintain its structure and function under the permutation of its nodes. When nodes of a system are rearranged, the system's behavior and response remain consistent, ensuring the preservation of underlying symmetries. This property is crucial in applications involving graphs and networks, as it ensures that the model's predictions and inferences are independent of the specific labeling or ordering of the nodes, offering robustness and generalizability. For instance, in graph neural networks, equivariance under node permutation allows the models to effectively learn and represent graph-structured data, irrespective of the node ordering, enhancing their applicability across diverse graph-based problems.

![image](https://i.imgur.com/5hSc6je.png)

An Equivariant Quantum Circuit integrates symmetries into quantum machine learning models, ensuring that the circuit's response is invariant under certain transformational operations that align with the symmetry properties relevant to the problem domain. These circuits are tailored for learning on structures like weighted graphs by embedding the equivalent representation of the graph's symmetry group into the circuit. This approach is instrumental for modeling systems where these symmetries are an intrinsic feature and is a valuable contribution to the quantum machine learning field.

![image](https://i.imgur.com/xcCkSzA.png)

The EQC is an advanced type of quantum circuit designed to maintain symmetries found in the quantum system it represents. This is achieved by embedding the symmetry group of the problem domain, such as a weighted graph, into the quantum circuit. The circuit's response is invariant under transformational operations that align with the inherent symmetries of the system. Notably, this approach is pivotal for modeling physical systems where specific symmetries must be preserved during quantum computations. The implementation of EQC involves the integration of group theory, quantum physics, and machine learning, making it a multifaceted and sophisticated methodology.

### Reinforcement Learning Model
Our Reinforcement Learning model is calibrated to solve the Traveling Salesperson Problem (TSP) through a procedure informed by the worst-case heuristic analysis. The model iteratively seeks the optimal tour, guided by a reward structure that encourages the identification of shorter paths. With the incorporation of quantum machine learning algorithms, such as variational quantum circuits, the model's robustness to hardware errors is enhanced, thus making it a resilient framework suited for near-term quantum devices. The model's convergence characteristics demonstrate the algorithm's efficacy in learning to approximate solutions for TSP efficiently.

![image](https://i.imgur.com/1yJaV6P.png)

The Reinforcement Learning model, which leverages the Christofides algorithm, is specifically tailored to converge towards an optimal tour for the Traveling Salesperson Problem (TSP). This is achieved through iterative search processes guided by reward structures to encourage the identification of shorter paths. The incorporation of quantum machine learning algorithms, such as variational quantum circuits, into the model enhances its resilience to hardware errors, making it well-suited for near-term quantum devices. The convergence characteristics of this model illustrate its efficacy in efficiently approximating solutions for TSP, demonstrating promising progress in addressing complex combinatorial optimization problems.

### Results
Our empirical findings exhibit significant advancement in TSP solution methodology, with the reinforcement learning model yielding an approximation to an optimal tour within a mean of 300 iterations, as guided by the principles of the Christofides algorithm. These results are indicative of the model's capability to effectively leverage quantum computation paradigms to address complex combinatorial optimization problems and contribute to the broader spectrum of quantum machine learning research. 

![image](https://i.imgur.com/CvaLd6d.png)

## Project Files
This repository contains the code and documentation for the Bachelor's Degree Final Project (Catalan: *Treball de Final de Grau*; TFG) in Computer Engineering at Universitat Autònoma de Barcelona (UAB) titled *Quantum Graph Neural Network for solving TSP*. This project explores the exciting intersection of quantum computing and machine learning to tackle the so called Travelling Salesperson Problem (TSP).

### File structure

The project is organized as follows:

- `data/`: This directory contains the image datasets used for training and testing during initial testing phases.
- `docs/`: Project documentation and presentation files.
- `imgs/`: Project images and plots.
- `notebooks/`: Jupyter notebooks demonstrating the experiments and results.
- `utils/`: Utility functions and helper scripts.
- `.gitignore`: Github .gitignore file to get rid of annoying files.
- `LICENSE.txt`: The project's license file.
- `README.md`: This documentation file.
- `requirements.txt`: The Python requirements text file.

### Tutorial
For the step-by-step implementation of the code, please refer to [this tutorial](https://github.com/yeray142/QML-QGNN/blob/master/notebooks/q_learning_tutorial.ipynb). The tutorial provides a comprehensive guide with detailed explanations and code samples for easy understanding of the implementation process.

## Installation

### Prerequisites
> [!WARNING]
> If you are using a Python version lower than 3.9, you are required to import the specific definitions from the typing module for the composite type hints.

Before you can run the code in this repository, you will need the following prerequisites:

- Basic understanding of quantum computing and machine learning.
- Python 3.11 or higher (lower versions may also work well, but we will not support them).
- Access to a quantum computing platform or simulator (e.g., IBM Qiskit, Google Cirq).
- Required Python libraries, which can be installed using `pip`:
```
pip install -r requirements.txt
```

### Usage

To replicate the experiments or use the implemented models, follow these steps:

1. Clone this repository to your local machine:
```
git clone https://github.com/yeray142/QML-QGNN.git
```
2. Navigate to the project directory:
```
cd <directory_where_the_cloned_repository_is_located>
```
3. Open the Jupyter notebooks in the `notebooks/` directory to replicate the experiments and results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

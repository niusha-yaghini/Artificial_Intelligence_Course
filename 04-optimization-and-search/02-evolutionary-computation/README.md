# Evolutionary Computation

This section focuses on population-based optimization algorithms inspired by biological evolution. 
Unlike classical local search methods that improve a single solution, evolutionary algorithms maintain a population of candidate solutions and iteratively improve them through selection, recombination, and variation operators.

The main goal of this section is to understand the design principles behind evolutionary optimization, including:

- Solution representation
- Fitness evaluation
- Population initialization
- Selection strategies
- Crossover / recombination operators
- Mutation mechanisms
- Exploration–exploitation trade-off


## Project Structure

03-evolutionary-computation/

│
├── problems/
│
├── src/
│
├── 01_genetic_algorithm.ipynb
├── 02_differential_evolution.ipynb
├── 03_genetic_programming.ipynb
│
└── README.md



## Problems

The `problems` directory contains optimization problem definitions that are independent from the optimization algorithms.

Separating problems from algorithms allows:

- Reusing the same problem across multiple algorithms
- Comparing different optimization strategies fairly
- Keeping algorithm implementations problem-independent
- Supporting different solution representations

Each problem defines the required components such as:

- Solution representation
- Objective / fitness function
- Constraints
- Evaluation mechanism


Examples of possible problems:

- 0/1 Knapsack Problem
- Traveling Salesman Problem (TSP)
- Symbolic Regression
- Benchmark Optimization Functions


## Source Algorithms

The `src` directory contains the implementation of evolutionary algorithms.

Each algorithm focuses on the optimization mechanism rather than a specific application.

Implemented algorithms:

### Genetic Algorithm (GA)

A population-based evolutionary algorithm that uses:

- Selection
- Crossover
- Mutation

to evolve better solutions over generations.


### Differential Evolution (DE)

A continuous optimization algorithm based on:

- Population vectors
- Mutation through vector differences
- Recombination
- Selection


### Genetic Programming (GP)

An evolutionary approach where solutions are represented as programs or expression trees.

It is commonly used for:

- Symbolic regression
- Automated program generation


## Notebook Organization

Each notebook introduces one evolutionary algorithm and follows a consistent workflow:

1. Problem formulation
2. Solution representation
3. Algorithm intuition
4. Implementation details
5. Experimental evaluation
6. Parameter analysis
7. Visualization and result interpretation


## Algorithm Comparison

When multiple algorithms can solve the same problem, their performance can be compared based on:

- Solution quality
- Convergence behavior
- Computational cost
- Parameter sensitivity


The goal is not only to obtain good solutions, but also to understand the strengths, limitations, and suitable applications of each evolutionary method.



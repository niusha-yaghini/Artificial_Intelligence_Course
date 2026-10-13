# Genetic Algorithm — Final Summary

## Overview

Genetic Algorithm (GA) is an evolutionary optimization method inspired by natural selection and genetic evolution. Instead of improving a single solution, GA maintains a population of candidate solutions and evolves them through multiple generations.

The main idea is to balance:

- **Exploration:** discovering new regions of the search space
- **Exploitation:** improving already promising solutions

---

## GA Workflow

The implemented GA follows this pipeline:

Population Initialization

↓

Fitness Evaluation

↓

Elitism (Preserve Best Individuals)

↓

Parent Selection

↓

Crossover

↓

Mutation

↓

New Generation Creation


Each component has a specific role:

| Component | Purpose |
|---|---|
| Population | Represents multiple candidate solutions |
| Fitness Function | Measures solution quality |
| Elitism | Preserves the best solutions between generations |
| Selection | Chooses promising parents |
| Crossover | Combines information from multiple parents |
| Mutation | Introduces diversity and prevents premature convergence |

---

## Representation

For the 0/1 Knapsack problem, each solution was represented as a binary chromosome.

Example:
[1,0,1,1,0,1,...]


Each gene represents whether an item is selected:

- 1 → item included
- 0 → item excluded

---

## Selection Strategies

Two selection methods were implemented and compared:

### Tournament Selection

Tournament selection chooses a subset of individuals and selects the best one.

Advantages:

- Higher selection pressure
- Faster convergence
- Strong exploitation capability

Disadvantage:

- Excessive pressure may reduce diversity

---

### Roulette Wheel Selection

Roulette selection assigns selection probability proportional to fitness.

Advantages:

- Maintains more diversity
- Provides stronger exploration capability

Disadvantage:

- Usually converges slower

---

## Genetic Operators

### Crossover

Single-point crossover was implemented.

Two parent chromosomes exchange genetic information to create new offspring.

Purpose:

- Combine useful characteristics from different solutions
- Accelerate search toward promising regions


### Mutation

Binary mutation was applied after offspring generation.

Purpose:

- Introduce new genetic information
- Maintain population diversity
- Reduce the risk of premature convergence

---

# Experimental Analysis

The GA was evaluated on a large-scale 0/1 Knapsack problem with 100 binary decision variables.

Several experiments were performed.

---

## 1. Selection Strategy Comparison

Tournament selection showed faster convergence and achieved a slightly better final fitness.

Roulette wheel selection maintained more diversity due to its probabilistic nature.

Conclusion:

Tournament selection provided stronger exploitation, while Roulette selection provided better exploration.

---

## 2. Mutation Rate Analysis

Different mutation rates were evaluated.

Results showed:

- Very low mutation reduced diversity and slowed convergence.
- Very high mutation disturbed useful genetic patterns.
- A moderate mutation rate provided the best balance.

The mutation rate of:
0.01


provided the best exploration-exploitation trade-off in this experiment.

---

## 3. Elite Ratio Analysis

Different elitism ratios were compared.

Results showed:

- Without elitism, good solutions could be lost.
- Moderate elitism improved stability.
- Excessive elitism reduced exploration.

An elite ratio between:
5% - 10%


provided a good balance between preserving quality solutions and maintaining diversity.

---

## 4. Population Size Analysis

Increasing population size improved exploration capability and convergence speed.

However, after a certain point, increasing population size provided limited improvement while increasing computational cost.

This demonstrates the trade-off between:

- Solution quality
- Computational complexity

---

# Final Remarks

The implemented Genetic Algorithm successfully optimized a large-scale combinatorial problem by combining:

- Population-based search
- Selection pressure
- Genetic recombination
- Random mutation
- Elitist preservation

The experiments demonstrated that GA performance strongly depends on parameter configuration and that proper balancing between exploration and exploitation is essential for effective optimization.
# Ant Colony Optimization (ACO) — Final Summary

## 1. Introduction

Ant Colony Optimization (ACO) is a population-based metaheuristic algorithm inspired by the foraging behavior of real ants.

In nature, ants are able to find short paths between their nest and food sources by depositing pheromone trails. Other ants are more likely to follow paths with stronger pheromone concentrations, which creates a collective learning mechanism.

ACO transforms this behavior into an optimization algorithm where artificial ants construct solutions and improve them through pheromone-based communication.

---

# 2. Optimization Problem Representation

ACO is particularly suitable for combinatorial optimization problems such as:

- Travelling Salesman Problem (TSP)
- Vehicle Routing Problem (VRP)
- Scheduling problems
- Network routing problems

Unlike continuous optimization algorithms such as PSO and DE, the solution in ACO is usually represented as a discrete structure.

For TSP, a solution is represented as a permutation:

$Route=[c_0,c_1,c_2,...,c_n]$

where each city is visited exactly once and the route returns to the starting city.

Example:

$0 \rightarrow 3 \rightarrow 1 \rightarrow 5 \rightarrow 0$

---

# 3. Main Components of ACO

## 3.1 Pheromone Matrix

ACO maintains a pheromone matrix:

$\tau_{ij}$

where each element represents the learned desirability of traveling from city \(i\) to city \(j\).

Initially, all edges have equal pheromone values:

$\tau_{ij}=\tau_0$

During optimization, good routes receive more pheromone reinforcement.

---

## 3.2 Heuristic Information

ACO also uses problem-specific information.

For TSP:

$\eta_{ij}=\frac{1}{d_{ij}}$

where:

- \(d_{ij}\) is the distance between two cities.
- shorter edges receive higher heuristic values.

---

# 4. Transition Probability

The probability of selecting the next city is calculated as:

$P_{ij}
=
\frac{
\tau_{ij}^{\alpha}
\eta_{ij}^{\beta}
}
{
\sum
\tau_{ik}^{\alpha}
\eta_{ik}^{\beta}
}$


where:

- \(\alpha\): importance of pheromone information.
- \(\beta\): importance of heuristic information.


Large values of:

- α increase exploitation of previous experience.
- β increase preference toward shorter distances.

---

# 5. Pheromone Update

After ants construct their solutions, pheromone values are updated.

## 5.1 Evaporation

To avoid unlimited accumulation:

$\tau_{ij}
\leftarrow
(1-\rho)\tau_{ij}$

where:

$\rho$

is the evaporation rate.

Evaporation provides exploration by reducing the influence of old solutions.

---

## 5.2 Deposit

Ants reinforce the edges belonging to good routes:

$\Delta\tau_{ij}
=
\frac{Q}{L}$

where:

- \(Q\) is pheromone intensity.
- \(L\) is route length.

Shorter routes receive larger pheromone updates.

---

# 6. Implementation Steps

The implemented ACO algorithm consists of:

1. Generate initial pheromone matrix.
2. Construct routes for multiple ants.
3. Evaluate route lengths.
4. Update global best solution.
5. Apply pheromone evaporation.
6. Deposit pheromone on promising routes.
7. Repeat until convergence.

---

# 7. Experimental Evaluation

## 7.1 Small TSP Experiment

ACO was first tested on an 8-city TSP problem.

Results showed:

- feasible Hamiltonian cycle generation.
- rapid convergence.
- successful pheromone reinforcement.

Because the search space was small:

$8!=40320$

possible routes exist, causing fast convergence.

---

## 7.2 Large TSP Experiment

A larger problem with 30 cities was evaluated.

The search space becomes:

$30! \approx 2.65\times10^{32}$

possible routes.

ACO successfully found a feasible route:

$L=5.6579$

The convergence curve demonstrated that:

- the solution improved rapidly during early iterations.
- pheromone learning guided ants toward better routes.
- the algorithm reached a stable solution.

---

# 8. Parameter Analysis

## 8.1 Alpha Analysis

Different values:

$\alpha \in \{0.5,1,2,3\}$

were tested.

The best performance was obtained at:

$\alpha=1$

This indicates that a balanced influence of pheromone information provides better exploration-exploitation trade-off.

---

## 8.2 Beta Analysis

Different values:

$\beta \in \{1,2,3,5\}$

were evaluated.

The best result was achieved at:

$\beta=3$

Moderate heuristic influence improved route selection, while very large values increased greedy behavior.

---

## 8.3 Evaporation Rate Analysis

Different evaporation rates:

$\rho \in \{0.1,0.3,0.5,0.7\}$

were tested.

Results showed that ACO performance was relatively robust to evaporation changes.

The evaporation parameter mainly controls:

- memory preservation,
- exploration capability.

---

# 9. Strengths of ACO

Advantages:

- Effective for combinatorial optimization problems.
- Naturally handles discrete solutions.
- Does not require gradient information.
- Provides adaptive learning through pheromone updates.
- Can incorporate problem constraints easily.

---

# 10. Limitations of ACO

Limitations:

- Computational cost increases with problem size.
- Parameter tuning is important.
- Can suffer from premature convergence.
- Usually slower than continuous optimization algorithms on simple numerical problems.

---

# 11. Final Conclusion

In this project, an Ant Colony Optimization algorithm was implemented and evaluated on TSP problems.

The experiments demonstrated that:

- artificial ants can construct feasible solutions.
- pheromone-based learning improves route quality.
- ACO successfully scales from small to larger TSP instances.
- parameter selection affects the balance between exploration and exploitation.

Overall, ACO is a powerful metaheuristic method for discrete optimization problems, especially when the solution space is combinatorial and traditional optimization methods become impractical.
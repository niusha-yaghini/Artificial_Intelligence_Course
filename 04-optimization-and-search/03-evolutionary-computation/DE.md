# Differential Evolution (DE) — Final Summary

## Overview

Differential Evolution (DE) is a population-based evolutionary optimization algorithm designed mainly for continuous optimization problems.

Unlike Genetic Algorithms that manipulate chromosomes using genetic operators, DE operates directly on real-valued vectors and uses differences between population members to guide the search.

The main idea of DE is:

> Use the relative distance and direction between solutions to generate new candidate solutions.

This allows DE to perform directed exploration of the search space instead of relying only on random modifications.

---

# Difference Between GA and DE

Although both algorithms belong to Evolutionary Computation, their search mechanisms are different.

| | Genetic Algorithm | Differential Evolution |
|-|-|-|
|Representation|Chromosome|Real-valued vector|
|Typical problems|Discrete / combinatorial|Continuous optimization|
|Main operator|Crossover + Mutation|Differential Mutation|
|Search information|Genetic combination|Vector differences|
|Replacement|Generation based|Individual competition|

---

# DE Representation

Each solution is represented as a continuous vector.

Example:

$X=[x_1,x_2,...,x_D]$

where each element represents a decision variable.

For example:
[2.5, -1.3, 4.8]


represents one candidate solution.

The population is a collection of such vectors:


[
[1.2,3.4],
[-2.1,0.5],
[4.0,-1.7]
]


---

# Differential Evolution Workflow

The implemented DE follows this pipeline:

Initialize Population

    ↓

Mutation (DE/rand/1)

    ↓

Boundary Handling

    ↓

Binomial Crossover

    ↓

Selection

    ↓

Next Generation


---

# 1. Population Initialization

The initial population is randomly generated inside the defined search bounds.

For example:

$-5 \leq x_i \leq 5$

Each individual represents one possible solution.

---

# 2. Differential Mutation

The main mechanism of DE is mutation based on vector differences.

The implemented strategy:

## DE/rand/1

$V_i=X_{r1}+F(X_{r2}-X_{r3})$


where:

- $X_{r1}$ is the base vector
- $X_{r2},X_{r3}$ generate the differential direction
- $F$ controls the step size


The difference:

$X_{r2}-X_{r3}$

provides both:

- movement direction
- movement magnitude

---

# 3. Boundary Handling

Mutation may create solutions outside the allowed search space.

Therefore, boundary handling was applied.

Example:

Before:
[8,-7]

with bounds:
[-5,5]

After clipping:
[5,-5]



This ensures all candidate solutions remain feasible.

---

# 4. Binomial Crossover

After creating the mutant vector, DE combines it with the original target vector.

The crossover rate:

$
CR
$

controls the probability of taking components from the mutant vector.

The result is:

$
Trial\ Vector
$

which becomes a new candidate solution.

---

# 5. Selection

DE uses greedy selection.

Each individual competes only with its own trial vector:

$
Trial_i \quad vs \quad X_i
$


For minimization:

$
X_i^{new}
=
\begin{cases}
Trial_i & f(Trial_i)<f(X_i)\\
X_i & otherwise
\end{cases}
$


This provides a natural elitism mechanism because better solutions are never replaced by worse ones.

---

# Benchmark Evaluation

The implemented DE algorithm was evaluated on three standard continuous optimization functions.

---

# 1. Sphere Function

$
f(x)=\sum x_i^2
$

Global optimum:

$
x^*=[0,0]
$

$
f(x^*)=0
$


Characteristics:

- Simple convex landscape
- No local optimum
- Tests basic convergence ability

Result:

DE rapidly converged to the global optimum.

---

# 2. Rosenbrock Function

$
f(x_1,x_2)=
(1-x_1)^2+
100(x_2-x_1^2)^2
$


Global optimum:

$
x^*=[1,1]
$


Characteristics:

- Narrow curved valley
- Requires accurate search direction
- Tests exploitation capability

Result:

DE successfully followed the valley and reached:

$
f(x^*)\approx0
$

---

# 3. Rastrigin Function

$
f(x)=10n+\sum
(x_i^2-10cos(2\pi x_i))
$


Global optimum:

$
x^*=[0,0]
$


Characteristics:

- Highly multimodal
- Many local optima
- Requires strong exploration capability

Result:

DE escaped local optima and successfully reached the global minimum.

---

# Parameter Effects

## Differential Weight (F)

Controls the magnitude of differential movement.

Increasing F:

- Larger search steps
- More exploration
- Better escape from local optima
- Possible instability near optimum


Decreasing F:

- Smaller movements
- Stronger exploitation
- Faster refinement
- Higher risk of premature convergence


---

## Crossover Rate (CR)

Controls how much information is transferred from the mutant vector.

Higher CR:

- More changes from mutation
- Higher diversity
- Stronger exploration


Lower CR:

- More conservative updates
- Stable convergence
- Reduced exploration


---

## Population Size

Controls search diversity.

Larger population:

- Better coverage of search space
- More robust optimization
- Higher computational cost


Smaller population:

- Faster execution
- Lower diversity
- Higher risk of premature convergence


---

# Final Remarks

The implemented Differential Evolution algorithm successfully optimized continuous benchmark functions by combining:

- Population-based search
- Vector-difference mutation
- Recombination through crossover
- Greedy selection

The experiments demonstrated that DE can effectively balance exploration and exploitation and is particularly powerful for continuous parameter optimization problems.

Unlike Genetic Algorithms that search through discrete representations, DE directly optimizes numerical parameters by exploiting the geometric relationship between candidate solutions.
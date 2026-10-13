# Final Summary — Particle Swarm Optimization (PSO)

## 1. Overview

Particle Swarm Optimization (PSO) is a population-based metaheuristic algorithm inspired by the social behavior of biological swarms such as bird flocking and fish schooling.

Unlike evolutionary algorithms that generate new individuals through genetic operators, PSO models optimization as a movement process where particles adjust their positions according to their own experience and the experience of the swarm.

The main idea is:

> Simple agents interacting with each other can produce an intelligent global search behavior.

---

# 2. Particle Representation

In PSO, each candidate solution is represented as a particle.

Each particle contains:

Particle

├── Position
├── Velocity
├── Fitness
├── Personal Best Position (pbest)
└── Personal Best Fitness


---

## Position

The position represents the current solution.

For a two-dimensional optimization problem:

$X_i=[x_1,x_2]$

Example:

$X_i=[2.5,-1.3]$


---

## Velocity

Velocity determines the direction and magnitude of particle movement.

$V_i=[v_1,v_2]$


The velocity controls how the particle moves through the search space.

---

# 3. Personal Best and Global Best

PSO uses two types of memory.

---

## Personal Best (pbest)

Each particle stores the best solution it has found so far.

$pbest_i$


This represents individual learning.

---

## Global Best (gbest)

The swarm stores the best solution found by all particles.

$gbest$


This represents social learning.

---

# 4. PSO Mathematical Model

The optimization process is based on two main equations.

---

## Velocity Update

$v_i(t+1)=
wv_i(t)
+
c_1r_1(pbest_i-x_i)
+
c_2r_2(gbest-x_i)$


The equation contains three components:


### 1. Inertia Component

$wv_i(t)$

Controls the tendency of the particle to continue its previous movement.

---

### 2. Cognitive Component

$c_1r_1(pbest_i-x_i)$

Represents learning from personal experience.

---

### 3. Social Component

$c_2r_2(gbest-x_i)$

Represents learning from the best solution found by the swarm.

---

## Position Update

After updating velocity:

$x_i(t+1)=x_i(t)+v_i(t+1)$


The particle moves to a new location in the search space.

---

# 5. PSO Algorithm Workflow

The complete PSO procedure is:

Initialize Particles

    ↓

Evaluate Fitness

    ↓

Update pbest

    ↓

Update gbest

    ↓

Update Velocity

    ↓

Update Position

    ↓

Boundary Handling

    ↓

Repeat until termination


---

# 6. Boundary Handling

Since particles can move outside the search domain, boundary handling is required.

In this implementation, a clipping strategy was used:

$x_i < x_{min}
\rightarrow x_{min}$


$x_i > x_{max}
\rightarrow x_{max}$


This keeps particles inside the feasible search region.

---

# 7. Benchmark Functions

The PSO implementation was evaluated using four standard optimization functions.

---

# Sphere Function

$f(x)=\sum_{i=1}^{n}x_i^2$


Characteristics:

- Convex function
- Single global minimum
- Tests basic convergence capability


Optimal solution:

$x^*=[0,0]$

---

# Rastrigin Function

$f(x)=10n+
\sum_{i=1}^{n}
(x_i^2-10cos(2\pi x_i))$


Characteristics:

- Highly multimodal
- Contains many local minima
- Tests exploration capability


Optimal solution:

$x^*=[0,0]$

---

# Rosenbrock Function

$f(x)=
\sum_{i=1}^{n-1}
[
100(x_{i+1}-x_i^2)^2
+
(1-x_i)^2
]$


Characteristics:

- Narrow curved valley
- Difficult exploitation problem


Optimal solution:

$x^*=[1,1]$

---

# Ackley Function

$f(x)=
-20e^{-0.2\sqrt{\frac1n\sum x_i^2}}
-
e^{\frac1n\sum cos(2\pi x_i)}
+
20+e$


Characteristics:

- Complex multimodal landscape
- Requires balance between exploration and exploitation


Optimal solution:

$x^*=[0,0]$

---

# 8. Experimental Results

| Function | Best Solution | Fitness |
|---|---|---|
| Sphere | ≈ [0,0] | \(7.47\times10^{-15}\) |
| Rastrigin | ≈ [0,0] | \(2.55\times10^{-13}\) |
| Rosenbrock | ≈ [1,1] | \(7.29\times10^{-7}\) |
| Ackley | ≈ [0,0] | \(3.47\times10^{-7}\) |

---

# 9. Results Analysis

## Sphere

PSO quickly converged toward the global optimum because the function has a simple convex landscape.

The strong attraction toward gbest resulted in stable convergence.

---

## Rastrigin

Despite multiple local minima, PSO successfully reached the global optimum.

This demonstrates the ability of the swarm mechanism to maintain exploration and avoid premature convergence.

---

## Rosenbrock

The algorithm successfully followed the narrow valley and reached a solution close to:

$(1,1)$

The slightly higher error compared with Sphere is expected due to the difficult landscape.

---

## Ackley

PSO achieved a solution very close to the global minimum.

The result confirms that the combination of personal and social learning is effective for complex nonlinear landscapes.

---

# 10. Advantages of PSO

Advantages:

- Simple implementation
- Few control parameters
- No gradient information required
- Effective for continuous optimization
- Natural balance between exploration and exploitation

---

# 11. Limitations of PSO

Limitations:

- Sensitive to parameter selection
- Can suffer from premature convergence
- Performance depends on swarm diversity
- Mainly designed for continuous optimization problems

---

# 12. Comparison with Evolutionary Algorithms

|Algorithm|Representation|Main Mechanism|
|-|-|-|
|GA|Chromosome|Selection, Crossover, Mutation|
|DE|Vector|Differential Mutation|
|GP|Expression Tree|Subtree Evolution|
|PSO|Particle|Velocity-based Movement|

---

# Final Conclusion

Particle Swarm Optimization provides an alternative approach to evolutionary computation by modeling optimization as a collective movement process.

Instead of generating new solutions through genetic operators, PSO particles continuously adjust their positions using personal experience and social knowledge.

The experiments demonstrated that PSO can effectively solve different continuous optimization problems, from simple convex functions to complex multimodal landscapes.

The performance of PSO depends on achieving a proper balance between:

- Exploration of new regions
- Exploitation of promising solutions

through the interaction between pbest and gbest mechanisms.
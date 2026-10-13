COURSE:


An optimization problem usually consists of three main components:

1. Decision Variables
The variables that need to be determined.

2) Objective Function
A function that measures the quality of the solution.

3) Constraints
Sometimes not all solutions are allowed.


What is Search Space?
    The space where all possible solutions lie.

Feasible Region:
    Out of the entire Search Space: Only the points that satisfy the Constraints.

Global Optimum:
    The best solution in the entire space.

Local Optimum:
    A good peak but not the best.


The main optimization problem

Many real-world problems:
Are non-convex.
Have very large spaces.
Are not differentiable.


Exploration vs Exploitation
This is the most important idea of ​​all Metaheuristics.


Exploitation means:
Make the best use of the information we have.

Exploration means:
Explore new areas.

If we only Exploit:
We get stuck:
Local optimum

If we only Explore:
We never converge.


What is a metaheuristic?
A general algorithm that:
Does not guarantee to find the exact answer.
But finds very good answers in a reasonable amount of time.


| Feature            | Hill Climbing | Simulated Annealing | Tabu Search |
| ------------------ | ------------- | ------------------- | ----------- |
| Has memory?        | No            | No                  | Yes         |
| Accepts bad moves? | No            | Yes                 | Possible    |
| Is it random?      | Low           | High                | Medium      |
| Local Optimum      | Main problem  | Reduces             | Reduces     |
| Complexity         | Low           | Medium              | Medium      |


Hill Climbing:
    Just look at the present.

Simulated Annealing:
    Make mistakes sometimes to find better.

Tabu Search:
    Remember the past so that mistakes are not repeated.


One more important point to keep going
You should see a big pattern here:
All of these algorithms have a common problem:
How to balance Exploration and Exploitation?
And each one gives a different answer:

| Algorithm           | Solution                   |
| ------------------- | -------------------------- |
| Hill Climbing       | Strong Exploitation        |
| Simulated Annealing | Random exploration         |
| Tabu Search         | Memory-based exploration   |
| Genetic Algorithm   | Population diversity       |
| PSO                 | Social information sharing |
| ACO                 | Collective memory          |

# Benders Decomposition Methods Documentation

## Overview

Benders Decomposition is a mathematical optimization technique used to solve large-scale problems by decomposing them into smaller, more manageable subproblems. The method is particularly effective for problems with a specific structure, typically involving both complicating and non-complicating variables. Benders Decomposition iteratively solves a master problem and subproblems, refining the solution at each step.

The general structure of a problem suitable for Benders Decomposition includes:
- **Master Problem:** Contains the complicating variables.
- **Subproblems:** Contain the non-complicating variables and depend on the solution of the master problem.

The method alternates between solving the master problem and the subproblems, generating Benders cuts (constraints) to improve the master problem solution iteratively.

### Methodology

Benders Decomposition divides the problem into:
1. **Master Problem:** Optimizes over complicating variables.
2. **Subproblems:** Solve for the non-complicating variables given the solution from the master problem.

The iterative process involves:
1. **Solving the Master Problem:** 
   - The master problem is solved to obtain values for the complicating variables.
2. **Solving the Subproblems:**
   - Given the master problem's solution, the subproblems are solved to obtain values for the non-complicating variables.
3. **Generating Benders Cuts:**
   - If the subproblem solutions indicate that the current solution of the master problem is not feasible or optimal, Benders cuts (constraints) are generated and added to the master problem.
4. **Updating the Master Problem:**
   - The master problem is updated with the new Benders cuts and resolved.

This process is repeated until convergence, i.e., no more Benders cuts are generated, and the master problem solution is optimal.

### Formulation

Consider a mixed-integer linear programming (MILP) problem in the form:

$$
\begin{equation}
\begin{aligned}
\min \quad & c^T x + d^T y \\
\textrm{s.t.} \quad & A x + B y \leq b, \\
  & G y \leq h, \\
  & E x \leq f, \\
  & x \in \mathbb{R}^n, \\
  & y \in \mathbb{R}^m,
\end{aligned}
\end{equation}
$$

where:
- $x$ represents the non-complicating variables, they are continous variables
- $y$ represents the complicating variables, can be both iteger or continous 
- $c$ and $d$ are cost vectors,
- $A$ and $B$ are constraint matrices,
- $b$, $h$, and $f$ are right-hand side vectors,
- $g y \leq h$ represents constraints that depend only on $y$,
- $E x \leq f$ represents constraints that depend only on $x$.

#### Master Problem

The master problem focuses on the complicating variables $y$:

$$
\begin{equation}
\begin{aligned}
\min \quad & d^T y + \theta \\
\textrm{s.t.} \quad & G y \leq h, \\
  & \theta \geq \pi_k^T (b - B y), \quad k \in K\\
  & \sigma_s^T (b - B y), \quad s \in S, \\
\end{aligned}
\end{equation}
$$

where $\theta$ is a lower bound on the objective the subproblems, $\pi_k$  are dual multipliers from the constraints, and $\sigma_k$ are extreme ray (retrived with the Farkas lemma) from the  infeasible constraints.

#### Subproblems

Given a fixed $y$ from the master problem, the subproblem optimizes over the non-complicating variables $x$:

$$
\begin{equation}
\begin{aligned}
\min \quad & c^T x \\
\textrm{s.t.} \quad & A x \leq b - B y, \\
  & E x \leq f, \\
  & x \in \mathbb{R}^n.
\end{aligned}
\end{equation}
$$

If the subproblem is infeasible, a feasibility cut is generated. If feasible, an optimality cut is generated based on the solution.

#### Generating Benders Cuts

1. **Feasibility Cut:**
   If the subproblem is infeasible, a feasibility cut is added to the master problem to eliminate the current solution of $y$:

   $$
   \begin{equation}
   \begin{aligned}
   \sigma^T (b - B y) \leq 0,
   \end{aligned}
   \end{equation}
   $$

   where $\sigma$ are the Farkas duals corresponding to the infeasible constraints.

   **Farkas' Lemma:**
   Farkas' Lemma is a result from linear algebra used to prove the infeasibility of a system of linear inequalities. It states that for any matrix \(A\) and vector \(b\), exactly one of the following statements is true:
   - There exists an $x \geq 0$ such that $A x \eq b$,
   - There exists a $\sigma$ such that $\sigma^T A \leq 0$ and $\sigma^T b \geq 0$.

   In the context of Benders Decomposition, Farkas' Lemma is used to generate the extreme rays $\sigma$ that define the feasibility cuts.

2. **Optimality Cut:**
   If the subproblem is feasible, an optimality cut is generated:

   $$
   \begin{equation}
   \begin{aligned}
   \theta \geq \pi^T (b - B y),
   \end{aligned}
   \end{equation}
   $$

   where $\pi$ are the dual multipliers from the subproblem solution.

By iteratively solving the master problem and subproblems, and adding the appropriate Benders cuts, the method converges to the optimal solution.

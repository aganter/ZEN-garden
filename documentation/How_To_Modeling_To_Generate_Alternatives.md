# ZEN-garden Modeling to Generate Alternatives Documentation

<table><tr><td valign="center"> 
  <img align="left" height="25px" src="https://github.com/RRE-ETH/ZEN-garden/actions/workflows/pytest_with_conda.yml/badge.svg?branch=development"> 
  <img align="left" height="25px" src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/jafluri/5d0d79e86182cd9ccd785d824b1f9ac7/raw/zen_coverage.json">
  <b> (Development Branch) </b>
</td></tr></table>
<hr style="height: 5px; background-color: black;">
<img src="https://github.com/ZEN-universe/ZEN-garden/assets/114185605/d6a9aca9-74b0-4a82-8295-43e6a78b8450" alt="drawing" width="200"/>


## Overview

Modeling to Generate Alternatives (MGA) is a method aimed at identifying a range of solutions that are close to the minimum cost within a specified margin. This margin, denoted as ε (epsilon), represents a cost fraction and is typically set between 5% and 20%. The goal is to explore the solution space defined by:

- **Original Problem Constraint:** $Ax = b$
- **Cost Constraint:** $c^T x ≤ (1 + ε)f_{opt}$
- **Non-negativity Constraint:** $x \geq 0$

where:
- $A$ and $b$ are the system's constraints,
- $c^T x$ represents the system cost,
- $f_{opt}$ is the optimal cost,
- $\epsilon$ is the slack parameter defining the cost margin.

The inclusion of the cost deviation constraint $c^T x ≤ (1 + ε)f_{opt}$ expands the original problem's constraints to include solutions within ε of the optimal cost, $f_{opt}$. This approach ensures the solution space is convex, as it is defined by linear constraints.


### Methodology

Most established MGA methods identify vertices of the projected near-optimal space by solving repeated, independent optimization problems. These problems typically are of the form:

$$
\begin{equation}
\begin{aligned}
\min_{x} \quad & w^T x\\
\textrm{s.t.} \quad & Ax = b\\
  & c^T x \leq (1 + \epsilon)f_{opt}\\ 
  &x\geq0    \\
\end{aligned}
\end{equation}
$$

The feasible region of this optimization problem is simply the full near-optimal space. The objective function is used to seek boundary points where the variables take on extreme, efficient values. Each solution of the above optimization problem yields a vertex of the near-optimal space. The space can therefore be explored by repeatedly solving the optimization problem for different weights $w$. Existing methods typically differ in how they choose the objective coefficients to optimize.

### Random Directions

In the ZEN-garden repository, we developed the Random Directions method, which identifies the near-optimal space using randomly generated objectives. 
The objective weights of these production variables are determined randomly on the interval $[0,1]$. Let $β_i ∼ Unif(0,1)$ be such a randomly generated objective coefficient. The Random Directions MGA formulation is then:

$$
\begin{equation}
\begin{aligned}
\min_{x} \quad & \sum β_i x_i\\
\textrm{s.t.} \quad & Ax = b\\
  & c^T x \leq (1 + \epsilon)f_{opt}\\
  &x\geq0    \\
\end{aligned}
\end{equation}
$$

Random Directions repeatedly solves this equation to obtain different boundary points. Thi method is not iterative. Each optimization problem is completely independent of previous ones, allowing for a broad exploration of the near-optimal spacen and possible parallelization.
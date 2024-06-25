# ZEN-garden Modeling to Generate Alternatives Documentation

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
\textrm{s.t.} \quad & Ax \leq b\\
  & c^T x \leq (1 + \epsilon)f_{opt}\\ 
  &x\geq0    \\
\end{aligned}
\end{equation}
$$

The feasible region of this optimization problem is simply the full near-optimal space. The objective function is used to seek boundary points where the variables take on extreme, efficient values. Each solution of the above optimization problem yields a vertex of the near-optimal space. The space can therefore be explored by repeatedly solving the optimization problem for different weights $w$. Existing methods typically differ in how they choose the objective coefficients to optimize.

### Random Directions

In the ZEN-garden repository, we developed the Random Directions method, which identifies the near-optimal space using randomly generated objectives. The objective weights of these production variables are determined randomly on the interval $[0,1]$. Let $d_i \sim Norm(0,1)$ be such a randomly generated objective coefficient. However, in the context of generating directions on a hypersphere, $d$ is not just a set of coefficients but represents the direction vector of research. The Random Directions MGA formulation is then:

$$
\begin{equation}
\begin{aligned}
\min_{x} \quad & f(x)= \sum d_i x_i\\
\textrm{s.t.} \quad & Ax \leq b\\
  & c^T x \leq (1 + \epsilon)f_{opt}\\
  &x\geq0    \\
\end{aligned}
\end{equation}
$$

Let $x = [x_{1}, x_{2}, ..., x_{{N_d}}]$ be the decision variables of interest. The "real" objective function is then:

$$
f(x) = \sum_{i=1}^{N_d} \frac{d_i}{L_i} x_{i}
$$

where we just add the term $L_i$, the characteristic scale that approximately normalizes the variables, helping improve performance in cases where the variables are vastly different scales.

Random Directions repeatedly solves this equation to obtain different boundary points. This method is not iterative. Each optimization problem is completely independent of previous ones, allowing for a broad exploration of the near-optimal space and possible parallelization.


## How to MGA

### MGA class

The **ModelingToGenerateAlternatives** class provides functionalities to implement the **Modeling to Generate Alternatives (MGA)** method. This class ensures the exploration of near-optimal solutions by generating alternative solutions that are within a specified cost margin from the optimal solution. Here’s a detailed breakdown of the class functionalities:

#### Key Functionalities
- **Sanity Checks for Input Data:**
  - **Sanity Checks on MGA Iteration Scenario:** Ensures that the input data structure is correct and consistent with the expected format. It verifies the dictionary structure, existence of necessary keys, and that values match the expected coordinate system of the objective variables.
  - **Sanity Checks on Characteristic Scales File:** Confirms that the characteristic scales dictionary is correctly formatted, with appropriate keys and values, ensuring that scales are properly defined for normalizing variables.

- **Loading and Storing Input Data:**
  - **Store Input Data:** Reads and stores the input data necessary for the MGA scenario. This involves updating the configuration and decision variables to ensure all necessary data is available for the MGA process.

- **Generating Weights for MGA Objective Function:**
  - **Generate Random Directions:** Produces random direction vectors from a normal distribution for each decision variable. These directions are used to explore different regions of the solution space.
  - **Generate Characteristic Scales:** Creates characteristic scales for the decision variables, which are used to normalize the variables. This helps in handling variables of different scales more effectively.
  - **Generate Weights:** Combines the random direction vectors and characteristic scales to generate weights for the MGA objective function. These weights are necessary in formulating the objective function for exploring near-optimal solutions as seen above.

- **Adding Cost Constraint:**
  - **Add Cost Constraint:** Adds a constraint to the optimization problem to limit the total cost based on the optimal solution's cost and the allowed deviation defined by the cost slack variables. This ensures that the generated alternatives remain within the specified cost margin.

- **Solving the Optimization Problem:**
  - **Run:** Executes the MGA process by solving the optimization problem. For each iteration, it generates weights, constructs the optimization problem with the cost constraint and the new objective function, and solves it to find a near-optimal solution. The results are saved for further analysis.


### Configurate MGA method 
This is how the data folder looks like:


<img src="https://github.com/ZEN-universe/ZEN-garden/raw/28b9d472debae1b6d739abe6ee4968fecfb59669/documentation/images/Data_Folder_General.png" alt="Data Folder Structure" width="400" />


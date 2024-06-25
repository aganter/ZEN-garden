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


### Challenges

One of the challenges in MGA is identifying all feasible solutions that satisfy the cost deviation constraint, especially for large-scale problems. Computing the entire near-optimal space for all variables is often not practical due to computational limitations. Therefore, most analyses focus on a subset of variables, projecting the near-optimal space onto a smaller, more manageable set of aggregate variables ($x_{agg} = g(x)$). This projection maintains the convexity of the solution space, facilitating the exploration of near-optimal solutions.


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


### Configure MGA method 
This is how the data folder looks like:

<p align="center">
    <img src="https://github.com/ZEN-universe/ZEN-garden/raw/28b9d472debae1b6d739abe6ee4968fecfb59669/documentation/images/Data_Folder_General.png" alt="Data Folder Structure" width="600" />
</p>

Let's analyze it:

- `config.py`: 

  In the `default_config.py` there is a new class called `ModelingToGenerateAlternatives`

  ```python
  class ModelingToGenerateAlternatives(Subscriptable):
      """
      This class is used to model the behavior of the system to generate alternatives.
      """
      
      modeling_to_generate_alternatives: bool = False
      objective_variables: str = "technologies"
      analysis: Analysis = Analysis()
      solver: Solver = Solver()
      system: System = System()

      characteristic_scales_path: str = ""
      cost_slack_variables: float = 0.0
      folder_path: Path = Path("data/")
      # Keep the same name for code consistency and usability: this are the MGA iterations
      scenarios: dict[str, Any] = {"": {}}
      immutable_system_elements: dict = {
          "conduct_scenario_analysis": True,
          "run_default_scenario": False,
      }
      allowed_mga_objective_objects: list[str] = [
          "set_carriers",
          "set_technologies",
          "set_conversion_technologies",
          "set_storage_technologies",
          "set_transport_technologies",
      ]
      allowed_mga_objective_locations: list[str] = ["set_nodes", "set_location", "set_edges", "set_supernodes"]
  ```

  This class handles all the necessary paramters to set up the the MGA algorithm, in particular, in the `config.py` the user needs to set:
  ``` 
  mga = config.mga
  ```
  <p align="center">
    <img src="https://github.com/ZEN-universe/ZEN-garden/raw/28b9d472debae1b6d739abe6ee4968fecfb59669/documentation/images/Config_File.png" alt="Config File" width="600" />
</p>

- The parameter `modeling_to_generate_alternatives` equals to True to activate the MGA method
- The user needs to define the path for both the `modeling_to_generate_alternatives` folder and the `characteristic_scale.json` file paths
- `analysis`, `system`, and `solver` are deep copy of the default config ones, this is just for robustness against dumb errors
-  It is fundamental that the ```mga.analysis['folder_output']``` is the same of the ```config.analysis['folder_output']```
- The parameter `cost_slack_variables` that is the allowed percetage of cost deviation
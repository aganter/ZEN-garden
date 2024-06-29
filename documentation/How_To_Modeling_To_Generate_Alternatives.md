# ZEN-garden Modeling to Generate Alternatives Documentation

## Overview

Modeling to Generate Alternatives (MGA) identifies a range of solutions close to the minimum cost within a specified margin, ε (epsilon), typically set between 5% and 20%. This method explores the solution space defined by:

- **Original Problem Constraint:** $Ax = b$
- **Cost Constraint:** $c^T x ≤ (1 + ε)f_{opt}$
- **Non-negativity Constraint:** $x \geq 0$

where:
- $A$ and $b$ represent the system's constraints,
- $c^T x$ denotes the system cost,
- $f_{opt}$ is the optimal cost,
- ε is the slack parameter defining the cost margin.

Including the cost deviation constraint $c^T x ≤ (1 + ε)f_{opt}$ expands the original problem's constraints to encompass solutions within ε of the optimal cost, ensuring a convex solution space defined by linear constraints.

### Methodology

Established MGA methods identify vertices of the near-optimal space by solving repeated, independent optimization problems of the form:

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

The feasible region of this problem is the full near-optimal space. The objective function seeks boundary points where variables take on extreme, efficient values, with each solution yielding a vertex of the near-optimal space. Methods typically vary in how they select the objective coefficients.

### Random Directions

The ZEN-garden repository introduces the Random Directions method, identifying the near-optimal space using randomly generated objectives. Objective weights for production variables are randomly determined within the [0,1] interval. Let $d_i \sim Norm(0,1)$ represent a randomly generated objective coefficient. In this context, $d$ represents the direction vector of research. The Random Directions MGA formulation is:

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

Let $x = [x_{1}, x_{2}, ..., x_{{N_d}}]$ be the decision variables. The objective function is:

$$
f(x) = \sum_{i=1}^{N_d} \frac{d_i}{L_i} x_{i}
$$

where $L_i$ is the characteristic scale that normalizes the variables, enhancing performance when variables have vastly different scales.

Random Directions solves this equation repeatedly to obtain different boundary points, allowing for broad exploration of the near-optimal space and potential parallelization. Characteristic scales are derived from the $x$ values in the optimal solution or estimated to match the expected magnitude of variables in the near-optimal space if $x$ values are zero.

### Challenges

A challenge in MGA is identifying all feasible solutions that meet the cost deviation constraint, especially in large-scale problems. Computing the entire near-optimal space for all variables is often impractical due to computational limits. Thus, analyses typically focus on a subset of variables, projecting the near-optimal space onto a smaller set of aggregate variables $x_{agg} = g(x)$, maintaining the solution space's convexity and facilitating exploration of near-optimal solutions.

## How to MGA
### MGA Class

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

- **Adding Cost Deviation Constraint:**
  - **Add Cost Deviation Constraint:** Adds a constraint to the optimization problem to limit the total cost based on the optimal solution's cost and the allowed deviation defined by the cost slack variables. This ensures that the generated alternatives remain within the specified cost margin.

- **Solving the Optimization Problem:**
  - **Run:** Executes the MGA process by solving the optimization problem. It generates weights, constructs the optimization problem with the cost deviation constraint and the new objective function, and solves it to find a near-optimal solution. The results are saved for further analysis.

### Configure MGA Method
This is what the data folder looks like:

<p align="center">
    <img src="https://github.com/ZEN-universe/ZEN-garden/raw/28b9d472debae1b6d739abe6ee4968fecfb59669/documentation/images/Data_Folder_General.png" alt="Data Folder Structure" width="600" />
</p>

Let's analyze it:

#### Config
In the `default_config.py`, there is a new class called `ModelingToGenerateAlternatives`:

```python
class ModelingToGenerateAlternatives(Subscriptable):
    """
    This class is used to model the behavior of the system to generate alternatives.
    """
    
    modeling_to_generate_alternatives: bool = False
    analysis: Analysis = Analysis()
    solver: Solver = Solver()
    system: System = System()

    characteristic_scales_path: str = ""
    cost_slack_variables: float = 0.0
    folder_path: Path = Path("data/")
    # Keep the same name for code consistency and usability: these are the MGA iterations
    scenarios: dict[str, Any] = {"": {}}
    immutable_system_elements: dict = {
        "conduct_scenario_analysis": True,
        "run_default_scenario": False,
    }
    allowed_mga_objective_objects: list[str] = [
        "set_carriers",
        "set_technologies",
    ]
    allowed_mga_objective_locations: list[str] = [
       "set_nodes",
        "set_location",
        "set_supernodes",
        "set_superlocation",
    ]
```

This class handles all the necessary parameters to set up the MGA algorithm. In particular, in the `config.py`, the user needs to set:

```mga = config.mga```

  <p align="center">
    <img src="https://github.com/ZEN-universe/ZEN-garden/blob/1d6c06a3669d0ec06b8d581a0f7fd31fffe9d891/documentation/images/Config_File.png" alt="Config File" width="800" />
</p>

- The parameter `modeling_to_generate_alternatives` should be set to True to activate the MGA method.
  - The user needs to define the paths for both the `modeling_to_generate_alternatives` folder and the `characteristic_scale.json` file.
  - `analysis`, `system`, and `solver` are deep copies of the default config, which is done for robustness against errors.
  - It is fundamental that `mga.analysis['folder_output']` is the same as `config.analysis['folder_output']`.
  - The parameter `cost_slack_variables` defines the allowed percentage of cost deviation.

#### MGA Folder
The `modeling_to_generate_alternatives` folder looks like:

<p align="center">
    <img src="https://github.com/ZEN-universe/ZEN-garden/blob/development_ZENx_MC_AG/documentation/images/MGA_Folder.png" width="600" />
</p>

The `1_carbon_storage.json` file is the dictionary where we set the parameters for the new objective function, specifically:
```python
{
  "objective_variables": "capacity_supernodes",
  "objective_set": {
      "set_technologies": [
          "carbon_storage"
      ],
      "set_supernodes": [
          "CH",
          "DE"
      ]
  }
}
```
- The element types of the objective variables: the user can choose among the variables defined in the optimization problem. For example, in the dictionary above, we opt for `"capacity_supernodes"`. At the moment, it is not possible to simultaneously optimize carriers and technologies.
- The `objective_set` key must contain two keys that refer to the coordinates of the `"capacity_supernodes"` variable array:
  1. The first specifies the object's coordinate. The user can choose among `"set_carriers", "set_technologies", "set_conversion_technologies", "set_storage_technologies", "set_transport_technologies"`. For example, in the dictionary above, we opt to optimize `carbon_storage`, which is a technology. Therefore, the correct key for the list will be `set_technologies`. Using any other key will result in an error in the code.
  2. The second specifies the location coordinate. The user can choose among `"set_nodes", "set_location", "set_edges", "set_supernodes"`. For example, in the dictionary above, the correct coordinate for `carbon_storage` is `"set_supernodes"`. Using any other coordinate will result in an error in the code.

#### Characteristic Scale
The `characteristic_scale.json` file is structured as follows:
```python
{
  "carbon_storage": {
    "default_value": 4,
    "unit": "GW"
  }
}
```
It must contain as primary keys all the variables we would like to optimize in the objective function, so in this case, it is enough to have just `"carbon_storage"`. Each primary key defines a dictionary with two keys: `"default_value"` and `"unit"` to approximate the value of the variable, as explained in the Methodology section.

It is important to note that the MGA objective function, if the problem is a multi year optimization, has as decision variables the sum of the variables over the optimized years. The latter is the multiply by the weight. So, the characteristic scale should be an approximation of the value of the variables considering all the optimized years.

#### MGA Iterations
The `mga_iterations.py` resembles the `scenarios.py` in format:
```python
NUMBER_OF_ITERATIONS = 20
scenarios = dict()

scenarios["1_carbon_storage"] = {
  "analysis": {"objective": "mga"},
  "ModelingToGenerateAlternatives": {
    "objective_elements_definition": {
      "file": "1_carbon_storage",
      "default_op": [1 for _ in range(NUMBER_OF_ITERATIONS)],
    }
  },
}
```

#### MGA Iterations
The scenario name is the same as the file we need to load to build the MGA objective function.
This format exploits the existing functionalities for scenarios, particularly:
- With the key `"analysis"`, the user needs to change the objective function to `"mga"`.
- `"file"` sets the file name from which the values must be read.
- `"default_opt"` is a list of ones with a length equal to the number of iterations of Random MGA the user wants to perform.

For each defined scenario, it is fundamental to have a corresponding `.json` file inside the `modeling_to_generate_alternatives` folder.

## Supernodes
At the beginning of this document, one of the challenges of MGA is mentioned. To overcome this problem, there are new functionalities to decrease the number of decision variables and aggregate them together.

1. The MGA objective function can account for a user-defined number of variables as explained above.
2. The supernodes aggregation gives the possibility to aggregate nodes by country. In the `system.py` file, the user can now set `system["run_supernodes"] = True`. In this way, in the optimization problem, new sets are defined as supernodes, aggregations of nodes by country. For example, nodes `"BE10", "BE21", "BE22"` belong to node `"BE"`. And consequently, new aggregated variables for `capacity` and `flow_import` are defined as the sum of those variable for all the nodes belonging to a supernode. These are called respectively `capacity_supernodes` and `flow_import_supernodes`.

N.B. 1: In order to be able to perform the supernodes aggregation, the file `all_nodes.csv` and `all_edges.csv` inside the folder `energy_system` must contain an additional column, to be `supernode` and `superedge` respectively, as shown in the following pictures:
<p align="center">
    <img src="https://github.com/ZEN-universe/ZEN-garden/blob/7052e6cc624742e5f7902fc55c16799a888c9eba/documentation/images/supernodes.png" width="400" />
</p>
<p align="center">
    <img src="https://github.com/ZEN-universe/ZEN-garden/blob/development_ZENx_MC_AG/documentation/images/speredgeds.png" width="400" />
</p>
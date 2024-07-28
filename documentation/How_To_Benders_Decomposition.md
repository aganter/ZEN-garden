# ZEN-garden Benders Decomposition Method Documentation

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
- $G y \leq h$ represents constraints that depend only on $y$,
- $E x \leq f$ represents constraints that depend only on $x$.

#### Master Problem

The master problem focuses on the complicating variables $y$:

$$
\begin{equation}
\begin{aligned}
\min \quad & d^T y + \theta \\
\textrm{s.t.} \quad & G y \leq h, \\
  & \theta \geq \pi_k^T (b - B y), \quad k \in K\\
  & \sigma_s^T (b - B y) \leq 0 , \quad s \in S, \\
\end{aligned}
\end{equation}
$$

where $\theta$ is a lower bound on the objective the subproblems, $\pi_k$  are dual multipliers from the constraints, and $\sigma_s$ are extreme ray (retrived with the Farkas lemma) from the  infeasible constraints.

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
   If the subproblem is infeasible, a feasibility cut is added to the master problem to eliminate the current solution of $y$ 

    $$
    f(x) = \sum_{i=1}^{N_d} \frac{d_i}{L_i} x_{i}
    $$

   where $\sigma$ are the Farkas duals corresponding to the infeasible constraints.

   **Farkas' Lemma:**
   Farkas' Lemma is a result from linear algebra used to prove the infeasibility of a system of linear inequalities. It states that for any matrix \(A\) and vector \(b\), exactly one of the following statements is true:
   - There exists an $x \geq 0$ such that $A x = b$,
   - There exists a $\sigma$ such that $\sigma^T A \leq 0$ and $\sigma^T b \geq 0$.

   In the context of Benders Decomposition, Farkas' Lemma is used to generate the extreme rays $\sigma$ that define the feasibility cuts.

2. **Optimality Cut:**
   If the subproblem is feasible, an optimality cut is generated:

   $$
   \theta \geq \pi^T (b - B y)
   $$

   where $\pi$ are the dual multipliers from the subproblem solution.

By iteratively solving the master problem and subproblems, and adding the appropriate Benders cuts, the method converges to the optimal solution.

### Challenges

A significant challenge is addressing optimization problems under scenario uncertainties, particularly when considering parametric uncertainties. Benders Decomposition is utilized to solve such optimization problems by breaking them down into subproblems representing different scenarios. Each subproblem corresponds to a distinct scenario, allowing the method to handle a wide range of uncertainties effectively.

Using Benders Decomposition is crucial for large systems because it enables the inclusion of uncertainties without overwhelming computational resources. By decomposing the problem into manageable subproblems, Benders Decomposition can efficiently explore different scenarios and provide robust solutions. This is particularly important in large-scale optimization problems where direct inclusion of all uncertainties would be computationally infeasible. The iterative nature of Benders Decomposition ensures that the solution converges to an optimal and robust decision that accounts for various uncertainties, making it a powerful tool for large systems.

## How to Benders Decomposition
### BendersDecomposition Class

**TO NOTE:** Benders Decomposition is implemented using Gurobi functionalities so it can be used only if the solver chosen is indeed Gurobi.

The **BendersDecomposition** class provides functionalities to implement the **Benders Decomposition** method. This class ensures the exploration of different subproblems. Here’s a detailed breakdown of the class functionalities:

#### Key Functionalities

- **Instantiate the Master and the Different Subproblems:**
  - **Master Class:** Child of OptimizationSetUp class. It creates the master problem and removes the operational variables and constraints if necessary.
  - **Subproblem Class:** Child of OptimizationSetUp class. It creates a subproblem and removes the operational constraints if necessary.

- **Design and Operational Variables and Constraints:**
  - **Map the Variables from Linopy to Gurobi:** Linopy does not yet offer key functionalities to implement Benders Decomposition, particularly for generating feasibility cuts. Therefore, Gurobi is leveraged. For robustness, a mapping is created between the names of the variables in Linopy and Gurobi.
  - **Separate Design and Operational Variables and Constraints:** The user can define which variables and constraints belong to the design or operational class, and these are separated accordingly.

- **Solve the Problems:**
  - **Solve Master Problem:** Uses the solve method of the parent class to solve the master problem.
  - **Fix Design Variables in the Subproblems:** Sets the lower and upper bounds of the design variables in the subproblems to be equal to the values of the same variables in the master problem just solved.
  - **Solve Subproblems:** Uses the solve method of the parent class to solve the different subproblems.

- **Generate Cuts:**
  - **Generate Feasibility Cuts and Add Them to the Master:** Adds feasibility constraints to the master problem to shrink the feasible space and move the master solution towards a feasible one.
  - **Generate Optimality Cuts and Add Them to the Master:** Adds optimality constraints to the master problem to shrink the feasible space and move the master solution towards the optimal one.
  - **Check for Oscillatory Behavior:** Ensures the solver does not fall into an infinite loop due to numerical issues by checking for oscillatory behavior caused by feasibility and optimality tolerance.

- **Termination:**
  - **Termination Criteria:** Checks the lower bound and the upper bounds of the objective function of the subproblems. The lower bound is given by the outer approximation of the objective in the master, while the upper bounds are given by the solution of the subproblems. If all the differences between the two for all the subproblems are lower than a user-defined absolute gap, the algorithm stops. If the objective function of the subproblem is a mock function, the termination criteria are reached when all the subproblems are feasible.
  - **Save:** Post-processes the results and saves them.

- **Fit:** The fit method combines the above-described methods into the iterative Benders approach.


### Objective Functions

#### Master Objective

The master problem is the design problem. It includes only the design constraints, and the objective function is defined as follows:

- **When the Objective Function is "mga":**
  - **Optimizing "capacity" (design):** In this case, the objective function of the master problem remains identical to the objective function of the monolithic (combined) problem. This ensures that the design decisions made in the master problem are aligned with the overall optimization goals of the entire system.
  - **Optimizing "flow_import" (operational):** Here, the master problem's objective function is set to be the outer approximation of the subproblem's objective function. This means that the master problem uses a simplified, but still representative, version of the operational objective to guide the design decisions. This approximation helps in effectively linking the design and operational problems while ensuring computational efficiency.

- **When the Objective Function is "total_cost" or "total_carbon_emissions":**
  - In these scenarios, the objective function of the master problem is also set as the outer approximation of the subproblem's objective function. This approach ensures that the master problem focuses on minimizing either the total cost or the total carbon emissions of the system, considering the outer bounds provided by the subproblem solutions. The outer approximation provides a feasible estimate that guides the design decisions towards the overall optimization goals. Ordinaray, in the Benders Decomposition method the separation is made so the investment costs are in the Master problem. Here, we aim to address parametric uncertainties so all the element of the total cost (capex included) go into the Subproblems.

#### Subproblem Objective

The subproblem is the operational problem. It includes only the operational constraints, and the objective function is defined as follows:

- **When the Objective Function is "mga":**
  - **Optimizing "capacity" (design):** The subproblem's objective function is set to a mock constant objective function. This means that the subproblem does not perform actual optimization for capacity during each iteration. Instead, it focuses on verifying the feasibility of the design provided by the master problem.
  - **Optimizing "flow_import" (operational):** In this case, the subproblem's objective function is the same as that of the monolithic problem. This ensures that the operational decisions made in the subproblem are directly aligned with the overall optimization goals of the system, providing accurate feedback to the master problem.

- **When the Objective Function is "total_cost" or "total_carbon_emissions":**
  - In these scenarios, the subproblem's objective function remains identical to the monolithic problem's objective function. This alignment ensures that the subproblem optimally manages the operational aspects of the system, focusing on minimizing either the total cost or the total carbon emissions, as specified. The results from the subproblem provide precise guidance for the master problem to adjust its design decisions accordingly.


### Configure Benders Method
This is what the data folder looks like:

<p align="center">
    <img src="https://github.com/aganter/ZEN-garden/blob/development_ZENx_MC_AG/documentation/images/Data_Folder_General_Benders.png" alt="Data Folder Structure Benders" width="600" />
</p>

Let's analyze it:

#### Config
In the `default_config.py`, there is a new class called `BendersDecomposition`:

```python
class BendersDecomposition(Subscriptable):
    """
    Class defining the Benders Decomposition method.
    """

    benders_decomposition: bool = False
    analysis: Analysis = Analysis()
    system: System = System()
    input_path: Path = Path("data/")
    scenarios: dict[str, Any] = {"": {}}
    immutable_system_elements: dict = {
        "conduct_scenario_analysis": True,
        "run_default_scenario": True,
    }
    use_monolithic_solution: bool = False
    absolute_optimality_gap: int = 1e-4
    max_number_of_iterations: int = 1e4
```

This class handles all the necessary parameters to set up the Benders Decomposition algorithm. In particular, in the `config.py`, the user needs to set:

```benders = config.benders```

  <p align="center">
    <img src="https://github.com/aganter/ZEN-garden/blob/4594ea136a94092836be40d276079acf0668d209/documentation/images/Config_File_Benders.png" alt="Config File Benders" width="800" />
</p>


- The parameter `benders_decomposition` should be set to True to activate the Benders Decomposition method.
- The parameter `use_monolithic_solution` should be set to True if the user wants to exploit the solution of the monolithic problem as starting point of the Benders Decomposition.
- The parameter `run_default_scenario` should be set to True if the user wants to consider as one of the subproblems also the default scenario.

#### Benders Decompsotion Folder
The `benders_decomposition` folder looks like:

<p align="center">
    <img src="https://github.com/aganter/ZEN-garden/blob/development_ZENx_MC_AG/documentation/images/BD_Folder.png" width="600" />
</p>

The `constraints.csv` file is the csv file with two columns: 
- `constraint_name`: the name of the constraints in the problem;
- `constraint_type`: if the constraint is operational or design.

The `variables.csv` file is the csv file with two columns: 
- `variable_name`: the name of the variables in the problem;
- `variable_type`: if the variables is operational or design.

The `not_coupling_variables.csv` file is the csv file with two columns: 
- `variable_name`: the name of the design variables that are not coupling;
- `variable_type`: if the variables is operational or design (all must set to design)


#### Benders Decomposition Scenarios
The `benders_scenarios.py` resembles the `scenarios.py` in format, for example:
```python
scenarios = dict()

scenarios["high_demand"] = {"set_carriers": {"demand_yearly_variation": {"file": "demand_yearly_variation_high"}}}
scenarios["low_demand"] = {"set_carriers": {"demand_yearly_variation": {"file": "demand_yearly_variation_low"}}}
```
Where the user can specify all the different parametric uncertainties scenario to be run. This dictionary can be also empty if the user want simply to run the monolithic problem with Benders. In this case the solution of the Benders method will be the same as the solution of the monolithic problem.

### Summary

Benders Decomposition is crucial for solving large-scale optimization problems, especially under scenario uncertainties. By breaking down the problem into more manageable subproblems, it efficiently includes uncertainties without overwhelming computational resources, ensuring robust and optimal solutions. The **BendersDecomposition** class provides all necessary functionalities to implement this method effectively.

### Proposed Separation of Design/Operational Constraints and Variables and Possible Objective Function

The following tables explain the default setting for the separation of design and operational constraints and variables for the **BendersDecomposition** class.

#### Constraints

| Constraint Name                                      | Constraint Type |
|------------------------------------------------------|-----------------|
| constraint_carbon_emissions_cumulative               | operational     |
| constraint_carbon_emissions_annual_limit             | operational     |
| constraint_carbon_emissions_budget                   | operational     |
| constraint_net_present_cost                          | operational     |
| constraint_carbon_emissions_annual                   | operational     |
| constraint_cost_carbon_emissions_total               | operational     |
| constraint_cost_total                                | operational     |
| constraint_carbon_emissions_budget_overshoot         | operational     |
| constraint_carbon_emissions_annual_overshoot         | operational     |
| constraint_availability_import                       | operational     |
| constraint_availability_export                       | operational     |
| constraint_availability_import_yearly                | operational     |
| constraint_availability_export_yearly                | operational     |
| constraint_cost_carrier                              | operational     |
| constraint_cost_shed_demand                          | operational     |
| constraint_limit_shed_demand                         | operational     |
| constraint_cost_carrier_total                        | operational     |
| constraint_carbon_emissions_carrier                  | operational     |
| constraint_carbon_emissions_carrier_total            | operational     |
| constraint_nodal_energy_balance                      | operational     |
| constraint_carrier_flow_import_supernodes            | operational     |
| constraint_technology_capacity_limit_not_reached     | design          |
| constraint_technology_capacity_limit_reached         | design          |
| constraint_technology_min_capacity_addition          | design          |
| constraint_technology_max_capacity_addition          | design          |
| constraint_technology_construction_time              | design          |
| constraint_technology_construction_time_outside      | design          |
| constraint_technology_lifetime                       | design          |
| constraint_technology_lifetime_previous              | design          |
| constraint_capex_yearly                              | operational     |
| constraint_cost_capex_total                          | operational     |
| constraint_opex_yearly                               | operational     |
| constraint_cost_opex_total                           | operational     |
| constraint_carbon_emissions_technology_total         | operational     |
| constraint_technologies_capacity_supernodes          | design          |
| constraint_technology_diffusion_limit                | design          |
| constraint_technology_diffusion_limit_total          | design          |
| constraint_capacity_factor_conversion                | operational     |
| constraint_opex_technology_conversion                | operational     |
| constraint_carbon_emissions_technology_conversion    | operational     |
| constraint_carrier_conversion                        | operational     |
| constraint_linear_capex                              | operational     |
| constraint_capacity_coupling                         | operational     |
| constraint_capex_coupling                            | operational     |
| retrofit_flow_coupling                               | operational     |
| constraint_capacity_factor_transport                 | operational     |
| constraint_opex_technology_transport                 | operational     |
| constraint_carbon_emissions_technology_transport     | operational     |
| constraint_transport_technology_losses_flow          | operational     |
| constraint_transport_technology_capex                | operational     |
| constraint_capacity_factor_storage                   | operational     |
| constraint_opex_technology_storage                   | operational     |
| constraint_carbon_emissions_technology_storage       | operational     |
| constraint_storage_level_max                         | operational     |
| constraint_couple_storage_level                      | operational     |
| constraint_capacity_energy_to_power_ratio            | design          |
| constraint_storage_technology_capex                  | operational     |
| constraint_optimal_cost_total_deviation              | operational     |

#### Variables

##### Design Variables

| Variable Name              | Variable Type |
|----------------------------|---------------|
| capacity_previous          | design        |
| capacity_investment        | design        |
| capacity_supernodes        | design        |
| technology_installation    | design        |
| capacity                   | design        |
| capacity_addition          | design        |

##### Operational Variables

| Variable Name                             | Variable Type   |
|-------------------------------------------|-----------------|
| carbon_emissions_annual                   | operational     |
| carbon_emissions_cumulative               | operational     |
| carbon_emissions_budget_overshoot         | operational     |
| carbon_emissions_annual_overshoot         | operational     |
| cost_carbon_emissions_total               | operational     |
| cost_total                                | operational     |
| net_present_cost                          | operational     |
| flow_import                               | operational     |
| flow_export                               | operational     |
| cost_carrier                              | operational     |
| cost_carrier_total                        | operational     |
| carbon_emissions_carrier                  | operational     |
| carbon_emissions_carrier_total            | operational     |
| shed_demand                               | operational     |
| cost_shed_demand                          | operational     |
| flow_import_supernodes                    | operational     |
| cost_capex                                | operational     |
| capex_yearly                              | operational     |
| cost_capex_total                          | operational     |
| cost_opex                                 | operational     |
| cost_opex_total                           | operational     |
| opex_yearly                               | operational     |
| carbon_emissions_technology               | operational     |
| carbon_emissions_technology_total         | operational     |
| flow_conversion_input                     | operational     |
| flow_conversion_output                    | operational     |
| capacity_approximation                    | operational     |
| capex_approximation                       | operational     |
| flow_transport                            | operational     |
| flow_transport_loss                       | operational     |
| flow_storage_charge                       | operational     |
| flow_storage_discharge                    | operational     |
| storage_level                             | operational     |

### Non-Coupling Variables

| Variable Name           | Variable Type |
|-------------------------|---------------|
| capacity_previous       | design        |
| capacity_investment     | design        |
| capacity_supernodes     | design        |
| technology_installation | design        |

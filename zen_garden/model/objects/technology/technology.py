"""
:Title:          ZEN-GARDEN
:Created:        October-2021
:Authors:        Alissa Ganter (aganter@ethz.ch),
                Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Class defining the parameters, variables and constraints that hold for all technologies.
The class takes the abstract optimization model as an input, and returns the parameters, variables and
constraints that hold for all technologies.
"""
import logging
import time

import linopy as lp
import numpy as np
import pandas as pd
import xarray as xr
from linopy.constraints import AnonymousConstraint

#anyaxie
import piecewise_regression

from zen_garden.utils import lp_sum
from ..component import ZenIndex, IndexSet
from ..element import Element, GenericRule

class Technology(Element):
    """
    Class defining the parameters, variables and constraints that hold for all technologies.
    """
    # set label
    label = "set_technologies"
    location_type = None

    def __init__(self, technology: str, optimization_setup):
        """init generic technology object

        :param technology: technology that is added to the model
        :param optimization_setup: The OptimizationSetup the element is part of """

        super().__init__(technology, optimization_setup)

    def store_carriers(self):
        """ retrieves and stores information on reference """
        self.reference_carrier = self.data_input.extract_carriers(carrier_type="reference_carrier")
        self.energy_system.set_technology_of_carrier(self.name, self.reference_carrier)

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # store scenario dict
        super().store_scenario_dict()
        # set attributes of technology
        set_location = self.location_type
        self.capacity_addition_min = self.data_input.extract_input_data("capacity_addition_min", index_sets=[], unit_category={"energy_quantity": 1, "time": -1})
        self.capacity_addition_max = self.data_input.extract_input_data("capacity_addition_max", index_sets=[], unit_category={"energy_quantity": 1, "time": -1})
        self.capacity_addition_unbounded = self.data_input.extract_input_data("capacity_addition_unbounded", index_sets=[], unit_category={"energy_quantity": 1, "time": -1})
        self.lifetime = self.data_input.extract_input_data("lifetime", index_sets=[], unit_category={})
        self.construction_time = self.data_input.extract_input_data("construction_time", index_sets=[], unit_category={})
        # maximum diffusion rate
        self.max_diffusion_rate = self.data_input.extract_input_data("max_diffusion_rate", index_sets=["set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={})

        # add all raw time series to dict
        self.raw_time_series = {}
        self.raw_time_series["min_load"] = self.data_input.extract_input_data("min_load", index_sets=[set_location, "set_time_steps"], time_steps="set_base_time_steps_yearly", unit_category={})
        self.raw_time_series["max_load"] = self.data_input.extract_input_data("max_load", index_sets=[set_location, "set_time_steps"], time_steps="set_base_time_steps_yearly", unit_category={})
        self.raw_time_series["opex_specific_variable"] = self.data_input.extract_input_data("opex_specific_variable", index_sets=[set_location, "set_time_steps"], time_steps="set_base_time_steps_yearly", unit_category={"money": 1, "energy_quantity": -1})
        # non-time series input data
        self.capacity_limit = self.data_input.extract_input_data("capacity_limit", index_sets=[set_location, "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"energy_quantity": 1, "time": -1})
        self.carbon_intensity_technology = self.data_input.extract_input_data("carbon_intensity_technology", index_sets=[set_location], unit_category={"emissions": 1, "energy_quantity": -1})
        # extract existing capacity
        self.set_technologies_existing = self.data_input.extract_set_technologies_existing()
        self.capacity_existing = self.data_input.extract_input_data("capacity_existing", index_sets=[set_location, "set_technologies_existing"], unit_category={"energy_quantity": 1, "time": -1})
        self.capacity_investment_existing = self.data_input.extract_input_data("capacity_investment_existing", index_sets=[set_location, "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"energy_quantity": 1, "time": -1})
        self.lifetime_existing = self.data_input.extract_lifetime_existing("capacity_existing", index_sets=[set_location, "set_technologies_existing"])
        # anyaxie
        # endogenous learning input data
        # segments for pwa of cumulative cost for each technology
        if self.optimization_setup.system["use_endogenous_learning"]:
            self.num_pwa_segments = int(self.data_input.extract_attribute("num_pwa_segments", unit_category={})["value"])
            self.learning_rate = self.data_input.extract_attribute("learning_rate", unit_category={})["value"]
            self.global_share_factor = self.data_input.extract_attribute("global_share", unit_category={})["value"]
            self.learning_curve_lb = self.data_input.extract_attribute("learning_curve_lower_bound", unit_category={"energy_quantity": 1, "time": -1})["value"]
            self.learning_curve_ub = self.data_input.extract_attribute("learning_curve_upper_bound", unit_category={"energy_quantity": 1, "time": -1})["value"]
            self.learning_curve_npts = self.data_input.extract_attribute("learning_curve_npts", unit_category={})["value"]
            self.global_initial_capacity= self.data_input.extract_attribute("global_initial_capacity", unit_category={"energy_quantity": 1, "time": -1})["value"]
            self.cum_capacity_row = self.data_input.extract_input_data("cum_capacity_row", index_sets=["set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"energy_quantity": 1, "time": -1})

    def calculate_capex_of_capacities_existing(self, storage_energy=False):
        """ this method calculates the annualized capex of the existing capacities

        :param storage_energy: boolean if energy storage
        :return: capex of existing capacities
        """
        if self.__class__.__name__ == "StorageTechnology":
            if storage_energy:
                capacities_existing = self.capacity_existing_energy
            else:
                capacities_existing = self.capacity_existing
            capex_capacity_existing = capacities_existing.to_frame().apply(
                lambda _capacity_existing: self.calculate_capex_of_single_capacity(_capacity_existing.squeeze(), _capacity_existing.name, storage_energy), axis=1)
        else:
            capacities_existing = self.capacity_existing
            capex_capacity_existing = capacities_existing.to_frame().apply(lambda _capacity_existing: self.calculate_capex_of_single_capacity(_capacity_existing.squeeze(), _capacity_existing.name), axis=1)
        return capex_capacity_existing

    def calculate_capex_of_single_capacity(self, *args):
        """ this method calculates the annualized capex of the existing capacities. Is implemented in child class

        :param args: arguments
        """
        raise NotImplementedError

    def calculate_fraction_of_year(self):
        """calculate fraction of year"""
        # only account for fraction of year
        fraction_year = self.optimization_setup.system["unaggregated_time_steps_per_year"] / self.optimization_setup.system["total_hours_per_year"]
        return fraction_year

    def add_new_capacity_addition_tech(self, capacity_addition: pd.Series, capex: pd.Series, step_horizon: int):
        """ adds the newly built capacity to the existing capacity

        :param capacity_addition: pd.Series of newly built capacity of technology
        :param capex: pd.Series of capex of newly built capacity of technology
        :param step_horizon: current horizon step """
        system = self.optimization_setup.system
        # reduce lifetime of existing capacities and add new remaining lifetime
        self.lifetime_existing = (self.lifetime_existing - system["interval_between_years"]).clip(lower=0)
        # new capacity
        new_capacity_addition = capacity_addition[step_horizon]
        new_capex = capex[step_horizon]
        # if at least one value unequal to zero
        if not (new_capacity_addition == 0).all():
            # add new index to set_technologies_existing
            index_new_technology = max(self.set_technologies_existing) + 1
            self.set_technologies_existing = np.append(self.set_technologies_existing, index_new_technology)
            # add new remaining lifetime
            lifetime = self.lifetime_existing.unstack()
            lifetime[index_new_technology] = self.lifetime[0] - system["interval_between_years"]
            self.lifetime_existing = lifetime.stack()

            for type_capacity in list(set(new_capacity_addition.index.get_level_values(0))):
                # if power
                if type_capacity == system["set_capacity_types"][0]:
                    energy_string = ""
                # if energy
                else:
                    energy_string = "_energy"
                capacity_existing = getattr(self, "capacity_existing" + energy_string)
                capex_capacity_existing = getattr(self, "capex_capacity_existing" + energy_string)
                # add new existing capacity
                capacity_existing = capacity_existing.unstack()
                capacity_existing[index_new_technology] = new_capacity_addition.loc[type_capacity]
                setattr(self, "capacity_existing" + energy_string, capacity_existing.stack())
                # calculate capex of existing capacity
                capex_capacity_existing = capex_capacity_existing.unstack()
                capex_capacity_existing[index_new_technology] = new_capex.loc[type_capacity]
                setattr(self, "capex_capacity_existing" + energy_string, capex_capacity_existing.stack())

    def add_new_capacity_investment(self, capacity_investment: pd.Series, step_horizon):
        """ adds the newly invested capacity to the list of invested capacity

        :param capacity_investment: pd.Series of newly built capacity of technology
        :param step_horizon: optimization time step """
        system = self.optimization_setup.system
        new_capacity_investment = capacity_investment[step_horizon]
        new_capacity_investment = new_capacity_investment.fillna(0)
        if not (new_capacity_investment == 0).all():
            for type_capacity in list(set(new_capacity_investment.index.get_level_values(0))):
                # if power
                if type_capacity == system["set_capacity_types"][0]:
                    energy_string = ""
                # if energy
                else:
                    energy_string = "_energy"
                capacity_investment_existing = getattr(self, "capacity_investment_existing" + energy_string)
                # add new existing invested capacity
                capacity_investment_existing = capacity_investment_existing.unstack()
                capacity_investment_existing[step_horizon] = new_capacity_investment.loc[type_capacity]
                setattr(self, "capacity_investment_existing" + energy_string, capacity_investment_existing.stack())

    # anyaxie
    def update_initial_global_cost_capacity_tech(self, total_global_cost, global_cumulative_capacity, step_horizon):
        """ updates initial global cost of the technology

        :param total_global_cost: pd.Series of total cost """

        for type_capacity in list(set(total_global_cost.index.get_level_values(0))):
            _global_initial_cost = total_global_cost.loc[type_capacity].loc[step_horizon]
            _global_cum_capacity = global_cumulative_capacity.loc[type_capacity].loc[step_horizon]
            # if power
            if type_capacity == self.optimization_setup.system["set_capacity_types"][0]:
                energy_string = ""
            # if energy
            else:
                energy_string = "_energy"
            # new initial cost and capacity
            setattr(self, "total_cost_pwa_initial_global_cost" + energy_string, _global_initial_cost)
            setattr(self, "global_initial_capacity" + energy_string, _global_cum_capacity)


    def perform_total_cost_pwa(self, capacity_types=False):
        """
        perform pwa approximation for total cost
        :return: slope and intersect of each segment, interpolation points
        """

        def fun_total_cost(u, c_initial: float, q_initial: float,
                           learning_rate: float) -> object:  # u is a vector
            """
            Total cumulative Cost for Learning Curve
            :param u: Cumulative Capacity
            :param c_initial: Initial Cost
            :param q_initial: Initital Capacity
            :param learning_rate: Learning Rate
            :return: Total cumulative cot
            """
            alpha = c_initial / np.power(q_initial, learning_rate)
            exp = 1 + learning_rate
            TC = alpha / exp * ( np.power(u, exp) )
            return TC

        def pwa_zero(x_values, y_function):
            """
            Linear interpolation of total cost function when capex specific is zero
            """
            # Create a linear space for the interpolation
            interpolated_x = np.linspace(min(x_values), max(x_values), 3)
            intersect = [0]*(len(interpolated_x)-1)
            slope = [0]*(len(interpolated_x)-1)
            y_values = []
            for x in interpolated_x:
                y_values.append(y_function(x))

            segments = list(range(len(slope)))
            index_tech = pd.Index(segments, name="set_total_cost_pwa_segments")

            return interpolated_x, y_values, intersect, slope, index_tech


        def pwa_equidistant(x_values, y_function, num_interpolation_points):
            """
            Linear interpolation of a function based on interpolation points.

            Parameters:
            - x_values: List or array of x-values of the interpolation points.
            - y_function: Function that takes x as input and returns y.
            - num_interpolation_points: Number of points for the linearized function.

            Returns:
            - interpolated_x: X-values of the interpolated points.
            - interpolated_y: Y-values of the interpolated points.
            - coefficients: List of tuples (a, b) representing the coefficients of each linear segment.
            """
            # Create a linear space for the interpolation
            interpolated_x = np.linspace(min(x_values), max(x_values), num_interpolation_points)

            # Initialize lists to store coefficients and y values
            intersect = []
            slope = []
            y_values = []

            # Calculate y values using the provided function
            for x in interpolated_x:
                y_values.append(y_function(x))

            # Perform linear interpolation and calculate coefficients
            for i in range(len(interpolated_x) - 1):
                x_segment = interpolated_x[i:i + 2]
                y_segment = y_values[i:i + 2]

                b = (y_segment[1] - y_segment[0]) / (x_segment[1] - x_segment[0])
                a = y_segment[0] - b * x_segment[0]

                intersect.append(a)
                slope.append(b)

            return interpolated_x, y_values, intersect, slope

        def pwa_non_equidistant(x_values, y_function, min_num_segments, max_num_segments):
            """
            Linear interpolation of a function with dynamically spaced segments
            Requires piecewise regression package
            """
            # Initialize lists to store coefficients and y values
            intersect = []
            slope = []
            interpolated_x = []
            TC_interpolated = []
            y_values = []

            # Calculate y values using the provided function
            for x in x_values:
                y_values.append(y_function(x))

            # Initialize number of intermediary breakpoints
            n = min_num_segments-1
            while n <= max_num_segments-1:
                pw_fit = piecewise_regression.Fit(x_values, y_values, n_breakpoints=n)
                # check if successful fit
                if pw_fit.get_params()["converged"]:
                    print(f"Successful fit with {n} breakpoints.")
                    error = 1 - pw_fit.best_muggeo.best_fit.r_squared
                    if error < 0.001:
                        print(f"R squared error is {error}")
                        break
                    else:
                        n += 1
                else:
                    print(f"Failed fit with {n} breakpoints.")
                    n -= 1
                    pw_fit = piecewise_regression.Fit(x_values, y_values, n_breakpoints=n)
                    break

            raw_params = pw_fit.best_muggeo.best_fit.raw_params
            breakpoints = pw_fit.best_muggeo.best_fit.next_breakpoints

            # Find the interpolated_x
            interpolated_x_first = np.array([x_values[0]])
            interpolated_x_next = breakpoints
            interpolated_x_last = np.array([x_values[-1]])
            interpolated_x = np.concatenate((interpolated_x_first, interpolated_x_next, interpolated_x_last))

            # Find TC_interpolated
            TC_interpolated = pw_fit.predict(interpolated_x)

            # Find the y-intersects
            tossed_params = raw_params[1:]

            # Initialise list of slope with first raw param from fit
            slope = []
            # Iteratively sum each pair of values in tossed_params with the result from the previous sum
            result = 0
            for i in range(0, len(breakpoints) + 1):
                result += tossed_params[i]
                slope.append(result)

            # Determine intersect
            intersect = TC_interpolated[:-1] - slope * interpolated_x[:-1]

            # Create index for segments
            segments = list(range(len(slope)))
            index_tech = pd.Index(segments, name="set_total_cost_pwa_segments")

            return interpolated_x, TC_interpolated, intersect, slope, index_tech


        def perform_pwa(capacity_existing, initial_cost, max_capacity):
            """
            perform pwa approximation for total cost
            :return: slope and intersect of each segment, interpolation points # TODO: Implement case for preprocessing
            """
            # Same for power and energy-rated
            learning_rate = self.learning_rate
            global_share_factor = self.global_share_factor
            global_initial_capacity = self.global_initial_capacity
            # Update global_initial_capacity if it is smaller than capacity_existing
            if global_initial_capacity < capacity_existing:
                self.global_initial_capacity = capacity_existing
                global_initial_capacity = capacity_existing
                logging.warning("Global initial capacity must be greater than existing capacity in model.")

            # todo: implement case for lower bound of PWA
            # Lower and Upper bound for global cumulative capacity
            npts = int(self.learning_curve_npts)
            lb = self.learning_curve_lb
            ub = self.learning_curve_ub

            q_values = np.linspace(lb, ub, npts)
            function = lambda u: fun_total_cost(u, initial_cost, global_initial_capacity, learning_rate)


            if self.optimization_setup.system["equidistant_total_cost_pwa"]:
                equidistant = True
            else:
                equidistant = False

            if equidistant and initial_cost != 0:
                # Use given set of segments
                segments = self.num_pwa_segments
                index_tech = pd.Index(list(range(segments)), name="set_total_cost_pwa_segments")
                num_interpolation_points = segments + 1
                # todo: remove the time logger later
                t0 = time.perf_counter()
                interpolated_q, interpolated_TC, intersect, slope = pwa_equidistant(q_values, function, num_interpolation_points)
                t1 = time.perf_counter()
                logging.info(f"Time to perform equidistant pwa for tech {self.name}: {t1 - t0:0.4f} seconds")
            elif not equidistant and initial_cost != 0:
                # Use given max number of segments
                min_num_segments = self.optimization_setup.system["min_num_segments"]
                max_num_segments = self.optimization_setup.system["max_num_segments"]
                # todo: remove the time logger later
                t0 = time.perf_counter()
                interpolated_q, interpolated_TC, intersect, slope, index_tech = pwa_non_equidistant(q_values, function, min_num_segments, max_num_segments)
                t1 = time.perf_counter()
                logging.info(f"Time to perform non-equidistant pwa for tech {self.name}: {t1 - t0:0.4f} seconds")
            else:
                # If technology for free
                interpolated_q, interpolated_TC, intersect, slope, index_tech = pwa_zero(q_values, function)

            # Determines which segment the initial capacity is on, to calculate total global initial cost
            index_segment = np.where(
                (interpolated_q[:-1] <= global_initial_capacity) & (global_initial_capacity < interpolated_q[1:]))[0][0]
            pwa_initial_total_global_cost = intersect[index_segment] + slope[index_segment] * global_initial_capacity


            return interpolated_q, interpolated_TC, intersect, slope, pwa_initial_total_global_cost, initial_cost, index_tech


        # Different for energy-rated and power-rated
        try:
            initial_cost = self.capex_specific.loc[:, 0].sum() / len(self.capex_specific.loc[:, 0])
        except:
            initial_cost = self.capex_specific_transport.loc[:, 0].sum() / len(self.capex_specific_transport.loc[:, 0])
        # Take the year with the maximum capacity limit across the nodes as the maximum capacity
        max_capacity = self.capacity_limit.groupby('year').sum().max() # todo: Implement case for when capacity_limit in nodes = 0
        capacity_existing = self.capacity_existing.sum() # todo: Implement case for decomissioning

        [interpolated_q, interpolated_TC, intersect, slope, pwa_initial_total_global_cost, initial_cost, index_tech] = perform_pwa(capacity_existing, initial_cost, max_capacity)

        self.total_cost_pwa_points_lower_bound = pd.Series(index=index_tech, data=interpolated_q[:-1], dtype=float)
        self.total_cost_pwa_points_upper_bound = pd.Series(index=index_tech, data=interpolated_q[1:], dtype=float)
        self.total_cost_pwa_TC_lower_bound = pd.Series(index=index_tech, data=interpolated_TC[:-1], dtype=float)
        self.total_cost_pwa_TC_upper_bound = pd.Series(index=index_tech, data=interpolated_TC[1:], dtype=float)
        self.total_cost_pwa_slope = pd.Series(index=index_tech, data=slope, dtype=float)
        self.total_cost_pwa_intersect = pd.Series(index=index_tech, data=intersect, dtype=float)
        self.total_cost_pwa_initial_global_cost = pwa_initial_total_global_cost
        self.total_cost_pwa_initial_unit_cost = initial_cost
        self.set_total_cost_pwa_segments = index_tech

        # for energy-rated technologies
        if capacity_types:
            # Different for energy-rated and power-rated
            initial_cost = self.capex_specific_energy.loc[:, 0].sum() / len(self.capex_specific_energy.loc[:, 0])
            max_capacity = self.capacity_limit_energy.groupby('year').sum().max()
            capacity_existing = self.capacity_existing_energy.sum()

            [interpolated_q, interpolated_TC, intersect, slope, pwa_initial_total_global_cost, initial_cost] = perform_pwa(capacity_existing, initial_cost, max_capacity,equidistant)

            self.total_cost_pwa_points_lower_bound_energy = pd.Series(index=index_tech, data=interpolated_q[:-1], dtype=float)
            self.total_cost_pwa_points_upper_bound_energy = pd.Series(index=index_tech, data=interpolated_q[1:], dtype=float)
            self.total_cost_pwa_TC_lower_bound_energy = pd.Series(index=index_tech, data=interpolated_TC[:-1], dtype=float)
            self.total_cost_pwa_TC_upper_bound_energy = pd.Series(index=index_tech, data=interpolated_TC[1:], dtype=float)
            self.total_cost_pwa_slope_energy = pd.Series(index=index_tech, data=slope, dtype=float)
            self.total_cost_pwa_intersect_energy = pd.Series(index=index_tech, data=intersect, dtype=float)
            self.total_cost_pwa_initial_global_cost_energy = pwa_initial_total_global_cost
            self.total_cost_pwa_initial_unit_cost_energy = initial_cost
            self.set_total_cost_pwa_segments = index_tech


    ### --- classmethods
    @classmethod
    def get_available_existing_quantity(cls, optimization_setup, tech, capacity_type, loc, year, type_existing_quantity):
        """ returns existing quantity of 'tech', that is still available at invest time step 'time'.
        Either capacity or capex.

        :param optimization_setup: The OptimizationSetup the element is part of
        :param tech: name of technology
        :param capacity_type: type of capacity
        :param loc: location (node or edge) of existing capacity
        :param year: current yearly time step
        :param type_existing_quantity: capex or capacity
        :return existing_quantity: existing capacity or capex of existing capacity
        """
        params = optimization_setup.parameters.dict_parameters
        sets = optimization_setup.sets
        existing_quantity = 0
        if type_existing_quantity == "capacity":
            existing_variable = params.capacity_existing
        elif type_existing_quantity == "cost_capex":
            existing_variable = params.capex_capacity_existing
        else:
            raise KeyError(f"Wrong type of existing quantity {type_existing_quantity}")

        for id_capacity_existing in sets["set_technologies_existing"][tech]:
            is_existing = cls.get_if_capacity_still_existing(optimization_setup, tech, year, loc=loc, id_capacity_existing=id_capacity_existing)
            # if still available at first base time step, add to list
            if is_existing:
                existing_quantity += existing_variable[tech, capacity_type, loc, id_capacity_existing]
        return existing_quantity

    @classmethod
    def get_if_capacity_still_existing(cls,optimization_setup, tech, year,loc,id_capacity_existing):
        """
        returns boolean if capacity still exists at yearly time step 'year'.
        :param optimization_setup: The optimization setup to add everything
        :param tech: name of technology
        :param year: yearly time step
        :param loc: location
        :param id_capacity_existing: id of existing capacity
        :return: boolean if still existing
        """
        # get params and system
        params = optimization_setup.parameters.dict_parameters
        system = optimization_setup.system
        # get lifetime of existing capacity
        lifetime_existing = params.lifetime_existing[tech, loc, id_capacity_existing]
        lifetime = params.lifetime[tech]
        delta_lifetime = lifetime_existing - lifetime
        # reference year of current optimization horizon
        current_year_horizon = optimization_setup.energy_system.set_time_steps_yearly[0]
        if delta_lifetime >= 0:
            cutoff_year = (year-current_year_horizon)*system["interval_between_years"]
            return cutoff_year >= delta_lifetime
        else:
            cutoff_year = (year-current_year_horizon+1)*system["interval_between_years"]
            return cutoff_year <= lifetime_existing

    @classmethod
    def get_lifetime_range(cls, optimization_setup, tech, year):
        """ returns lifetime range of technology.

        :param optimization_setup: OptimizationSetup the technology is part of
        :param tech: name of the technology
        :param year: yearly time step
        :return: lifetime range of technology
        """
        first_lifetime_year = cls.get_first_lifetime_time_step(optimization_setup, tech, year)
        first_lifetime_year = max(first_lifetime_year, optimization_setup.sets["set_time_steps_yearly"][0])
        return range(first_lifetime_year, year + 1)

    @classmethod
    def get_first_lifetime_time_step(cls,optimization_setup,tech,year):
        """
        returns first lifetime time step of technology,
        i.e., the earliest time step in the past whose capacity is still available at the current time step
        :param optimization_setup: The optimization setup to add everything
        :param tech: name of technology
        :param year: yearly time step
        :return: first lifetime step
        """
        # get params and system
        params = optimization_setup.parameters.dict_parameters
        system = optimization_setup.system
        lifetime = params.lifetime[tech]
        # conservative estimate of lifetime (floor)
        del_lifetime = int(np.floor(lifetime/system["interval_between_years"])) - 1
        return year - del_lifetime

    @classmethod
    def get_investment_time_step(cls,optimization_setup,tech,year):
        """
        returns investment time step of technology, i.e., the time step in which the technology is invested considering the construction time
        :param optimization_setup: The optimization setup to add everything
        :param tech: name of technology
        :param year: yearly time step
        :return: investment time step
        """
        # get params and system
        params = optimization_setup.parameters.dict_parameters
        system = optimization_setup.system
        construction_time = params.construction_time[tech]
        # conservative estimate of construction time (ceil)
        del_construction_time = int(np.ceil(construction_time/system["interval_between_years"]))
        return year - del_construction_time

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Technology --- ###
    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <Technology>

        :param optimization_setup: The OptimizationSetup """
        # construct the pe.Sets of the class <Technology>
        energy_system = optimization_setup.energy_system

        # conversion technologies
        optimization_setup.sets.add_set(name="set_conversion_technologies", data=energy_system.set_conversion_technologies,
                                        doc="Set of conversion technologies. Indexed by set_technologies")
        # retrofitting technologies
        optimization_setup.sets.add_set(name="set_retrofitting_technologies", data=energy_system.set_retrofitting_technologies,
                                        doc="Set of retrofitting technologies. Indexed by set_conversion_technologies")
        # transport technologies
        optimization_setup.sets.add_set(name="set_transport_technologies", data=energy_system.set_transport_technologies,
                                        doc="Set of transport technologies. Indexed by set_technologies")
        # storage technologies
        optimization_setup.sets.add_set(name="set_storage_technologies", data=energy_system.set_storage_technologies,
                                        doc="Set of storage technologies. Indexed by set_technologies")
        # existing installed technologies
        optimization_setup.sets.add_set(name="set_technologies_existing", data=optimization_setup.get_attribute_of_all_elements(cls, "set_technologies_existing"),
                                        doc="Set of existing technologies. Indexed by set_technologies",
                                        index_set="set_technologies")
        # reference carriers
        optimization_setup.sets.add_set(name="set_reference_carriers", data=optimization_setup.get_attribute_of_all_elements(cls, "reference_carrier"),
                                        doc="set of all reference carriers correspondent to a technology. Indexed by set_technologies",
                                        index_set="set_technologies")
        # anyaxie
        if optimization_setup.system["use_endogenous_learning"]:
            # segments for pwa of cumulative cost
            optimization_setup.sets.add_set(name="set_total_cost_pwa_segments",
                                            data=optimization_setup.get_attribute_of_all_elements(cls,
                                                                                                  "set_total_cost_pwa_segments"),
                                            doc="Set of segments for pwa of total cost function",
                                            index_set="set_technologies")

        # add pe.Sets of the child classes
        for subclass in cls.__subclasses__():
            subclass.construct_sets(optimization_setup)

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <Technology>

        :param optimization_setup: The OptimizationSetup """
        # construct pe.Param of the class <Technology>

        # existing capacity
        optimization_setup.parameters.add_parameter(name="capacity_existing", index_names=["set_technologies", "set_capacity_types", "set_location", "set_technologies_existing"], capacity_types=True, doc='Parameter which specifies the existing technology size', calling_class=cls)
        # existing capacity
        optimization_setup.parameters.add_parameter(name="capacity_investment_existing", index_names=["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly_entire_horizon"], capacity_types=True, doc='Parameter which specifies the size of the previously invested capacities', calling_class=cls)
        # minimum capacity addition
        optimization_setup.parameters.add_parameter(name="capacity_addition_min", index_names=["set_technologies", "set_capacity_types"], capacity_types=True, doc='Parameter which specifies the minimum capacity addition that can be installed', calling_class=cls)
        # maximum capacity addition
        optimization_setup.parameters.add_parameter(name="capacity_addition_max", index_names=["set_technologies", "set_capacity_types"], capacity_types=True, doc='Parameter which specifies the maximum capacity addition that can be installed', calling_class=cls)
        # unbounded capacity addition
        optimization_setup.parameters.add_parameter(name="capacity_addition_unbounded", index_names=["set_technologies"], doc='Parameter which specifies the unbounded capacity addition that can be added each year (only for delayed technology deployment)', calling_class=cls)
        # lifetime existing technologies
        optimization_setup.parameters.add_parameter(name="lifetime_existing", index_names=["set_technologies", "set_location", "set_technologies_existing"], doc='Parameter which specifies the remaining lifetime of an existing technology', calling_class=cls)
        # lifetime existing technologies
        optimization_setup.parameters.add_parameter(name="capex_capacity_existing", index_names=["set_technologies", "set_capacity_types", "set_location", "set_technologies_existing"], capacity_types=True, doc='Parameter which specifies the total capex of an existing technology which still has to be paid', calling_class=cls)
        # variable specific opex
        optimization_setup.parameters.add_parameter(name="opex_specific_variable", index_names=["set_technologies","set_location","set_time_steps_operation"], doc='Parameter which specifies the variable specific opex', calling_class=cls)
        # fixed specific opex
        optimization_setup.parameters.add_parameter(name="opex_specific_fixed", index_names=["set_technologies", "set_capacity_types","set_location","set_time_steps_yearly"], capacity_types=True, doc='Parameter which specifies the fixed annual specific opex', calling_class=cls)
        # lifetime newly built technologies
        optimization_setup.parameters.add_parameter(name="lifetime", index_names=["set_technologies"], doc='Parameter which specifies the lifetime of a newly built technology', calling_class=cls)
        # construction_time newly built technologies
        optimization_setup.parameters.add_parameter(name="construction_time", index_names=["set_technologies"], doc='Parameter which specifies the construction time of a newly built technology', calling_class=cls)
        # maximum diffusion rate, i.e., increase in capacity
        optimization_setup.parameters.add_parameter(name="max_diffusion_rate", index_names=["set_technologies", "set_time_steps_yearly"], doc="Parameter which specifies the maximum diffusion rate which is the maximum increase in capacity between investment steps", calling_class=cls)
        # capacity_limit of technologies
        optimization_setup.parameters.add_parameter(name="capacity_limit", index_names=["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], capacity_types=True, doc='Parameter which specifies the capacity limit of technologies', calling_class=cls)
        # minimum load relative to capacity
        optimization_setup.parameters.add_parameter(name="min_load", index_names=["set_technologies", "set_capacity_types", "set_location", "set_time_steps_operation"], capacity_types=True, doc='Parameter which specifies the minimum load of technology relative to installed capacity', calling_class=cls)
        # maximum load relative to capacity
        optimization_setup.parameters.add_parameter(name="max_load", index_names=["set_technologies", "set_capacity_types", "set_location", "set_time_steps_operation"], capacity_types=True, doc='Parameter which specifies the maximum load of technology relative to installed capacity', calling_class=cls)
        # carbon intensity
        optimization_setup.parameters.add_parameter(name="carbon_intensity_technology", index_names=["set_technologies", "set_location"], doc='Parameter which specifies the carbon intensity of each technology', calling_class=cls)
        # calculate additional existing parameters
        optimization_setup.parameters.add_parameter(name="existing_capacities", data=cls.get_existing_quantity(optimization_setup, type_existing_quantity="capacity"),
                                                    doc="Parameter which specifies the total available capacity of existing technologies at the beginning of the optimization",calling_class=cls)
        optimization_setup.parameters.add_parameter(name="existing_capex",data=cls.get_existing_quantity(optimization_setup,type_existing_quantity="cost_capex"),
                                                    doc="Parameter which specifies the total capex of existing technologies at the beginning of the optimization",
                                                    calling_class=cls)
        # anyaxie
        if optimization_setup.system["use_endogenous_learning"]:
            # Add learning rate parameter
            optimization_setup.parameters.add_parameter(name="learning_rate", index_names=["set_technologies"], doc='Parameter which specifies the learning rate of the technology', calling_class=cls)
            # Add global share factor
            optimization_setup.parameters.add_parameter(name="global_share_factor",index_names=["set_technologies"], doc='Parameter which specifies the global share factor of the technology', calling_class=cls)
            # Add lower bound of PWA
            optimization_setup.parameters.add_parameter(name="total_cost_pwa_points_lower_bound", index_names=["set_technologies", "set_capacity_types", "set_total_cost_pwa_segments"], capacity_types=True, doc='Parameter which specifies the lower bound of the pwa of total cost function', calling_class=cls)
            # Add upper bound of PWA
            optimization_setup.parameters.add_parameter(name="total_cost_pwa_points_upper_bound", index_names=["set_technologies", "set_capacity_types", "set_total_cost_pwa_segments"], capacity_types=True, doc='Parameter which specifies the upper bound of the pwa of total cost function', calling_class=cls)
            # Add interpolation points of TC lower bound of PWA
            optimization_setup.parameters.add_parameter(name="total_cost_pwa_TC_lower_bound", index_names=["set_technologies", "set_capacity_types", "set_total_cost_pwa_segments"], capacity_types=True, doc='Parameter which specifies the lower bound of the pwa of total cost function', calling_class=cls)
            # Add interpolation points of TC upper bound of PWA
            optimization_setup.parameters.add_parameter(name="total_cost_pwa_TC_upper_bound", index_names=["set_technologies", "set_capacity_types", "set_total_cost_pwa_segments"], capacity_types=True, doc='Parameter which specifies the upper bound of the pwa of total cost function', calling_class=cls)
            # Add intersect of PWA
            optimization_setup.parameters.add_parameter(name="total_cost_pwa_intersect", index_names=[ "set_technologies", "set_capacity_types", "set_total_cost_pwa_segments"], capacity_types=True, doc='Parameter which specifies the intersect of the pwa of total cost function', calling_class=cls)
            # Add slope of PWA
            optimization_setup.parameters.add_parameter(name="total_cost_pwa_slope", index_names=["set_technologies","set_capacity_types","set_total_cost_pwa_segments"], capacity_types=True, doc='Parameter which specifies the slope of the pwa of total cost function', calling_class=cls)
            # Add initial global cost of PWA
            optimization_setup.parameters.add_parameter(name="total_cost_pwa_initial_global_cost", index_names=[ "set_technologies", "set_capacity_types"], capacity_types=True, doc='Parameter which specifies the initital total global cost of the pwa of total cost function', calling_class=cls)
            # Add initial unit cost of PWA
            optimization_setup.parameters.add_parameter(name="total_cost_pwa_initial_unit_cost", index_names=[ "set_technologies", "set_capacity_types"], capacity_types=True, doc='Parameter which specifies the initital unit cost of the technology', calling_class=cls)
            # Add global initial capacity
            optimization_setup.parameters.add_parameter(name="global_initial_capacity", index_names=["set_technologies"], doc='Parameter which specifies the global initial capacity of the technology', calling_class=cls)
            # Add parameter for  cumulative capacity in the rest of the world
            optimization_setup.parameters.add_parameter(name="cum_capacity_row", index_names=["set_technologies", "set_time_steps_yearly"], doc='Parameter which specifies the global initial capacity of the technology', calling_class=cls) # todo: Add capacity types? Same capacity addition energy and power rated?
        # add pe.Param of the child classes
        for subclass in cls.__subclasses__():
            subclass.construct_params(optimization_setup)

    @classmethod
    def construct_vars(cls, optimization_setup):
        """ constructs the pe.Vars of the class <Technology>
        :param optimization_setup: The OptimizationSetup """

        model = optimization_setup.model
        variables = optimization_setup.variables
        sets = optimization_setup.sets

        def capacity_bounds(tech, capacity_type, loc, time):
            """ 
            # TODO: This could be vectorized
            return bounds of capacity for bigM expression
            :param tech: tech index
            :param capacity_type: either power or energy
            :param loc: location of capacity
            :param time: investment time step
            :return bounds: bounds of capacity"""
            # bounds only needed for Big-M formulation, thus if any technology is modeled with on-off behavior
            if tech in techs_on_off:
                system = optimization_setup.system
                params = optimization_setup.parameters.dict_parameters
                if capacity_type == system["set_capacity_types"][0]:
                    energy_string = ""
                else:
                    energy_string = "_energy"
                capacity_existing = getattr(params, "capacity_existing" + energy_string)
                capacity_addition_max = getattr(params, "capacity_addition_max" + energy_string)
                capacity_limit = getattr(params, "capacity_limit" + energy_string)
                capacities_existing = 0
                for id_technology_existing in sets["set_technologies_existing"][tech]:
                    if params.lifetime_existing[tech, loc, id_technology_existing] > params.lifetime[tech]:
                        if time > params.lifetime_existing[tech, loc, id_technology_existing] - params.lifetime[tech]:
                            capacities_existing += capacity_existing[tech, capacity_type, loc, id_technology_existing]
                    elif time <= params.lifetime_existing[tech, loc, id_technology_existing] + 1:
                        capacities_existing += capacity_existing[tech, capacity_type, loc, id_technology_existing]

                capacity_addition_max = len(sets["set_time_steps_yearly"]) * capacity_addition_max[tech, capacity_type]
                max_capacity_limit = capacity_limit[tech, capacity_type, loc, time]
                bound_capacity = min(capacity_addition_max + capacities_existing, max_capacity_limit + capacities_existing)
                return 0, bound_capacity
            else:
                return 0, np.inf

        # bounds only needed for Big-M formulation, thus if any technology is modeled with on-off behavior
        techs_on_off = cls.create_custom_set(["set_technologies", "set_on_off"], optimization_setup)[0]
        # construct pe.Vars of the class <Technology>
        # capacity technology
        variables.add_variable(model, name="capacity", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup),
            bounds=capacity_bounds, doc='size of installed technology at location l and time t', unit_category={"energy_quantity": 1, "time": -1})
        # built_capacity technology
        variables.add_variable(model, name="capacity_addition", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup),
            bounds=(0,np.inf), doc='size of built technology (invested capacity after construction) at location l and time t', unit_category={"energy_quantity": 1, "time": -1})
        # invested_capacity technology
        variables.add_variable(model, name="capacity_investment", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup),
            bounds=(0,np.inf), doc='size of invested technology at location l and time t',unit_category={"energy_quantity": 1, "time": -1})
        # anyaxie:
        # if not optimization_setup.system["use_endogenous_learning"]:
        # capex of building capacity
        variables.add_variable(model, name="cost_capex", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup),
            bounds=(0, np.inf), doc='capex for building technology at location l and time t', unit_category={"money": 1})
        # else:
        #     variables.add_variable(model, name="cost_capex", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_time_steps_yearly"],optimization_setup),
        #         bounds=(0, np.inf), doc='capex for building technology at location l and time t',unit_category={"money": 1})
        # annual capex of having capacity
        variables.add_variable(model, name="capex_yearly", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup),
            bounds=(0,np.inf), doc='annual capex for having technology at location l', unit_category={"money": 1})
        # total capex
        variables.add_variable(model, name="cost_capex_total", index_sets=sets["set_time_steps_yearly"],
            bounds=(0,np.inf), doc='total capex for installing all technologies in all locations at all times', unit_category={"money": 1})
        # opex
        variables.add_variable(model, name="cost_opex", index_sets=cls.create_custom_set(["set_technologies", "set_location", "set_time_steps_operation"], optimization_setup),
            bounds=(0,np.inf), doc="opex for operating technology at location l and time t", unit_category={"money": 1, "time": -1})
        # total opex
        variables.add_variable(model, name="cost_opex_total", index_sets=sets["set_time_steps_yearly"],
            bounds=(0,np.inf), doc="total opex all technologies and locations in year y", unit_category={"money": 1})
        # yearly opex
        variables.add_variable(model, name="opex_yearly", index_sets=cls.create_custom_set(["set_technologies", "set_location", "set_time_steps_yearly"], optimization_setup),
            bounds=(0,np.inf), doc="yearly opex for operating technology at location l and year y", unit_category={"money": 1})
        # carbon emissions
        variables.add_variable(model, name="carbon_emissions_technology", index_sets=cls.create_custom_set(["set_technologies", "set_location", "set_time_steps_operation"], optimization_setup),
            doc="carbon emissions for operating technology at location l and time t", unit_category={"emissions": 1, "time": -1})
        # total carbon emissions technology
        variables.add_variable(model, name="carbon_emissions_technology_total", index_sets=sets["set_time_steps_yearly"],
           doc="total carbon emissions for operating technology at location l and time t",unit_category={"emissions": 1})

        # anyaxie
        if optimization_setup.system["use_endogenous_learning"]:
            # yearly capex as sum over all nodes
            # variables.add_variable(model, name="capex_yearly_all_positions", index_sets=cls.create_custom_set(
            #     ["set_technologies", "set_capacity_types", "set_time_steps_yearly"], optimization_setup),
            #                        bounds=(0, np.inf), doc="yearly capex of technology h over all positions", unit_category={"money": 1})
            # investment cost sum over all nodes
            variables.add_variable(model, name="cost_capex_all_positions", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_time_steps_yearly"], optimization_setup),
                                   bounds=(0, np.inf), doc="investment cost of technology h over all positions", unit_category={"money": 1})

            # global cumulative capacity
            variables.add_variable(model, name="global_cumulative_capacity", index_sets=cls.create_custom_set(
                ["set_technologies", "set_capacity_types", "set_time_steps_yearly"], optimization_setup),
                                   bounds=(0, np.inf), doc="learning-inducing capacity of technology h in period y", unit_category={"energy_quantity": 1, "time": -1})
            # segement selection for pwa total cost binary variables
            variables.add_variable(model, name="total_cost_pwa_segment_selection", index_sets=cls.create_custom_set(
                ["set_technologies", "set_capacity_types", "set_time_steps_yearly", "set_total_cost_pwa_segments"],optimization_setup),
                                   binary=True, doc="segment selection binary variable", unit_category={})
            # segment position variable for cumulative global capacity
            variables.add_variable(model, name="total_cost_pwa_cum_capacity_segment_position",index_sets=cls.create_custom_set(
                ["set_technologies", "set_capacity_types", "set_time_steps_yearly","set_total_cost_pwa_segments"], optimization_setup),
                                   bounds=(0, np.inf), doc="cumulative global capacity of technology y in year y in segment w", unit_category={"energy_quantity": 1, "time": -1})
            # total global cost variable
            variables.add_variable(model, name="total_cost_pwa_global_cost", index_sets=cls.create_custom_set(
                ["set_technologies", "set_capacity_types", "set_time_steps_yearly"], optimization_setup),
                                     bounds=(0, np.inf), doc="total global cost of technology h in period y", unit_category={"money": 1})
            # yearly capex as sum over all nodes
            variables.add_variable(model, name="cost_capex_global", index_sets=cls.create_custom_set(
                ["set_technologies", "set_capacity_types", "set_time_steps_yearly"], optimization_setup),
                                   bounds=(0, np.inf), doc="yearly capex of technology h over all positions",  unit_category={"money": 1})
            # European cumulative capacity
            variables.add_variable(model, name="european_cumulative_capacity", index_sets=cls.create_custom_set(
                ["set_technologies", "set_capacity_types", "set_time_steps_yearly"], optimization_setup),
                                   bounds=(0, np.inf), doc="learning-inducing capacity of technology h in period y",unit_category={"energy_quantity": 1, "time": -1})
            # segement selection for pwa total cost binary variables of European cost
            variables.add_variable(model, name="total_cost_pwa_segment_selection_eu", index_sets=cls.create_custom_set(
                ["set_technologies", "set_capacity_types", "set_time_steps_yearly", "set_total_cost_pwa_segments"],optimization_setup),
                                   binary=True, doc="segment selection binary variable", unit_category={})
            # segment position variable for cumulative European capacity
            variables.add_variable(model, name="total_cost_pwa_cum_capacity_segment_position_eu",index_sets=cls.create_custom_set(
                ["set_technologies", "set_capacity_types", "set_time_steps_yearly","set_total_cost_pwa_segments"], optimization_setup),
                                   bounds=(0, np.inf), doc="cumulative global capacity of technology y in year y in segment w", unit_category={})
            # total European cost variable
            variables.add_variable(model, name="total_cost_pwa_european_cost", index_sets=cls.create_custom_set(
                ["set_technologies", "set_capacity_types", "set_time_steps_yearly"], optimization_setup),
                                     bounds=(0, np.inf), doc="total global cost of technology h in period y",unit_category={"money": 1} )



        # install technology
        # Note: binary variables are written into the lp file by linopy even if they are not relevant for the optimization,
        # which makes all problems MIPs. Therefore, we only add binary variables, if really necessary. Gurobi can handle this
        # by noting that the binary variables are not part of the model, however, only if there are no binary variables at all,
        # it is possible to get the dual values of the constraints.
        mask = cls._technology_installation_mask(optimization_setup)
        if mask.any():
            variables.add_variable(model, name="technology_installation", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup),
                                   binary=True, doc='installment of a technology at location l and time t', mask=mask, unit_category=None)

        # add pe.Vars of the child classes
        for subclass in cls.__subclasses__():
            subclass.construct_vars(optimization_setup)

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <Technology>

        :param optimization_setup: The OptimizationSetup """
        model = optimization_setup.model
        constraints = optimization_setup.constraints
        sets = optimization_setup.sets
        # construct pe.Constraints of the class <Technology>
        rules = TechnologyRules(optimization_setup)
        #  technology capacity_limit
        constraints.add_constraint_block(model, name="constraint_technology_capacity_limit",
                                         constraint=rules.constraint_technology_capacity_limit_block(),
                                         doc='limited capacity of  technology depending on loc and time')
        # minimum capacity
        constraints.add_constraint_block(model, name="constraint_technology_min_capacity_addition",
                                         constraint=rules.constraint_technology_min_capacity_addition_block(),
                                         doc='min capacity of technology that can be installed')
        # maximum capacity
        constraints.add_constraint_block(model, name="constraint_technology_max_capacity_addition",
                                         constraint=rules.constraint_technology_max_capacity_addition_block(),
                                         doc='max capacity of technology that can be installed')
        # construction period
        constraints.add_constraint_block(model, name="constraint_technology_construction_time",
                                         constraint=rules.constraint_technology_construction_time_block(),
                                         doc='lead time in which invested technology is constructed')
        # lifetime
        constraints.add_constraint_block(model, name="constraint_technology_lifetime",
                                         constraint=rules.constraint_technology_lifetime_block(),
                                         doc='calculate all existing capacity in certain year')
        # limit diffusion rate
        constraints.add_constraint_block(model, name="constraint_technology_diffusion_limit",
                                         constraint=rules.constraint_technology_diffusion_limit_block(),
                                         doc="limit the newly built capacity by the existing knowledge stock")
        # limit diffusion rate total
        constraints.add_constraint_block(model, name="constraint_technology_diffusion_limit_total",
                                         constraint=rules.constraint_technology_diffusion_limit_total_block(),
                                         doc="limit the newly built capacity by the existing knowledge stock for the entire energy system")
        # limit max load by installed capacity
        constraints.add_constraint_block(model, name="constraint_capacity_factor",
                                         constraint=rules.constraint_capacity_factor_block(),
                                         doc='limit max load by installed capacity')
        # annual capex of having capacity
        # if not optimization_setup.system["use_endogenous_learning"]:
        constraints.add_constraint_block(model, name="constraint_capex_yearly",
                                             constraint=rules.constraint_capex_yearly_block(),
                                             doc='annual capex of having capacity of technology.')
        # total capex of all technologies
        constraints.add_constraint_rule(model, name="constraint_cost_capex_total", index_sets=sets["set_time_steps_yearly"], rule=rules.constraint_cost_capex_total_rule,
            doc='total capex of all technology that can be installed.')
        # calculate opex
        constraints.add_constraint_block(model, name="constraint_opex_technology",
                                         constraint=rules.constraint_opex_technology_block(),
                                         doc="opex for each technology at each location and time step")
        # yearly opex
        constraints.add_constraint_block(model, name="constraint_opex_yearly",
                                         constraint=rules.constraint_opex_yearly_block(),
                                         doc='total opex of all technology that are operated.')
        # total opex of all technologies
        constraints.add_constraint_rule(model, name="constraint_cost_opex_total", index_sets=sets["set_time_steps_yearly"], rule=rules.constraint_cost_opex_total_rule, doc='total opex of all technology that are operated.')
        # carbon emissions of technologies
        constraints.add_constraint_block(model, name="constraint_carbon_emissions_technology",
                                         constraint=rules.constraint_carbon_emissions_technology_block(),
                                         doc="carbon emissions for each technology at each location and time step")
        # total carbon emissions of technologies
        constraints.add_constraint_block(model, name="constraint_carbon_emissions_technology_total", constraint=rules.constraint_carbon_emissions_technology_total_block(),
                                         doc="total carbon emissions for each technology at each location and time step")

        # anyaxie
        if optimization_setup.system["use_endogenous_learning"]:
            # segment capacity sum
            constraints.add_constraint_block(model, name="constraint_pwa_total_cost_global_cum_capacity_segment",
                                             constraint=rules.constraint_pwa_total_cost_global_cum_capacity_segment_block(),
                                                doc="segment capacity sum for pwa of cumulative cost")

            # segment capacity upper bounds
            constraints.add_constraint_block(model, name="constraint_pwa_total_cost_cum_capacity_upper_bound",
                                             constraint=rules.constraint_pwa_total_cost_cum_capacity_upper_bound_block(),
                                             doc="segment capacity upper bounds for pwa of cumulative cost")

            # segment capacity lower bounds
            constraints.add_constraint_block(model, name="constraint_pwa_total_cost_cum_capacity_lower_bound",
                                             constraint=rules.constraint_pwa_total_cost_cum_capacity_lower_bound_block(),
                                                doc="segment capacity lower bounds for pwa of cumulative cost")


            # segment selection pwa of total cost
            constraints.add_constraint_block(model, name="constraint_pwa_total_cost_segment_selection",
                                             constraint=rules.constraint_pwa_total_cost_segment_selection_block(),
                                             doc="segment selection with binary variable for pwa of cumulative cost")


            # pwa approximation for total cumulative cost
            constraints.add_constraint_block(model, name="constraint_approximate_total_global_cost",
                                             constraint=rules.constraint_approximate_total_global_cost_block(),
                                             doc="approximation of cumulative cost with pwa")

            # constraint for cumulative global capacity
            constraints.add_constraint_block(model, name="constraint_cum_global_capacity",
                                             constraint=rules.constraint_global_cum_capacity_block(),
                                             doc="constraint for cumulative global capacity")

            # # sum of capacities as difference of total cumulative cost over all nodes
            # constraints.add_constraint_block(model, name="constraint_capex_yearly_all_positions",
            #                                  constraint=rules.constraint_capex_yearly_all_positions_block(),
            #                                  doc="yearly capex of all nodes as difference of total cumulative cost between timesteps")
            # segment capacity sum Europe
            constraints.add_constraint_block(model, name="constraint_pwa_total_cost_european_cum_capacity_segment",
                                             constraint=rules.constraint_pwa_total_cost_european_cum_capacity_segment_block(),
                                             doc="segment capacity sum for pwa of cumulative cost")

            # segment capacity upper bounds Europe
            constraints.add_constraint_block(model, name="constraint_pwa_total_cost_cum_capacity_upper_bound_eu",
                                             constraint=rules.constraint_pwa_total_cost_cum_capacity_upper_bound_eu_block(),
                                             doc="segment capacity upper bounds for pwa of cumulative cost")

            # segment capacity lower bounds Europe
            constraints.add_constraint_block(model, name="constraint_pwa_total_cost_cum_capacity_lower_bound_eu",
                                             constraint=rules.constraint_pwa_total_cost_cum_capacity_lower_bound_eu_block(),
                                             doc="segment capacity lower bounds for pwa of cumulative cost")

            # segment selection pwa of total cost Europe
            constraints.add_constraint_block(model, name="constraint_pwa_total_cost_segment_selection_eu",
                                             constraint=rules.constraint_pwa_total_cost_segment_selection_eu_block(),
                                             doc="segment selection with binary variable for pwa of cumulative cost")

            # pwa approximation for total cumulative cost Europe
            constraints.add_constraint_block(model, name="constraint_approximate_total_european_cost",constraint=rules.constraint_approximate_total_european_cost_block(),
                                             doc="approximation of cumulative cost with pwa")
            # constraint for cumulative global capacity Europe
            constraints.add_constraint_block(model, name="constraint_european_addition",constraint=rules.constraint_european_addition_block(),
                                             doc="constraint for cumulative global capacity")
            # cost capex constraint
            constraints.add_constraint_block(model, name="constraint_cost_capex", constraint=rules.constraint_cost_capex_block(),
                                                         doc="Calculate european investment cost")

            # split cost capex constraint
            constraints.add_constraint_block(model, name="constraint_cost_capex_split", constraint=rules.constraint_split_capex_across_all_positions_block(),
                                                         doc="Split european investment cost across all nodes")


        # disjunct if technology is on
        # the disjunction variables
        variables = optimization_setup.variables
        index_vals, _ = cls.create_custom_set(["set_technologies", "set_on_off", "set_capacity_types", "set_location", "set_time_steps_operation"], optimization_setup)
        index_names = ["on_off_technologies", "on_off_capacity_types", "on_off_locations", "on_off_time_steps_operation"]
        variables.add_variable(model, name="tech_on_var",
                               index_sets=(index_vals, index_names),
                               doc="Binary variable which equals 1 when technology is switched on at location l and time t", binary=True, unit_category=None)
        variables.add_variable(model, name="tech_off_var",
                               index_sets=(index_vals, index_names),
                               doc="Binary variable which equals 1 when technology is switched off at location l and time t", binary=True, unit_category=None)
        model.add_constraints(model.variables["tech_on_var"] + model.variables["tech_off_var"] == 1, name="tech_on_off_cons")
        n_cons = model.constraints.ncons

        # disjunct if technology is on
        constraints.add_constraint_rule(model, name="disjunct_on_technology",
            index_sets=cls.create_custom_set(["set_technologies", "set_on_off", "set_capacity_types", "set_location", "set_time_steps_operation"], optimization_setup), rule=rules.disjunct_on_technology_rule,
            doc="disjunct to indicate that technology is on")
        # disjunct if technology is off
        constraints.add_constraint_rule(model, name="disjunct_off_technology",
            index_sets=cls.create_custom_set(["set_technologies", "set_on_off", "set_capacity_types", "set_location", "set_time_steps_operation"], optimization_setup), rule=rules.disjunct_off_technology_rule,
            doc="disjunct to indicate that technology is off")

        # if nothing was added we can remove the tech vars again
        if model.constraints.ncons == n_cons:
            model.constraints.remove("tech_on_off_cons")
            model.variables.remove("tech_on_var")
            model.variables.remove("tech_off_var")

        # add pe.Constraints of the child classes
        for subclass in cls.__subclasses__():
            logging.info(f"Construct pe.Constraints of {subclass.__name__}")
            subclass.construct_constraints(optimization_setup)

    @classmethod
    def _technology_installation_mask(cls, optimization_setup):
        """check if the binary variable is necessary

        :param optimization_setup: optimization setup object"""
        params = optimization_setup.parameters
        model = optimization_setup.model
        sets = optimization_setup.sets

        mask = xr.DataArray(False, coords=[model.variables.coords["set_time_steps_yearly"],
                                           model.variables.coords["set_technologies"],
                                           model.variables.coords["set_capacity_types"],
                                           model.variables.coords["set_location"], ])

        # used in transport technology
        techs = list(sets["set_transport_technologies"])
        if len(techs) > 0:
            edges = list(sets["set_edges"])
            sub_mask = (params.distance.loc[techs, edges] * params.capex_per_distance_transport.loc[techs, edges] != 0)
            sub_mask = sub_mask.rename({"set_transport_technologies": "set_technologies", "set_edges": "set_location"})
            mask.loc[:, techs, :, edges] |= sub_mask

        # used in constraint_technology_min_capacity_addition
        mask = mask | (params.capacity_addition_min.notnull() & (params.capacity_addition_min != 0))

        # used in constraint_technology_max_capacity_addition
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup)
        index = ZenIndex(index_values, index_names)
        sub_mask = (params.capacity_addition_max.notnull() & (params.capacity_addition_max != np.inf) & (params.capacity_addition_max != 0))
        for tech, capacity_type in index.get_unique([0, 1]):
            locs = index.get_values(locs=[tech, capacity_type], levels=2, unique=True)
            mask.loc[:, tech, capacity_type, locs] |= sub_mask.loc[tech, capacity_type]

        return mask

    @classmethod
    def get_existing_quantity(cls, optimization_setup, type_existing_quantity):
        """
        get existing capacities of all technologies
        :param optimization_setup: The OptimizationSetup the element is part of
        :param type_existing_quantity: capacity or cost_capex
        :return: The existing capacities
        """

        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup)
        # get all the capacities
        index_arrs = IndexSet.tuple_to_arr(index_values, index_names)
        coords = [optimization_setup.sets.get_coord(data, name) for data, name in zip(index_arrs, index_names)]
        existing_quantities = xr.DataArray(np.nan, coords=coords, dims=index_names)
        values = np.zeros(len(index_values))
        for i, (tech, capacity_type, loc, time) in enumerate(index_values):
            values[i] = Technology.get_available_existing_quantity(optimization_setup, tech, capacity_type, loc, time,
                                                                   type_existing_quantity=type_existing_quantity)
        existing_quantities.loc[index_arrs] = values
        return existing_quantities


class TechnologyRules(GenericRule):
    """
    Rules for the Technology class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules
        :param optimization_setup: OptimizationSetup of the element
        """

        super().__init__(optimization_setup)

    # Disjunctive Constraints
    # -----------------------

    def disjunct_on_technology_rule(self, tech, capacity_type, loc, time):
        """definition of disjunct constraints if technology is On
        iterate through all subclasses to find corresponding implementation of disjunct constraints

        :param tech: technology
        :param capacity_type: capacity type
        :param loc: location
        :param time: time step
        """
        for subclass in Technology.__subclasses__():
            if tech in self.optimization_setup.get_all_names_of_elements(subclass):
                # extract the relevant binary variable (not scalar, .loc is necessary)
                binary_var = self.optimization_setup.model.variables["tech_on_var"].loc[tech, capacity_type, loc, time]
                subclass.disjunct_on_technology_rule(self.optimization_setup, tech, capacity_type, loc, time, binary_var)
                return None

    def disjunct_off_technology_rule(self, tech, capacity_type, loc, time):
        """definition of disjunct constraints if technology is off
        iterate through all subclasses to find corresponding implementation of disjunct constraints

        :param tech: technology
        :param capacity_type: capacity type
        :param loc: location
        :param time: time step
        """
        for subclass in Technology.__subclasses__():
            if tech in self.optimization_setup.get_all_names_of_elements(subclass):
                # extract the relevant binary variable (not scalar, .loc is necessary)
                binary_var = self.optimization_setup.model.variables["tech_off_var"].loc[tech, capacity_type, loc, time]
                subclass.disjunct_off_technology_rule(self.optimization_setup, tech, capacity_type, loc, time, binary_var)
                return None

    # Rule-based constraints
    # -----------------------

    def constraint_cost_capex_total_rule(self, year):
        """ sums over all technologies to calculate total capex

        .. math::
            CAPEX_y = \\sum_{h\\in\mathcal{H}}\\sum_{p\\in\mathcal{P}}A_{h,p,y}+\\sum_{k\\in\mathcal{K}}\\sum_{n\\in\mathcal{N}}A^\mathrm{e}_{k,n,y}

        :param year: yearly time step
        :return: linopy constraint
        """

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        # if self.system["use_endogenous_learning"]:
            # term_sum_yearly = self.variables["capex_yearly_all_positions"].loc[..., year].sum()
        # else:
        term_sum_yearly = self.variables["capex_yearly"].loc[..., year].sum()

        ### formulate constraint
        lhs = (self.variables["cost_capex_total"].loc[year]
               - term_sum_yearly)
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_cost_opex_total_rule(self, year):
        """ sums over all technologies to calculate total opex

        .. math::
            OPEX_y = \sum_{h\in\mathcal{H}}\sum_{p\in\mathcal{P}} OPEX_{h,p,y}

        :param year: yearly time step
        :return: linopy constraint
        """

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        term_sum_yearly = self.variables["opex_yearly"].loc[..., year].sum()

        ### formulate constraint
        lhs = (self.variables["cost_opex_total"].loc[year]
               - term_sum_yearly)
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    # Block-based constraints
    # -----------------------

    def constraint_technology_capacity_limit_block(self):
        """limited capacity_limit of technology

        .. math::
            \mathrm{if\ existing\ capacities\ < capacity\ limit}\ s^\mathrm{max}_{h,p,y} \geq S_{h,p,y}
        .. math::
            \mathrm{else}\ \Delta S_{h,p,y} = 0

        :return: linopy constraints
        """

        ### index sets
        # not necessary

        ### masks
        # take the maximum of the capacity limit and the existing capacities.
        # If the capacity limit is 0 (or lower than existing capacities), the maximum is the existing capacity
        maximum_capacity_limit = np.maximum(self.parameters.existing_capacities,self.parameters.capacity_limit)
        # create mask so that skipped if capacity_limit is inf
        m = maximum_capacity_limit != np.inf

        ### index loop
        # not necessary

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        lhs = self.variables["capacity"].where(m)
        rhs = maximum_capacity_limit.where(m,0.0)
        constraints = lhs <= rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_technology_min_capacity_addition_block(self):
        """ min capacity addition of technology

        .. math::
            s^\mathrm{add, min}_{h} B_{i,p,y} \le \Delta S_{h,p,y}

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)
        tech_arr, capacity_type_arr = index.get_unique(["set_technologies", "set_capacity_types"], as_array=True)

        ### masks
        # we create a mask here only to avoid having constraints with binary variables when it's not necessary
        # passing constraints with binary variables to gurobi, even of the type 0 * binary_var, means that no
        # dual variables are returned
        mask = xr.zeros_like(self.parameters.capacity_addition_min, dtype=bool)
        mask.loc[tech_arr, capacity_type_arr] = True
        mask &= self.parameters.capacity_addition_min != 0

        ### index loop
        # not necessary

        ### auxiliary calculations
        # if the mask is empty, we don't need to do anything and abort here
        if not mask.any():
            return []

        ### formulate constraint
        lhs = mask * (self.parameters.capacity_addition_min * self.variables["technology_installation"]
                      - self.variables["capacity_addition"])
        rhs = 0
        constraints = lhs <= rhs

        ### return
        return self.constraints.return_contraints(constraints, mask=mask)

    def constraint_technology_max_capacity_addition_block(self):
        """max capacity addition of technology

        .. math::
            s^\mathrm{add, max}_{h} B_{i,p,y} \ge \Delta S_{h,p,y}

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        constraints = []
        for tech, capacity_type in index.get_unique(["set_technologies", "set_capacity_types"]):
            # not that the else here is just a dummy
            if self.parameters.capacity_addition_max.loc[tech, capacity_type] != np.inf:

                ### auxiliary calculations
                # we only want a constraints with a binary variable if the corresponding max_built_capacity is not zero
                if np.any(self.parameters.capacity_addition_max.loc[tech, capacity_type].notnull() & (self.parameters.capacity_addition_max.loc[tech, capacity_type] != 0)):
                    term_installation = self.parameters.capacity_addition_max.loc[tech, capacity_type].item() * self.variables["technology_installation"].loc[tech, capacity_type]
                else:
                    # dummy
                    term_installation = self.variables["capacity_addition"].loc[tech, capacity_type].where(False)

                ### formulate constraint
                lhs = (- self.variables["capacity_addition"].loc[tech, capacity_type]
                       + term_installation)
                rhs = 0
                constraints.append(lhs >= rhs)

            else:
                # a dummy
                constraints.append(np.nan*self.variables["capacity_addition"].loc[tech, capacity_type].where(False) == np.nan)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies", "set_capacity_types"]),
                                                  index_names=["set_technologies", "set_capacity_types"])

    def constraint_technology_construction_time_block(self):
        """ construction time of technology, i.e., time that passes between investment and availability

        .. math::
            \mathrm{if\ start\ time\ step\ in\ set\ time\ steps\ yearly}\ \Delta S_{h,p,y} = S_{h,p,y}^\mathrm{invest}
        .. math::
            \mathrm{elif\ start\ time\ step\ in\ set\ time\ steps\ yearly\ entire\ horizon}\ \Delta S_{h,p,y} = s^\mathrm{invest, exist}_{h,p,y}
        .. math::
            \mathrm{else}\ \Delta S_{h,p,y} = 0

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over technologies and years, because the conditions depend on the year and the technology
        # we vectorize over capacity types and locations
        constraints = []
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):

            ### auxiliary calculations
            investment_time = Technology.get_investment_time_step(self.optimization_setup, tech, year)
            ### formulate constraint
            if investment_time in self.sets["set_time_steps_yearly"]:
                lhs = (self.variables["capacity_addition"].loc[tech, :, :, year]
                       - self.variables["capacity_investment"].loc[tech, :, :, investment_time])
                rhs = 0
            elif investment_time in self.sets["set_time_steps_yearly_entire_horizon"]:
                lhs = self.variables["capacity_addition"].loc[tech, :, :, year]
                rhs = self.parameters.capacity_investment_existing.loc[tech, :, :, investment_time]
            else:
                lhs = self.variables["capacity_addition"].loc[tech, :, :, year]
                rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

    def constraint_technology_lifetime_block(self):
        """ limited lifetime of the technologies

        .. math::
            S_{h,p,y} = \\sum_{\\tilde{y}=\\max(y_0,y-\\lceil\\frac{l_h}{\\Delta^\mathrm{y}}\\rceil+1)}^y \\Delta S_{h,p,\\tilde{y}}
            + \\sum_{\\hat{y}=\\psi(\\min(y_0-1,y-\\lceil\\frac{l_h}{\\Delta^\mathrm{y}}\\rceil+1))}^{\\psi(y_0)} \\Delta s^\mathrm{ex}_{h,p,\\hat{y}}

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # this mask is just to make sure we only get constraints where we want them, no if-condition
        mask = self.variables["capacity"].mask

        ### index loop
        # we loop over technologies and years, because we need to cycle over the lifetime range of the technology
        # which requires the technology and the year, we vectorize over capacity types and locations
        constraints = []
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):

            ### auxiliary calculations
            term_neg_previous_capacity_additions = []
            for previous_year in Technology.get_lifetime_range(self.optimization_setup, tech, year):
                term_neg_previous_capacity_additions.append(-1.0 * self.variables["capacity_addition"].loc[tech, :, :, previous_year])

            ### formulate constraint
            lhs = lp_sum([1.0 * self.variables["capacity"].loc[tech, :, :, year],
                          *term_neg_previous_capacity_additions])
            rhs = self.parameters.existing_capacities.loc[tech, :, :, year]
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  mask=mask,
                                                  index_values=index.get_unique(["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

    def constraint_technology_diffusion_limit_block(self):
        """limited technology diffusion based on the existing capacity in the previous year

        .. math::
                \\Delta S_{j,e,y}\\leq ((1+\\vartheta_j)^{\\Delta^\mathrm{y}}-1)K_{j,e,y}
                +\\Delta^\mathrm{y}(\\xi\\sum_{\\tilde{j}\\in\\tilde{\mathcal{J}}}S_{\\tilde{j},e,y} + \\zeta_j)

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over technologies, capacity types and time steps, to accurately capture the conditions in the constraint
        # we vectorize over locations
        constraints = []
        for tech, time in index.get_unique(["set_technologies", "set_time_steps_yearly"]):
            # skip if max diffusion rate = inf
            if self.parameters.max_diffusion_rate.loc[tech, time] != np.inf:
                ### auxiliary calculations
                # mask for the capacity types that are not considered
                capacity_types = index.get_values([tech, slice(None), slice(None), time], "set_capacity_types", unique=True)
                mask = xr.DataArray(np.nan, coords=[self.variables.coords["set_capacity_types"]], dims=["set_capacity_types"])
                mask.loc[capacity_types] = 1

                interval_between_years = self.system["interval_between_years"]
                knowledge_depreciation_rate = self.system["knowledge_depreciation_rate"]
                reference_carrier = self.sets["set_reference_carriers"][tech][0]
                if tech in self.sets["set_transport_technologies"]:
                    set_locations = self.sets["set_edges"]
                    set_technology = self.sets["set_transport_technologies"]
                    knowledge_spillover_rate = 0
                else:
                    set_locations = self.sets["set_nodes"]
                    knowledge_spillover_rate = self.parameters.knowledge_spillover_rate
                    if tech in self.sets["set_conversion_technologies"]:
                        set_technology = self.sets["set_conversion_technologies"]
                    else:
                        set_technology = self.sets["set_storage_technologies"]

                    # add capacity addition of entire previous horizon
                end_time = time - 1

                # actual years between first invest time step and end_time
                delta_time = interval_between_years * (end_time - self.sets["set_time_steps_yearly"][0])
                existing_time = self.sets["set_technologies_existing"][tech]
                # Note: instead of summing over all but one location, we sum over all and then subtract one
                term_total_capacity_knowledge_existing = ((self.parameters.capacity_existing.loc[tech, :, set_locations, existing_time]  # add spillover from other regions
                                                      + knowledge_spillover_rate * (self.parameters.capacity_existing.loc[tech, :, set_locations, existing_time].sum("set_location") - self.parameters.capacity_existing.loc[tech, :, set_locations, existing_time]))
                                                     * (1 - knowledge_depreciation_rate) ** (delta_time + self.parameters.lifetime.loc[tech].item() - self.parameters.lifetime_existing.loc[tech, set_locations, existing_time])).sum("set_technologies_existing")
                # round to avoid numerical errors
                # if self.optimization_setup.solver["round_parameters"]:
                #     rounding_value = 10 ** (-self.optimization_setup.solver["rounding_decimal_points_capacity"])
                #     term_total_capacity_knowledge_existing = term_total_capacity_knowledge_existing.where(term_total_capacity_knowledge_existing > rounding_value, 0)

                horizon_time = np.arange(self.sets["set_time_steps_yearly"][0], end_time + 1)
                horizon_time = self.variables.coords["set_time_steps_yearly"][self.variables.coords["set_time_steps_yearly"].isin(horizon_time)]
                if len(horizon_time) >= 1:
                    term_total_capacity_knowledge_addition = ((self.variables["capacity_addition"].loc[tech, :, set_locations, horizon_time]  # add spillover from other regions
                                                          + knowledge_spillover_rate * (self.variables["capacity_addition"].loc[tech, :, set_locations, horizon_time].sum("set_location") - self.variables["capacity_addition"].loc[tech, :, set_locations, horizon_time]))
                                                         * (1 - knowledge_depreciation_rate) ** (interval_between_years * (end_time - horizon_time))).sum("set_time_steps_yearly")
                else:
                    # dummy term
                    term_total_capacity_knowledge_addition = self.variables["capacity_investment"].loc[tech, :, set_locations, time].where(False)

                # total capacity in previous year; if time is first time step of interval, use existing capacities of present year
                other_techs = [other_tech for other_tech in set_technology if self.sets["set_reference_carriers"][other_tech][0] == reference_carrier]
                if time != self.optimization_setup.energy_system.set_time_steps_yearly[0]:
                    term_total_capacity_all_techs_var = self.variables["capacity"].loc[other_techs, :, set_locations, time-1].sum("set_technologies")
                    term_total_capacity_all_techs_param = xr.zeros_like(self.parameters.existing_capacities.loc[tech,:,set_locations,time])
                else:
                    term_total_capacity_all_techs_param = self.parameters.existing_capacities.loc[other_techs,:,set_locations,time].sum("set_technologies")
                    term_total_capacity_all_techs_var = self.variables["capacity"].loc[tech, :, set_locations, time].where(False)

                ### formulate constraint
                # build the lhs
                lhs = (self.variables["capacity_investment"].loc[tech, :, set_locations, time]
                       + self.variables["capacity_investment"].loc[tech, :, :, time].where(False)
                       - ((1 + self.parameters.max_diffusion_rate.loc[tech, time].item()) ** interval_between_years - 1) * term_total_capacity_knowledge_addition
                       - self.parameters.market_share_unbounded * term_total_capacity_all_techs_var)
                lhs *= mask

                # build the rhs
                rhs = xr.zeros_like(self.parameters.existing_capacities).loc[tech, :,:, time]
                rhs.loc[:, set_locations] += ((1 + self.parameters.max_diffusion_rate.loc[tech, time].item()) ** interval_between_years - 1) * term_total_capacity_knowledge_existing
                # add initial market share until which the diffusion rate is unbounded
                rhs.loc[:, set_locations] += self.parameters.market_share_unbounded * term_total_capacity_all_techs_param + self.parameters.capacity_addition_unbounded.loc[tech]
                rhs *= mask

                # combine
                constraints.append(lhs <= rhs)

        ### return
        return self.constraints.return_contraints(constraints, model=self.model, stack_dim_name="diffusion_limit_dim")

    def constraint_technology_diffusion_limit_total_block(self):
        """limited technology diffusion based on the existing capacity in the previous year for the entire energy system

        .. math:: #TODO
                \\Delta S_{j,e,y}\\leq ((1+\\vartheta_j)^{\\Delta^\mathrm{y}}-1)K_{j,e,y}
                +\\Delta^\mathrm{y}(\\xi\\sum_{\\tilde{j}\\in\\tilde{\mathcal{J}}}S_{\\tilde{j},e,y} + \\zeta_j)

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over technologies, capacity types and time steps, to accurately capture the conditions in the constraint
        # we vectorize over locations
        constraints = []
        for tech, time in index.get_unique(["set_technologies", "set_time_steps_yearly"]):
            # skip if max diffusion rate = inf
            if self.parameters.max_diffusion_rate.loc[tech, time] != np.inf:
                ### auxiliary calculations
                # mask for the capacity types that are not considered
                capacity_types = index.get_values([tech, slice(None), slice(None), time], "set_capacity_types", unique=True)
                mask = xr.DataArray(np.nan, coords=[self.variables.coords["set_capacity_types"]], dims=["set_capacity_types"])
                mask.loc[capacity_types] = 1

                interval_between_years = self.system["interval_between_years"]
                knowledge_depreciation_rate = self.system["knowledge_depreciation_rate"]

                reference_carrier = self.sets["set_reference_carriers"][tech][0]
                if tech in self.sets["set_transport_technologies"]:
                    set_locations = self.sets["set_edges"]
                    set_technology = self.sets["set_transport_technologies"]
                else:
                    set_locations = self.sets["set_nodes"]
                    if tech in self.sets["set_conversion_technologies"]:
                        set_technology = self.sets["set_conversion_technologies"]
                    else:
                        set_technology = self.sets["set_storage_technologies"]

                # add capacity addition of entire previous horizon
                end_time = time - 1

                # actual years between first invest time step and end_time
                delta_time = interval_between_years * (end_time - self.sets["set_time_steps_yearly"][0])
                existing_time = self.sets["set_technologies_existing"][tech]
                term_total_capacity_knowledge_existing = (self.parameters.capacity_existing.loc[tech, :, set_locations, existing_time]
                                                     * (1 - knowledge_depreciation_rate) ** (delta_time + self.parameters.lifetime.loc[tech].item() - self.parameters.lifetime_existing.loc[tech, set_locations, existing_time])).sum(["set_technologies_existing","set_location"])
                # round to avoid numerical errors
                # if self.optimization_setup.solver["round_parameters"]:
                #     rounding_value = 10 ** (-self.optimization_setup.solver["rounding_decimal_points_capacity"])
                #     term_total_capacity_knowledge_existing = term_total_capacity_knowledge_existing.where(term_total_capacity_knowledge_existing > rounding_value, 0)

                horizon_time = np.arange(self.sets["set_time_steps_yearly"][0], end_time + 1)
                horizon_time = self.variables.coords["set_time_steps_yearly"][self.variables.coords["set_time_steps_yearly"].isin(horizon_time)]
                if len(horizon_time) >= 1:
                    term_total_capacity_knowledge_addition = (self.variables["capacity_addition"].loc[tech, :, set_locations, horizon_time]
                        * (1 - knowledge_depreciation_rate) ** (interval_between_years * (end_time - horizon_time))).sum(["set_time_steps_yearly","set_location"])
                else:
                    # dummy term
                    term_total_capacity_knowledge_addition = self.variables["capacity_investment"].loc[tech, :, set_locations, time].where(False).sum("set_location")

                # total capacity in previous year; if time is first time step of interval, use existing capacities of present year
                other_techs = [other_tech for other_tech in set_technology if self.sets["set_reference_carriers"][other_tech][0] == reference_carrier]
                if time != self.optimization_setup.energy_system.set_time_steps_yearly[0]:
                    term_total_capacity_all_techs_var = self.variables["capacity"].loc[other_techs, :,set_locations, time - 1].sum(["set_technologies","set_location"])
                    term_total_capacity_all_techs_param = xr.zeros_like(self.parameters.existing_capacities.loc[tech,:,set_locations,time]).sum("set_location")
                else:
                    term_total_capacity_all_techs_param = self.parameters.existing_capacities.loc[other_techs, :, set_locations, time].sum(["set_technologies","set_location"])
                    term_total_capacity_all_techs_var = self.variables["capacity"].loc[tech, :, set_locations,time].where(False).sum("set_location")

                ### formulate constraint
                # build the lhs
                lhs = (self.variables["capacity_investment"].loc[tech, :, set_locations, time].sum("set_location")
                       - ((1 + self.parameters.max_diffusion_rate.loc[tech, time].item()) ** interval_between_years - 1) * term_total_capacity_knowledge_addition
                       - self.parameters.market_share_unbounded * term_total_capacity_all_techs_var)
                lhs *= mask

                # build the rhs
                rhs = (((1 + self.parameters.max_diffusion_rate.loc[tech, time].item()) ** interval_between_years - 1) * term_total_capacity_knowledge_existing
                       # add initial market share until which the diffusion rate is unbounded
                       + self.parameters.market_share_unbounded * term_total_capacity_all_techs_param + self.parameters.capacity_addition_unbounded.loc[tech]*len(set_locations))
                rhs *= mask

                constraints.append(lhs <= rhs)

        ### return
        # reording takes too much memory!
        return self.constraints.return_contraints(constraints, model=self.model, stack_dim_name="diffusion_limit_total_dim")

    def constraint_capex_yearly_block(self):
        """ aggregates the capex of built capacity and of existing capacity

        .. math::
            A_{h,p,y} = f_h (\\sum_{\\tilde{y} = \\max(y_0,y-\\lceil\\frac{l_h}{\\Delta^\mathrm{y}}\\rceil+1)}^y \\alpha_{h,y}\\Delta S_{h,p,\\tilde{y}}
            + \\sum_{\\hat{y}=\\psi(\\min(y_0-1,y-\\lceil\\frac{l_h}{\\Delta^\mathrm{y}}\\rceil+1))}^{\\psi(y_0)} \\alpha_{h,y_0}\\Delta s^\mathrm{ex}_{h,p,\\hat{y}})

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies and yearly time steps because we need to calculate the lifetime range
        # we vectorize over capacities and locations
        constraints = []
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):

            ### auxiliary calculations
            discount_rate = self.parameters.discount_rate
            lifetime = self.parameters.lifetime.loc[tech].item()
            if discount_rate != 0:
                annuity = ((1+discount_rate)**lifetime * discount_rate)/((1+discount_rate)**lifetime - 1)
            else:
                annuity = 1/lifetime
            term_neg_annuity_cost_capex_previous = []
            for previous_year in Technology.get_lifetime_range(self.optimization_setup, tech, year):
                term_neg_annuity_cost_capex_previous.append(-annuity * self.variables["cost_capex"].loc[tech, :, :, previous_year])

            ### formulate constraint
            lhs = lp_sum([1.0 * self.variables["capex_yearly"].loc[tech, :, :, year],
                          *term_neg_annuity_cost_capex_previous])
            rhs = annuity * self.parameters.existing_capex.loc[tech, :, :, year]
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

    def constraint_opex_technology_block(self):
        """ calculate opex of each technology

        .. math::
            \mathrm{if\ tech\ is\ conversion\ tech}\ OPEX_{h,p,t}^\mathrm{cost} = \\beta_{h,p,t} G_{i,n,t,y}^\mathrm{r}
        .. math::
            \mathrm{if\ tech\ is\ transport\ tech}\ OPEX_{h,p,t}^\mathrm{cost} = \\beta_{h,p,t} F_{j,e,t}
        .. math::
            \mathrm{if\ tech\ is\ storage\ tech}\ OPEX_{h,p,t}^\mathrm{cost} = \\beta_{h,p,t} (\\underline{H}_{k,n,t} + \\overline{H}_{k,n,t})

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_location", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies because of the reference carrier and flow which depend on the technology
        # we vectorize over locations and time steps
        constraints = []
        for tech in index.get_unique(["set_technologies"]):
            locs = index.get_values([tech], "set_location", unique=True)
            reference_carrier = self.sets["set_reference_carriers"][tech][0]
            if tech in self.sets["set_conversion_technologies"]:
                if reference_carrier in self.sets["set_input_carriers"][tech]:
                    reference_flow = self.variables["flow_conversion_input"].loc[tech, reference_carrier, locs].to_linexpr()
                    reference_flow = reference_flow.rename({"set_nodes": "set_location"})
                else:
                    reference_flow = self.variables["flow_conversion_output"].loc[tech, reference_carrier, locs].to_linexpr()
                    reference_flow = reference_flow.rename({"set_nodes": "set_location"})
            elif tech in self.sets["set_transport_technologies"]:
                reference_flow = self.variables["flow_transport"].loc[tech, locs].to_linexpr()
                reference_flow = reference_flow.rename({"set_edges": "set_location"})
            else:
                reference_flow = self.variables["flow_storage_charge"].loc[tech, locs] + self.variables["flow_storage_discharge"].loc[tech, locs]
                reference_flow = reference_flow.rename({"set_nodes": "set_location"})

            term_reference_flow = - self.parameters.opex_specific_variable.loc[tech, locs] * reference_flow

            ### formulate constraint
            # the first term is just to ensure full shape
            lhs = lp.merge(self.variables["cost_opex"].loc[tech].where(False).to_linexpr(),
                           self.variables["cost_opex"].loc[tech, locs].to_linexpr(),
                           term_reference_flow,
                           compat="broadcast_equals")
            rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies"]),
                                                  index_names=["set_technologies"])

    def constraint_opex_yearly_block(self):
        """ yearly opex for a technology at a location in each year

        .. math::
            OPEX_{h,p,y} = \sum_{t\in\mathcal{T}}\tau_t OPEX_{h,p,t}^\mathrm{cost}
            + \gamma_{h,y} S_{h,p,y}
            #TODO complete constraint (second summation symbol)

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_location", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies and yearly time steps because some auxillary calculations depend on the technology
        # we vectorize over locations
        constraints = []
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):

            ### auxiliary calculations
            times = self.time_steps.get_time_steps_year2operation(year)

            term_neg_summed_cost_opex = - (self.variables["cost_opex"].loc[tech, :, times] * self.parameters.time_steps_operation_duration.loc[times]).sum(["set_time_steps_operation"])
            term_neg_summed_capacities = - lp_sum([self.parameters.opex_specific_fixed.loc[tech, capacity_type, :, year]*self.variables["capacity"].loc[tech, capacity_type, :, year]
                                                   for capacity_type in self.system["set_capacity_types"] if tech in self.sets["set_storage_technologies"] or capacity_type == self.system["set_capacity_types"][0]])

            ### formulate constraint
            lhs = lp_sum([1.0 * self.variables["opex_yearly"].loc[tech, :, year],
                          term_neg_summed_cost_opex,
                          term_neg_summed_capacities])
            rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

    def constraint_carbon_emissions_technology_block(self):
        """ calculate carbon emissions of each technology

        .. math::
            \mathrm{if\ tech\ is\ conversion\ tech}\ E_{h,p,t} = \\epsilon_h G_{i,n,t,y}^\mathrm{r}
        .. math::
            \mathrm{if\ tech\ is\ transport\ tech}\ E_{h,p,t} = \\epsilon_h F_{j,e,t}
        .. math::
            \mathrm{if\ tech\ is\ storage\ tech}\ E_{h,p,t} = \\epsilon_h (\\underline{H}_{k,n,t} + \\overline{H}_{k,n,t})

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_location", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies because of the reference carrier and flow which depend on the technology
        # we vectorize over locations and time steps
        constraints = []
        for tech in index.get_unique(["set_technologies"]):

            ### auxiliary calculations
            locs = index.get_values([tech], 1, unique=True)
            reference_carrier = self.sets["set_reference_carriers"][tech][0]
            if tech in self.sets["set_conversion_technologies"]:
                if reference_carrier in self.sets["set_input_carriers"][tech]:
                    reference_flow = self.variables["flow_conversion_input"].loc[tech, reference_carrier, locs].to_linexpr()
                    reference_flow = reference_flow.rename({"set_nodes": "set_location"})
                else:
                    reference_flow = self.variables["flow_conversion_output"].loc[tech, reference_carrier, locs].to_linexpr()
                    reference_flow = reference_flow.rename({"set_nodes": "set_location"})
            elif tech in self.sets["set_transport_technologies"]:
                reference_flow = self.variables["flow_transport"].loc[tech, locs].to_linexpr()
                reference_flow = reference_flow.rename({"set_edges": "set_location"})
            else:
                reference_flow = self.variables["flow_storage_charge"].loc[tech, locs] + self.variables["flow_storage_discharge"].loc[tech, locs]
                reference_flow = reference_flow.rename({"set_nodes": "set_location"})

            term_reference_flow = - self.parameters.carbon_intensity_technology.loc[tech, locs] * reference_flow

            ### formulate constraint
            # the first term is just to ensure full shape
            lhs = lp.merge(self.variables["carbon_emissions_technology"].loc[tech].where(False).to_linexpr(),
                           self.variables["carbon_emissions_technology"].loc[tech, locs].to_linexpr(),
                           term_reference_flow,
                           compat="broadcast_equals")
            rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies"]),
                                                  index_names=["set_technologies"])

    def constraint_carbon_emissions_technology_total_block(self):
        """ calculate total carbon emissions of each technology

        .. math::
            E_y^{\mathcal{H}} = \sum_{t\in\mathcal{T}}\sum_{h\in\mathcal{H}} E_{h,p,t} \\tau_{t}

        :return: linopy constraints
        """

        ### index sets
        years = self.sets["set_time_steps_yearly"]
        # this index is just for the sums in the auxiliary calculations
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_location"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we cycle over the years, because the sum of the operational time steps depends on the year
        constraints = []
        for year in years:

            ### auxiliary calculations
            term_summed_carbon_emissions_technology = []
            for tech in index.get_unique(["set_technologies"]):
                locs = index.get_values([tech], "set_location", unique=True)
                times = self.time_steps.get_time_steps_year2operation(year)
                term_summed_carbon_emissions_technology.append((self.variables["carbon_emissions_technology"].loc[tech, locs, times] * self.parameters.time_steps_operation_duration.loc[times]).sum())
            term_summed_carbon_emissions_technology = lp_sum(term_summed_carbon_emissions_technology)

            ### formulate constraint
            lhs = self.variables["carbon_emissions_technology_total"].loc[year] - term_summed_carbon_emissions_technology
            rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=years,
                                                  index_names=["set_time_steps_yearly"])

    def constraint_capacity_factor_block(self):
        """ Load is limited by the installed capacity and the maximum load factor

        .. math::
            \mathrm{if\ tech\ is\ conversion\ tech}\ G_{i,n,t,y}^\mathrm{r} \\leq m_{i,n,t,y}S_{i,n,y}
        .. math::
            \mathrm{if\ tech\ is\ transport\ tech}\ F_{j,e,t,y}^\mathrm{r} \\leq m_{j,e,t,y}S_{j,e,y}
        .. math::
            \mathrm{if\ tech\ is\ storage\ tech}\ \\underline{H}_{k,n,t,y}+\\overline{H}_{k,n,t,y}\\leq m_{k,n,t,y}S_{k,n,y}

        :return: linopy constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_operation"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we oop over all technologies for the conditions and vectorize over the rest
        constraints = []
        for tech in index.get_unique(["set_technologies"]):

            ### auxiliary calculations
            capacity_types, locs, times = index.get_values([tech], [1, 2, 3], unique=True)
            # to actual coords to avoid renaming
            capacity_types = self.variables.coords["set_capacity_types"].loc[capacity_types]
            locs = self.variables.coords["set_location"].loc[locs]
            times = self.variables.coords["set_time_steps_operation"].loc[times]
            # the reference carrier
            reference_carrier = self.sets["set_reference_carriers"][tech][0]
            # get invest time step
            time_step_year = xr.DataArray([self.optimization_setup.energy_system.time_steps.convert_time_step_operation2year(t) for t in times.data], coords=[times])
            # we create the capacity term (the dimension reassignment does not change the variables, just the broadcasting)
            term_capacity = self.parameters.max_load.loc[tech, capacity_types, locs, times] * self.variables["capacity"].loc[tech, capacity_types, locs, time_step_year].to_linexpr()

            # this term is just to ensure full shape
            full_shape_term = self.variables["capacity"].loc[tech, ..., time_step_year].where(False).to_linexpr()

            # conversion technology
            if tech in self.sets["set_conversion_technologies"]:
                if reference_carrier in self.sets["set_input_carriers"][tech]:
                    term_flow = -1.0 * self.variables["flow_conversion_input"].loc[tech, reference_carrier, locs, times]
                else:
                    term_flow = -1.0 * self.variables["flow_conversion_output"].loc[tech, reference_carrier, locs, times]
            # transport technology
            elif tech in self.sets["set_transport_technologies"]:
                term_flow = -1.0 * self.variables["flow_transport"].loc[tech, locs, times]
            # storage technology
            elif tech in self.sets["set_storage_technologies"]:
                system = self.optimization_setup.system
                # if limit power
                mask = (capacity_types == system["set_capacity_types"][0]).astype(float)
                # where true
                term_flow = mask*(-1.0 * self.variables["flow_storage_charge"].loc[tech, locs, times] - 1.0 * self.variables["flow_storage_discharge"].loc[tech, locs, times])

                # TODO integrate level storage here as well

            ### formulate constraint
            lhs = lp.merge(lp.merge(term_capacity, term_flow), full_shape_term)
            rhs = 0
            constraints.append(lhs >= rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(["set_technologies"]),
                                                  index_names=["set_technologies"])

    # anyaxie
    def constraint_approximate_total_global_cost_block(self):
        """Approximate total cumulative cost for each technology.

        .. math::
            \\tilde{TC}_{h,y} = \\sum_{w\\in \\mathcal{W}} \\kappa_{h,w} \\cdot Z_{h,y,w} + \\sigma_{h,w} \\cdot
            S^{\\mathrm{seg}}_{h,y,w}

        :return: List of constraints
        """
        ### index sets
        index_values, index_names = Element.create_custom_set(
            ["set_technologies", "set_capacity_types", "set_time_steps_yearly", "set_total_cost_pwa_segments"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies and timesteps for the conditions and vectorize over the rest

        # Initialize an empty list to store the constraints
        constraints = []
        # Iterate over technologies
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):
            # Get the unique segments for the current technology
            segments = index.get_unique(["set_total_cost_pwa_segments"])

            # todo: what if only one segment?
            # Calculate the linear combination for Z and S using pwa parameters intersect and slope
            term_Z = sum(self.parameters.total_cost_pwa_intersect.loc[tech, :, segment]
                         * self.variables['total_cost_pwa_segment_selection'].loc[tech, :, year, segment]
                         for segment in segments)

            term_X = sum(self.parameters.total_cost_pwa_slope.loc[tech, :, segment]
                * self.variables['total_cost_pwa_cum_capacity_segment_position'].loc[tech, :, year, segment]
                for segment in segments)

            # Formulate the constraint
            lhs = (self.variables['total_cost_pwa_global_cost'].loc[tech, :, year]
                   - term_Z
                   - term_X)
            rhs = 0

            # Append the constraint to the list
            constraints.append(lhs == rhs)

        # Return the list of constraints
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

    def constraint_pwa_total_cost_segment_selection_block(self):
        """Ensure that for each technology and each year, the sum over segments of Z equals 1.

        . math::
            \sum_{w\in\mathcal{W}}&Z_{h,y,w} = 1

        :return: List of constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types",
                                       "set_time_steps_yearly", "set_total_cost_pwa_segments"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # Initialize an empty list to store the constraints
        constraints = []

        # Iterate over technologies
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):
            # Get the unique segments for the current technology
            segments = index.get_unique(["set_total_cost_pwa_segments"])

            # Sum up all binary variables for the segment selection for each timestep and each technology
            sum_segments_z = sum(self.variables['total_cost_pwa_segment_selection']
                                 .loc[tech, :, year, segment] for segment in segments)

            # Formulate the constraint
            lhs = sum_segments_z
            rhs = 1

            # Append the constraint to the list
            constraints.append(lhs == rhs)

        # Return the list of constraints
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly","set_total_cost_pwa_segments"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly", "set_total_cost_pwa_segments"])


    def constraint_pwa_total_cost_cum_capacity_upper_bound_block(self):
        """Ensure that for each technology and each year, the segment capacity is within the interpolation points.

        . math::
            S^{\mathrm{seg}}_{h,y,w} \leq \overline{s}_{h,w}^{\mathrm{glo}}\cdot Z_{h,y,w}

        :return: List of constraints
        """
        ### index sets
        index_values, index_names = Element.create_custom_set(
            ["set_technologies","set_capacity_types", "set_time_steps_yearly", "set_total_cost_pwa_segments"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary


        ### index loop
        # Initialize an empty list to store the constraints
        constraints = []
        for tech, year, segment in index.get_unique(["set_technologies",
                                                                    "set_time_steps_yearly", "set_total_cost_pwa_segments"]):

            lhs = (self.variables['total_cost_pwa_cum_capacity_segment_position'].loc[tech, :, year, segment]
                - self.parameters.total_cost_pwa_points_upper_bound.loc[tech, :, segment]
                   *self.variables['total_cost_pwa_segment_selection'].loc[tech, :, year, segment])
            rhs = 0

            # Append the constraint to the list
            constraints.append(lhs <= rhs)

        # Return the list of constraints
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly", "set_total_cost_pwa_segments"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly", "set_total_cost_pwa_segments"])
    def constraint_pwa_total_cost_cum_capacity_lower_bound_block(self):
        """Ensure that for each technology and each year, the segment capacity is within the interpolation points.

        . math::
            {s}_{h,w}^{\mathrm{glo}}\cdot Z_{h,y,w} \leq S^{\mathrm{seg}}_{h,y,w}

        :return: List of constraints
        """
        ### index sets
        index_values, index_names = Element.create_custom_set(
            ["set_technologies", "set_capacity_types", "set_time_steps_yearly", "set_total_cost_pwa_segments"],
            self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # Initialize an empty list to store the constraints
        constraints = []
        for tech, year, segment in index.get_unique(["set_technologies", "set_time_steps_yearly",
                                                                    "set_total_cost_pwa_segments"]):
            lhs = (self.parameters.total_cost_pwa_points_lower_bound.loc[tech, :, segment]
                   * self.variables['total_cost_pwa_segment_selection'].loc[tech, :, year, segment]
                   - self.variables['total_cost_pwa_cum_capacity_segment_position'].loc[tech, :, year, segment])
            rhs = 0

            # Append the constraint to the list
            constraints.append(lhs <= rhs)

        # Return the list of constraints
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly", "set_total_cost_pwa_segments"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly", "set_total_cost_pwa_segments"])

    def constraint_pwa_total_cost_global_cum_capacity_segment_block(self):
        """Ensures that the sum of all segment capacities is the capacity installed in that year. Needed for PWA

        . math::
            \sum_{w\in\mathcal{W}}S^{\mathrm{seg}}_{h,y,w}=S_{h,y}^{\mathrm{glo}}

        :return: List of constraints
        """
        ### index sets
        index_values, index_names = Element.create_custom_set(
            ["set_technologies", "set_capacity_types", "set_time_steps_yearly", "set_total_cost_pwa_segments"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        constraints = []
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):

            # Get the unique segments for the current technology
            segments = self.sets["set_total_cost_pwa_segments"][tech]

            # Iterate over the segments
            lhs = (self.variables['global_cumulative_capacity'].loc[tech, :, year]
                   - sum(self.variables['total_cost_pwa_cum_capacity_segment_position'].loc[tech, :, year, segment] for segment in segments))
            rhs = 0

            constraints.append(lhs == rhs)

        # Return the list of constraints
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

    def constraint_global_cum_capacity_block(self):
        """Calculates the cumulative global capacity for each technology in each year

               . math::
                   S_{h,y}^{glo} = \sum_{\tilde{y}=y_0}^y\sum_{p\in\mathcal{P}}\Delta S_{h,p,\tilde{y}}
                   +  s_{h,\hat{y}}^{\mathrm{row}}
                    + s_{\mathrm{initial}}^{\mathrm{glo}}



       :return: List of constraints
       """

        ### index sets
        index_values, index_names = Element.create_custom_set(
            ["set_technologies", "set_location", "set_capacity_types", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # this mask is just to make sure we only get constraints where we want them, no if-condition
        mask = self.variables["capacity"].mask

        ### index loop
        # we loop over technologies and time steps, because we need to cycle over the lifetime range of the technology
        # which requires the technology and the year, we vectorize over capacity types and locations
        constraints = []
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):

            ### auxiliary calculations
            global_share_factor = self.parameters.global_share_factor.loc[tech].item()
            term_neg_previous_capacity_additions = []

            # Sum over all previous capacity additions
            if self.system["global_active_capacity"]:
                # Case for active cumulative technologies with decomissioning
                for previous_year in Technology.get_lifetime_range(self.optimization_setup, tech, year):
                    if self.system["use_exogenous_cap_add_row"]:
                        term_neg_previous_capacity_additions.append((-1.0) * self.variables["capacity_addition"].loc[tech, :, :, previous_year].sum(dims="set_location"))
                        term_global_capacities = (self.parameters.global_initial_capacity.loc[tech]
                        + self.parameters.cum_capacity_row.loc[tech, year])
                    else:
                        term_neg_previous_capacity_additions.append((-1/global_share_factor)*self.variables["capacity_addition"].loc[tech, :, :,previous_year].sum(dims="set_location"))
                        term_global_capacities = self.parameters.global_initial_capacity.loc[tech]
            else:
                # No decommissioning
                year_range = year - index.get_unique(["set_time_steps_yearly"])[0] + 1
                for previous_year in self.sets["set_time_steps_yearly"][:year_range]:
                    if self.system["use_exogenous_cap_add_row"]:
                        term_neg_previous_capacity_additions.append((-1.0) * self.variables["capacity_addition"].loc[tech, :, :, previous_year].sum(dims="set_location"))
                        term_global_capacities = self.parameters.global_initial_capacity.loc[tech] + self.parameters.cum_capacity_row.loc[tech, year]
                    else:
                        term_neg_previous_capacity_additions.append((-1/global_share_factor)*self.variables["capacity_addition"].loc[tech, :, :,previous_year].sum(dims="set_location"))
                        term_global_capacities = self.parameters.global_initial_capacity.loc[tech]

            # Case for
            ### formulate constraint
            lhs = lp_sum([1.0 * self.variables["global_cumulative_capacity"].loc[tech, :, year],
                          *term_neg_previous_capacity_additions])
            rhs = term_global_capacities
            constraints.append(lhs == rhs)

        # Return the list of constraints
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

    def constraint_capex_yearly_all_positions_block(self):
        """ aggregates the capex of built capacity and of existing capacity

        .. math::
            A_{h,y} = f_h \left(\sum_{\tilde{y}=\max\left(y_0,y-\left\lceil\nicefrac{l_h}{\Delta^\mathrm{y}}\right
            \rceil+1\right)}^y g_h \left( TC_{h,y} - TC_{h,y-1} \right)\right.\nonumber+ \left.\sum_{\hat{y}=\psi\left(
            y-\left\lceil\nicefrac{l_h}{\Delta^\mathrm{y}}\right\rceil+1\right)}^{\psi(y_0-1)}
             \sum_{p\in\mathcal{P}}\alpha_{h,y_0}\Delta s^\mathrm{ex}_{h,p,\hat{y}} \right)

        :return:
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(
            ["set_technologies", "set_capacity_types", "set_time_steps_yearly"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies and yearly time steps because we need to calculate the lifetime range
        # we vectorize over capacities and locations
        constraints = []
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):

            ### auxiliary calculations
            discount_rate = self.parameters.discount_rate
            lifetime = self.parameters.lifetime.loc[tech].item()
            if discount_rate != 0:
                annuity = ((1 + discount_rate) ** lifetime * discount_rate) / ((1 + discount_rate) ** lifetime - 1)
            else:
                annuity = 1 / lifetime
            term_neg_annuity_cost_capex_previous = []
            for previous_year in Technology.get_lifetime_range(self.optimization_setup, tech, year):
                term_neg_annuity_cost_capex_previous.append(-annuity * self.variables["cost_capex"].loc[tech, :, previous_year])

            ### formulate constraint
            lhs = lp_sum([1.0 * self.variables["capex_yearly_all_positions"].loc[tech, :, year],
                          *term_neg_annuity_cost_capex_previous])
            rhs = annuity * self.parameters.existing_capex.loc[tech, :, :, year].sum(dim=["set_location"])
            constraints.append(lhs == rhs)

            ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])


    def constraint_cost_capex_block(self):
        """ calculates the capex of each technology

        .. math::
            Cost_Capex{h,y} = TC{h,y} - TC{h,y-1}

        :return:
        """
        ### index sets
        index_values, index_names = Element.create_custom_set(
            ["set_technologies", "set_capacity_types", "set_time_steps_yearly"],
            self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies because we need to get the global share factor
        # we vectorize over capacity types and years
        constraints = []
        if self.system["use_exogenous_cap_add_row"]:
            for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):
                if year == index.get_unique(["set_time_steps_yearly"])[0]:  # if first year of model horizon
                    lhs = (1.0 * self.variables["cost_capex_all_positions"].loc[tech, :, year]
                           - self.variables["total_cost_pwa_european_cost"].loc[tech,:,year])
                    rhs = (-1.0) * self.parameters.total_cost_pwa_initial_global_cost.loc[tech, :]
                else:
                    lhs = (1.0 * self.variables["cost_capex_all_positions"].loc[tech, :, year]
                           - self.variables["total_cost_pwa_european_cost"].loc[tech,:,year]
                           + self.variables["total_cost_pwa_global_cost"].loc[tech, :, year - 1])
                    rhs = 0

                constraints.append(lhs == rhs)
        else:
            for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):
                global_share_factor = self.parameters.global_share_factor.loc[tech].item()
                if year == index.get_unique(["set_time_steps_yearly"])[0]:  # if first year of model horizon
                    lhs = (1.0 * self.variables["cost_capex_all_positions"].loc[tech, :, year]
                           - global_share_factor * self.variables["total_cost_pwa_global_cost"].loc[tech, :, year])
                    rhs = (-global_share_factor) * self.parameters.total_cost_pwa_initial_global_cost.loc[tech, :]

                else:
                    lhs = (1.0 * self.variables["cost_capex_all_positions"].loc[tech, :, year]
                           - global_share_factor * (self.variables["total_cost_pwa_global_cost"].loc[tech, :, year]
                                    - self.variables["total_cost_pwa_global_cost"].loc[tech, :, year - 1]))
                    rhs = 0

                constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

    def constraint_split_capex_across_all_positions_block(self):
        '''
        Splits the capex across all nodes equally
        '''

        ### index sets
        index_values, index_names = Element.create_custom_set(
            ["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"],
            self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies because we need to get the global share factor
        # we vectorize over capacity types and years
        constraints = []

        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):
            lhs = (1.0 * self.variables["cost_capex"].loc[tech, :, :, year]
                   - self.variables["cost_capex_all_positions"].loc[tech, :, year]/len(index.get_unique(["set_location"])))
            rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])



    def constraint_european_addition_block(self):
        """ Calculates the capacity addition under neglection of the ROW capacity addition

        .. math::
            S_{h,y}^{EU} = S_{h,y-1}^{glo} + \Delta S_{h,y}

        :return:
        """
        ### index sets
        index_values, index_names = Element.create_custom_set(
            ["set_technologies", "set_capacity_types", "set_time_steps_yearly"],
            self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies and yearly time steps because we need to calculate the lifetime range
        # we vectorize over capacities and locations
        constraints = []
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):
            if year == index.get_unique(["set_time_steps_yearly"])[0]:  # if first year of model horizon
                lhs = (self.variables["european_cumulative_capacity"].loc[tech, :, year]
                       - self.variables["capacity_addition"].loc[tech, :, :, year].sum(dims="set_location"))  # European addition
                rhs = 1.0 * self.parameters.global_initial_capacity.loc[tech]  # Global capacity
            else:
                lhs = (self.variables["european_cumulative_capacity"].loc[tech,:,year]
                       - self.variables["global_cumulative_capacity"].loc[tech, :, year-1]  # Global capacity
                       - self.variables["capacity_addition"].loc[tech, :, :, year].sum(dims="set_location")) # European addition
                rhs = 0

            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])


    def constraint_approximate_total_european_cost_block(self):
        """Approximate total cumulative cost for each technology.

        .. math::
            \\tilde{TC}_{h,y} = \\sum_{w\\in \\mathcal{W}} \\kappa_{h,w} \\cdot Z_{h,y,w} + \\sigma_{h,w} \\cdot
            S^{\\mathrm{seg}}_{h,y,w}

        :return: List of constraints
        """
        ### index sets
        index_values, index_names = Element.create_custom_set(
            ["set_technologies", "set_capacity_types", "set_time_steps_yearly", "set_total_cost_pwa_segments"], self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # we loop over all technologies and timesteps for the conditions and vectorize over the rest

        # Initialize an empty list to store the constraints
        constraints = []
        # Iterate over technologies
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):
            # Get the unique segments for the current technology
            segments = index.get_unique(["set_total_cost_pwa_segments"])

            # Calculate the linear combination for Z and S using pwa parameters intersect and slope
            term_Z = sum(self.parameters.total_cost_pwa_intersect.loc[tech, :, segment]
                         * self.variables['total_cost_pwa_segment_selection_eu'].loc[tech, :, year, segment]
                         for segment in segments)

            term_X = sum(self.parameters.total_cost_pwa_slope.loc[tech, :, segment]
                * self.variables['total_cost_pwa_cum_capacity_segment_position_eu'].loc[tech, :, year, segment]
                for segment in segments)

            # Formulate the constraint
            lhs = (self.variables['total_cost_pwa_european_cost'].loc[tech, :, year]
                   - term_Z
                   - term_X)
            rhs = 0

            # Append the constraint to the list
            constraints.append(lhs == rhs)

        # Return the list of constraints
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

    def constraint_pwa_total_cost_segment_selection_eu_block(self):
        """Ensure that for each technology and each year, the sum over segments of Z equals 1.

        . math::
            \sum_{w\in\mathcal{W}}&Z_{h,y,w} = 1

        :return: List of constraints
        """

        ### index sets
        index_values, index_names = Element.create_custom_set(["set_technologies", "set_capacity_types",
                                                               "set_time_steps_yearly", "set_total_cost_pwa_segments"],
                                                              self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # Initialize an empty list to store the constraints
        constraints = []

        # Iterate over technologies
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):
            # Get the unique segments for the current technology
            segments = index.get_unique(["set_total_cost_pwa_segments"])

            # Sum up all binary variables for the segment selection for each timestep and each technology
            sum_segments_z = sum(self.variables['total_cost_pwa_segment_selection_eu']
                                 .loc[tech, :, year, segment] for segment in segments)

            # Formulate the constraint
            lhs = sum_segments_z
            rhs = 1

            # Append the constraint to the list
            constraints.append(lhs == rhs)

        # Return the list of constraints
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly",
                                                       "set_total_cost_pwa_segments"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly",
                                                               "set_total_cost_pwa_segments"])

    def constraint_pwa_total_cost_cum_capacity_upper_bound_eu_block(self):
        """Ensure that for each technology and each year, the segment capacity is within the interpolation points.

        . math::
            S^{\mathrm{seg}}_{h,y,w} \leq \overline{s}_{h,w}^{\mathrm{glo}}\cdot Z_{h,y,w}

        :return: List of constraints
        """
        ### index sets
        index_values, index_names = Element.create_custom_set(
            ["set_technologies", "set_capacity_types", "set_time_steps_yearly", "set_total_cost_pwa_segments"],
            self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # Initialize an empty list to store the constraints
        constraints = []
        for tech, year, segment in index.get_unique(["set_technologies",
                                                     "set_time_steps_yearly", "set_total_cost_pwa_segments"]):
            lhs = (self.variables['total_cost_pwa_cum_capacity_segment_position_eu'].loc[tech, :, year, segment]
                   - self.parameters.total_cost_pwa_points_upper_bound.loc[tech, :, segment]
                   * self.variables['total_cost_pwa_segment_selection_eu'].loc[tech, :, year, segment])
            rhs = 0

            # Append the constraint to the list
            constraints.append(lhs <= rhs)

        # Return the list of constraints
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly",
                                                       "set_total_cost_pwa_segments"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly",
                                                               "set_total_cost_pwa_segments"])

    def constraint_pwa_total_cost_cum_capacity_lower_bound_eu_block(self):
        """Ensure that for each technology and each year, the segment capacity is within the interpolation points.

        . math::
            {s}_{h,w}^{\mathrm{glo}}\cdot Z_{h,y,w} \leq S^{\mathrm{seg}}_{h,y,w}

        :return: List of constraints
        """
        ### index sets
        index_values, index_names = Element.create_custom_set(
            ["set_technologies", "set_capacity_types", "set_time_steps_yearly", "set_total_cost_pwa_segments"],
            self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        # Initialize an empty list to store the constraints
        constraints = []
        for tech, year, segment in index.get_unique(["set_technologies", "set_time_steps_yearly",
                                                     "set_total_cost_pwa_segments"]):
            lhs = (self.parameters.total_cost_pwa_points_lower_bound.loc[tech, :, segment]
                   * self.variables['total_cost_pwa_segment_selection_eu'].loc[tech, :, year, segment]
                   - self.variables['total_cost_pwa_cum_capacity_segment_position_eu'].loc[tech, :, year, segment])
            rhs = 0

            # Append the constraint to the list
            constraints.append(lhs <= rhs)

        # Return the list of constraints
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly",
                                                       "set_total_cost_pwa_segments"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly",
                                                               "set_total_cost_pwa_segments"])

    def constraint_pwa_total_cost_european_cum_capacity_segment_block(self):
        """Ensures that the sum of all segment capacities is the capacity installed in that year. Needed for PWA

        . math::
            \sum_{w\in\mathcal{W}}S^{\mathrm{seg}}_{h,y,w}=S_{h,y}^{\mathrm{glo}}

        :return: List of constraints
        """
        ### index sets
        index_values, index_names = Element.create_custom_set(
            ["set_technologies", "set_capacity_types", "set_time_steps_yearly", "set_total_cost_pwa_segments"],
            self.optimization_setup)
        index = ZenIndex(index_values, index_names)

        ### masks
        # not necessary

        ### index loop
        constraints = []
        for tech, year in index.get_unique(["set_technologies", "set_time_steps_yearly"]):
            # Get the unique segments for the current technology
            segments = self.sets["set_total_cost_pwa_segments"][tech]

            # Iterate over the segments
            lhs = (self.variables['european_cumulative_capacity'].loc[tech, :, year]
                   - sum(self.variables['total_cost_pwa_cum_capacity_segment_position_eu'].loc[tech, :, year, segment] for
                         segment in segments))
            rhs = 0

            constraints.append(lhs == rhs)

        # Return the list of constraints
        return self.constraints.return_contraints(constraints,
                                                  model=self.model,
                                                  index_values=index.get_unique(
                                                      ["set_technologies", "set_time_steps_yearly"]),
                                                  index_names=["set_technologies", "set_time_steps_yearly"])

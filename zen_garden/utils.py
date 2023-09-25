"""
:Title:        ZEN-GARDEN
:Created:      October-2021
:Authors:      Janis Fluri (janis.fluri@id.ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Class is defining to read in the results of an Optimization problem.
"""

import logging
import os
import sys
import warnings
from collections import UserDict
from contextlib import contextmanager
from datetime import datetime
from ordered_set import OrderedSet

import h5py
import linopy as lp
from linopy.config import options
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from numpy import string_
from copy import deepcopy
from collections import defaultdict


def setup_logger(log_path=None, level=logging.INFO):
    # SETUP LOGGER
    log_format = '%(asctime)s %(filename)s: %(message)s'

    if log_path is None:
        log_path = os.path.join('outputs', 'logs')
        os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_path, 'valueChain.log'), level=level, format=log_format,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.captureWarnings(True)
    # we don't want to add this multiple times
    if not any([handle.name == "STDOUT" for handle in logging.getLogger().handlers]):
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name("STDOUT")
        handler.setLevel(level)
        logging.getLogger().addHandler(handler)


def get_inheritors(klass):
    """
    Get all child classes of a given class

    :param klass: The class to get all children
    :return: All children as a set
    """

    subclasses = OrderedSet()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


# This redirects output streams to files
# --------------------------------------

class RedirectStdStreams(object):
    """
    A context manager that redirects the output to a file
    """

    def __init__(self, stdout=None, stderr=None):
        """
        Initializes the context manager

        :param stdout: Stream for stdout
        :param stderr: Stream for stderr
        """
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        """
        The exit function of the context manager

        :param exc_type: Type of the exit
        :param exc_value: Value of the exit
        :param traceback:  traceback of the error
        """
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


# This functionality is for the IIS constraints
# ---------------------------------------------

class IISConstraintParser(object):
    """
    This class is used to parse the IIS constraints and print them in a nice way
    Most functions here are just copied from linopy 0.2.x
    """

    EQUAL = "="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    SIGNS = {EQUAL, GREATER_EQUAL, LESS_EQUAL}
    SIGNS_pretty = {EQUAL: "=", GREATER_EQUAL: "≥", LESS_EQUAL: "≤"}

    def __init__(self, iis_file, model):
        self.iis_file = iis_file
        self.model = model

        self.labels = self.read_labels()

    def write_parsed_output(self, outfile=None, manual_display_max_terms=100):
        """
        Writes the parsed output to a file
        :param outfile: The file to write to
        :param manual_display_max_terms: the max number of terms that are displayed, overwriting the default
        """
        # avoid truncating the expression
        # default_config_options = lp.config.options
        # lp.config.options = lp.config.OptionSettings(display_max_terms=manual_display_max_terms)
        # write the outfile
        if outfile is None:
            fname, _ = os.path.splitext(self.iis_file)
            outfile = fname + "_linopy.ilp"
        seen_constraints = []
        with open(outfile, "w") as f:
            constraints = self.model.constraints
            for label in self.labels:
                name, coord = self.get_label_position(constraints, label)
                constraint = constraints[name]
                expr_str = self.print_single_constraint(constraint, coord)
                coords_str = self.print_coord(coord)
                cons_str = f"\t{coords_str}:\t{expr_str}\n"
                if name not in seen_constraints:
                    seen_constraints.append(name)
                    cons_str = f"\n{name}:\n{cons_str}"
                f.write(cons_str)

    def read_labels(self):
        """
        Reads the labels from the IIS file
        :return: A list of labels
        """

        labels = []
        with open(self.iis_file, "rb") as f:
            for line in f.readlines():
                line = line.decode()
                if line.startswith(" c"):
                    labels.append(int(line.split(":")[0][2:]))
        return labels

    def print_single_constraint(self, constraint, coord):
        coeffs, vars, sign, rhs = xr.broadcast(constraint.coeffs,
                                               constraint.vars,
                                               constraint.sign,
                                               constraint.rhs)
        coeffs = coeffs.sel(coord).values
        vars = vars.sel(coord).values
        sign = sign.sel(coord)[0].item()
        rhs = rhs.sel(coord)[0].item()

        expr = self.print_single_expression(coeffs, vars, self.model)
        # sign = self.SIGNS_pretty[sign]

        return f"{expr} {sign} {rhs:.12g}"

    @staticmethod
    def print_coord(coord):
        if isinstance(coord, dict):
            coord = coord.values()
        return "[" + ", ".join([str(c) for c in coord]) + "]" if len(coord) else ""

    @staticmethod
    def get_label_position(constraints, value):
        """
        Get tuple of name and coordinate for variable labels.
        """

        name = constraints.get_name_by_label(value)
        con = constraints[name]
        indices = [i[0] for i in np.where(con.values == value)]

        # Extract the coordinates from the indices
        coord = {
            dim: con.labels.indexes[dim][i] for dim, i in zip(con.labels.dims, indices)
        }

        return name, coord

    @staticmethod
    def print_single_expression(c, v, model):
        """
        This is a linopy routine but without max terms
        Print a single linear expression based on the coefficients and variables.
        """

        # catch case that to many terms would be printed
        def print_line(expr):
            res = []
            for i, (coeff, (name, coord)) in enumerate(expr):
                coord_string = IISConstraintParser.print_coord(coord)
                if i:
                    # split sign and coefficient
                    coeff_string = f"{float(coeff):+.4}"
                    res.append(f"{coeff_string[0]} {coeff_string[1:]} {name}{coord_string}")
                else:
                    res.append(f"{float(coeff):.4} {name}{coord_string}")
            return " ".join(res) if len(res) else "None"

        if isinstance(c, np.ndarray):
            mask = v != -1
            c, v = c[mask], v[mask]

        expr = list(zip(c, model.variables.get_label_position(v)))
        return print_line(expr)


# This class is for the scenario analysis
# ---------------------------------------

class ScenarioDict(dict):
    """
    This is a dictionary for the scenario analysis that has some convenience functions
    """

    _param_dict_keys = {"file", "file_op", "default", "default_op"}
    _special_elements = ["system", "analysis", "base_scenario", "sub_folder", "param_map"]

    def __init__(self, init_dict, system, analysis):
        """
        Initializes the dictionary from a normal dictionary
        :param init_dict: The dictionary to initialize from
        :param system: The system to which the dictionary belongs
        """

        # avoid circular imports
        from . import inheritors
        self.element_classes = reversed(inheritors.copy())

        # set the attributes and expand the dict
        self.system = system
        self.analysis = analysis
        self.init_dict = init_dict
        expanded_dict = self.expand_subsets(init_dict)
        self.validate_dict(expanded_dict)
        self.dict = expanded_dict

        # super init
        super().__init__(self.dict)

        # finally we update the analysis and system
        self.update_analysis_system()

    def update_analysis_system(self):
        """
        Updates the analysis and system
        """

        if "analysis" in self.dict:
            for key, value in self.dict["analysis"].items():
                if type(self.analysis[key]) == type(value):
                    self.analysis[key] = value
                else:
                    raise ValueError(
                        f"Trying to update analysis with key {key} and value {value} of type {type(value)}, "
                        f"but the analysis has already a value of type {type(self.analysis[key])}")
        if "system" in self.dict:
            for key, value in self.dict["system"].items():
                if type(self.system[key]) == type(value):
                    self.system[key] = value
                else:
                    raise ValueError(f"Trying to update system with key {key} and value {value} of type {type(value)}, "
                                     f"but the system has already a value of type {type(self.system[key])}")

    @staticmethod
    def expand_lists(scenarios: dict):
        """
        Expands all lists of parameters in the all scenarios and returns a new dict
        :param scenarios: The initial dict of scenarios
        :return: The expanded dict, where all necessary parameters are expanded and subpaths are set
        """

        # Important, all for-loops through keys or items in this routine should be sorted!

        expanded_scenarios = dict()
        for scenario_name, scenario_dict in sorted(scenarios.items(), key=lambda x: x[0]):
            scenario_dict["base_scenario"] = scenario_name
            scenario_dict["sub_folder"] = ""
            scenario_dict["param_map"] = dict()
            scenario_list = ScenarioDict._expand_scenario(scenario_dict)

            # add the scenarios to the dict
            for scenario in scenario_list:
                if scenario["sub_folder"] == "":
                    name = scenario["base_scenario"]
                else:
                    name = scenario["base_scenario"] + "_" + scenario["sub_folder"]
                expanded_scenarios[name] = scenario

        return expanded_scenarios

    @staticmethod
    def _expand_scenario(scenario: dict, param_map=None, counter=0):

        # get the default
        if param_map is None:
            param_map = dict()

        # list for the expanded scenarios
        expanded_scenarios = []

        # iterate over all elements
        for element, element_dict in sorted(scenario.items(), key=lambda x: x[0]):
            # we do not expand these
            if element in ScenarioDict._special_elements:
                continue

            for param, param_dict in sorted(element_dict.items(), key=lambda x: x[0]):
                for key in sorted(ScenarioDict._param_dict_keys):
                    if key in param_dict and isinstance(param_dict[key], list):
                        # get the old param dict entry
                        if scenario["sub_folder"] != "":
                            old_param_map_entry = param_map.pop(scenario["sub_folder"])
                        else:
                            old_param_map_entry = dict()

                        # we need to expand this
                        for num, value in enumerate(param_dict[key]):
                            # copy the scenario
                            new_scenario = deepcopy(scenario)

                            # set the new value
                            new_scenario[element][param][key] = value

                            # create the name
                            if key + "_fmt" in param_dict:
                                if "{}" not in param_dict[key + "_fmt"]:
                                    raise SyntaxError("When setting a format for a name, you need to include a "
                                                      "placeholder '{}' for its value! No placeholder found in "
                                                      f"for {key} in {param} in {element} in {scenario['base_scenario']}")
                                name = param_dict[key + "_fmt"].format(value)
                                del new_scenario[element][param][key + "_fmt"]
                                # we don't need to increment the param for the next expansion
                                param_up = 0
                            else:
                                name = f"p{counter:02d}_{num:03d}"
                                # we need to increment the param for the next expansion
                                param_up = 1

                            # set the sub_folder
                            if new_scenario["sub_folder"] == "":
                                new_scenario["sub_folder"] = name
                            else:
                                new_scenario["sub_folder"] += "_" + name

                            # update the param_map
                            param_map[new_scenario["sub_folder"]] = deepcopy(old_param_map_entry)
                            if element not in param_map[new_scenario["sub_folder"]]:
                                param_map[new_scenario["sub_folder"]][element] = dict()
                            if param not in param_map[new_scenario["sub_folder"]][element]:
                                param_map[new_scenario["sub_folder"]][element][param] = dict()
                            param_map[new_scenario["sub_folder"]][element][param][key] = value

                            # set the param_map of the scenario
                            new_scenario["param_map"] = param_map

                            # expand this scenario as well
                            expanded_scenarios.extend(
                                ScenarioDict._expand_scenario(new_scenario, param_map, counter + param_up))

                        # expansion done
                        return expanded_scenarios

        # nothing was expanded, so we just return the scenario
        expanded_scenarios.append(scenario)

        # return the list
        return expanded_scenarios

    def expand_subsets(self, init_dict):
        """
        Expands a dictionary, e.g. expands sets etc.
        :param init_dict: The initial dict
        :return: A new dict which can be used for the scenario analysis
        """

        new_dict = init_dict.copy()
        for element_class in self.element_classes:
            current_set = element_class.label
            if current_set in new_dict:
                for param, param_dict in new_dict[current_set].items():
                    # dict for expansion
                    base_dict = param_dict

                    # get the exlusion list
                    if "exclude" in base_dict:
                        exclude_list = base_dict["exclude"]
                        del base_dict["exclude"]
                    else:
                        exclude_list = []

                    # expand the sets
                    for element in self.system.get(current_set, []):
                        if element not in exclude_list:
                            # create dicts if necessary
                            if element not in new_dict:
                                new_dict[element] = {}
                            # we only set the param dict if it is not already set
                            if param not in new_dict[element]:
                                new_dict[element][param] = base_dict.copy()
                # delete the old set
                del new_dict[current_set]
        return new_dict

    def validate_dict(self, vali_dict):
        """
        Validates a dictionary, raises an error if it is not valid
        :param vali_dict: The dictionary to validate
        """

        for element, element_dict in vali_dict.items():
            if element in self._special_elements:
                continue

            if not isinstance(element_dict, dict):
                raise ValueError(f"The entry for {element} is not a dictionary!")

            for param, param_dict in element_dict.items():
                if len(diff := (set(param_dict.keys()) - self._param_dict_keys)) > 0:
                    raise ValueError(
                        f"The entry for element {element} and param {param} contains invalid entries: {diff}!")

    @staticmethod
    def validate_file_name(fname):
        """
        Checks if the file name has an extension, it is expected to not have an extension
        :param fname: The file name to validte
        :return: The validated file name
        """

        fname, ext = os.path.splitext(fname)
        if ext != "":
            warnings.warn(f"The file name {fname}{ext} has an extension {ext}, removing it.")
        return fname

    def get_default(self, element, param):
        """
        Return the name where the default value should be read out
        :param element: The element name
        :param param: The parameter of the element
        :return: If the entry is overwritten by the scenario analysis the entry and factor are returned, otherwise
                 the default entry is returned with a factor of 1
        """

        # These are the default values
        default_f_name = "attributes"
        default_factor = 1.0

        if element in self.dict and param in (element_dict := self.dict[element]):
            param_dict = element_dict[param]
            default_f_name = param_dict.get("default", default_f_name)
            default_f_name = self.validate_file_name(default_f_name)
            default_factor = param_dict.get("default_op", default_factor)

        return default_f_name, default_factor

    def get_param_file(self, element, param):
        """
        Return the file name where the parameter values should be read out
        :param element: The element name
        :param param: The parameter of the element
        :return: If the entry is overwritten by the scenario analysis the entry and factor are returned, otherwise
                 the default entry is returned with a factor of 1
        """

        # These are the default values
        default_f_name = param
        default_factor = 1.0

        if element in self.dict and param in (element_dict := self.dict[element]):
            param_dict = element_dict[param]
            default_f_name = param_dict.get("file", default_f_name)
            default_f_name = self.validate_file_name(default_f_name)
            default_factor = param_dict.get("file_op", default_factor)

        return default_f_name, default_factor


# linopy helpers
# --------------

def lp_sum(exprs, dim='_term'):
    """
    Sum of linear expressions with lp.expressions.merge, returns 0 if list is emtpy
    :param exprs: The expressions to sum
    :param dim: Along which dimension to merge
    :return: The sum of the expressions
    """

    # emtpy sum
    if len(exprs) == 0:
        return 0
    # no sum
    if len(exprs) == 1:
        return exprs[0]
    # normal sum
    return lp.expressions.merge(exprs, dim=dim)


def linexpr_from_tuple_np(tuples, coords, model):
    """
    Transforms tuples of (coeff, var) into a linopy linear expression, but uses numpy broadcasting
    :param tuples: Tuple of (coeff, var)
    :param coords: The coordinates of the final linear expression
    :param model: The model to which the linear expression belongs
    :return: A linear expression
    """

    # get actual coords
    if not isinstance(coords, xr.core.dataarray.DataArrayCoordinates):
        coords = xr.DataArray(coords=coords).coords

    # numpy stack everything
    coefficients = []
    variables = []
    for coeff, var in tuples:
        var = var.labels.data
        if isinstance(coeff, (float, int)):
            coeff = np.full(var.shape, 1.0 * coeff)
        coefficients.append(coeff)
        variables.append(var)

    # to linear expression
    variables = xr.DataArray(np.stack(variables, axis=0), coords=coords, dims=["_term", *coords])
    coefficients = xr.DataArray(np.stack(coefficients, axis=0), coords=coords, dims=["_term", *coords])
    xr_ds = xr.Dataset({"coeffs": coefficients, "vars": variables}).transpose(..., "_term")

    return lp.LinearExpression(xr_ds, model)


def xr_like(fill_value, dtype, other, dims):
    """
    Creates an xarray with fill value and dtype like the other object but only containing the given dimensions
    :param fill_value: The value to fill the data with
    :param dtype: dtype of the data
    :param other: The other object to use as base
    :param dims: The dimensions to use
    :return: An object like the other object but only containing the given dimensions
    """

    # get the coords
    coords = {}
    for dim in dims:
        coords[dim] = other.coords[dim]

    # create the data array
    da = xr.DataArray(np.full([len(other.coords[dim]) for dim in dims], fill_value, dtype=dtype), coords=coords,
                      dims=dims)

    # return
    return da


# This is to lazy load h5 file most of it is taken from the hdfdict package
###########################################################################

TYPEID = '_type_'


@contextmanager
def hdf_file(hdf, lazy=True, *args, **kwargs):
    """
    Context manager yields h5 file if hdf is str,
    otherwise just yield hdf as is.

    :param hdf: #TODO describe parameter/return
    :param lazy: #TODO describe parameter/return
    :param args: #TODO describe parameter/return
    :param kwargs: #TODO describe parameter/return
    """
    if isinstance(hdf, str):
        if not lazy:
            with h5py.File(hdf, *args, **kwargs) as hdf:
                yield hdf
        else:
            yield h5py.File(hdf, *args, **kwargs)
    else:
        yield hdf


def unpack_dataset(item):
    """
    Reconstruct a hdfdict dataset.
    Only some special unpacking for yaml and datetime types.

    :param item: h5py.Dataset
    :return: Unpacked Data
    """

    value = item[()]
    if TYPEID in item.attrs:
        if item.attrs[TYPEID].astype(str) == 'datetime':
            if hasattr(value, '__iter__'):
                value = [datetime.fromtimestamp(
                    ts) for ts in value]
            else:
                value = datetime.fromtimestamp(value)

        if item.attrs[TYPEID].astype(str) == 'yaml':
            value = yaml.safe_load(value.decode())

    # bytes to strings
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    return value


class LazyHdfDict(UserDict):
    """
    Helps loading data only if values from the dict are requested.
    This is done by reimplementing the __getitem__ method.
    """

    def __init__(self, _h5file=None, *args, **kwargs):
        """

        :param _h5file: #TODO describe parameter/return
        :param args: #TODO describe parameter/return
        :param kwargs: #TODO describe parameter/return
        """
        super().__init__(*args, **kwargs)
        self._h5file = _h5file  # used to close the file on deletion.

    def __getitem__(self, key):
        """Returns item and loads dataset if needed.

        :param key: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """
        item = super().__getitem__(key)
        if isinstance(item, h5py.Dataset):
            item = unpack_dataset(item)
            self.__setitem__(key, item)
        return item

    def unlazy(self, return_dict=False):
        """
        Unpacks all datasets.
        You can call dict(this_instance) then to get a real dict.

        :param return_dict: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """
        load(self, lazy=False)

        # Load loads all the data but we need to transform the lazydict into normal dicts
        def _recursive(lazy_dict):
            for k in list(lazy_dict.keys()):
                if isinstance(lazy_dict[k], LazyHdfDict):
                    _recursive(lazy_dict[k])
                    lazy_dict[k] = dict(lazy_dict[k])

        _recursive(self)

        if return_dict:
            return dict(self)

    def close(self):
        """
        Closes the h5file if provided at initialization.
        """
        if self._h5file and hasattr(self._h5file, 'close'):
            self._h5file.close()

    def __del__(self):
        """
        delete
        """
        self.close()

    def _ipython_key_completions_(self):
        """
        Returns a tuple of keys.
        Special Method for ipython to get key completion

        :return: #TODO describe parameter/return
        """
        return tuple(self.keys())


def fill_dict(hdfobject, datadict, lazy=True, unpacker=unpack_dataset):
    """
    Recursivley unpacks a hdf object into a dict

    :param hdfobject: Object to recursively unpack
    :param datadict: A dict option to add the unpacked values to
    :param lazy: If True, the datasets are lazy loaded at the moment an item is requested.
    :param unpacker: Unpack function gets `value` of type h5py.Dataset. Must return the data you would like to
                     have it in the returned dict.
    :return: a dict
    """

    for key, value in hdfobject.items():
        if type(value) == h5py.Group or isinstance(value, LazyHdfDict):
            if lazy:
                datadict[key] = LazyHdfDict()
            else:
                datadict[key] = {}
            datadict[key] = fill_dict(value, datadict[key], lazy, unpacker)
        elif isinstance(value, h5py.Dataset):
            if not lazy:
                value = unpacker(value)
            datadict[key] = value

    return datadict


def load(hdf, lazy=True, unpacker=unpack_dataset, *args, **kwargs):
    """
    Returns a dictionary containing the groups as keys and the datasets as values from given hdf file.

    :param hdf: string (path to file) or `h5py.File()` or `h5py.Group()`
    :param lazy: If True, the datasets are lazy loaded at the moment an item is requested.
    :param unpacker: Unpack function gets `value` of type h5py.Dataset. Must return the data you would like to
                     have it in the returned dict.
    :param args: Additional arguments for the hdf_file handler
    :param kwargs: Additional keyword arguments for the hdf_file handler
    :return: The dictionary containing all groupnames as keys and datasets as values.
    """

    with hdf_file(hdf, lazy=lazy, *args, **kwargs) as hdf:
        if lazy:
            data = LazyHdfDict(_h5file=hdf)
        else:
            data = {}
        return fill_dict(hdf, data, lazy=lazy, unpacker=unpacker)


def pack_dataset(hdfobject, key, value):
    """
    Packs a given key value pair into a dataset in the given hdfobject.

    :param hdfobject: #TODO describe parameter/return
    :param key: #TODO describe parameter/return
    :param value: #TODO describe parameter/return
    """

    isdt = None
    if isinstance(value, datetime):
        value = value.timestamp()
        isdt = True

    if hasattr(value, '__iter__'):
        if all(isinstance(i, datetime) for i in value):
            value = [item.timestamp() for item in value]
            isdt = True

    try:
        ds = hdfobject.create_dataset(name=key, data=value)
        if isdt:
            ds.attrs.create(
                name=TYPEID,
                data=string_("datetime"))
    except TypeError:
        # Obviously the data was not serializable. To give it
        # a last try; serialize it to yaml
        # and save it to the hdf file:
        ds = hdfobject.create_dataset(
            name=key,
            data=string_(yaml.safe_dump(value))
        )
        ds.attrs.create(
            name=TYPEID,
            data=string_("yaml"))
        # if this fails again, restructure your data!


def dump(data, hdf, packer=pack_dataset, *args, **kwargs):
    """
    Adds keys of given dict as groups and values as datasets to the given hdf-file (by string or object) or group object.

    :param data: The dictionary containing only string keys and data values or dicts again.
    :param hdf: string (path to file) or `h5py.File()` or `h5py.Group()`
    :param packer: Callable gets `hdfobject, key, value` as input.
                   `hdfobject` is considered to be either a h5py.File or a h5py.Group.
                   `key` is the name of the dataset.
                   `value` is the dataset to be packed and accepted by h5py.
    :param args: Additional arguments for the hdf_file handler
    :param kwargs: Additional keyword arguments for the hdf_file handler
    :return: `h5py.Group()` or `h5py.File()` instance
    """

    def _recurse(datadict, hdfobject):
        for key, value in datadict.items():
            if isinstance(key, tuple):
                key = '_'.join((str(i) for i in key))
            if isinstance(value, (dict, LazyHdfDict)):
                hdfgroup = hdfobject.create_group(key)
                _recurse(value, hdfgroup)
            else:
                packer(hdfobject, key, value)

    with hdf_file(hdf, *args, **kwargs) as hdf:
        _recurse(data, hdf)
        return hdf


# This is to lazy load h5 file most
###################################

class LazyEntry(object):
    """
    This is a lazy entry from a store that is loaded only when it is requested.
    """

    def __init__(self, path, dtype, store, value=None):
        """
        Initializes the class.

        :param value: The value to store
        :param path: The path to the leave of the store
        :param dtype: The type of the leave
        :param store: The store to load the data from
        :param value: The value to store, can be None, if given, this is returned when deserialized and the store is not
                      accessed.
        """
        self.path = path
        self.dtype = dtype
        self.store = store
        self.value = value

    def desarialize(self):
        """
        Deserializes the data from the store.

        :return: The deserialized data
        """

        # if we have a value, return it
        if self.value is not None:
            return self.value

        # get the data
        df = self.store.get(self.path)

        # go through the different types
        if self.dtype == "pandas":
            return df
        elif self.dtype == "scalar":
            return df.values[0]
        elif self.dtype == "vector" or self.dtype == "matrix":
            return df.values
        else:
            raise TypeError(f"Unkon type {self.dtype}")


class LazyDict(dict):
    """
    This class is a dictionary that loads the values lazily.
    """

    def __getitem__(self, item):
        """
        Returns the item from the dictionary.

        :param item: The item to return
        :return: The item
        """

        value = super().__getitem__(item)

        if isinstance(value, LazyEntry):
            value = value.desarialize()
            super().__setitem__(item, value)

        return value


class HDFPandasSerializer(LazyDict):
    """
    This class saves dictionaries with a pandas store as a hdf file.
    """

    def __init__(self, file_name, lazy=True):
        """
        Initializes the class to read a hdf file, potentially lazily. For writing files, use the classmethod
        "serialize_dict".

        :param file_name: The file name of the hdf file.
        :param lazy: Boolean if lazy selection
        """

        # super init
        super().__init__()

        # attributes
        self.file_name = file_name
        self.store = pd.HDFStore(file_name, mode="r")

        # raise a key error if the file is empty
        if not self.store.keys():
            raise KeyError(f"This file does not contain any keys: {file_name}")

        self._lazy = lazy

        # load all keys
        self._load()

    def _load(self):
        """
        Loads the hdf file into the dictionary.
        """

        for path, groups, leaves in self.store.walk():
            # get the right dict
            previous_keys = path.split("/")[1:]
            current_dict = self
            for key in previous_keys:
                current_dict = current_dict[key]

            # create the groups
            for group_key in groups:
                current_dict[group_key] = LazyDict()

            # load the leaves
            for leave_key in leaves:
                leave_path = f"{path}/{leave_key}"
                attrs = self.store.get_storer(f"{path}/{leave_key}").attrs
                dtype = attrs.type

                # if its a scalar we read it out now
                value = None
                if dtype == "scalar":
                    value = attrs.value

                # create the entry
                entry = LazyEntry(leave_path, dtype, self.store, value=value)

                if self._lazy:
                    current_dict[leave_key] = entry
                else:
                    current_dict[leave_key] = entry.desarialize()

        # no need to keep the file open
        if not self._lazy:
            self.close()

    def close(self):
        """
        Closes the hdf file.
        """

        self.store.close()

    @classmethod
    def _recurse(cls, store, dictionary, previous_key=""):
        """
        Recursively saves the dictionary into the store.

        :param store: The store to save the dictionary into.
        :param dictionary: The dictionary to save.
        :param previous_key: The key of the dictionary.
        """

        for key, value in dictionary.items():
            if not isinstance(key, str):
                raise TypeError("All dictionary keys must be strings!")

            key = f"{previous_key}/{key}"
            if isinstance(value, dict):
                cls._recurse(store, value, key)
            elif isinstance(value, (pd.DataFrame, pd.Series)):
                # make a proper multi index to save memory
                store.put(key, value)
                store.get_storer(key).attrs.type = "pandas"
            elif isinstance(value, (float, str, int)):
                store.put(key, pd.Series([], dtype=int))
                store.get_storer(key).attrs.value = value
                store.get_storer(key).attrs.type = "scalar"
            elif isinstance(value, (list, tuple)) or isinstance(value, np.ndarray) and value.ndim == 1:
                store.put(key, pd.Series(value))
                store.get_storer(key).attrs.type = "vector"
            elif isinstance(value, np.ndarray) and value.ndim == 2:
                store.put(key, pd.DataFrame(value))
                store.get_storer(key).attrs.type = "matrix"
            else:
                raise TypeError(f"Type {type(value)} is not supported.")

    @classmethod
    def serialize_dict(cls, file_name, dictionary, overwrite=True):
        """
        Serialized a dictionary of dataframes and other objects into a hdf file.

        :param file_name: The file name of the hdf file.
        :param dictionary: The dictionary to serialize
        :param overwrite: If True, the file will be overwritten.
        """

        if not overwrite and os.path.exists(file_name):
            raise FileExistsError("File already exists. Please set overwrite=True to overwrite the file.")

        with pd.HDFStore(file_name, mode='w', complevel=4) as store:
            cls._recurse(store, dictionary)


class InputDataChecks:
    """
    This class checks if the input data (folder/file structure, system.py settings, element definitions, etc.) is defined correctly
    """

    def __init__(self, config, optimization_setup):
        """
        Initialize the class

        :param config: config object used to extract the analysis, system and solver dictionaries
        """
        self.system = config.system
        self.analysis = config.analysis
        self.optimization_setup = optimization_setup

    def check_technology_selected(self):
        """
        Checks if at least one technology is selected in the system.py file
        """
        assert len(self.system["set_conversion_technologies"] + self.system["set_transport_technologies"] + self.system["set_storage_technologies"]) > 0, f"No technology selected in stystem.py"

    def check_year_definitions(self):
        """
        Check if year-related parameters are defined correctly
        """
        #assert that number of optimized years is a positive integer
        assert isinstance(self.system["optimized_years"], int) and self.system["optimized_years"] > 0, f"Number of optimized years must be a positive integer, however it is {self.system['optimized_years']}"
        #assert that interval between years is a positive integer
        assert isinstance(self.system["interval_between_years"], int) and self.system["interval_between_years"] > 0, f"Interval between years must be a positive integer, however it is {self.system['interval_between_years']}"
        assert isinstance(self.system["reference_year"], int) and self.system["reference_year"] >= self.analysis["earliest_year_of_data"], f"Reference year must be an integer and larger than the defined earliest_year_of_data: {self.analysis['earliest_year_of_data']}"
        #check if the number of years in the rolling horizon isn't larger than the number of optimized years
        if self.system["years_in_rolling_horizon"] > self.system["optimized_years"] and self.system["use_rolling_horizon"]:
            warnings.warn(f"The chosen number of years in the rolling horizon step is larger than the total number of years optimized!")

    def check_primary_folder_structure(self):
        """
        Checks if the primary folder structure (set_conversion_technology, set_transport_technology, ..., system_specification) is provided correctly

        :param analysis: dictionary defining the analysis framework
        """
        for technology_subset in self.analysis["subsets"]["set_technologies"]:
            if not os.path.exists(os.path.join(self.analysis["dataset"], technology_subset)):
                raise AssertionError(f"Folder {technology_subset} does not exist!")
        if not os.path.exists(os.path.join(self.analysis["dataset"], "set_carriers")):
            raise AssertionError(f"Folder set_carriers does not exist!")
        if not os.path.exists(os.path.join(self.analysis["dataset"], "system_specification")):
            raise AssertionError(f"Folder system_specification does not exist!")
        for file_name in ["attributes.csv", "base_units.csv", "set_edges.csv", "set_nodes.csv", "unit_definitions.txt"]:
            if file_name not in os.listdir(os.path.join(self.analysis["dataset"], "system_specification")):
                raise FileNotFoundError(f"File {file_name} is missing in the system_specification directory")

    def check_existing_technology_data(self):
        """
        This method checks the existing technology input data and only regards those technology elements for which folders containing the attributes.csv file exist.
        """
        self.optimization_setup.system["set_technologies"] = []
        for technology_subset in self.optimization_setup.analysis["subsets"]["set_technologies"]:
            for technology in self.optimization_setup.system[technology_subset]:
                if technology not in self.optimization_setup.paths[technology_subset].keys():
                    logging.warning(f"Technology {technology} selected in config does not exist in input data, excluded from model.")
                    self.optimization_setup.system[technology_subset].remove(technology)
                elif "attributes.csv" not in self.optimization_setup.paths[technology_subset][technology]:
                    raise FileNotFoundError(f"The file attributes.csv does not exist for the technology {technology}")
            self.optimization_setup.system["set_technologies"].extend(self.optimization_setup.system[technology_subset])
            # check subsets of technology_subset
            if technology_subset in self.optimization_setup.analysis["subsets"].keys():
                for subset in self.optimization_setup.analysis["subsets"][technology_subset]:
                    for technology in self.optimization_setup.system[subset]:
                        if technology not in self.optimization_setup.paths[technology_subset].keys():
                            logging.warning(f"Technology {technology} selected in config does not exist in input data, excluded from model.")
                            self.optimization_setup.system[subset].remove(technology)
                        elif "attributes.csv" not in self.optimization_setup.paths[technology_subset][technology]:
                            raise FileNotFoundError(f"The file attributes.csv does not exist for the technology {technology}")
                    self.optimization_setup.system[technology_subset].extend(self.optimization_setup.system[subset])
                    self.optimization_setup.system["set_technologies"].extend(self.optimization_setup.system[subset])

    def check_existing_carrier_data(self):
        """
        Checks the existing carrier data and only regards those carriers for which folders exist
        """
        # check if carriers exist
        for carrier in self.optimization_setup.system["set_carriers"]:
            assert carrier in self.optimization_setup.paths["set_carriers"].keys(), f"Carrier {carrier} does not exist in input data."
            assert "attributes.csv" in self.optimization_setup.paths["set_carriers"][carrier], f"Attributes.csv file does not exist for the carrier {carrier}"

    def check_dataset(self):
        """
        Ensures that the dataset chosen in the config does exist and contains a system.py file
        """
        dataset = os.path.basename(self.analysis["dataset"])
        found = False
        if dataset in os.listdir(os.getcwd()):
            found = True
        else:
            for directory in next(os.walk(os.getcwd()))[1]:
                if dataset in os.listdir(os.path.join(os.getcwd(), directory)) and directory != "outputs":
                    found = True
                    break
        if not found:
            raise AssertionError(f"The chosen dataset {dataset} does not exist at {self.analysis['dataset']} as it is specified in the config")
        #check if chosen dataset contains a system.py file
        if not os.path.exists(os.path.join(self.analysis['dataset'], "system.py")):
            raise FileNotFoundError(f"system.py not found in dataset: {self.analysis['dataset']}")

    def check_carrier_configuration(self, input_carrier, output_carrier, reference_carrier, name):
        """
        Checks if the chosen input/output and reference carrier combination is reasonable

        :param input_carrier: input carrier of conversion technology
        :param output_carrier: output carrier of conversion technology
        :param reference_carrier: reference carrier of technology
        :param name: name of conversion technology
        """
        #assert that conversion technology has at least an input or an output carrier
        assert len(input_carrier+output_carrier) > 0, f"Conversion technology {name} has neither an input nor an output carrier!"
        #check if reference carrier in input and output carriers and set technology to correspondent carrier
        assert reference_carrier[0] in (input_carrier + output_carrier), f"reference carrier {reference_carrier} of technology {name} not in input and output carriers {input_carrier + output_carrier}"
        set_input_carrier = set(input_carrier)
        set_output_carrier = set(output_carrier)
        #assert that input and output carrier of conversion tech are different
        assert not set_input_carrier & set_output_carrier, f"The conversion technology {name} has the same input ({input_carrier[0]}) and output ({output_carrier[0]}) carrier!"

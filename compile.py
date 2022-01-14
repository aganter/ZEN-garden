"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
              Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Compilation  of the optimization problem.
==========================================================================================================================================================================="""

import os
import logging
import config
from preprocess.prepare import Prepare
from model.model import Model
from postprocess.results import Postprocess

# SETUP LOGGER
log_format = '%(asctime)s %(filename)s: %(message)s'
if not os.path.exists('outputs/logs'):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    os.mkdir('outputs/logs')
logging.basicConfig(filename='outputs/logs/valueChain.log', level=logging.CRITICAL, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
# prevent double printing
logging.propagate = True

# CREATE INPUT FILE
prepare = Prepare(config.analysis, config.system)
# FORMULATE AND SOLVE THE OPTIMIZATION PROBLEM
model = Model(config.analysis, config.system, prepare.pyoDict)
model.solve(config.solver, prepare.pyoDict)

# EVALUATE RESULTS
evaluation = Postprocess(model, prepare.pyoDict, modelName = 'test')

# CREATE DICTIONARY TO VISUALIZE RESULTS IN DASHBOARD



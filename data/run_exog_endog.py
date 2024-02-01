"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      January-2024
Authors:      Anya Xie (anyxie@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Analysis of results of exog vs. endog model
==========================================================================================================================================================================="""

import os
from zen_garden._internal import main
import time


def endog(config, folder_path, data_set_name):
    # run the test
    config.analysis["dataset"] = os.path.join(folder_path, data_set_name + "_endog")
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name + "_endog"))

    return optimization_setup

def exog(config, folder_path, data_set_name):
    # run the test
    config.analysis["dataset"] = os.path.join(folder_path, data_set_name + "_exog")
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name + "_exog"))

    return optimization_setup


if __name__ == "__main__":
    from config import config
    config.solver["keep_files"] = False
    folder_path = os.path.dirname(__file__)

    data_set_name = "20240201_Hydrogen_8760_cap_addition_max"

    # Run endog and exog models
    t0 = time.perf_counter()
    try:
        optimization_setup = endog(config=config, folder_path=folder_path, data_set_name=data_set_name)
        optimization_setup = exog(config=config, folder_path=folder_path, data_set_name=data_set_name)
        t1 = time.perf_counter()
        message = f'Optimization completed after {(t1-t0)/60:0.2f} min for dataset {data_set_name} :)'
    except:
        t1 = time.perf_counter()
        message = f'Optimization failed after {(t1-t0)/60:0.2f} min for dataset {data_set_name} :('

    import asyncio
    from telegram import Bot

    botToken = '6585556167:AAGDTMRiK9BZEsjjTBT1_m57c-92USqVdRI'
    bot = Bot(token=botToken)

    chatId = '5778346437'
    asyncio.run(bot.send_message(chat_id=chatId, text=message))

"""
@Purpose: Main script for initializing environment settings and start procesing the CorpusOrales project, handling main modes:
@Usage: Run `python mainProcess.py`.
"""

from sources.common.common import processControl, logger, log_
from sources.common.paramsManager import getConfigs
from sources.preseeaCorpus import processCrawler
from sources.voiceProcess import processAudio
from sources.analisis.analysis import processAnalisis
import os


def mainProcess():
    #preseeaAccess()
    resultsPath = os.path.join(processControl.env['outputDir'], "results.json")
    if processControl.args.proc == "C":
        processCrawler(resultsPath)
    elif processControl.args.proc == "A":
        processAudio(resultsPath)
    elif processControl.args.proc == "R":
        processAnalisis(resultsPath)

    return True


if __name__ == '__main__':
    """
    Entry point for starting the main image caption process.

    This block of code is executed when the script is run directly. It logs the start of the process, retrieves configuration settings,
    and then triggers the main process. After the main process completes, it logs the completion of the task.

    The function performs the following steps:
    - Logs the start of the process.
    - Calls `getConfigs()` to retrieve necessary configurations.
    - Executes `mainProcess()` to handle model training or application.
    - Logs the completion of the process.

    :return: None
    """

    log_("info", logger, "********** STARTING Main Corpus Orales Process **********")
    getConfigs()
    mainProcess()
    log_("info", logger, "********** PROCESS COMPLETED **********")

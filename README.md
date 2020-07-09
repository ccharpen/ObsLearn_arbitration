# ObsLearn_arbitration
Code to run task and analyses for study of arbitration between imitation and emulation during observational learning.

See https://osf.io/49ws3/ for raw behavioral data files and aggregate-level data (behavioral and fMRI summaries).

## 'task' folder
The task was run with PsychoPy v.1.85 (https://www.psychopy.org/).
- practice_before_scanner.py runs the instructions and practice trials that participants completed before the fMRI portion of the study
- main_task_fmri.py runs each task run while participants are in the MRI scanner

## 'behavioral_analyses' folder
This contains data and code to perform behavioral model-fitting analyses. 'Data_for_models_S1.mat' and 'Data_for_models_S2.mat' contain the data from Study 1 and Study 2, respectively, and are being used by the wrapper script 'models_parameter_estimation_MLE.m' to perform model-fitting of all 10 models using maximum likelihood estimation. The output file of that script, 'Models_Parameters_MLE.mat', is also saved here. Individual model functions (loglikelihood functions, and generative functions) are in the model_function folder.

Code and outputs for hierarchical model-fitting, using expectation maximization method, are in the hierarchical_fit folder. The 'run_EM.m' function is the wrapper that runs model-fitting for all 10 models.

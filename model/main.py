import os
import warnings

from train_module import *
from experimental_setup import *
warnings.filterwarnings("ignore", category=UserWarning)



def run(mode: str, path="experiments.json"):
    #launch_tensorboard()
    errors = []
    if mode == "dev":
        exp_setup = load_experiments()
        exp_setup = exp_setup["test"]
        print(exp_setup)
        try:
            train_model(exp_setup)
        except:
            errors.append(f"Experiment no. {exp_setup['EXPERIMENT_ID']} with model {exp_setup['MODEL']} failed on ds {exp_setup['DATA_DIR']}.")

    if mode == "train":
        #create_exp()
        exp_setup = load_experiments(path)
        for key in exp_setup:
            exp = exp_setup[key]
            print(exp)
            try:
                train_model(exp)
            except:
                errors.append(f"Experiment no. {exp['EXPERIMENT_ID']} with model {exp['MODEL']} failed on ds {exp['DATA_DIR']}.")
    for error in errors: print(error)
    return


def run_mp(exp):
    errors = []
    try:
        train_model(exp)
    except:
        errors.append(f"Experiment no. {exp['EXPERIMENT_ID']} with model {exp['MODEL']} failed on ds {exp['DATA_DIR']}.")
    for error in errors:
        print(error)


# Quickrun
if __name__ == "__main__":
 
    set = load_experiments("experiments.json")
    for experiment in set:
        print(set[experiment])
        train_model(set[experiment])

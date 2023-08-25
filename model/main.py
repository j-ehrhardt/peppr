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
    """            
    exp_setup =  {
        "EXPERIMENT_ID": "test",
        "DEVICE": 1,
        "MODEL": "cconv_inv",
        "DATA_DIR": "../data/physical/ds1/",
        "LOG_DIR": "logs/test/", # "logs/test/multi/ode/ds2/cconv_inv/",
        "SEED": 2,
        "SEQ_LEN": 512,
        "BATCH_SIZE": 512,
        "NUM_WORKERS": 8,
        "MAX_EPOCHS": 100,
        "LEARNING_RATE": 1e-05,
        "WEIGHT_DECAY": 1e-07,
        "CONV_N_FILT": 4,
        "CONV_ENCS": [
            2,
            2,
            2
        ],
        "CONV_N_KERNEL": 3,
        "CONV_STRIDE": 2,
        "CONV_DIL": 1,
        "VAR_N_LAT": 32,
        "CORE_ENCS": [
            0.8,
            0.4,
            0.2,
            0.2
        ],
        "KL_WEIGHT": 0.0001,
        "LAMBDA_X": 2.5,
        "LAMBDA_Y": 0.5,
        "LAMBDA_Z": 0.0001,
        "DS_SAMPLING": 1
    }

    #try:
    train_model(exp_setup)
    """
    """    
    run(mode="train", path="experiments/experiments_theoretical_multi_seed_2.json")
    """
    """
    PATH = "experiments/real"
    for element in os.listdir(PATH):
        set = load_experiments(PATH + "/" + element)
        for experiment in set:
            print(set[experiment])
            train_model(set[experiment])
    """
    set = load_experiments("experiments.json")
    for experiment in set:
        print(set[experiment])
        train_model(set[experiment])
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train model:")
    parser.add_argument('--path', metavar='path', required=True,
                        help='the path to the experimental file')
    args = parser.parse_args()

    exp_file = load_experiments(args.path)
    #train_model(exp_file["0"])
    for experiment in exp_file:
        print(exp_file[experiment])
        train_model(experiment)
    
    """
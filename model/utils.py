import os
import json
import torch
import numpy as np
import torch.nn as nn


def launch_tensorboard(log_dir = "logs/ds3/"):
    os.system('tensorboard --logdir="' + log_dir + '"')
    return

def list_mean(list):
    return sum(list)/len(list)

def save_results(hparam: dict, metrics: dict):
    """
    saving results and experimental setup as .json file in log-dir
    metrics must be passed as dict: [train_y_MSE, train_X_MSE, test_y_MSE, test_X_MSE]
    """
    results = {"setup": hparam, "results":metrics}

    with open(hparam["LOG_DIR"] + "logs_exp" + str(hparam["EXPERIMENT_ID"]) + "/results.json", "w") as handle:
        json.dump(results, handle, indent=4)
        handle.close()
    return print(f"Logged results of experiment ", hparam["EXPERIMENT_ID"])

def get_out_shape(model, x, y):
    # calculating the output dimension of a model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = x.to(device), y.to(device)
    model.to(device)
    _, y_ = model.forward(x, y, 1)
    return y_


def init_weights(m):
    if type(m) == nn.Linear:
        try:
            torch.nn.init.normal_(m.weight, mean=0, std=1/torch.sqrt(m.in_features))
        except:
            pass

class AutoStop:
    def __init__(self, tolerance_e=10, min_delta=0.0001, max_delta=0.001):
        self.tolerance = tolerance_e
        self.max_delta = max_delta
        self.min_delta = min_delta
        self.counter_max = 0
        self.counter_min = 0
        self.min_loss = np.inf

    def auto_stop(self, l_x_val, l_y_val):
        if l_x_val < self.min_loss and l_y_val < self.min_loss:
            if (l_x_val - self.min_loss) < self.min_delta and (l_y_val - self.min_loss) < self.min_delta:
                self.counter_min += 1
                if self.counter_min > self.tolerance*2:
                    print("\n\nbreak: convergence criterion\n\n")
                    return True
            self.min_loss = max([l_x_val, l_y_val])
            self.counter_min, self.counter_max = 0, 0
        elif l_x_val > (self.min_loss + self.max_delta) and l_y_val > (self.min_loss + self.max_delta):
            self.counter_max += 1
            if self.counter_max >= self.tolerance:
                print("\n\nbreak: divergence criterion\n\n")
                return True
        elif l_x_val == np.inf or l_y_val == np.inf:
            return True
        return False


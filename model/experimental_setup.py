import json

def create_exp(i, seed_):
    """
    creates a grid of experiments from the parameter lists batch_size, learning_rate, and max_epochs

    :return experiments_dict:
    """

    device = 0
    run = "real"
    mode = "ind"
    ds = ["physical/ds1", "physical/ds2", "physical/ds3", "physical/ds4", "biochemical/ds1", "biochemical/ds2", "biochemical/ds3"]    # , , "physical/ds5""physical/ds6"
    model = ["cconv_core", "cconv_vae", "cconv_inv"]
    seed = [seed_] #, 2, 3, 4, 5, 6, 7, 42]
    seq_len = [512]
    batch_size = [512]
    max_epochs = [1000]
    lr = [1e-5]
    wd = [1e-7]
    n_filt = [4]#[2, 4, 8]
    conv_enc = [[4,4,8]] #, [8, 8, 16]]
    conv_kernel = [3]
    conv_stride = [2]
    conv_dil = [1]
    var_n_lat = [10]
    core_encs = [[0.8,0.4,0.2,0.2]]
    kl_weight = [0.0001]
    lambda_x = 2.5
    lambda_y = 0.5
    lambda_z = 0.0001
    sampling = [1]# [1, 2, 3, 4, 5, 6, 7, 8]

    experiments_dict = {}
    for a in ds:
        for b in model:
            for c in seed:
                for d in seq_len:
                    for e in batch_size:
                        for f in max_epochs:
                            for g in lr:
                                for h in wd:
                                    for j in n_filt:
                                        for k in conv_enc:
                                            for l in conv_kernel:
                                                for m in conv_stride:
                                                    for n in conv_dil:
                                                        for o in var_n_lat:
                                                            for p in core_encs:
                                                                for q in kl_weight:
                                                                    for ds_sample in sampling:
                                                                        experiments_dict[i] = {
                                                                            "EXPERIMENT_ID": i,                     # int
                                                                            "DEVICE": device,                            # int
                                                                            "MODEL": b,                             # str
                                                                            "DATA_DIR": "../data/" + a + "/",       # str
                                                                            "LOG_DIR": "logs/" + run + "/" + mode + "/" + a + "/" + b + "/", # str # "logs/" + run + "/" + mode + "/" + a + "/" + b + "_" + str(ds_sample) +  "/", # str, #"logs/" + run + "/" + mode + "/" + a + "/" + b + "_" + str(ds_sample) +  "/", # str
                                                                            "SEED": c,                              # int
                                                                            "SEQ_LEN": d,                           # int
                                                                            "BATCH_SIZE": e,                        # int
                                                                            "NUM_WORKERS": 8,                       # int
                                                                            "MAX_EPOCHS": f,                        # int
                                                                            "LEARNING_RATE": g,                     # int
                                                                            "WEIGHT_DECAY": h,                      # int
                                                                            "CONV_N_FILT": j,                       # int
                                                                            "CONV_ENCS": k,                         # list(int, 5)
                                                                            "CONV_N_KERNEL": l,                     # int
                                                                            "CONV_STRIDE": m,                       # int
                                                                            "CONV_DIL": n,                          # int
                                                                            "VAR_N_LAT": o,                         # int
                                                                            "CORE_ENCS": p,                         # list(float, 4)
                                                                            "KL_WEIGHT": q,                         # float
                                                                            "LAMBDA_X": lambda_x,                   # float
                                                                            "LAMBDA_Y": lambda_y,                   # float
                                                                            "LAMBDA_Z": lambda_z,                   # float
                                                                            "DS_SAMPLING": ds_sample
                                                                        }
                                                                        i += 1

    #filename = "experiments_" + run + "_" + mode + "_seed_" + str(seed[0]) + ".json"
    filename = "experiments.json"
    with open(filename, "w") as json_file:
        json.dump(experiments_dict, json_file, indent=4)

    print("experimental grid was created and saved in experiments.json\n\n")
    return i


def load_experiments(path="experiments.json"):
    with open(path) as json_file:
        hparam = json.load(json_file)

    return hparam


# Quickrun
if __name__ == "__main__":
    seeds = [1]# [1, 2, 3, 4, 5, 6, 7, 42]
    i = 2010

    for seed in seeds:
        i = create_exp(i, seed)
    #hparam = load_experiments()

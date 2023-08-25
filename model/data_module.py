import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, default_collate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, PowerTransformer
from sklearn.model_selection import train_test_split


class DataModule():
    def __init__(self, hparam:dict):
        """
        datamodule to convert csv to datasets
        including slicer, train- test- val- sampler, and dataloader.

        :param hparam: [dict] containing experimental setup
        """
        self.hparam = hparam
        self.create_dirs()
        self.scaler = PowerTransformer(method="yeo-johnson", standardize=False)

    def create_dirs(self):
        if not os.path.exists(self.hparam["LOG_DIR"] + "/logs_exp" + str(self.hparam["EXPERIMENT_ID"])):
            os.makedirs(self.hparam["LOG_DIR"] + "/logs_exp" + str(self.hparam["EXPERIMENT_ID"]))

    def scale(self, df):
        cols = df.columns
        df = pd.DataFrame(self.scaler.fit_transform(df), columns=cols, index=None)
        return df

    def sliding_window(self, array, seq_len):
        """
        Applying sliding, sampling an array into an array of arrays with length seq_len
        :return: array of arrays
        """
        X = []
        for i in range(len(array) - seq_len - 1):
            _X = array[i:(i + seq_len)]
            X.append(np.transpose(_X))
        return np.array(X)

    def static_window(self, array, seq_len, num_samples=-1):
        """
        Applying static  window sampling, sampling an array into an array of arrays with length seq_len
        :return: array of arrays
        """
        X = []
        # condition for single sample
        if num_samples == -1:
            num_samples = int(len(array)/seq_len)

        for i in range(int(len(array)/seq_len)):
            _X = array[(i*seq_len):((i*seq_len) + seq_len)]
            X.append(np.transpose(_X))
        return np.array(X)

    def f_train_loader(self, x_train, y_train):
        ds_train = TensorDataset(x_train, y_train)
        return DataLoader(ds_train, batch_size=self.hparam["BATCH_SIZE"], num_workers=self.hparam["NUM_WORKERS"], shuffle=True, drop_last=True, pin_memory=True)

    def f_val_loader(self, x_val, y_val):
        ds_val = TensorDataset(x_val, y_val)
        return DataLoader(ds_val, batch_size=self.hparam["BATCH_SIZE"], num_workers=self.hparam["NUM_WORKERS"], shuffle=False, drop_last=True, pin_memory=True)

    def f_test_loader(self, x_test, y_test):
        ds_test = TensorDataset(x_test, y_test)
        return DataLoader(ds_test, batch_size=self.hparam["BATCH_SIZE"], num_workers=self.hparam["NUM_WORKERS"], shuffle=True, drop_last=True, pin_memory=True)

    def f_pred_loader(self, x_test, y_test):
        num_samples = -1
        ds_pred = TensorDataset(
            torch.Tensor(self.static_window(x_test, seq_len=self.hparam["SEQ_LEN"], num_samples=num_samples)),
            torch.Tensor(self.static_window(y_test, seq_len=self.hparam["SEQ_LEN"], num_samples=num_samples)))
        return DataLoader(ds_pred, batch_size=1, num_workers=self.hparam["NUM_WORKERS"], shuffle=False, drop_last=True, pin_memory=True)


class DataModuleSingleStep(DataModule):
    def __init__(self, hparam:dict):
        super().__init__(hparam)
        self.hparam = hparam

        df_in, df_out, df_param = self.load_df()
        df_in, df_out, df_param = self.scale(df_in), self.scale(df_out), self.scale(df_param) # TODO remove scaling, if necessary
        x_train, x_val, x_test, y_train, y_val, y_test = self.sampler(df_in, df_out, df_param)

        self.train_loader = self.f_train_loader(x_train, y_train)
        self.val_loader = self.f_val_loader(x_val, y_val)
        self.test_loader = self.f_test_loader(x_test, y_test)

    def load_df(self):
        df_in = pd.read_csv(self.hparam["DATA_DIR"] + "ds_in.csv", index_col=0)
        df_out = pd.read_csv(self.hparam["DATA_DIR"] + "ds_out.csv", index_col=0)
        df_param = pd.read_csv(self.hparam["DATA_DIR"] + "ds_param.csv", index_col=0)

        """        """
        # only for theoretical testings to triple the dataset size
        #if "theoretical" in self.hparam["LOG_DIR"]:
        df_in = pd.concat([df_in, df_in], axis=0)
        df_out = pd.concat([df_out, df_out], axis=0)
        df_param = pd.concat([df_param, df_param], axis=0)

        df_in = df_in.iloc[::self.hparam["DS_SAMPLING"], :].reset_index(drop=True)
        df_out = df_out.iloc[::self.hparam["DS_SAMPLING"], :].reset_index(drop=True)
        df_param = df_param.iloc[::self.hparam["DS_SAMPLING"], :].reset_index(drop=True)
        return df_in, df_out, df_param

    def sampler(self, df_in, df_out, df_param):
        x = pd.concat([df_in, df_param], axis=1).values
        y = df_out.values

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.5, shuffle=False, random_state=self.hparam["SEED"])
        x_val, x_test, y_val, y_test = train_test_split(x, y, test_size=.5, shuffle=False, random_state=self.hparam["SEED"])

        return torch.Tensor(self.sliding_window(x_train, seq_len=self.hparam["SEQ_LEN"])), \
               torch.Tensor(self.sliding_window(x_val, seq_len=self.hparam["SEQ_LEN"])), \
               torch.Tensor(self.sliding_window(x_test, seq_len=self.hparam["SEQ_LEN"])), \
               torch.Tensor(self.sliding_window(y_train, seq_len=self.hparam["SEQ_LEN"])), \
               torch.Tensor(self.sliding_window(y_val, seq_len=self.hparam["SEQ_LEN"])), \
               torch.Tensor(self.sliding_window(y_test, seq_len=self.hparam["SEQ_LEN"]))


class DataModuleMultiStep(DataModule):
    def __init__(self, hparam:dict):
        super().__init__(hparam)
        self.hparam = hparam

        x_train, x_val, x_test, y_train, y_val, y_test = self.sampler()

        self.train_loader = self.f_train_loader(x_train, y_train)
        self.val_loader = self.f_val_loader(x_val, y_val)
        self.test_loader = self.f_test_loader(x_test, y_test)

    def load_y_df(self, path):
        df_out = pd.read_csv(path, index_col=0)
        df_out = self.scale(df_out)  # TODO remove scaling, if necessary

        # only for theoretical testings to triple the dataset size
        if "theoretical" in self.hparam["LOG_DIR"]:
            df_out = pd.concat([df_out, df_out, df_out], axis=0)

        df_out = df_out.iloc[::self.hparam["DS_SAMPLING"], :].reset_index(drop=True).values


        y_train, y_val = train_test_split(df_out, test_size=.5, shuffle=False, random_state=self.hparam["SEED"])
        y_val, y_test = train_test_split(y_val, test_size=.5, shuffle=False, random_state=self.hparam["SEED"])
        return self.sliding_window(y_train, seq_len=self.hparam["SEQ_LEN"]), \
               self.sliding_window(y_val, seq_len=self.hparam["SEQ_LEN"]), \
               self.sliding_window(y_test, seq_len=self.hparam["SEQ_LEN"])

    def sampler(self):
        df_in = pd.read_csv(self.hparam["DATA_DIR"] + "ds_in.csv", index_col=0)
        df_param = pd.read_csv(self.hparam["DATA_DIR"] + "ds_param.csv", index_col=0)
        df_in, df_param = self.scale(df_in), self.scale(df_param)  # TODO remove scaling, if necessary

        # only for theoretical testings to triple the dataset size
        if "theoretical" in self.hparam["LOG_DIR"]:
            df_in = pd.concat([df_in, df_in, df_in], axis=0)
            df_param = pd.concat([df_param, df_param, df_param], axis=0)

        df_in = df_in.iloc[::self.hparam["DS_SAMPLING"], :].reset_index(drop=True)
        df_param = df_param.iloc[::self.hparam["DS_SAMPLING"], :].reset_index(drop=True)

        x = pd.concat([df_in, df_param], axis=1).values
        x1_train, x1_val = train_test_split(x, test_size=.5, shuffle=False, random_state=self.hparam["SEED"])
        x1_val, x1_test = train_test_split(x1_val, test_size=.5, shuffle=False, random_state=self.hparam["SEED"])
        x1_train, x1_val, x1_test = self.sliding_window(x1_train, seq_len=self.hparam["SEQ_LEN"]), self.sliding_window(x1_val, seq_len=self.hparam["SEQ_LEN"]), self.sliding_window(x1_test, seq_len=self.hparam["SEQ_LEN"])

        y1_train, y1_val, y1_test = self.load_y_df(path=self.hparam["DATA_DIR"] + "ds_out.csv")
        y_train, y_val, y_test = y1_train, y1_val, y1_test
        x_train, x_val, x_test = x1_train, x1_val, x1_test

        try:
            y2_train, y2_val, y2_test = self.load_y_df(path=self.hparam["DATA_DIR"] + "ds_out2.csv")
            try:
                y_train, y_val, y_test = np.stack((y1_train, y2_train), axis=1), np.stack((y1_val, y2_val), axis=1), np.stack((y1_test, y2_test), axis=1)
            except:
                add_dims = y1_train.shape[1] - y2_train.shape[1]
                y2_train, y2_val, y2_test = np.pad(y2_train, ((0,0), (0,add_dims), (0,0)), "constant", constant_values=0), np.pad(y2_val, ((0, 0), (0, add_dims), (0, 0)), "constant", constant_values=0), np.pad(y2_test, ((0, 0), (0, add_dims), (0, 0)), "constant", constant_values=0)
                y_train, y_val, y_test = np.stack((y1_train, y2_train), axis=1), np.stack((y1_val, y2_val), axis=1), np.stack((y1_test, y2_test), axis=1)
        except:
            print("no ds2")
        try:
            y3_train, y3_val, y3_test = self.load_y_df(path=self.hparam["DATA_DIR"] + "ds_out3.csv")
            try:
                y_train, y_val, y_test = np.stack((y1_train, y2_train, y3_train), axis=1), np.stack((y1_val, y2_val, y3_val), axis=1), np.stack((y1_test, y2_test, y3_test), axis=1)
            except:
                add_dims = y1_train.shape[1] - y3_train.shape[1]
                y2_train, y2_val, y2_test = np.pad(y3_train, ((0,0), (0,add_dims), (0,0)), "constant", constant_values=0), np.pad(y3_val, ((0, 0), (0, add_dims), (0, 0)), "constant", constant_values=0), np.pad(y3_test, ((0, 0), (0, add_dims), (0, 0)), "constant", constant_values=0)
                y_train, y_val, y_test = np.stack((y1_train, y2_train, y3_train), axis=1), np.stack((y1_val, y2_val, y3_val), axis=1), np.stack((y1_test, y2_test, y3_test), axis=1)
        except:
            print("no ds3")
        return torch.Tensor(x_train), torch.Tensor(x_val), torch.Tensor(x_test), \
               torch.Tensor(y_train), torch.Tensor(y_val), torch.Tensor(y_test)


class SinglePredModule(DataModuleSingleStep):
    def __init__(self, hparam:dict):
        """
        loading single slices for vis predictions
        :param hparam: [dict] hyperparameters
        """
        super(SinglePredModule, self).__init__()
        self.hparam = hparam

        df_in, df_out, df_param = self.load_df()
        _, _, x_test, _, _, y_test = self.sampler(df_in, df_out, df_param)

        self.pred_loader = self.f_pred_loader(x_test, y_test)


# quick test and quick run
if __name__ == "__main__":
    exp_setup = {
        "EXPERIMENT_ID": "test",
        "MODEL": "pepper_sconv",
        "DATA_DIR": "../data/ode/ds3/",
        "LOG_DIR": "logs/datatest_sconvs/",
        "SEED": 42,
        "SEQ_LEN": 512,
        "BATCH_SIZE": 1024,
        "NUM_WORKERS": 4,
        "MAX_EPOCHS": 500,
        "LEARNING_RATE": 0.0001,
        "WEIGHT_DECAY": 1e-05,
        "CONV_N_FILT": 4,
        "CONV_ENCS": [2, 4, 8, 4, 4],
        "CONV_N_KERNEL": 3,
        "CONV_STRIDE": 2,
        "CONV_DIL": 1,
        "VAR_N_LAT": 4,
        "CORE_ENCS": [1.0, 0.9, 0.8, 0.7],
        "KL_WEIGHT": 0.0001,
        "XY_LOSS_WEIGHT": 0.5,
        "DS_SAMPLING":1
    }

    print(exp_setup)
    #data = DataModuleSingleStep(hparam=exp_setup)
    data = DataModuleMultiStep(hparam=exp_setup)
    train_data = data.train_loader
    val_data   = data.val_loader
    test_data  = data.test_loader

    # print first sample in train_dataloader
    X_train, y_train = [x for x in iter(train_data).next()]
    X_val, y_val = [x for x in iter(val_data).next()]
    X_test, y_test = [x for x in iter(test_data).next()]

    print(len(train_data), len(val_data), len(test_data))

    # Single y out: Tensor(batch, channels, values)
    # Multiple y out: Tensor(batch, step, channels, values)
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_val shape: ", X_val.shape)
    print("y_val shape: ", y_val.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)
    """
    for i, (x, y) in enumerate(train_data):
        print(x.shape)
        print(y.shape)
    """
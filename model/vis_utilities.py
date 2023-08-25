import os
import re
import glob
import torch
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from data_module import *

# import nets
import nets as pn


class VisGen():
    def __init__(self, exp_path, hparam):
        self.exp_path = exp_path
        self.hparam = hparam
        self.ds_path = hparam["DATA_DIR"]

    def read_ds(self, ds_path):
        """
        loading all csv files in a dir and converting them to df
        :return: df
        """
        df = pd.DataFrame()
        df_buffer = pd.DataFrame()
        # appending all csv files from one ds into one dfy
        for file in os.listdir(ds_path):
            if ".csv" in file:
                df_buffer = pd.read_csv(ds_path + file)
                df = pd.concat([df_buffer, df], axis=0)
        return df

    def slice_ds(self): #TODO change to appropriate slicer according to all ds
        df = self.read_ds(self.ds_path)
        df_input, df_output, df_param = DataModule.slicer(self, df=df)
        return {"input": df_input, "output": df_output, "param": df_param}

    def launch_tensorboard(self):
        os.system('tensorboard --logdir="' + self.hparam["LOG_DIR"] + '"')
        return


class VisDs():
    """
    class for visualizing the ds for experiments.
    the exp_path must be passed in the form of "./model/logs/[MODEL]/logs_exp[NO]/"
    """
    def __init__(self, exp_path: str):
        super(VisDs, self).__init__()
        # parameters + results
        self.exp_path = exp_path

        with open(self.exp_path + "results.json", "r") as handle:
            res_dict = json.load(handle)
        self.hparam = res_dict["setup"]
        self.results = res_dict["results"]

        # general
        self.df_dict = VisGen(exp_path=self.exp_path, hparam=self.hparam).slice_ds()
        self.cols_in = self.df_dict["input"].columns
        self.cols_out = self.df_dict["output"].columns
        self.cols_param = self.df_dict["param"].columns

    def plot_input(self):
        self.df_dict["input"].plot(kind="line", subplots=True, title="input", legend=True, xlabel="sample", figsize=(30, 1.5 * len(self.cols_in)))

    def plot_output(self):
        self.df_dict["output"].plot(kind="line", subplots=True, title="output", legend=True, xlabel="sample", figsize=(30, 1.5 * len(self.cols_out)))

    def plot_param(self):
        self.df_dict["param"].plot(kind="line", subplots=True, title="param", legend=True, xlabel="sample", figsize=(30, 1.5 * len(self.cols_param)))


class VisRes():
    """
    class for visualizing the results for experiments in PATH.
    the exp_path must be passed in the form of "./model/logs/[MODEL]/logs_exp[NO]/"

    the class runs a single prediction with prediction dataloader on the saved and loaded model
    the results are visuaized in plots
    """
    def __init__(self, exp_path):
        super(VisRes, self).__init__()
        # parameters + results
        with open(exp_path + "results.json", "r") as handle:
            res_dict = json.load(handle)
        self.hparam = res_dict["setup"]
        self.results = res_dict["results"]

        # paths
        self.exp_path = exp_path
        self.log_path = self.hparam["LOG_DIR"]
        self.data_path = self.hparam["DATA_DIR"]
        self.model_path = exp_path + "model.pth"
        self.results_path = exp_path + "results.json"

        # general
        self.df_dict = VisGen(exp_path=self.exp_path, hparam=self.hparam).slice_ds()
        self.cols_in = list(self.df_dict["input"].columns)
        self.cols_out = list(self.df_dict["output"].columns)
        self.cols_param = list(self.df_dict["param"].columns)

        self.df_x, self.df_x_hat, self.df_y, self.df_y_hat = self.single_pred()
        self.df_dict = {"df_x": self.df_x, "df_x_hat": self.df_x_hat, "df_y": self.df_y, "df_y_hat": self.df_y_hat}

    def single_pred(self):
        random.seed(self.hparam["SEED"])
        np.random.seed(self.hparam["SEED"])
        torch.manual_seed(self.hparam["SEED"])
        torch.cuda.manual_seed(self.hparam["SEED"])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"run on {device}...\n")

        # dataloader for single prediction
        if self.hparam["MODEL"] == "pn_raw_s": # linewise prediction
            dl = SinglePredModule(hparam=self.hparam, flag="single", num_samples=10).gen_samples()
            in_, out_ = [x for x in iter(dl).next()]
        elif self.hparam["MODEL"] != "pn_raw_s":
            dl = SinglePredModule(hparam=self.hparam, flag="multi", num_samples=10).gen_samples()
            in_, out_ = [x for x in iter(dl).next()]

        # TODO append nets
        # load model
        if self.hparam["MODEL"] == "pepper_core":
            model = pn.PepperNetCore(hparam=self.hparam, in_=in_, out_=out_)
        elif self.hparam["MODEL"] == "pepper_varcore":
            model = pn.PepperNetVarCore(hparam=self.hparam, in_=in_, out_=out_)
        elif self.hparam["MODEL"] == "pepper_convcore":
            model = pn.PepperNetConvCore(hparam=self.hparam, in_=in_, out_=out_)
        elif self.hparam["MODEL"] == "pepper_convvar":
            model = pn.PepperNetConvVar(hparam=self.hparam, in_=in_, out_=out_)
        elif self.hparam["MODEL"] == "pepper_convvarcore":
            model = pn.PepperNetConvVarCore(hparam=self.hparam, in_=in_, out_=out_)
        elif self.hparam["MODEL"] == "pepper_convconvcore":
            model = pn.PepperNetConvConvCore(hparam=self.hparam, in_=in_, out_=out_)
        elif self.hparam["MODEL"] == "pepper_convconvvar":
            model = pn.PepperNetConvConvVar(hparam=self.hparam, in_=in_, out_=out_)
        elif self.hparam["MODEL"] == "pepper_convconvvarcore":
            model = pn.PepperNetConvConvVarCore(hparam=self.hparam, in_=in_, out_=out_)
        elif self.hparam["MODEL"] == "pepper_cconvcore":
            model = pn.PepperNetCausalConvCore(hparam=self.hparam, in_=in_, out_=out_)
        elif self.hparam["MODEL"] == "pepper_cconvvar":
            model = pn.PepperNetCausalConvVar(hparam=self.hparam, in_=in_, out_=out_)
        elif self.hparam["MODEL"] == "pepper_cconvvarcore":
            model = pn.PepperNetCausalConvVarCore(hparam=self.hparam, in_=in_, out_=out_)
        elif self.hparam["MODEL"] == "pepper_cconvcconvcore":
            model = pn.PepperNetCausalConvCore2(hparam=self.hparam, in_=in_, out_=out_)
        elif self.hparam["MODEL"] == "pepper_cconvcconvvar":
            model = pn.PepperNetCausalConvVar2(hparam=self.hparam, in_=in_, out_=out_)
        elif self.hparam["MODEL"] == "pepper_cconvcconvvarcore":
            model = pn.PepperNetCausalConvVarCore2(hparam=self.hparam, in_=in_, out_=out_)
        else:
            print("no model selected!\n")
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device(device)))
        model.eval()

        # init tensor shapes
        x_orig, y_orig = [i for i in iter(dl).next()]
        x_hat_all, x_gt_all, y_hat_all, y_gt_all = torch.empty(x_orig.shape[1], x_orig.shape[2]), \
                                                   torch.empty(x_orig.shape[1], x_orig.shape[2]), \
                                                   torch.empty(y_orig.shape[1], y_orig.shape[2]), \
                                                   torch.empty(y_orig.shape[1], y_orig.shape[2])
        # predict
        with torch.no_grad():
            for x, y in dl:
                # init pred tensor shapes
                x_hat, x_gt, y_hat, y_gt = torch.empty(x_orig.shape[0], x_orig.shape[1], 0), \
                                           torch.empty(x_orig.shape[0], x_orig.shape[1], 0), \
                                           torch.empty(y_orig.shape[0], y_orig.shape[1], 0), \
                                           torch.empty(y_orig.shape[0], y_orig.shape[1], 0)
                # predict
                x.to(device)
                pred_x, pred_y = model(x)

                # reshape prediction tensors into original tensor shapes
                x_buff = torch.chunk(torch.squeeze(pred_x), x_orig.shape[2])
                y_buff = torch.chunk(torch.squeeze(pred_y), y_orig.shape[2])
                for i in x_buff:
                    x_hat = torch.cat((x_hat, torch.reshape(i, (x_hat.shape[0], x_hat.shape[1], 1))), dim=2)
                for j in y_buff:
                    y_hat = torch.cat((y_hat, torch.reshape(j, (y_hat.shape[0], y_hat.shape[1], 1))), dim=2)

                # append predictions to *_hat_all prediction tensor ... get rid of batch
                x, x_hat, y, y_hat = torch.squeeze(x, dim=0), torch.squeeze(x_hat, dim=0), torch.squeeze(y, dim=0), torch.squeeze(y_hat, dim=0)
                x_hat_all = torch.cat((x_hat_all, x_hat), dim=0)
                x_gt_all = torch.cat((x_gt_all, x), dim=0)
                y_hat_all = torch.cat((y_hat_all, y_hat), dim=0)
                y_gt_all = torch.cat((y_gt_all, y), dim=0)

        x_hat_all = x_hat_all[self.hparam["SEQ_LEN"]:][:]
        x_gt_all = x_gt_all[self.hparam["SEQ_LEN"]:][:]
        y_hat_all = y_hat_all[self.hparam["SEQ_LEN"]:][:]
        y_gt_all = y_gt_all[self.hparam["SEQ_LEN"]:][:]

        # save as df for individual predictions and stacked predictions
        self.df_x_ind, self.df_x_hat_ind, self.df_y_ind, self.df_y_hat_ind = pd.DataFrame(x), pd.DataFrame(x_hat), pd.DataFrame(y), pd.DataFrame(y_hat)
        df_x, df_x_hat, df_y, df_y_hat = pd.DataFrame(x_gt_all, columns=self.cols_in + self.cols_param), \
                                                             pd.DataFrame(x_hat_all, columns=self.cols_in + self.cols_param), \
                                                             pd.DataFrame(y_gt_all, columns=self.cols_out), \
                                                             pd.DataFrame(y_hat_all, columns=self.cols_out)
        return df_x, df_x_hat, df_y, df_y_hat

    def print_dfs(self, df_name):
        for name in df_name:
            print(self.df_dict[name])

    def plot_gt_x_only(self):
        self.df_x.plot(kind="line", subplots=True, title="ground truth x", legend=True, xlabel="sample", figsize=(30, 1.5 * len(self.df_x_hat.columns)))

    def plot_gt_y_only(self):
        self.df_y.plot(kind="line", subplots=True, title="ground truth y", legend=True, xlabel="sample", figsize=(30, 1.5 * len(self.df_x_hat.columns)))

    def plot_pred_x_only(self):
        self.df_x_hat.plot(kind="line", subplots=True, title="predictions x", legend=True, xlabel="sample", figsize=(30, 1.5 * len(self.df_x_hat.columns)))

    def plot_pred_y_only(self):
        self.df_y_hat.plot(kind="line", subplots=True, title="predictions y", legend=True, xlabel="sample", figsize=(30, 1.5 * len(self.df_y_hat.columns)))

    def plot_pred_x_diff(self):
        for val in self.df_x_hat:
            fig = plt.figure(figsize=(30, 1.5))
            plt.plot(np.array(self.df_x_hat[val]), label=val + " pred")
            plt.plot(np.array(self.df_x[val]), label=val + " gt")
            plt.title("predictions and ground truths for X")
            plt.legend()
            plt.show()

    def plot_pred_y_diff(self):
        for val in self.df_y_hat:
            fig = plt.figure(figsize=(30, 1.5))
            plt.plot(self.df_y_hat[val], label=val + " pred")
            plt.plot(self.df_y[val], label=val + " gt")
            plt.title("predictions and ground truths for y")
            plt.legend()
            plt.show()


class VisTrain():
    """
    Class for visualizing the training progress of the nn
    Can be imported in network class, and applied during training
    """
    def __init__(self, hparam):
        super(VisTrain, self).__init__()
        # parameters
        self.hparam = hparam

        # paths
        self.log_dir = self.hparam["LOG_DIR"] + "logs_exp" + str(self.hparam["EXPERIMENT_ID"]) + "/"
        self.data_dir = self.hparam["DATA_DIR"]
        self.vis_dir = self.create_dir()
        self.model_path = self.log_dir + "model.pth"

    def create_dir(self):
        if not os.path.exists(self.hparam["LOG_DIR"] + "/logs_exp" + str(self.hparam["EXPERIMENT_ID"]) + "/vis"):
            os.makedirs(self.hparam["LOG_DIR"] + "/logs_exp" + str(self.hparam["EXPERIMENT_ID"]) + "/vis")
        return self.hparam["LOG_DIR"] + "/logs_exp" + str(self.hparam["EXPERIMENT_ID"]) + "/vis/"

    def plot_pred_xy(self, x, x_hat, y, y_hat, epoch):
        # must be applied inside training epoch
        # plot only first element of minibatch
        x, x_hat, y, y_hat = x.cpu(), x_hat.cpu(), y.cpu(), y_hat.cpu()
        x, x_hat = x[0].detach().numpy(), x_hat[0].detach().numpy()
        y, y_hat = y[0].detach().numpy(), y_hat[0].detach().numpy()
        fig, (ax1, ax2) = plt.subplots(1,2)

        ax1.plot(x_hat, label="x pred")
        ax1.plot(x, label="x gt")
        ax1.legend(loc="upper right")

        ax2.plot(y_hat, label="y pred")
        ax2.plot(y, label="y gt")
        ax2.legend(loc="upper right")
        plt.suptitle("X, y predictions and ground truth epoch " + str(epoch) + "\n")
        plt.savefig(self.vis_dir + "xy_" + str(epoch) + ".png")
        plt.close()

    def plot_multi_pred_xy(self, x, x_hat, y, y_hat, epoch):
        # must be applied inside training epoch
        # plot only first element of minibatch
        x, x_hat, y, y_hat = x.cpu(), x_hat.cpu(), y.cpu(), y_hat.cpu()
        x, x_hat = x[0].detach().numpy(), x_hat[0].detach().numpy()
        y, y_hat = y[0].detach().numpy(), y_hat[0].detach().numpy()
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(15, 5))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.plot(x_hat, label="x pred")
        ax1.plot(x, label="x gt")
        ax1.legend(loc="upper right")

        ax2.plot(y_hat[:,0].T, label="y1 pred")
        ax2.plot(y[:,0].T, label="y1 gt")
        ax2.legend(loc="upper right")

        ax3.plot(y_hat[:,1].T, label="y2 pred")
        ax3.plot(y[:,1].T, label="y2 gt")
        ax3.legend(loc="upper right")

        """
        ax4.plot(y_hat[:,2].T, label="y3 pred")
        ax4.plot(y[:,2].T, label="y3 gt")
        ax4.legend(loc="upper right")
        """

        plt.suptitle("X, y1, y2, y3 predictions and ground truth epoch " + str(epoch) + "\n")
        plt.savefig(self.vis_dir + "xy_" + str(epoch) + ".png")
        plt.close()

    def plot_pred_x(self, x, x_hat, epoch, flag="x"):
        # plot only first element of minibatch
        x, x_hat = x.cpu(), x_hat.cpu()
        x, x_hat = np.array(x[0]), np.array(x_hat[0])
        fig = plt.figure(figsize=(5,5))
        plt.plot(x_hat, label="x pred")
        plt.plot(x, label="x gt")
        plt.legend(loc="upper right")
        plt.title(flag + " predictions and ground truth, epoch " + str(epoch))
        plt.savefig(self.vis_dir + flag + "_" + str(epoch) + ".png")
        plt.close()

    def create_pred_gif(self, flag="x"):
        # sorting all images in directory for creating the gif
        def to_int(str):
            return int(str) if str.isdigit() else str

        def natural_keys(str):
            return [to_int(c) for c in re.split(r'(\d+)', str)]

        # Create the frames
        frames = []
        imgs = glob.glob(self.vis_dir + "*.png")
        imgs.sort(key=natural_keys)
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)

        # Save into a GIF file that loops forever
        frames[0].save(self.vis_dir + "training_" + flag + ".gif", format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
        return print("\nTraining GIF was saved... \n")



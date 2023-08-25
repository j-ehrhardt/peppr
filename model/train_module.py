import gc
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from eval_module import *
from utils import *
from vis_utilities import *


def train_step(model, dl, device, optimizer, logger, i, e):
    ll2_x, ll2_y = [], [] # list objects to store intermediate errors

    model.train()
    #with tqdm(dl, unit="batch") as tbatch:
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        # Forward
        x_hat, y_hat, z = model(x, y, e)
        l2_x, l2_y, = model.hparam["LAMBDA_X"] * l2_loss(x, x_hat), model.hparam["LAMBDA_Y"] * l2_loss(y, y_hat)

        if "vae" in model.hparam["MODEL"]:
            l2_x += model.hparam["KL_WEIGHT"] * (model.kl_f + model.kl_b)
            l2_y += model.hparam["KL_WEIGHT"] * (model.kl_f + model.kl_b)
        if "inv" in model.hparam["MODEL"]:
            l2_y += model.hparam["LAMBDA_Z"] * kl_div_loss(z, device)

        # Backward
        optimizer.zero_grad()
        l2_x.backward(retain_graph=False)
        l2_y.backward(retain_graph=False)
        optimizer.step()

        # logging
        logger.add_scalar("loss_train_step_x", l2_x, i)
        logger.add_scalar("loss_train_step_y", l2_y, i)
        ll2_x.append(l2_x.item())
        ll2_y.append(l2_y.item())

        #tbatch.set_postfix(loss_x=l2_x.item(), loss_y=l2_y.item())
        i+=1
    return list_mean(ll2_x), list_mean(ll2_y), i


def val_step(model, dl, device, vstr, e):
    l1_x, l1_y, l2_x, l2_y, rmse_x, rmse_y = 0, 0, 0, 0, 0, 0
    num_batches = len(dl)

    model.eval()
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            x_hat, y_hat, _ = model(x, y, e)

            l1_y += l1_loss(y, y_hat).item()
            l1_x += l1_loss(x, x_hat).item()
            l2_y += l2_loss(y, y_hat).item()
            l2_x += l2_loss(x, x_hat).item()
            rmse_y += rmse_loss(y, y_hat).item()
            rmse_x += rmse_loss(x, x_hat).item()

    l1_y /= num_batches
    l1_x /= num_batches
    l2_y /= num_batches
    l2_x /= num_batches
    rmse_y /=num_batches
    rmse_x /=num_batches

    #print(f"Val L2 Error X, MSE: ", l2_x)
    #print(f"Val L2 Error y, MSE: ", l2_y)

    x, y = x.transpose(1, 2), y.transpose(1, 2)
    x_hat, y_hat = x_hat.transpose(1, 2), y_hat.transpose(1, 2)
    # saving pred plots in vis dir
    if not e % 5:
        try:
            vstr.plot_pred_xy(x=x, x_hat=x_hat, y=y, y_hat=y_hat, epoch=e)
        except:
            vstr.plot_multi_pred_xy(x=x, x_hat=x_hat, y=y, y_hat=y_hat, epoch=e)
    return l1_x, l1_y, l2_x, l2_y, rmse_x, rmse_y


def test_step(model, dl, device, vstr, e):
    l1_x, l1_y, l2_x, l2_y, rmse_x, rmse_y = 0, 0, 0, 0, 0, 0
    num_batches = len(dl)

    model.eval()
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            x_hat, y_hat, _ = model(x, y, e)

            l1_y += l1_loss(y, y_hat).item()
            l1_x += l1_loss(x, x_hat).item()
            l2_y += l2_loss(y, y_hat).item()
            l2_x += l2_loss(x, x_hat).item()
            rmse_y += rmse_loss(y, y_hat).item()
            rmse_x += rmse_loss(x, x_hat).item()

        l1_y /= num_batches
        l1_x /= num_batches
        l2_y /= num_batches
        l2_x /= num_batches
        rmse_y /= num_batches
        rmse_x /= num_batches
    print(f"Test Error X, MSE: ", l2_x)
    print(f"Test Error y, MSE: ", l2_y)

    x, y = x.transpose(1, 2), y.transpose(1, 2)
    x_hat, y_hat = x_hat.transpose(1, 2), y_hat.transpose(1, 2)
    try:
        vstr.plot_pred_xy(x=x, x_hat=x_hat, y=y, y_hat=y_hat, epoch=e)
    except:
        vstr.plot_multi_pred_xy(x=x, x_hat=x_hat, y=y, y_hat=y_hat, epoch=e)
    return l1_x, l1_y, l2_x, l2_y, rmse_x, rmse_y



def train_model(hparam: dict):
    # prelim
    random.seed(hparam["SEED"])
    np.random.seed(hparam["SEED"])
    torch.manual_seed(hparam["SEED"])
    torch.cuda.manual_seed(hparam["SEED"])

    # init visualization and loggers
    vstr = VisTrain(hparam=hparam)
    logger = SummaryWriter(log_dir=hparam["LOG_DIR"] + "/logs_exp" + str(hparam["EXPERIMENT_ID"]) + "/")
    fct = AutoStop(tolerance_e=10, min_delta=0.0001, max_delta=0.1)
    i = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"run on {device}...\n")

    #with torch.cuda.device(hparam["DEVICE"]) if torch.cuda.device_count() > 1 else torch.cuda.device(0):  # for server
    # data
    if "multi" in hparam["MODEL"]:
        dl_train = DataModuleMultiStep(hparam).train_loader
        dl_val = DataModuleMultiStep(hparam).val_loader
        dl_test = DataModuleMultiStep(hparam).test_loader
    else:
        dl_train = DataModuleSingleStep(hparam).train_loader
        dl_val = DataModuleSingleStep(hparam).val_loader
        dl_test = DataModuleSingleStep(hparam).test_loader
    in_, out_ = next(iter(dl_train)) #= [x for x in iter(dl_train).next()]

    # models
    if hparam["MODEL"] == "core":
        model = pn.PepperNetCore(hparam=hparam, in_=in_, out_=out_)
    elif hparam["MODEL"] == "vae":
        model = pn.PepperNetVAE(hparam=hparam, in_=in_, out_=out_)
    elif hparam["MODEL"] == "inv":
        model = pn.PepperNetInv(hparam=hparam, in_=in_, out_=out_)

    elif hparam["MODEL"] == "oconv_core":
        model = pn.PepperNetConvCore(hparam=hparam, in_=in_, out_=out_)
    elif hparam["MODEL"] == "oconv_vae":
        model = pn.PepperNetConvVAE(hparam=hparam, in_=in_, out_=out_)
    elif hparam["MODEL"] == "oconv_inv":
        model = pn.PepperNetConvInv(hparam=hparam, in_=in_, out_=out_)

    elif hparam["MODEL"] == "sconv_core":
        model = pn.PepperNetSConvCore(hparam=hparam, in_=in_, out_=out_)
    elif hparam["MODEL"] == "sconv_vae":
        model = pn.PepperNetSConvVAE(hparam=hparam, in_=in_, out_=out_)
    elif hparam["MODEL"] == "sconv_inv":
        model = pn.PepperNetSConvInv(hparam=hparam, in_=in_, out_=out_)

    elif hparam["MODEL"] == "cconv_core":
        model = pn.PepperNetCConvCore(hparam=hparam, in_=in_, out_=out_)
    elif hparam["MODEL"] == "cconv_vae":
        model = pn.PepperNetCConvVAE(hparam=hparam, in_=in_, out_=out_)
    elif hparam["MODEL"] == "cconv_inv":
        model = pn.PepperNetCConvInv(hparam=hparam, in_=in_, out_=out_)

    # multi models
    elif hparam["MODEL"] == "multi_core2":
        model = pn.PepperNetMultiCore2(hparam=hparam, in_=in_, out_=out_, n_=2)
    elif hparam["MODEL"] == "multi_vae2":
        model = pn.PepperNetMultiVAE2(hparam=hparam, in_=in_, out_=out_, n_=2)
    elif hparam["MODEL"] == "multi_inv2":
        model = pn.PepperNetMultiInv2(hparam=hparam, in_=in_, out_=out_, n_=2)

    elif hparam["MODEL"] == "multi_core3":
        model = pn.PepperNetMultiCore3(hparam=hparam, in_=in_, out_=out_, n_=3)
    elif hparam["MODEL"] == "multi_vae3":
        model = pn.PepperNetMultiVAE3(hparam=hparam, in_=in_, out_=out_, n_=3)
    elif hparam["MODEL"] == "multi_inv3":
        model = pn.PepperNetMultiInv3(hparam=hparam, in_=in_, out_=out_, n_=3)

    # do fancy weight initialization
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparam["LEARNING_RATE"], weight_decay=hparam["WEIGHT_DECAY"])
    #optimizer = torch.optim.SGD(model.parameters(), lr=hparam["LEARNING_RATE"], momentum=0.1, dampening=0.01, weight_decay=hparam["WEIGHT_DECAY"])
    model.to(device)

    print(hparam, "\n\n")

    print(f"**********************************************************************\nModel structure: {model}\n")
    print(f"Number of parameters: {sum(param.numel() for param in model.parameters())}")
    print("***********************************************************************\n")

    # list objects for loss loggings
    ll2_x_train, ll2_y_train = [], []
    ll1_x_val, ll1_y_val, ll2_x_val, ll2_y_val, rmsel_x_val, rmsel_y_val = [], [], [], [], [], []

    with tqdm(range(hparam["MAX_EPOCHS"]), unit="epoch") as tepoch:
        for e in tepoch:
            #print("Epoch " + str(e + 1) + " of " + str(hparam["MAX_EPOCHS"]))
            # train step and logging
            l2_x_train, l2_y_train, i = train_step(model=model, dl=dl_train, device=device, optimizer=optimizer, logger=logger, i=i, e=e)
            logger.add_scalar("loss_train_epoch_x", l2_x_train, e)
            logger.add_scalar("loss_train_epoch_y", l2_y_train, e)
            ll2_x_train.append(l2_x_train)
            ll2_y_train.append(l2_y_train)

            # val step and logging
            l1_x_val, l1_y_val, l2_x_val, l2_y_val, rmse_x_val, rmse_y_val = val_step(model=model, dl=dl_val, device=device, vstr=vstr, e=e)
            logger.add_scalar("loss_val_epoch_x", l2_x_val, e)
            logger.add_scalar("loss_val_epoch_y", l2_y_val, e)
            ll1_x_val.append(l1_x_val)
            ll1_y_val.append(l1_y_val)
            ll2_x_val.append(l2_x_val)
            ll2_y_val.append(l2_y_val)
            rmsel_x_val.append(rmse_x_val)
            rmsel_y_val.append(rmse_y_val)
            tepoch.set_postfix(loss_x=rmse_x_val, loss_y=rmse_y_val)
            if fct.auto_stop(rmse_x_val, rmse_y_val):
                break

    # test step
    l1_x_test, l1_y_test, l2_x_test, l2_y_test, rmse_x_test, rmse_y_test = test_step(model=model, dl=dl_test, device=device, vstr=vstr, e=e)

    logger.flush()
    logger.close()
    torch.save(model.state_dict(), hparam["LOG_DIR"] + "/logs_exp" + str(hparam["EXPERIMENT_ID"]) + "/model.pth")
    save_results(hparam=hparam, metrics={"l2_train_X":list_mean(ll2_x_train), "l2_train_y":list_mean(ll2_y_train),
                                         "l1_val_x":list_mean(ll1_x_val), "l1_val_y":list_mean(ll1_y_val),
                                         "l2_val_x":list_mean(ll2_x_val), "l2_val_y":list_mean(ll2_y_val),
                                         "rmse_val_x":list_mean(rmsel_x_val), "rmse_val_y":list_mean(rmsel_y_val),
                                         "l1_test_x":l1_x_test, "l1_test_y":l1_y_test, "l2_test_x":l2_x_test,
                                         "l2_test_y":l2_y_test, "rmse_test_x":rmse_x_test, "rmse_test_y":rmse_y_test})
    vstr.create_pred_gif(flag="xy")

    # freeing memory on GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()


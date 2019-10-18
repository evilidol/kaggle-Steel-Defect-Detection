from Dataset import TGS_Dataset
from Models import UNetResNet34, UNetResNet34_SE_Hyper, UNetResNet34_SE_Hyper_v2, UNetResNet34_SE, UNetResNet34_SE_Hyper_SPP, UNetResNet50_SE, FPNetResNet34, RefineNetResNet34
from unet_model import Res34Unetv4
from Network import SegmentationNetwork
from contextlib import contextmanager
import time
import os
from model import Unet

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

##############################
TRAIN_PATH = '/home/wdx/Downloads/severstal-steel-defect-detection'
AUX_PATH = './Data/auxiliary_data'
# LOAD_PATHS = None
LOAD_PATHS = [None, None, None, None, './Saves/bceLoss_1e-2/2019-09-23 01:10_Fold5_Epoch66_reset0_val0.528']

# LOAD_PATHS = None
DEBUG = False
##############################
# LOSS = 'lovasz'
LOSS = 'bce'
# OPTIMIZER = 'SGD'
OPTIMIZER = 'Adam'
PRETRAINED = True
N_EPOCH = 150
BATCH_SIZE = 8
MODEL = Res34Unetv4
ACTIVATION = 'relu'
###########OPTIMIZER###########
# LR = 5e-2
LR = 5e-5
USE_SCHEDULER = 'CosineAnneling'
MILESTONES = [20, 40, 75]
GAMMA = 0.5
PATIENCE = 10
T_MAX = 70
T_MUL = 1
LR_MIN = 0
##############################
COMMENT = 'SGDR (Tmax40, Tmul1), Lovasz, relu, pretrained'

if __name__ == '__main__':
    train_dataset = TGS_Dataset(TRAIN_PATH, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # train_dataset.load_images()
    loaders = train_dataset.yield_dataloader(num_workers=0, batch_size=BATCH_SIZE,
                                                  # auxiliary_df=TGS_Dataset.create_dataset_df(AUX_PATH)
                                                  )

    for i, (train_loader, val_loader) in enumerate(loaders, 1):
        with timer('Fold {}'.format(i)):
            if i < 5:
                continue
            # net = NET(lr=LR, debug=DEBUG, pretrained=PRETRAINED, fold=i, activation=ACTIVATION, comment=COMMENT)
            # model = MODEL(classes=4)
            model = Unet("resnet34", encoder_weights="imagenet", classes=5, activation=None)
            train = SegmentationNetwork(model, lr=LR, debug=DEBUG, fold=i, comment=COMMENT)
            train.define_criterion(LOSS)
            train.create_optmizer(optimizer=OPTIMIZER, use_scheduler=USE_SCHEDULER, milestones=MILESTONES,
                                gamma=GAMMA, patience=PATIENCE, T_max=T_MAX, T_mul=T_MUL, lr_min=LR_MIN)

            # if LOAD_PATHS is not None:
            #     if LOAD_PATHS[i - 1] is not None:
            #         train.load_model(LOAD_PATHS[i - 1])
            #         print("Flod_{} mode is loaded!".format(i))

            train.train_network(train_loader, val_loader, n_epoch=N_EPOCH)
            train.plot_training_curve(show=True)




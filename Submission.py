import gc
import os
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from Dataset import TGS_Dataset
from Evaluation import do_length_decode, do_length_encode
from Models import UNetResNet34_SE, UNetResNet34_SE_Hyper, UNetResNet34_SE_Hyper_v2, UNetResNet34_SE_Hyper_SPP
from Augmentation import do_horizontal_flip
from model import Unet
from Network import SegmentationNetwork
from rle_mask_utils import mask2rle, make_mask, post_process

# UNTESTED, NEED TO COMPARE WITH NATIVE MEAN
def average_fold_predictions(path_list, H=101, W=101, fill_value=255, threshold=0.5):
    '''Load rle from df, average them and return a new rle df'''
    folds = []
    # decode
    for p in path_list:
        df = pd.read_csv(p)
        im = []
        for i in range(len(df)):
            im.append(do_length_decode(str(df.rle_mask.iloc[i]), H, W, fill_value))
        folds.append(im)
    # average
    avg = np.mean(folds, axis=0)
    avg = avg > threshold
    # encode
    rle = []
    for i in range(len(avg)):
        rle.append(do_length_encode(avg[i]))
    # create sub
    df = pd.DataFrame(dict(id=df.id, rle_mask=rle))
    return df

def load_net_and_predict(net, folder_path, model_path, batch_size=32, tta_transform=None, threshold=0.5, min_size=3000):
    test_dataset = TGS_Dataset(folder_path, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # test_dataset.load_images(data='test')
    test_loader = test_dataset.yield_dataloader(data='test', num_workers=0, batch_size=batch_size)
    # predict
    for i in tqdm(range(len(model_path))):
        net.load_model(model_path[i])
        p = net.predict(test_loader, threshold=0.45, tta_transform=tta_transform, return_rle=False)
        if not i:
            avg = np.zeros_like(p['pred'])

        avg = (i * avg + p['pred']) / (i + 1)

    # avg = avg > threshold
    # if min_size > 0:
    #     for j in range(len(avg)):
    #         if avg[j].sum() <= min_size:
    #             avg[i] = avg[i] * 0
    predictions = []
    for fname, batch_pred in zip(p['ids'], avg):
        for cls, pred in enumerate(batch_pred):
            # pred = augmented(image=pred)['image']
            pred, num = post_process(pred, threshold, min_size)
            rle = mask2rle(pred)
            name = fname + f"_{cls + 1}"
            predictions.append([name, rle])

    # free some memory
    del test_dataset, test_loader
    gc.collect()
    # encode
    rle = []
    for i in range(len(avg)):
        rle.append(do_length_encode(avg[i]))
    # create sub
    df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
    return df

def tta_transform(images, mode):
    out = []
    if mode == 'out':
        images = images[0]
    images = images.transpose((0, 2, 3, 1))  # (n, h, w, c)
    tta = []
    for i in range(len(images)):
        t = np.fliplr(images[i])
        tta.append(t)
    tta = np.transpose(tta, (0, 3, 1, 2))
    out.append(tta)
    return np.asarray(out)

if __name__ == '__main__':
    FOLDER_PATH = '/home/wdx/Downloads/severstal-steel-defect-detection'
    DEBUG = False
    # net = UNetResNet34_SE_Hyper_SPP(debug=DEBUG)
    model = Unet("resnet34", encoder_weights="imagenet", classes=4, activation=None)
    train = SegmentationNetwork(model, lr=0.5, debug=DEBUG)
    NET_NAME = type(model).__name__
    THRESHOLD = 0.45
    MIN_SIZE = 3000
    BATCH_SIZE = 8

    LOAD_PATHS = [
        './Saves/bceLoss_1e-2/2019-09-23 04:26_Fold5_Epoch97_reset1_val0.545'
    ]


    ################################################
    # df = average_fold_predictions(SUB_PATHS)
    df = load_net_and_predict(train, FOLDER_PATH, LOAD_PATHS,
                              tta_transform=tta_transform,
                              batch_size=BATCH_SIZE,
                              threshold=THRESHOLD,
                              min_size=MIN_SIZE)
    # SUB_PATHS = ['/media/data/Kaggle/Kaggle-TGS-Salt-Identification/Saves/UNetResNet34_SE_PPM/sub_fold0_val0.792.csv']
    df.to_csv(os.path.join(
        './Saves',
        NET_NAME,
        '{}_5foldAvg.csv'.format(NET_NAME)),
        index=False)
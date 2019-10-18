import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, sampler
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.torch import ToTensor
from sklearn.model_selection import KFold, StratifiedKFold
from Augmentation import *
from rle_mask_utils import mask2rle, make_mask, make_mask_cls


# IM_SIZE = 101

class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask_cls(idx, self.df)
        if self.phase == 'train':
            image_path = os.path.join(self.root, "train_images", image_id)
        else:
            image_path = os.path.join(self.root, "test_images", image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']  # 4x256x1600
        mask = augmented['mask']  # 256x1600x4
        # mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        mask = mask[0]
        return img, mask

    def __len__(self):
        return len(self.fnames)


class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, df, mean, std, phase):
        self.root = root
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transforms = get_transforms(phase, mean, std)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image_path = os.path.join(self.root, "test_images", fname)
        image = cv2.imread(image_path)
        images = self.transforms(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(),  # only horizontal flip as of now
                Resize(256, 512),
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            Resize(256, 512),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


class TGS_Dataset():

    def __init__(self, folder_path, mean, std):
        self.folder_path = folder_path
        self.mean = mean
        self.std = std
        self.df_train = self.create_dataset_df(self.folder_path, phase='train')
        self.df_test = self.create_dataset_df(self.folder_path, phase='test')
        try:
            empty = np.array([np.sum(m) for m in self.df.masks])
            print('{} empty masks out of {} total masks'.format(np.sum(empty == 0), len(empty)))
        except AttributeError:
            pass


    @staticmethod
    def create_dataset_df(folder_path, load=True, phase='train'):
        '''Create a dataset for a specific dataset folder path'''
        # Walk and get paths
        walk = os.walk(folder_path)
        main_dir_path, subdirs_path, csv_path = next(walk)
        dir_im_path, _, im_path = next(walk)
        # Create dataframe
        # df = pd.DataFrame()
        # df['id'] = [im_p.split('.')[0] for im_p in im_path]
        # df['im_path'] = [os.path.join(dir_im_path, im_p) for im_p in im_path]
        train_csv_path = os.path.join(main_dir_path, csv_path[0])
        test_csv_path = os.path.join(main_dir_path, csv_path[1])
        df_train = pd.read_csv(train_csv_path)
        # some preprocessing
        # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
        df_train['ImageId'], df_train['ClassId'] = zip(*df_train['ImageId_ClassId'].str.split('_'))
        df_train['ClassId'] = df_train['ClassId'].astype(int)
        df_train = df_train.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
        df_train['defects'] = df_train.count(axis=1)

        df_test = pd.read_csv(test_csv_path)
        df_test['ImageId'] = df_test['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

        if phase == 'train':
            return df_train
        else:
            return df_test

    def yield_dataloader(self, data='train', nfold=5,
                         shuffle=True, seed=143, stratify=True,
                         num_workers=8, batch_size=10, auxiliary_df=None):

        if data == 'train':
            if stratify:
                kf = StratifiedKFold(n_splits=nfold,
                                     shuffle=True,
                                     random_state=seed)
            else:
                kf = KFold(n_splits=nfold,
                           shuffle=True,
                           random_state=seed)
            loaders = []
            idx = []
            for train_ids, val_ids in kf.split(self.df_train.index, self.df_train['defects'].values):
                train_dataset = SteelDataset(self.df_train.iloc[train_ids], self.folder_path, self.mean, self.std, phase=data)
                train_loader = DataLoader(train_dataset,
                                          shuffle=shuffle,
                                          num_workers=num_workers,
                                          batch_size=batch_size,
                                          pin_memory=False)

                val_dataset = SteelDataset(self.df_train.iloc[val_ids], self.folder_path, self.mean, self.std, phase=data)
                val_loader = DataLoader(val_dataset,
                                        shuffle=shuffle,
                                        num_workers=num_workers,
                                        batch_size=batch_size,
                                        pin_memory=False)
                # idx.append((self.df.id.iloc[train_ids], self.df.id.iloc[val_ids]))
                loaders.append((train_loader, val_loader))
            return loaders

        elif data == 'test':
            test_dataset = TestDataset(self.folder_path, self.df_test, self.mean, self.std, phase=data)
            test_loader = DataLoader(test_dataset,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     batch_size=batch_size,
                                     pin_memory=False)
            return test_loader

    def visualize_sample(self, sample_size):
        samples = np.random.choice(len(self.df_train), sample_size)
        # self.df.set_index('id', inplace=True)
        fig, axs = plt.subplots(5, sample_size)
        for i in range(sample_size):
            image_id, mask = make_mask(samples[i], self.df_train)
            image_path = os.path.join(self.folder_path, "train_images", image_id)
            im = cv2.imread(image_path, cv2.IMREAD_COLOR)
            print('Image shape: ', np.array(im).shape)
            # print('Mask shape: ', np.array(mask).shape)
            axs[0, i].imshow(im)
            axs[1, i].imshow(mask[:, :, 0])
            axs[2, i].imshow(mask[:, :, 1])
            axs[3, i].imshow(mask[:, :, 2])
            axs[4, i].imshow(mask[:, :, 3])

    def visualize_sample_cls(self, sample_size):
        transforms = get_transforms('train', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        samples = np.random.choice(len(self.df_train), sample_size)
        # self.df.set_index('id', inplace=True)
        fig, axs = plt.subplots(6, sample_size)
        # plt.suptitle(u'Lovász-Softmax training')
        for i in range(sample_size):
            image_id, mask = make_mask(samples[i], self.df_train)
            _, mask_cls = make_mask_cls(samples[i], self.df_train)
            image_path = os.path.join(self.folder_path, "train_images", image_id)
            im = cv2.imread(image_path, cv2.IMREAD_COLOR)
            augment = transforms(image=im, mask=mask)
            augment_cls = transforms(image=im, mask=mask_cls)
            # mask = augment['mask'][0].permute(2, 0, 1)
            mask = augment['mask'][0]
            mask_cls = augment_cls['mask'][0]
            im = augment['image'].permute(1, 2, 0)
            # print('Image shape: ', np.array(im).shape)
            print('Mask shape: ', np.array(mask_cls).shape)
            axs[0, i].imshow(im)
            axs[0, i].set_title(image_id)
            axs[1, i].imshow(mask[:, :, 0])
            axs[2, i].imshow(mask[:, :, 1])
            axs[3, i].imshow(mask[:, :, 2])
            axs[4, i].imshow(mask[:, :, 3])
            axs[5, i].imshow(mask_cls[:, :])
            # print(mask_cls)


if __name__ == '__main__':
    TRAIN_PATH = '/home/wdx/Downloads/severstal-steel-defect-detection'

    dataset = TGS_Dataset(TRAIN_PATH, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    dataset.visualize_sample_cls(3)
    # loaders = dataset.yield_dataloader(data='train', nfold=5,
    #                                         shuffle=True, seed=143,
    #                                         num_workers=0, batch_size=4)
    # ids = []
    # for i in loaders[0][0]:
    #     ids.append(i)
    #
    # print(len(ids))
    plt.show()

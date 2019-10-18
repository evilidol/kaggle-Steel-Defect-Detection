import torch.optim as optim
import torch.nn as nn
import torch

from contextlib import contextmanager
import datetime
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Loss
import Scheduler
from Analysis import show_image_mask, show_image_mask_pred, show_image_tta_pred
from Evaluation import  do_kaggle_metric, dice_accuracy, do_mAP, batch_encode, unpad_im
from rle_mask_utils import mask2rle, make_mask, post_process

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.3f}s".format(title, time.time() - t0))


class SegmentationNetwork(nn.Module):

    def __init__(self, model, lr=0.005, fold=None, debug=False, val_mode='max', comment=''):
        super(SegmentationNetwork, self).__init__()
        self.lr = lr
        self.fold = fold
        self.debug = debug
        self.scheduler = None
        self.best_model_path = None
        self.epoch = 0
        self.val_mode = val_mode
        self.comment = comment
        self.model = model.cuda()

        if self.val_mode == 'max':
            self.best_metric = -np.inf
        elif self.val_mode == 'min':
            self.best_metric = np.inf

        self.train_log = dict(loss=[], iou=[], mAP=[])
        self.val_log = dict(loss=[], iou=[], mAP=[])
        self.create_save_folder()
        self.learn_rate = []

    def create_optmizer(self, optimizer='SGD', use_scheduler=None, gamma=0.25, patience=4,
                        milestones=None, T_max=10, T_mul=2, lr_min=0):
        self.cuda()
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                  lr=self.lr, momentum=0.9, weight_decay=0.0001)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                       self.parameters()), lr=self.lr)

        if use_scheduler == 'ReduceOnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                  mode='max',
                                                                  factor=gamma,
                                                                  patience=patience,
                                                                  verbose=True,
                                                                  threshold=0.01,
                                                                  min_lr=1e-05,
                                                                  eps=1e-08)

        elif use_scheduler == 'Milestones':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=milestones,
                                                            gamma=gamma,
                                                            last_epoch=-1)

        elif use_scheduler == 'CosineAnneling':
            self.scheduler = Scheduler.CosineAnnealingLR(self.optimizer,
                                                         T_max=T_max,
                                                         T_mul=T_mul,
                                                         lr_min=lr_min,
                                                         val_mode=self.val_mode,
                                                         last_epoch=-1,
                                                         save_snapshots=True)


    def train_network(self, train_loader, val_loader, n_epoch=10):
        print('Model created, total of {} parameters'.format(
            sum(p.numel() for p in self.parameters())))
        while self.epoch < n_epoch:
            self.epoch += 1
            lr = np.mean([param_group['lr'] for param_group in self.optimizer.param_groups])
            self.learn_rate.append(lr)
            with timer('Train Epoch {:}/{:} - LR: {:.3E}'.format(self.epoch, n_epoch, lr)):
                # Training step
                train_loss, train_iou, train_mAP = self.training_step(train_loader)
                # train_loss, train_iou, train_mAP = (0, 0, 0)
                #  Validation
                val_loss, val_iou, val_mAP = self.perform_validation(val_loader)
                # val_loss, val_iou, val_mAP = (0, 0, 0)
                # Learning Rate Scheduler
                if self.scheduler is not None:
                    if type(self.scheduler).__name__ == 'ReduceLROnPlateau':
                        self.scheduler.step(np.mean(val_mAP))
                    elif type(self.scheduler).__name__ == 'CosineAnnealingLR':
                        self.scheduler.step(self.epoch,
                                            save_dict=dict(metric=np.mean(val_mAP),
                                                           save_dir=self.save_dir,
                                                           fold=self.fold,
                                                           state_dict=self.state_dict()))
                    else:
                        self.scheduler.step(self.epoch)
                # Save best model
                if type(self.scheduler).__name__ != 'CosineAnnealingLR':
                    self.save_best_model(np.mean(val_mAP))

            # Print statistics
            print(('train loss: {:.3f}  val_loss: {:.3f}  '
                   'train iou:  {:.3f}  val_iou:  {:.3f}  '
                   'train mAP:  {:.3f}  val_mAP:  {:.3f}').format(
                np.mean(train_loss),
                np.mean(val_loss),
                np.mean(train_iou),
                np.mean(val_iou),
                np.mean(train_mAP),
                np.mean(val_mAP)))

        self.save_training_log()

    def training_step(self, train_loader):
        # self.set_mode('train')
        self.model.train()
        train_loss = []
        train_iou = []
        train_mAP = []
        for i, (images, targets) in enumerate(train_loader):
            self.optimizer.zero_grad()
            images = images.cuda()
            targets = targets.long().cuda()
            # logit = self.forward(im, z)  self.model
            logit = self.model(images)
            pred = torch.sigmoid(logit)
            # pred = torch.softmax(logit, dim=1)
            pred, pred_ind = pred.data.max(1)

            loss = self.criterion(logit, targets)
            iou  = dice_accuracy(pred, targets, is_average=False).cpu().numpy()
            mAP = do_mAP(pred.data.cpu().numpy(), targets.cpu().numpy(), is_average=False)

            train_loss.append(loss.item())
            train_iou.extend(iou)
            train_mAP.extend(mAP)

            loss.backward()
            self.optimizer.step()

            if self.debug and not self.epoch % 5 and not i % 30:
            # if self.debug and self.epoch == 1 and i == 2:
                show_image_mask_pred(
                    images.cpu().data.numpy(), targets.cpu().data.numpy(), logit.cpu().data.numpy())
        # Append epoch data to metrics dict
        for metric in ['loss', 'iou', 'mAP']:
            self.train_log[metric].append(np.mean(eval('train_{}'.format(metric))))
        return train_loss, train_iou, train_mAP


    def perform_validation(self, val_loader):
        # self.set_mode('valid')
        self.model.eval()
        val_loss = []
        val_iou = []
        val_mAP = []
        for images, targets in val_loader:
            images = images.cuda()
            targets = targets.long().cuda()

            with torch.no_grad():
                logit = self.model(images)
                pred = torch.sigmoid(logit)
                # pred = torch.softmax(logit, dim=1)
                pred, pred_ind = pred.data.max(1)
                loss = self.criterion(logit, targets)
                iou  = dice_accuracy(pred, targets, is_average=False).cpu().numpy()
                mAP = do_mAP(pred.cpu().numpy(), targets.cpu().numpy(), is_average=False)

            val_loss.append(loss.item())
            val_iou.extend(iou)
            val_mAP.extend(mAP)
        # Append epoch data to metrics dict
        for metric in ['loss', 'iou', 'mAP']:
            self.val_log[metric].append(np.mean(eval('val_{}'.format(metric))))

        return val_loss, val_iou, val_mAP


    def predict(self, test_loader, return_rle=False, tta_transform=None, threshold=0.45, min_size=3500):
        # self.set_mode('test')
        self.model.eval()
        self.cuda()
        predictions = []
        for i, (fnames, img) in enumerate(test_loader):
            with torch.no_grad():
                # Apply TTA and predict
                batch_preds = []
                # TTA
                if tta_transform is not None:
                    tta_list = torch.FloatTensor(tta_transform(img.cpu().numpy(), mode='in'))
                    tta_pred = []
                    for t_im in tta_list:
                        t_im = t_im.cuda()
                        t_logit = self.model(t_im)
                        pred = torch.sigmoid(t_logit)
                        # pred = unpad_im(pred.cpu().numpy())
                        pred = pred.cpu().numpy()
                        tta_pred.append(pred)
                    batch_preds.extend(tta_transform(tta_pred, mode='out'))

                # Predict original batch
                img = img.cuda()
                logit = self.model(img)
                pred = torch.sigmoid(logit)
                # pred = unpad_im(pred.cpu().numpy())
                pred = pred.cpu().numpy()
                batch_preds.append(pred)

                # Average TTA results
                batch_preds = np.mean(batch_preds, 0)
                # Threshold result
                # for fname, batch_pred in zip(fnames, batch_preds):
                #     for cls, pred in enumerate(batch_pred):
                #         # pred = augmented(image=pred)['image']
                #         pred, num = post_process(pred, threshold, min_size)
                #         rle = mask2rle(pred)
                #         name = fname + f"_{cls + 1}"
                #         predictions.append([name, rle])

                if not i:
                    out = batch_preds
                    ids = fnames
                else:
                    out = np.concatenate([out, batch_preds], axis=0)
                    ids = np.concatenate([ids, fnames], axis=0)
                # if self.debug:
                #     show_image_tta_pred(
                #         img.cpu().data.numpy(), t_im.cpu().data.numpy(),
                #         logit.cpu().data.numpy(), t_logit.cpu().data.numpy())

        # save predictions to submission.csv
        # df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
        # df.to_csv("submission.csv", index=False)

        out = dict(ids=ids, pred=out)
        return out


    def define_criterion(self, name):
        if name.lower() == 'bce+dice':
            self.criterion = Loss.BCE_Dice()
        elif name.lower() == 'dice':
            self.criterion = Loss.DiceLoss()
        elif name.lower() == 'bce':
            # self.criterion = nn.BCEWithLogitsLoss()
            # self.criterion = nn.CrossEntropyLoss(size_average=False, reduction='sum')
            self.criterion = nn.CrossEntropyLoss()
            # self.criterion = Loss.CE_SOFT()
        elif name.lower() == 'robustfocal':
            self.criterion = Loss.RobustFocalLoss2d()
        elif name.lower() == 'lovasz-hinge' or name.lower() == 'lovasz':
            self.criterion = Loss.Lovasz_Hinge(per_image=True)
        elif name.lower() == 'bce+lovasz':
            self.criterion = Loss.BCE_Lovasz(per_image=True)
        else:
            raise NotImplementedError('Loss {} is not implemented'.format(name))


    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


    def save_best_model(self, metric):
        if (self.val_mode == 'max' and metric > self.best_metric) or (self.val_mode == 'min' and metric < self.best_metric):
            # Update best metric
            self.best_metric = metric
            # Remove old file
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
            # Save new best model weights
            date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
            if self.fold is not None:
                self.best_model_path = os.path.join(
                    self.save_dir,
                    '{:}_Fold{:}_Epoch{}_val{:.3f}'.format(date, self.fold, self.epoch, metric))
            else:
                self.best_model_path = os.path.join(
                    self.save_dir,
                    '{:}_Epoch{}_val{:.3f}'.format(date, self.epoch, metric))

            torch.save(self.state_dict(), self.best_model_path)


    def save_training_log(self):
        d = dict()
        for tk, vk in zip(self.train_log.keys(), self.val_log.keys()):
            d['train_{}'.format(tk)] = self.train_log[tk]
            d['val_{}'.format(vk)] = self.val_log[vk]

        df = pd.DataFrame(d)
        df.index += 1
        df.index.name = 'Epoch'

        date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
        if self.fold is not None:
            p = os.path.join(
                self.save_dir,
                '{:}_Fold{:}_TrainLog.csv'.format(date, self.fold))
        else:
            p = os.path.join(
                self.save_dir,
                '{:}_TrainLog.csv'.format(date))

        df.to_csv(p, sep=";")

        with open(p, 'a') as fd:
            fd.write(self.comment)


    def load_model(self, path=None, best_model=False):
        if best_model:
            self.load_state_dict(torch.load(self.best_model_path))
        else:
            self.load_state_dict(torch.load(path))

    def create_save_folder(self):
        name = type(self).__name__
        self.save_dir = os.path.join('./Saves', name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def plot_training_curve(self, show=True):
        fig, axs = plt.subplots(1, 4)
        for i, metric in enumerate(['loss', 'iou', 'mAP']):
            axs[i].plot(self.train_log[metric], 'ro-', label='Train')
            axs[i].plot(self.val_log[metric], 'bo-', label='Validation')
            axs[i].legend()
            axs[i].set_title(metric)
            axs[i].set_xlabel('Epochs')
            axs[i].set_ylabel(metric)
        axs[3].plot(self.learn_rate)
        axs[3].set_title('learning_rate')
        axs[3].set_xlabel('Epochs')
        if show:
            plt.show()

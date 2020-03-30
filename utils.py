
import os
import glob

import numpy as np
from skimage import io

try:
    import torch
except:
    pass

import shutil
from collections import Iterable
import matplotlib.pyplot as plt
from slacker import Slacker
import argparse

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0 # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val!=None: # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val**2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2/self.count - self.avg**2)
        else:
            pass

class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.',v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log

def save_checkpoint(state, is_best, work_dir, filename='checkpoint.pth'):
    checkpoint_path = os.path.join(work_dir, filename)
    if is_best:
        torch.save(state, checkpoint_path)
        shutil.copyfile(checkpoint_path,
                        os.path.join(work_dir, 'model_best.pth'))



def load_exam(exam_dir, ftype='png'):

    file_extension = '.'.join(['*', ftype])
    data_paths = glob.glob(os.path.join(exam_dir, file_extension))
    data_paths = sorted(data_paths, key=lambda x: x.split('/')[-1]) # sort by filename

    slices = []
    for data_path in data_paths:
        arr = io.imread(data_path)
        slices.append(arr)

    data_3d = np.stack(slices)

    return data_3d






def draw_curve(work_dir,logger1,logger2,labelname=None):

    logger1 = logger1.read()
    logger2 = logger2.read()


    if len(logger1[0]) == 3:
        epoch, trn_loss1, iou1 = zip(*logger1)
        epoch, trn_loss2, iou2 = zip(*logger2)
    elif len(logger1[0]) == 6:

        epoch, trn_loss1, iou1, dice1,acd1,asd1 = zip(*logger1)

        if len(logger1[1]) == 4:
            epoch, trn_loss2, iou2, dice2 = zip(*logger2)

        else:
            epoch, trn_loss2, iou2, dice2,acd2,asd2 = zip(*logger2)


    elif len(logger1[0]) == 7:

        epoch, trn_loss1,embedding_loss1,iou1, dice1,acd1,asd1 = zip(*logger1)
        epoch, trn_loss2, iou2, dice2,acd2,asd2 = zip(*logger2)

    elif len(logger1[0]) == 8:

        epoch, trn_loss1,bce_loss,dice_loss,iou1, dice1,acd1,asd1 = zip(*logger1)
        epoch, trn_loss2, iou2, dice2,acd2,asd2 = zip(*logger2)


    else :
        epoch, trn_loss1, iou1, dice1 = zip(*logger1)
        epoch, trn_loss2, iou2, dice2 = zip(*logger2)

    plt.figure(1)
    plt.plot(epoch, trn_loss1, '-b', label='train_total_loss')
    plt.plot(epoch, trn_loss2, '-r', label='val_loss')
    if len(logger1[0]) == 7:
        plt.plot(epoch, embedding_loss1, '-g', label='train_embedding_loss')
    elif len(logger1[0]) == 8:
        plt.plot(epoch, embedding_loss1, '-g', label='train_bce_loss')
        plt.plot(epoch, embedding_loss1, '-y', label='train_dice_loss')

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('compare_loss')
    plt.savefig(os.path.join(work_dir, 'trn_loss.png'))


    plt.figure(2)
    plt.plot(epoch, iou1, '-b', label='train-{}'.format(labelname))
    plt.plot(epoch, iou2, '-r', label='val-{}'.format(labelname))

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('compare_iou')
    plt.savefig(os.path.join(work_dir, 'compare_val_perf.png'))

    

    plt.close()


def draw_curve_3(work_dir,logger1,logger2,logger3):

    logger1 = logger1.read()
    logger2 = logger2.read()
    logger3 = logger3.read()

    if len(logger1[0]) == 6:

        epoch, trn_loss1, iou1, dice1,acd1,asd1 = zip(*logger1)
        epoch, trn_loss2, iou2, dice2,acd2,asd2= zip(*logger2)
        epoch, trn_loss3, iou3, dice3, acd3,asd3 = zip(*logger3)
    else:
        epoch, trn_loss1, iou1, dice1 = zip(*logger1)
        epoch, trn_loss2, iou2, dice2 = zip(*logger2)
        epoch, trn_loss3, iou3, dice3 = zip(*logger3)


    plt.figure(1)
    plt.plot(epoch, trn_loss1, '-b', label='train_loss')
    plt.plot(epoch, trn_loss2, '-r', label='val_loss')
    plt.plot(epoch, trn_loss3, '-g', label='mc_val_loss')

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('compare_loss')
    plt.savefig(os.path.join(work_dir, 'add_mc_trn_loss.png'))


    plt.figure(2)
    plt.plot(epoch, iou1, '-b', label='train-IoU')
    plt.plot(epoch, iou2, '-r', label='val-IoU')
    plt.plot(epoch, iou3, '-g', label='mc_val-IoU')

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('compare_iou')
    plt.savefig(os.path.join(work_dir, 'add_mc_compare_val_perf.png'))

    if len(logger1[0]) == 6:
        plt.figure(3)
        plt.plot(epoch, acd1, '-b', label='train-ACD')
        plt.plot(epoch, acd2, '-r', label='val-ACD')
        plt.plot(epoch, acd3, '-g', label='mc_val-ACD')

        plt.xlabel('Epoch')
        plt.legend()
        plt.title('compare_ACD')
        plt.savefig(os.path.join(work_dir, 'mc_compare_val_ACD.png'))
    plt.close()


def send_slack_message(token,channel,messge):
    token = token
    slack = Slacker(token)
    slack.chat.post_message(channel, messge)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
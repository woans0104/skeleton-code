import argparse
import torch.utils.data as data
import model
import ipdb
import os
import dataloader as loader
import torchvision.transforms as transforms
import numpy as np
from medpy.metric import binary
from utils import Logger, AverageMeter, save_checkpoint ,draw_curve ,draw_curve_3,send_slack_message,str2bool
import time
import shutil
import pickle
import torch.optim as optim
from predict import main_test
from model import *
from losses import DiceLoss,tversky_loss, NLL_OHEM
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser()

# arguments for dataset
parser.add_argument('--train-site',default='site2',help='site1,site2,site3,site4')
parser.add_argument('--test-site1',default='site1',help='site1,site2,site3,site4')
parser.add_argument('--test-site2',default='site3',help='site1,site2,site3,site4')
parser.add_argument('--test-site3',default='site4',help='site1,site2,site3,site4')



parser.add_argument('--work-dir', default='/data1/JM/spina_cord_segmentation')
parser.add_argument('--exp',default="test4", type=str)


parser.add_argument('--input-size',default=64,type=int)
parser.add_argument('--batch-size',default=4,type=int)
parser.add_argument('--arg-mode',default=False,type=str2bool)
parser.add_argument('--arg-thres',default=0.7,type=float)


# arguments for model
parser.add_argument('--arch', default='unet', type=str)


# arguments for optim & loss
parser.add_argument('--optim',default='sgd',choices=['adam','sgd'],type=str)
parser.add_argument('--eps',default=1e-08,type=float)
parser.add_argument('--weight-decay',default=1e-4,type=float)

parser.add_argument('--loss-function',default='bce',type=str)
parser.add_argument('--bce-weight', default=1, type=float)


parser.add_argument('--initial-lr',default=0.1,type=float)
parser.add_argument('--lr-schedule', default=[100,120], nargs='+', type=int)
parser.add_argument('--gamma',default=0.1,type=float)

# arguments for dataset
parser.add_argument('--train-size',default=0.8,type=float)
parser.add_argument('--val-size',default=0.2,type=float)
parser.add_argument('--train_mode',default=True,type=str2bool)


# arguments for test mode
parser.add_argument('--test_mode',default=True,type=str2bool)
parser.add_argument('--file-name', default='result_all', type=str)


# arguments for slack
parser.add_argument('--token',type=str)


args = parser.parse_args()


def main():
    # save input stats for later use

    print(args.work_dir, args.exp)
    work_dir = os.path.join(args.work_dir, args.exp)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # copy this file to work dir to keep training configuration
    shutil.copy(__file__, os.path.join(work_dir, 'main.py'))
    with open(os.path.join(work_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)




    # transform
    transform1 = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])

    # 1.train_dataset
    
    train_path, test_path = loader.make_dataset(args.train_site,train_size=args.train_size, mode='train')
                                                            
    np.save(os.path.join(work_dir, '{}_test_path.npy'.format(args.train_site)), test_path)

    train_image_path = train_path[0]
    train_label_path = train_path[1]
    test_image_path = test_path[0]
    test_label_path = test_path[1]

    train_dataset = loader.CustomDataset(train_image_path, train_label_path, args.train_site,args.input_size, transform1,
                                         arg_mode=args.arg_mode, arg_thres=args.arg_thres)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_dataset = loader.CustomDataset(test_image_path, test_label_path, args.train_site, args.input_size,transform1, arg_mode=False)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    Train_test_dataset = loader.CustomDataset(test_image_path, test_label_path, args.train_site, args.input_size,transform1)
    Train_test_loader = data.DataLoader(Train_test_dataset, batch_size=1, shuffle=True, num_workers=4)


   
    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))


    # 3.model_select
    my_net, model_name = model_select(args.arch,args.input_size,)


    # 4.gpu select
    my_net = nn.DataParallel(my_net).cuda()
    cudnn.benchmark = True

    # 5.optim

    if args.optim == 'adam':
        gen_optimizer = torch.optim.Adam(my_net.parameters(), lr=args.initial_lr, eps=args.eps)
    elif args.optim == 'sgd':
        gen_optimizer = torch.optim.SGD(my_net.parameters(), lr=args.initial_lr, momentum=0.9,weight_decay=args.weight_decay)



    # lr decay
    lr_schedule = args.lr_schedule
    lr_scheduler = optim.lr_scheduler.MultiStepLR(gen_optimizer,
                                                  milestones=lr_schedule[:-1],
                                                  gamma=args.gamma)

    # 6.loss
    if args.loss_function == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.bce_weight])).cuda()
    elif args.loss_function == 'mse':
        criterion = nn.MSELoss().cuda()



#####################################################################################

    # train

    send_slack_message(args.token, '#jm_private', '{} : starting_training'.format(args.exp))
    best_iou = 0
    try:
        if args.train_mode:
            for epoch in range(lr_schedule[-1]):

                train(my_net, train_loader, gen_optimizer, epoch, criterion,trn_logger, trn_raw_logger)
                iou = validate(val_loader, my_net, criterion, epoch, val_logger,save_fig=False,work_dir_name='jsrt_visualize_per_epoch')
                print('validation_iou **************************************************************')
               
                lr_scheduler.step()

                if args.val_size ==0:
                    is_best = 1
                else:
                    is_best = iou > best_iou
                best_iou = max(iou, best_iou)
                checkpoint_filename = 'model_checkpoint_{:0>3}.pth'.format(epoch + 1)
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': my_net.state_dict(),
                                 'optimizer': gen_optimizer.state_dict()},
                                is_best,
                                work_dir,
                                filename='checkpoint.pth')

        print("train end")
    except RuntimeError as e:
        send_slack_message(args.token, '#jm_private',
                       '-----------------------------------  error train : send to message JM  & Please send a kakao talk ----------------------------------------- \n error message : {}'.format(
                           e))
        import ipdb
        ipdb.set_trace()

    draw_curve(work_dir, trn_logger, val_logger)
    send_slack_message(args.token, '#jm_private', '{} : end_training'.format(args.exp))
                       

    if args.test_mode:
        print('Test mode ...')
        main_test(model=my_net,test_loader=test_data_list, args=args)




def train(model,train_loader,optimizer,epoch,criterion,logger, sublogger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()
    acds = AverageMeter()
    asds = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target,ori_input) in enumerate(train_loader):

        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

       
        output = model(input)

        loss = criterion(output, target)



        iou, dice,acd,asd = performance(output, target)
        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices.update(dice, input.size(0))
        acds.update(acd, input.size(0))
        asds.update(asd, input.size(0))



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'IoU {iou.val:.4f} ({iou.avg:.4f})\t'
              'Dice {dice.val:.4f} ({dice.avg:.4f})\t'
              'Acd {acd.val:.4f} ({acd.avg:.4f})\t'
              'Asd {asd.val:.4f} ({asd.avg:.4f})\t'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses,
            iou=ious, dice=dices,acd=acds,asd=asds))

        if i % 10 == 0:
            try:
                sublogger.write([epoch, i, loss.item(), iou, dice, acd, asd])
            except:
                # ipdb.set_trace()
                print('acd,asd : None')

    logger.write([epoch, losses.avg, ious.avg, dices.avg,acds.avg,asds.avg])



def validate(val_loader, model, criterion, epoch, logger,save_fig=False,work_dir_name=False):

    batch_time = AverageMeter()
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()
    acds = AverageMeter()
    asds = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target,ori_img) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            
            output = model(input)

            loss = criterion(output, target)

            iou, dice ,acd,asd = performance(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices.update(dice, input.size(0))
            acds.update(acd, input.size(0))
            asds.update(asd, input.size(0))


            work_dir = os.path.join(args.work_dir, args.exp)
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)

            if save_fig:
                if i % 2 == 0:
                    ae_save_fig(str(epoch), ori_img, target, output, iou, dice, acd,asd,work_dir,work_dir_name, image_name[0])

            batch_time.update(time.time() - end)
            end = time.time()

    print(' * IoU {ious.avg:.3f}({ious.std:.3f}) Dice {dices.avg:.3f}({dices.std:.3f}) Acd {acds.avg:3f}({acds.std:3f}) Asd {asds.avg:3f}({asds.std:3f})'.format(
           ious=ious, dices=dices,acds=acds,asds=asds))

    logger.write([epoch, losses.avg, ious.avg, dices.avg,acds.avg,asds.avg])

    return ious.avg




def model_select(network,input_size):

    # model_new

    if network == 'unet':
        my_net = Unet2D(in_shape=(1, input_size, input_size), padding=args.padding_size, momentum=args.batchnorm_momentum)
       
    else:
        raise ValueError('Not supported network.')

    model_name = str(my_net).split('(')[0]

    return my_net, model_name



def performance(output, target):
    pos_probs = torch.sigmoid(output)
    pos_preds = (pos_probs > 0.5).float()

    pos_preds = pos_preds.cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()

    if target.sum() == 0:  # background patch
        return 0, 0



    try:
        # ACD
        acd_se = binary.assd(pos_preds, target)
        # ASD
        d_sg = np.sqrt(binary.__surface_distances(pos_preds, target, 1))
        d_gs = np.sqrt(binary.__surface_distances(target, pos_preds, 1))
        asd_se = (d_sg.sum() + d_gs.sum()) / (len(d_sg) + len(d_gs))

    except:
        #pred == 0
        acd_se =None
        asd_se = None


    # IoU
    union = ((pos_preds + target) != 0).sum()
    intersection = (pos_preds * target).sum()
    iou = intersection / union

    # dice
    dice = (2 * intersection) / (pos_preds.sum() + target.sum())

    return iou, dice,acd_se,asd_se


if __name__ == '__main__':
    main()
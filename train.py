import argparse
import logging
import os
import datetime
import sys
import torch
import time
import random
import numpy as np
import math

from lib.datasets.dataset_crack import CrackDataset
from torch.utils.data import DataLoader


from torch.optim.lr_scheduler import PolynomialLR
from torch.nn.modules.loss import CrossEntropyLoss
from lib.losses.ohem_cross_entropy_loss import OhemCrossEntropyLoss

from lib.utils.loss_avg_meter import LossAverageMeter
from lib.utils.confusion_matrix_old import ConfusionMatrix

from lib.utils.save_log import save_log
from lib.utils.save_weight import save_weights

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    default='ddrnet_23', help='model name')
# D:\\data\\Crack_Forest_paddle\\Crack_Forest_paddle
# /home/user/data/lumianliefeng/Crack_Forest_paddle
# /home/user/data/liefeng/Crack_paddle_255
parser.add_argument('--dataset_root', type=str,
                    default='/home/user/data/lumianliefeng/Crack_Forest_paddle', help='dataset root directory')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,
                    default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int,
                    default=3407, help='random seed')
parser.add_argument('--log_path', type=str,
                    default='./run/crackforest', help='run path')
parser.add_argument('--log_iters', type=int,
                    default=500, help='log interval')
parser.add_argument('--eval', type=bool,
                    default=True, help='eval when train')
args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # create dataloader
    train_dataset = CrackDataset(args.dataset_root, args.img_size, "train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    data_length = len(train_dataset)
    max_iterations = data_length // args.batch_size * args.max_epochs

    val_dataset = CrackDataset(args.dataset_root, args.img_size, "val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    # create model
    if args.model == 'ddrnet':
        model = ddrnet_silm(args.num_classes)
        losses = [OhemCrossEntropyLoss()]
        loss_weights = [1]
    elif args.model == 'ddrnet_23':
        model = ddrnet_23(args.num_classes)
        losses = [OhemCrossEntropyLoss()]
        loss_weights = [1]
    elif args.model == 'bisenetv2':
        model = BiSeNetV2(args.num_classes)
        losses = [CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss()]
        loss_weights = [1, 1, 1, 1, 1]
    elif args.model == 'bisenetv1':
        model = BiSeNetV1(args.num_classes)
        losses = [OhemCrossEntropyLoss(), OhemCrossEntropyLoss(), OhemCrossEntropyLoss()]
        loss_weights = [1, 1, 1, 1]
    else:
        model = None
        raise KeyError("unknown model: {}".format(args.model))
    model.to('cuda')

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    lr = PolynomialLR(optimizer=optimizer, total_iters=max_iterations, power=0.9)

    # config log
    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.log_path = os.path.join(args.log_path, args.model + '_' + now_time)
    weight_path = os.path.join(args.log_path, "weights")
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    logging.basicConfig(filename=args.log_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # logging.info("model {}".format(args.model))
    logging.info("train img nums {}".format(data_length))
    logging.info("{} iterations per epoch. {} max iterations "
                 .format(data_length // args.batch_size, data_length // args.batch_size * args.max_epochs))

    trainlog_path = os.path.join(args.log_path, 'train.txt')
    vallog_path = os.path.join(args.log_path, 'val.txt')

    # metric
    iter_num = 0
    min_loss = 100
    best_miou = 0
    canny_loss = torch.tensor([0])

    # train
    start_time = time.time()
    for epoch in range(args.max_epochs):
        model.train()
        train_losses = LossAverageMeter()
        train_confmat = ConfusionMatrix(args.num_classes)
        for batch_idx, sampled_batch in enumerate(train_dataloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            if not isinstance(outputs, tuple):
                outputs = [outputs]
            train_confmat.update(label_batch.flatten(), outputs[0].argmax(1).flatten())
            loss_list = [loss_weights[i] * losses[i](outputs[i], label_batch.long()) for i in range(len(losses))]
            # print(loss_list)
            main_loss = loss_list[0]
            loss = main_loss.clone()
            for i in range(1, len(loss_list)):
                loss += loss_list[i]
            #canny_loss = loss_list[-1]
            train_losses.update(main_loss.item(), image_batch.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr.step()

            iter_num += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            progress = iter_num / max_iterations
            remaining_time = elapsed_time * (1 / progress - 1)
            remaining_time_minutes = math.floor(remaining_time / 60)
            remaining_time_seconds = math.floor(remaining_time - remaining_time_minutes * 60)
            current_lr = optimizer.param_groups[0]['lr']

            if iter_num % args.log_iters == 0:
                logging.info(
                    '[train]iteration %d / %d\tloss: %f\tmain_loss:%f\tcanny_loss:%f\tcurrent lr:%f\tremaining time : %d minutes  %d seconds'
                    % (iter_num, max_iterations, loss.item(), main_loss.item(),canny_loss.item() , current_lr, remaining_time_minutes,
                       remaining_time_seconds))

        trainmiou = save_log(train_confmat, train_losses, trainlog_path, epoch)
        if epoch >= 1:
            torch.save(model.state_dict(), os.path.join(weight_path, 'latest.pth'))

        if args.eval:
            logging.info("evaluating")
            model.eval()
            val_losses = LossAverageMeter()
            val_confmat = ConfusionMatrix(args.num_classes)
            with torch.no_grad():
                sum_main_loss = 0
                sum_loss = 0
                num = 0
                for batch_idx, sampled_batch in enumerate(val_dataloader):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    if not isinstance(outputs, tuple):
                        outputs = [outputs]

                    val_confmat.update(label_batch.flatten(), outputs[0].argmax(1).flatten())
                    loss_list = [loss_weights[i] * losses[i](outputs[i], label_batch.long()) for i in
                                 range(len(losses))]
                    main_loss = loss_list[0]
                    loss = main_loss
                    for i in range(1, len(loss_list)):
                        loss += loss_list[i]
                    val_losses.update(main_loss.item(), image_batch.shape[0])
                    sum_main_loss += main_loss.item()
                    sum_loss += loss.item()
                    num += 1

                current_time = time.time()
                elapsed_time = current_time - start_time
                progress = iter_num / max_iterations
                remaining_time = elapsed_time * (1 / progress - 1)
                remaining_time_minutes = math.floor(remaining_time / 60)
                remaining_time_seconds = math.floor(remaining_time - remaining_time_minutes * 60)
                logging.info('[ val ]iteration %d\tmain_loss : %f, loss: %f\tremaining time : %d minutes  %d seconds'
                             % (
                                 iter_num, sum_main_loss / num, sum_loss / num, remaining_time_minutes,
                                 remaining_time_seconds))

            valmiou = save_log(val_confmat, val_losses, vallog_path, epoch)
            min_loss, best_miou = save_weights(val_losses, valmiou, min_loss, best_miou, model.state_dict(),
                                               weight_path)
        else:
            min_loss, best_miou = save_weights(train_losses, trainmiou, min_loss, best_miou, model.state_dict(),
                                               weight_path)
    new_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.info("Training Started at " + now_time)
    logging.info("Training Finished at " + new_time_str)

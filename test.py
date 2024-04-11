import argparse
import os
import random
import torch
import logging
import sys

from lib.datasets.dataset_crack import CrackDataset
from torch.utils.data import DataLoader

from lib.models.ddrnet import ddrnet_silm
from lib.models.bisenetv1 import BiSeNetV1
from lib.models.bisenetv2 import BiSeNetV2

from lib.models.bisenetv1_global2taspp_ffm2fammul import BiSeNetV1_global2taspp_ffm2fammul
from lib.models.bisenetv1_global2taspp import BiSeNetV1_global2taspp
from lib.models.bisenetv1_mul import BiSeNetV1_mul

from torch.nn.modules.loss import CrossEntropyLoss
from lib.losses.ohem_cross_entropy_loss import OhemCrossEntropyLoss
from lib.losses.canny_loss import CannyLoss

from lib.utils.loss_avg_meter import LossAverageMeter
from lib.utils.confusion_matrix import ConfusionMatrix

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    default='bisenetv2', help='model name')
parser.add_argument('--log_path', type=str,
                    default='./run/crack500_20000/bisenetv2_20240403_093149', help='log path')
parser.add_argument('--checkpoint_type', type=str,
                    default='best_miou', help="best_miou or last or min_loss")
# D:\\data\\Crack_Forest_paddle\\Crack_Forest_paddle
# /home/user/data/lumianliefeng/Crack_Forest_paddlebisenetv1_global2taspp_20240323_201000
# /home/user/data/liefeng/Crack_paddle_255
parser.add_argument('--dataset_root', type=str,
                    default='/home/user/data/lumianliefeng/crack500_paddle_20000', help='dataset root directory')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--seed', type=int,
                    default=3407, help='random seed')
args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


if __name__ == '__main__':
    # create dataloader
    test_dataset = CrackDataset(args.dataset_root, args.img_size, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True,
                                num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    data_length = len(test_dataset)

    # create model
    if args.model == 'ddrnet':
        model = ddrnet_silm(args.num_classes)
        losses = [OhemCrossEntropyLoss()]
        loss_weights = [1]
    elif args.model == 'bisenetv2':
        model = BiSeNetV2(args.num_classes)
        losses = [CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss(), CrossEntropyLoss()]
        loss_weights = [1, 1, 1, 1]
    elif 'bisenetv1' in args.model:
        if args.model == 'bisenetv1':
            model = BiSeNetV1(args.num_classes)
        elif args.model == 'bisenetv1_global2taspp_ffm2fammul':
            model = BiSeNetV1_global2taspp_ffm2fammul(args.num_classes)
        elif args.model =='bisenetv1_global2taspp':
            model = BiSeNetV1_global2taspp(args.num_classes)
        elif args.model =='bisenetv1_mul':
            model = BiSeNetV1_mul(args.num_classes)

        else:
            raise KeyError("unknown model: {}".format(args.model))

        losses = [OhemCrossEntropyLoss(), OhemCrossEntropyLoss(), OhemCrossEntropyLoss(), CannyLoss()]
        loss_weights = [1, 1, 1, 1]
    else:
        model = None
        raise KeyError("unknown model: {}".format(args.model))

    if args.checkpoint_type == 'best_miou':
        model_type = "best_miou.pth"
    elif args.checkpoint_type == 'last':
        model_type = "lastest.pth"
    elif args.checkpoint_type == 'min_loss':
        model_type = "lowest_loss.path"
    else:
        raise KeyError("checkpoint_type should be 'best_miou' or 'last' or 'min_loss'")
    model.to('cuda')
    weight_path = os.path.join(args.log_path, "weights/" + model_type)
    model_state_dict = torch.load(weight_path)
    model.load_state_dict(model_state_dict)

    logging.basicConfig(filename=args.log_path + "/test_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("test img nums {}".format(data_length))
    # test
    test_losses = LossAverageMeter()
    test_confmat = ConfusionMatrix(args.num_classes)
    model.eval()
    with torch.no_grad():
        for batch_idx, sampled_batch in enumerate(test_dataloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            output = model(image_batch)
            if isinstance(output, tuple):
                output = output[0]
            test_confmat.update(label_batch.flatten(), output.argmax(1).flatten())
            loss_type = losses[0]
            loss = loss_type(output, label_batch.long())
            test_losses.update(loss.item(), image_batch.shape[0])

    testacc_global, testacc, testiu, testRec, testPre, testF1, mF1 = test_confmat.compute()
    testacc_global = testacc_global.item() * 100
    testaver_row_correct = ['{:.2f}'.format(i) for i in (testacc * 100).tolist()]
    testiou = ['{:.2f}'.format(i) for i in (testiu * 100).tolist()]
    testmiou = testiu.mean().item() * 100
    testF1 = testF1.item() * 100,
    testRec = testRec.item() * 100,
    testPre = testPre.item() * 100,
    mF1 = mF1.item() * 100
    logging.info("\n"+"losses.avg:" + str(test_losses.avg) + "\n" +
                 "miou:" + str(testmiou) + "\n" +
                 "acc_global:" + str(testacc_global) + "\n" +
                 "aver_row_correct:" + str(testaver_row_correct[0]) + "-" + str(testaver_row_correct[1]) + "\n" +
                 "iou:" + str(testiou[0]) + "-" + str(testiou[1]) + "\n" +
                 "f1:" + str(testF1) + "\n" +
                 "Rec:" + str(testRec) + "\n" +
                 "Rre:" + str(testPre) + "\n" +
                 "mF1:" + str(mF1) + "\n")
    logging.info("test Finished")

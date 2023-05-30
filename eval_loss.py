from model.fcos import FCOSDetector
import torch
from dataset.VOC_dataset import VOCDataset
import math, time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from torch.utils.tensorboard import SummaryWriter

log_dir = "logs/"
writer = SummaryWriter(log_dir)
parser = argparse.ArgumentParser()
# parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=3, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=3, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0,1,2', help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
transform = Transforms()
eval_dataset = VOCDataset(root_dir='/home/Disk1/zhangqiang/objectdetection/data/VOCdevkit/VOC2012',resize_size=[800,1333],
                           split='val',use_difficult=False,is_train=True,augment=transform)

model = FCOSDetector(mode="training").cuda()
model = torch.nn.DataParallel(model)
# model.load_state_dict(torch.load('/mnt/cephfs_new_wj/vc/zhangzhenghao/FCOS.Pytorch/output1/model_6.pth'))

BATCH_SIZE = opt.batch_size
# EPOCHS = opt.epochs
#WARMPUP_STEPS_RATIO = 0.12
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=eval_dataset.collate_fn,
                                           num_workers=opt.n_cpu, worker_init_fn=np.random.seed(0))
print("total_images : {}".format(len(eval_dataset)))


# steps_per_epoch = len(eval_dataset) // BATCH_SIZE
# TOTAL_STEPS = steps_per_epoch * EPOCHS
# WARMPUP_STEPS = 501
#
# GLOBAL_STEPS = 1
# LR_INIT = 2e-3
# LR_END = 2e-5
# optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)

# def lr_func():
#      if GLOBAL_STEPS < WARMPUP_STEPS:
#          lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
#      else:
#          lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
#              (1 + math.cos((GLOBAL_STEPS - WARMPUP_STEPS) / (TOTAL_STEPS - WARMPUP_STEPS) * math.pi))
#          )
#      return float(lr)


model.train()
# print('readytotrain')

for epoch in range(1, 21):
    modelpath = "./checkpoint/model_{}.pth".format(epoch)
    model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
    for epoch_step, data in enumerate(eval_loader):
        # print(epoch_step)
        batch_imgs, batch_boxes, batch_classes = data
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()

        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]
        loss.mean().backward()

        print(
            "epoch:%d steps:%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f total_loss:%.4f" % \
            (epoch + 1, epoch_step + 1, losses[0].mean(), losses[1].mean(),
             losses[2].mean(),  loss.mean()))
    writer.add_scalar('val_Loss', loss.mean(), epoch)

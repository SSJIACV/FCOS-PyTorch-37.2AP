from model.fcos import FCOSDetector
import torch
from dataset.odgt_dataset import OdgtDataset
import math, time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from config_dianwang import Config_myself

# rlaunch --cpu=12 --gpu=8 --memory=$((120*1024)) --max-wait-time 10h --preemptible=no --charged-group v_tracking -- python3 train_odgt_fangzhenchui.py

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=51, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=12, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0,1,2,3,4,5,6,7', help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu

save_path = Config_myself.save_path
os.makedirs(save_path, exist_ok=True)
root_dir = Config_myself.data_root_dir
set_class = Config_myself.set_class
CLASSES_NAME = Config_myself.CLASSES_NAME
resize_size = Config_myself.resize_size

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
transform = Transforms()


train_dataset = OdgtDataset(root_dir=root_dir,resize_size=resize_size,
                            set_class = set_class,CLASSES_NAME = CLASSES_NAME,is_train = True, augment = transform)

model = FCOSDetector(mode="training",config=Config_myself).cuda()
model = torch.nn.DataParallel(model)
if Config_myself.pretrained_coco:
    model.load_state_dict(torch.load(Config_myself.coco_weight_path))


BATCH_SIZE = opt.batch_size
EPOCHS = opt.epochs
#WARMPUP_STEPS_RATIO = 0.12
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=train_dataset.collate_fn,
                                           num_workers=opt.n_cpu, worker_init_fn=np.random.seed(0))
print("total_images : {}".format(len(train_dataset)))
steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMPUP_STEPS = Config_myself.WARMPUP_STEPS

GLOBAL_STEPS = 1
LR_INIT = Config_myself.LR_INIT3
LR_END = 2e-5
optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)

# def lr_func():
#      if GLOBAL_STEPS < WARMPUP_STEPS:
#          lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
#      else:
#          lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
#              (1 + math.cos((GLOBAL_STEPS - WARMPUP_STEPS) / (TOTAL_STEPS - WARMPUP_STEPS) * math.pi))
#          )
#      return float(lr)


model.train()

for epoch in range(EPOCHS):
    for epoch_step, data in enumerate(train_loader):

        batch_imgs, batch_boxes, batch_classes = data
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()

        #lr = lr_func()
        if GLOBAL_STEPS < WARMPUP_STEPS:
           lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
           for param in optimizer.param_groups:
               param['lr'] = lr
        # if GLOBAL_STEPS == 20001:
        #    lr = LR_INIT * 0.1
        #    for param in optimizer.param_groups:
        #        param['lr'] = lr
        # if GLOBAL_STEPS == 27001:
        #    lr = LR_INIT * 0.01
        #    for param in optimizer.param_groups:
        #       param['lr'] = lr
        if GLOBAL_STEPS == int(0.3*TOTAL_STEPS):
           lr = LR_INIT * 0.1
           for param in optimizer.param_groups:
               param['lr'] = lr
        if GLOBAL_STEPS == int(0.6*TOTAL_STEPS):
           lr = LR_INIT * 0.01
           for param in optimizer.param_groups:
              param['lr'] = lr
        start_time = time.time()

        optimizer.zero_grad()
        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]
        loss.mean().backward()
        optimizer.step()

        end_time = time.time()
        cost_time = int((end_time - start_time) * 1000)
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        print(
            "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
            (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
             losses[2].mean(), cost_time, lr, loss.mean()))

        GLOBAL_STEPS += 1

    if epoch%5==0 and epoch>0:
        torch.save(model.state_dict(), save_path + 'model_{}.pth'.format(epoch + 1))





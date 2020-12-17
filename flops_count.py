"""
This python script is used for 计算model的flops
"""
# @Time    : 2020/12/01 20:24
# @Author  : jss
# @email   : ssjia_cv@foxmail.com
# @File    : flops_count.py

import torch
import os
import numpy as np
import cv2
import json
from PIL import  Image
import yaml
from tqdm import tqdm
from torchvision import transforms
from model.fcos import FCOSDetector

from ptflops import get_model_complexity_info

# rlaunch --cpu=8 --gpu=1 --memory=$((80*1024)) --max-wait-time 10h --preemptible=no --charged-group v_tracking -- python3 flops_count.py

weight_path = "./checkpoint/zhongchui_2class_1202/model_21.pth"

if __name__ == '__main__':

    with torch.cuda.device(0):
        model=FCOSDetector(mode="inference")
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(weight_path,map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model=model.cuda().eval()
        macs, params = get_model_complexity_info(model, (3, 832, 1216), as_strings=True,
                                            print_per_layer_stat=False, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))



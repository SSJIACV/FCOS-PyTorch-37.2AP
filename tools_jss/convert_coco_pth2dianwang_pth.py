"""
This python script is used for 将COCO预训练模型的类别数（80类）转换为电网数据所需的类别数（1/2/4）
"""
# @Time    : 2020/12/07 16:21
# @Author  : jss
# @email   : ssjia_cv@foxmail.com
# @File    : convert_coco_pth2dianwang_pth.py

import torch

# rlaunch --cpu=12 --gpu=1 --memory=$((120*1024)) --max-wait-time 10h --preemptible=no --charged-group v_tracking -- python3 tools_jss/convert_coco_pth2dianwang_pth.py
def main():

    # gen coco pretrained weight

    # num_classes = 2
    # num_classes = 4
    num_classes = 1

    # weight
    model_coco = torch.load('/data/wurenji/code_new/FCOS-PyTorch-37.2AP/checkpoint/coco_37.2.pth')
    model_coco["module.fcos_body.head.cls_logits.weight"] = model_coco["module.fcos_body.head.cls_logits.weight"][ :num_classes, :]

    # bias
    model_coco["module.fcos_body.head.cls_logits.bias"] = model_coco["module.fcos_body.head.cls_logits.bias"][ :num_classes]

    # save new model
    torch.save(model_coco, "/data/wurenji/code_new/FCOS-PyTorch-37.2AP/checkpoint/coco_37.2_classes_%d.pth" % num_classes)

if __name__ == "__main__":
    main()
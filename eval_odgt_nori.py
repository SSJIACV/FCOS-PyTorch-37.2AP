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
from config_nori_dianwang import Config_nori_myself
from refile import smart_open
import nori2 as nori
from meghair.utils.imgproc import imdecode
nf = nori.Fetcher()

# rlaunch --cpu=8 --gpu=1 --memory=$((80*1024)) --max-wait-time 10h --preemptible=no --charged-group v_tracking -- python3 eval_odgt_nori.py
val_odgt_now = Config_nori_myself.val_odgt_now
det_save_odgt = Config_nori_myself.det_save_odgt
det_save_eval_log_txt = Config_nori_myself.det_save_eval_log_txt
threshold=0.05
CLASSES_NAME = Config_nori_myself.CLASSES_NAME
resize_size=Config_nori_myself.resize_size
weight_path = Config_nori_myself.weight_path

mean=[0.40789654, 0.44719302, 0.47026115]
std=[0.28863828, 0.27408164, 0.27809835]

def preprocess_img_boxes(image,resize_size):
    '''
    resize image and bboxes
    Returns
    image_paded: input_ksize
    bboxes: [None,4]
    '''
    min_side, max_side    = resize_size
    h,  w, _  = image.shape

    smallest_side = min(w,h)
    largest_side=max(w,h)
    scale=min_side/smallest_side
    if largest_side*scale>max_side:
        scale=max_side/largest_side
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w=32-nw%32
    pad_h=32-nh%32

    image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded, scale

def evaluate_odgt(records, model):
    all_result = []
    pbar = tqdm(total=len(records))
    for record in records:
        pbar.update(1)
        nori_id = record['ID']
        img = imdecode(nf.get(nori_id))
        # img = Image.open(image_path)
        img = np.array(img)
        img,scale=preprocess_img_boxes(img,resize_size)
        img=transforms.ToTensor()(img)
        img= transforms.Normalize(mean,std,inplace=True)(img)
        scores, labels,boxes  = model(img.unsqueeze(dim=0).cuda())
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()
        boxes /= scale
        # correct boxes for image scale
        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]
        # compute predicted labels and scores
        dtboxes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < threshold:
                continue
            score = float(score)
            tag = CLASSES_NAME[label]
            # only test bad 
            # if tag != 'zhongchui_bad':
            #     continue
            # if tag in ['normal_board', 'normal_driving_birds_device']:
            #     continue
            box = box.tolist()
            box_new = dict(box=box, score=score, tag=tag)
            dtboxes.append(box_new)
        record['dtboxes'] = dtboxes

        # new_gtboxes = []
        # for gtbox in record['gtboxes']:
        #     # if gtbox['tag'] == 'fangzhenchui_bad' or gtbox['tag'] == 'fangzhenchui_good':
        #     #     new_gtboxes.append(gtbox)
        #     if gtbox['tag'] == 'zhongchui_bad':
        #         new_gtboxes.append(gtbox)
        #     # if gtbox['tag'] in ['normal_board', 'normal_driving_birds_device']:
        #     #     continue
        #     # else:
        #     #     new_gtboxes.append(gtbox)
        # record['gtboxes'] = new_gtboxes
        all_result.append(record)

    fw = open(det_save_odgt, 'w')
    for res in all_result:
        res = json.dumps(res)
        fw.write(res + '\n')
    fw.close()

    # evaluation
    eval_script = '/data/wurenji/code_new/dianwang_detection/evalTookits2/eval.py'
    command = 'python3 -u %s --dt=%s --gt=%s --iou=%f | tee -a %s' % (eval_script, det_save_odgt, det_save_odgt, 0.2, det_save_eval_log_txt)
    os.system(command)
    print('done')

if __name__=="__main__":
    if os.path.exists(det_save_odgt):
        eval_script = '/data/wurenji/code_new/dianwang_detection/evalTookits2/eval.py'
        command = 'python3 -u %s --dt=%s --gt=%s --iou=%f | tee -a %s' % (eval_script, det_save_odgt, det_save_odgt, 0.2, det_save_eval_log_txt)
        os.system(command)
        print('done')
    else:
        val_odgt = val_odgt_now
        model=FCOSDetector(mode="inference",config=Config_nori_myself)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(weight_path,map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model=model.cuda().eval()
        print("===>success loading model")
        with smart_open(val_odgt, 'r') as f:
            lines = [l.rstrip() for l in f.readlines()]
        records = [json.loads(line) for line in lines]  # str to list
        evaluate_odgt(records, model)

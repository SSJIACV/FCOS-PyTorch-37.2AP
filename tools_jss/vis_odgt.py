import os
import cv2
import json
from tqdm import tqdm
import nori2 as nori
from meghair.utils.imgproc import imdecode
nf = nori.Fetcher()
from tqdm import tqdm
import math
import megbrain as mgb
import copy
from refile import smart_open

# gt_det_file = '/data/workspace/wurenji/data/xiaojinju/odgt/det2_train.odgt'
# gt_det_file = '/data/wurenji/code_new/RetinaNet/models/jueyuanzi_jss/det1_with_202007data_2019data/jueyuanzi_202006_append_good.odgt'
# gt_det_file = '/data/wurenji/code_new/RetinaNet/models/jueyuanzi_jss/det1_with_202007data_2019data/jueyuanzi_202006_append_good_new.odgt'
# gt_det_file = './test_fppi_in_ganta.odgt'
gt_det_file = '/data/wurenji/code_new/FCOS-PyTorch-37.2AP/data/daodixian_2class/annotations/daodixian_2class_val.odgt'
save_dir = '/data/wurenji/code_new/FCOS-PyTorch-37.2AP/data/vis_odgt_daodixian_1207/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(gt_det_file, 'r') as f:
    redis_keys = [l.rstrip() for l in f.readlines()]

pbar = tqdm(total=len(redis_keys))
all_tags = []
count = 0
# for redis_key in redis_keys[0:-1:20]:
for i,redis_key in enumerate(redis_keys):
    if i%5 !=0:
        continue
    count = count + 1
    if count>20:
        break
    pbar.update(1)
    item = json.loads(redis_key)
    # nid = item['image_info']['nori_id']
    nori_id = item['ID']
    fpath = item['fpath']
    image = imdecode(nf.get(nori_id))
    dst_path = os.path.join(save_dir, fpath)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    gtboxes = item.get('gtboxes')
    # dtboxes = item.get('gtboxes')
    # dtboxes = item['gtboxes']
    # for db in dtboxes:
    #     box = db['box']
    #     score = db['score']
    #     tag = db['tag']
    #     xmin, ymin, w, h, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), score
    #     if score > config.vis_thresh:
    #         cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 6)
    #         cv2.putText(image, '{}:{:.2f}'.format(tag, score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 6)
    for gtbox in gtboxes:
        all_tags.append(gtbox['tag'])
        box = [int(x) for x in gtbox['box']]
        cv2.rectangle(image, (box[0], box[1]), (box[2]+box[0], box[3]+box[1]), (0, 0, 255), 4)
        cv2.putText(image, '{}'.format(gtbox['tag']), (box[0], box[3]+box[1]), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 6)
    print(image.shape[:2])
    # image = cv2.resize(image, (1280, 768), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(dst_path, image)


print(set(all_tags))
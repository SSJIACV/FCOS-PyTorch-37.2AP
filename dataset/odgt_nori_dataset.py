import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import  Image
import random
import json
import time 
from refile import smart_open
import nori2 as nori
from meghair.utils.imgproc import imdecode
nf = nori.Fetcher()

def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:,2]
        xmax = w - boxes[:,0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes


class Odgt_Nori_Dataset(torch.utils.data.Dataset):
    def __init__(self,train_odgt='s3://jiashuaishuai/dianwang_data/dajinju/annotations/fangzhenchui_2class_train.odgt',resize_size=[800,1333],CLASSES_NAME = ["__background__ ","fangzhenchui_good","fangzhenchui_bad",],is_train = True, augment = None):
        # if is_train:
        #     self.odgt = train_odgt
        # else:
        #     self.odgt = val_odgt
        self.odgt = train_odgt
        with smart_open(self.odgt, 'r') as f:
            lines = [l.rstrip() for l in f.readlines()]
        records = [json.loads(line) for line in lines]  # str to list
        self.img_ids = records
        self.name2id=dict(zip(CLASSES_NAME,range(len(CLASSES_NAME))))
        self.id2name = {v:k for k,v in self.name2id.items()}
        self.resize_size=resize_size
        self.CLASSES_NAME = CLASSES_NAME
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.train = is_train
        self.augment = augment
        print("INFO=====>odgt dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self,index):

        img_id=self.img_ids[index]
        nori_id = img_id.get('ID', None)
        img = imdecode(nf.get(nori_id))
        img = Image.fromarray(img)
        # image_path = os.path.join(self._imgpath, img_id['fpath'])
        # img = Image.open(image_path)
        boxes=[]
        classes=[]
        for gtbox in img_id['gtboxes']:
            if gtbox['tag'] not in self.CLASSES_NAME:
                continue
            boxes.append(gtbox['box'])
            classes.append(self.name2id[gtbox['tag']])
        boxes=np.array(boxes,dtype=np.float32)
        #xywh-->xyxy
        boxes[...,2:]=boxes[...,2:]+boxes[...,:2]

        if self.train:
            if random.random() < 0.5:
                img, boxes = flip(img, boxes)
            if self.augment is not None:
                img, boxes = self.augment(img, boxes)
        img = np.array(img)
        img,boxes,scale=self.preprocess_img_boxes(img,boxes,self.resize_size)

        img=transforms.ToTensor()(img)
        boxes=torch.from_numpy(boxes)
        classes=torch.LongTensor(classes)


        return img,boxes,classes


    def preprocess_img_boxes(self,image,boxes,input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
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

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes, scale


    def collate_fn(self,data):
        imgs_list,boxes_list,classes_list=zip(*data)
        assert len(imgs_list)==len(boxes_list)==len(classes_list)
        batch_size=len(boxes_list)
        pad_imgs_list=[]
        pad_boxes_list=[]
        pad_classes_list=[]

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img=imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))


        max_num=0
        for i in range(batch_size):
            n=boxes_list[i].shape[0]
            if n>max_num:max_num=n
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))


        batch_boxes=torch.stack(pad_boxes_list)
        batch_classes=torch.stack(pad_classes_list)
        batch_imgs=torch.stack(pad_imgs_list)

        return batch_imgs,batch_boxes,batch_classes

if __name__=="__main__":
    # pass
    # train_dataset = OdgtDataset(root_dir='/data/wurenji/code_new/FCOS-PyTorch-37.2AP/data/dajinju/',resize_size=[800,1333],
    #                                 set_class = 'fangzhenchui_2class',CLASSES_NAME = ("__background__ ","fangzhenchui_good","fangzhenchui_bad",),is_train = True, augment = None)

    # print(train_dataset._annopath)
    now = time.time()
    print(now)
    # CLASSES_NAME = ("__background__ ","fangzhenchui_good","fangzhenchui_bad",)
    CLASSES_NAME = ["__background__ ","fangzhenchui_good","fangzhenchui_bad",]
    name2id=dict(zip(CLASSES_NAME,range(len(CLASSES_NAME))))
    print(name2id)
    #dataset=VOCDataset("/home/data/voc2007_2012/VOCdevkit/VOC2012",split='trainval')
    # for i in range(100):
    #     img,boxes,classes=dataset[i]
    #     img,boxes,classes=img.numpy().astype(np.uint8),boxes.numpy(),classes.numpy()
    #     img=np.transpose(img,(1,2,0))
    #     print(img.shape)
    #     print(boxes)
    #     print(classes)
    #     for box in boxes:
    #         pt1=(int(box[0]),int(box[1]))
    #         pt2=(int(box[2]),int(box[3]))
    #         img=cv2.rectangle(img,pt1,pt2,[0,255,0],3)
    #     cv2.imshow("test",img)
    #     if cv2.waitKey(0)==27:
    #         break
    #imgs,boxes,classes=eval_dataset.collate_fn([dataset[105],dataset[101],dataset[200]])
    # print(boxes,classes,"\n",imgs.shape,boxes.shape,classes.shape,boxes.dtype,classes.dtype,imgs.dtype)
    # for index,i in enumerate(imgs):
    #     i=i.numpy().astype(np.uint8)
    #     i=np.transpose(i,(1,2,0))
    #     i=cv2.cvtColor(i,cv2.COLOR_RGB2BGR)
    #     print(i.shape,type(i))
    #     cv2.imwrite(str(index)+".jpg",i)








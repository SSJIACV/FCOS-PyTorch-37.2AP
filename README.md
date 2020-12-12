## FCOS: Pytorch Implementation Support Odgt_Nori、PASCAL VOC and MS COCO
**FCOS的简洁实现：**本项目属于codebase建设的一部分，旨在支持odgt格式的数据标注以及Nori的image存储格式，在不用改任何参数（包括数据路径）的情况下，也能一键跑通本项目代码。

对于odgt标注格式的数据，本项目支持两种读取方式：一种是将odgt标注文件和images放在本地路径下（以下统称为**：Local**），另一种是odgt标注文件存于OSS中，image是nori存储方式（以下统称为**：OSS**），下面对这两种方式分别进行说明：

### OSS

#### 数据

不需要本地存储

#### 参数设置

对应 config_nori_dianwang.py 

```
train_odgt='s3://jiashuaishuai/dianwang_data/dajinju/annotations/zhongchui_2class_train.odgt' # 根据odgt和其中的nori_id来读取标注和Image

```

#### dataset.py

对应dataset文件夹中的odgt_nori_dataset.py

#### 训练脚本

对应train_odgt_nori.py

#### 评估脚本

对应eval_odgt_nori.py

### Local

#### 数据

```
# 你的数据结构应该按照以下格式：（以导地线为例）

data/
    -daodixian/
        -images/
            -*.jpg
        -annotations
            - daodixian_2class_train.odgt
            - daodixian_2class_val.odgt
注：images中存训练和验证的所有images
daodixian_2class就是config_dianwang.py 中的set_class参数，以 _train.odgt 和_val.odgt 结尾是固定的
```

#### 参数设置

对应 config_dianwang.py 

```
data_root_dir='./data/daodixian/' # 表示数据的根目录
set_class = 'daodixian_2class'  # 表示odgt标注文件除 _train.odgt以外的前缀
```

#### dataset.py

对应dataset文件夹中的odgt_dataset.py

#### 训练脚本

对应train_odgt.py

#### 评估脚本

对应eval_odgt.py

###  AP Result
| PASCAL VOC (800px) | COCO(800px) |
| :-----------: | :-----------------: |
|     78.7 (IoU.5)      |      **37.2**       |

### Requirements  
* opencv-python  
* pytorch >= 1.0  
* torchvision >= 0.4. 
* matplotlib
* cython
* numpy == 1.17
* Pillow
* tqdm
* pycocotools

### Results in coco 
Train coco2017 on 4 Tesla-V100, 4 imgs for each gpu, init lr=1e-2  using GN,central sampling,GIou.

You can run the train_coco.py, train 24 epoch and you can get the result. You need to change the coco2017 path.

You can download the 37.2 ap result in [Baidu driver link](https://pan.baidu.com/s/1tv0F_nmwiJ47C3zJ5v_C0g), password: cnwm,then put it in checkpoint folder, then run the coco_eval.py

### Results in Pascal Voc
Train Voc07+12 on 2 Tesla-V100 , 8 imgs for each gpu, init lr=1e-2  using GN,central sampling,GIou.  

You can run the train_voc.py, train 30 epoch and you can get the result. You need to change the PASCAL07+12 path, you can reference to this repo:https://github.com/YuwenXiong/py-R-FCN

You can download the 78.7 ap result in [Baidu driver link](https://pan.baidu.com/s/1aB0irfcJQM5WTlmiKFOfEA), password:s4cp, then put it in checkpoint folder, then run the eval_voc.py and

### Reference

thanks to [@zhenghao977](https://github.com/VectXmy), I referenced [his codes.](https://github.com/zhenghao977/FCOS-PyTorch-37.2AP)





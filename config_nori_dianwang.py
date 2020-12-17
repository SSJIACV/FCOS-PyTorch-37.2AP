class Config_nori_myself():
    #backbone
    pretrained=True
    freeze_stage_1=False
    freeze_bn=False

    #fpn
    fpn_out_channels=256
    use_p5=True
    
    #head
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]
    resize_size=[800,1333]
    save_path = "./checkpoint/zhongchui_2class_warmup101_epoch51_1214/"
    pretrained_coco = False
    coco_weight_path = './checkpoint/coco_37.2_classes_2.pth'
    train_odgt = 's3://jiashuaishuai/dianwang_data/dajinju/annotations/zhongchui_2class_train.odgt'
    CLASSES_NAME = ["__background__ ","zhongchui_good","zhongchui_bad",]
    class_num = len(CLASSES_NAME)-1
    WARMPUP_STEPS = 101
    LR_INIT = 2e-3

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000
    val_odgt_now = 's3://jiashuaishuai/dianwang_data/dajinju/annotations/zhongchui_2class_val.odgt'
    weight_path = "./checkpoint/zhongchui_2class_warmup101_epoch51_1214/model_26.pth"
    det_save_odgt = './val_results/zhongchui_2class_warmup101_epoch26_1214.odgt'
    det_save_eval_log_txt = './val_results/zhongchui_2class_warmup101_epoch26_1214.txt'
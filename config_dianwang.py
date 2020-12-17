class Config_myself():
    #backbone
    pretrained=True
    # freeze_stage_1=True
    # freeze_bn=True
    freeze_stage_1=False
    freeze_bn=False

    #fpn
    fpn_out_channels=256
    use_p5=True
    
    #head
    use_GN_head=True
    prior=0.01
    add_centerness=True
    # add_centerness=False
    # cnt_on_reg=True
    cnt_on_reg=False

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]
    resize_size=[800,1333]
    save_path = "./checkpoint/zhongchui_2class_cnt_reg_1208/"
    pretrained_coco = False
    coco_weight_path = './checkpoint/coco_37.2_classes_2.pth'
    data_root_dir='./data/daodixian/'
    set_class = 'daodixian_2class'
    # CLASSES_NAME = ["__background__ ","fangzhenchui_good","fangzhenchui_bad",]
    CLASSES_NAME = ["__background__ ","zhongchui_good","zhongchui_bad"]
    class_num = len(CLASSES_NAME)-1
    WARMPUP_STEPS = 501
    LR_INIT = 2e-3

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000
    # val_odgt_now = './data/dajinju/annotations/fangzhenchui_2class_val.odgt'
    val_odgt_now = './data/dajinju/annotations/zhongchui_2class_val.odgt'
    VAL_IMGS = './data/dajinju/images/'
    weight_path = "./checkpoint/zhongchui_2class_cnt_reg_1208/model_26.pth"
    det_save_odgt = './val_results/zhongchui_2class_cnt_reg_epoch26_1208.odgt'
    det_save_eval_log_txt = './val_results/zhongchui_2class_cnt_reg_epoch26_1208.txt'
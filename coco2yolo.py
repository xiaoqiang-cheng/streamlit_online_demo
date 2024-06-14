#COCO 格式的数据集转化为 YOLO 格式的数据集
#--json_path 输入的json文件路径
#--save_path 保存的文件夹名字，默认为当前目录下的labels。

import os
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
#这里根据自己的json文件位置，换成自己的就行
parser.add_argument('--json_path', default='/home/uisee/COCODetection/annotations/instances_train2017.json',type=str, help="input: coco format(json)")
#这里设置.txt文件保存位置
parser.add_argument('--save_path', default='/home/uisee/COCODetection/yolo/', type=str, help="specify where to save the output dir of labels")
arg = parser.parse_args()

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
#round函数确定(xmin, ymin, xmax, ymax)的小数位数
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x, y, w, h)

if __name__ == '__main__':
    json_file =   arg.json_path # COCO Object Instance 类型的标注
    ana_txt_save_path = arg.save_path  # 保存的路径

    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    id_map = {3: 0, 1: 1, 6:0, 8:0} # coco数据集的id不连续！重新映射一下再输出！
    # with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
    #     # 写入classes.txt
    #     for i, category in enumerate(data['categories']):
    #         f.write(f"{category['name']}\n")
    #         id_map[category['id']] = i
    # print(id_map)
    #这里需要根据自己的需要，更改写入图像相对路径的文件位置。
    new_anno_data = []
    # import ipdb
    # ipdb.set_trace()
    database = {}

    crown_ann_image_dict = set()
    for ann in data['annotations']:
        if ann["category_id"] in id_map.keys():
            if ann['image_id'] not in database.keys():
                database[ann['image_id']] = []
            database[ann['image_id']].append(ann)
            print(ann['image_id'])
            if ann['iscrowd']:
                crown_ann_image_dict.add(ann['image_id'])

    print(len(crown_ann_image_dict), "is crown")
    for crown_img in crown_ann_image_dict:
        database.pop(crown_img)


    list_file = open(os.path.join(ana_txt_save_path, 'train2017.txt'), 'w')
    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致

        anno_lines = ""
        # for ann in data['annotations']:
        # for ann in new_anno_data:
        #     if ann['image_id'] == img_id:
        if img_id in database.keys():
            for ann in database[img_id]:
                box = convert((img_width, img_height), ann["bbox"])
                if ann["category_id"] in id_map.keys():
                    anno_lines += "%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3])
        if anno_lines != "":

            f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
            f_txt.write(anno_lines)
            f_txt.close()
            #将图片的相对路径写入train2017或val2017的路径
            list_file.write('./images/%s.jpg\n' %(head))
    list_file.close()

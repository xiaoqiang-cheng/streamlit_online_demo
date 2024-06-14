
import numpy as np
import xml.etree.ElementTree as ET
import warnings
import glob
import copy
import json
import os, sys
import time

import pickle

global_compare_items_var = None

def serialize_data(data:dict, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
    except Exception as e:
        print("序列化数据时出现错误:", e)

def deserialize_data(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        print("反序列化数据时出现错误:", e)
        return None


if not os.path.exists("tmp"):
    os.makedirs("tmp")


def get_datatime_tail():
    from datetime import datetime
    # 获取当前时刻
    current_time = datetime.now()
    # 将时间对象格式化为字符串，精确到分
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M")
    return formatted_time

class redirect:
    content = ""
    def write(self,str):
        self.content += str
    def flush(self):
        self.content = ""


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def parse_json(filename):
    with open(filename, 'r', encoding="utf-8", errors='ignore') as f:
        content = ''.join(f.readlines())
    return json.loads(content)


def write_json(json_data,json_name):
    # Writing JSON data
    with open(json_name, 'w', encoding="utf-8") as f:
        json.dump(json_data, f,indent=4)


def cocogt_to_cocodet(gtfile, detfile):
    gt_dict = parse_json(gtfile)
    det_dict = copy.deepcopy(gt_dict['annotations'])
    for x in det_dict:
        x['score'] = 1
    write_json(det_dict, detfile)

def yolotxt_to_cocodet(uiseefloder, coco_gt_images_dict, outfile, ImageHeight = 720, ImageWidth = 1280):
    det_txt_list = [x for x in os.listdir(uiseefloder) if x.endswith(".txt")]
    ret_list = []
    import ipdb
    ipdb.set_trace()
    for x in coco_gt_images_dict["images"]:
        except_txt_name = x["file_name"][0:-4] + ".txt"
        # import ipdb
        # ipdb.set_trace()
        if except_txt_name in det_txt_list:
            txt_path = os.path.join(uiseefloder, except_txt_name)
            # txt: cls xmin, ymin, w, h score id ...
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print(txt_path)
                txt_ret = np.loadtxt(txt_path, dtype=np.float32)
            if len(txt_ret.shape) == 1 and txt_ret.shape[0] != 0:
                txt_ret = [txt_ret.tolist()]
            else:
                txt_ret = txt_ret.tolist()
            for num in txt_ret:
                image_result = {}
                if int(num[0]) == 0 or int(num[0]) == 4:
                    image_result["category_id"] = 4
                else:
                    image_result["category_id"] = int(num[0])
                image_result["score"] = 1
                image_result['file_name'] = x["file_name"]
                image_result["image_id"] = x["id"]

                cx = num[1] * ImageWidth
                cy = num[2] * ImageHeight
                box_w = num[3] * ImageWidth
                box_h = num[4] * ImageHeight

                # left top
                x0 = max(cx - box_w / 2, 0)
                y0 = max(cy - box_h / 2, 0)

                bbox = [x0, y0, box_w, box_h]
                # area = box_w * box_h
                image_result["bbox"] = bbox
                ret_list.append(image_result)
    write_json(ret_list, outfile)

def uiseetxt_to_cocodet(uiseefloder, coco_gt_images_dict, outfile, score_thres = 0.0):
    det_txt_list = [x for x in os.listdir(uiseefloder) if x.endswith(".txt")]
    ret_list = []
    for x in coco_gt_images_dict["images"]:
        except_txt_name = os.path.splitext(x["file_name"])[0] + ".txt"
        if except_txt_name in det_txt_list:
            txt_path = os.path.join(uiseefloder, except_txt_name)
            # txt: cls xmin, ymin, w, h score id ...

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                txt_ret = np.loadtxt(txt_path, dtype=np.float32)
            if len(txt_ret.shape) == 1 and txt_ret.shape[0] != 0:
                txt_ret = [txt_ret.tolist()]
            else:
                txt_ret = txt_ret.tolist()
            for num in txt_ret:

                image_result = {}
                image_result["category_id"] = int(num[0])
                image_result["score"] = num[5]
                image_result['file_name'] = x["file_name"]
                image_result["image_id"] = x["id"]
                image_result["bbox"] = num[1:5]

                if image_result["score"] < score_thres: continue

                # cx = num[1] * 1280
                # cy = num[2] * 720
                # box_w = num[3] * 1280
                # box_h = num[4] * 720

                # # left top
                # x0 = max(cx - box_w / 2, 0)
                # y0 = max(cy - box_h / 2, 0)

                # bbox = [x0, y0, box_w, box_h]
                # # area = box_w * box_h
                # image_result["bbox"] = bbox

                ret_list.append(image_result)

    write_json(ret_list, outfile)


def vocgt_to_cocogt(xml_dir, cocofile, PRE_DEFINE_CATEGORIES = None):
    START_BOUNDING_BOX_ID = 1
    START_IMAGE_ID = 1
    # If necessary, pre-define category and its id
    #  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
    #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
    #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
    #  "motorbike": 14, "person": 15, "pottedplant": 16,
    #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}

    def get(root, name):
        vars = root.findall(name)
        return vars

    def get_and_check(root, name, length):
        vars = root.findall(name)
        if len(vars) == 0:
            raise ValueError("Can not find %s in %s." % (name, root.tag))
        if length > 0 and len(vars) != length:
            raise ValueError(
                "The size of %s is supposed to be %d, but is %d."
                % (name, length, len(vars))
            )
        if length == 1:
            vars = vars[0]
        return vars

    def get_categories(xml_files):
        """Generate category name to id mapping from a list of xml files.

        Arguments:
            xml_files {list} -- A list of xml file paths.

        Returns:
            dict -- category name to id mapping.
        """
        classes_names = []
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall("object"):
                classes_names.append(member[0].text)
        classes_names = list(set(classes_names))
        classes_names.sort()
        return {name: i for i, name in enumerate(classes_names)}

    def convert(xml_files, json_file):
        json_dict = {"images": [], "annotations": [], "categories": []}
        if PRE_DEFINE_CATEGORIES is not None:
            categories = PRE_DEFINE_CATEGORIES
        else:
            categories = get_categories(xml_files)
        bnd_id = START_BOUNDING_BOX_ID
        image_id = START_IMAGE_ID
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            path = get(root, "path")
            if len(path) == 1:
                filename = os.path.basename(path[0].text)
            elif len(path) == 0:
                filename = get_and_check(root, "filename", 1).text
            else:
                raise ValueError("%d paths found in %s" % (len(path), xml_file))
            ## The filename must be a number
            size = get_and_check(root, "size", 1)
            width = int(get_and_check(size, "width", 1).text)
            height = int(get_and_check(size, "height", 1).text)
            image = {
                "file_name": filename,
                "height": height,
                "width": width,
                "id": image_id,
            }
            json_dict["images"].append(image)
            ## Currently we do not support segmentation.
            #  segmented = get_and_check(root, 'segmented', 1).text
            #  assert segmented == '0'
            for obj in get(root, "object"):
                category = get_and_check(obj, "name", 1).text
                if category not in categories:
                    new_id = len(categories)
                    categories[category] = new_id
                category_id = categories[category]
                bndbox = get_and_check(obj, "bndbox", 1)
                xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
                ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
                xmax = int(get_and_check(bndbox, "xmax", 1).text)
                ymax = int(get_and_check(bndbox, "ymax", 1).text)
                assert xmax > xmin
                assert ymax > ymin
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {
                    "area": o_width * o_height,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [xmin, ymin, o_width, o_height],
                    "category_id": category_id,
                    "id": bnd_id,
                    "ignore": 0,
                    "segmentation": [],
                }
                json_dict["annotations"].append(ann)
                bnd_id = bnd_id + 1
            image_id += 1
        for cate, cid in categories.items():
            cat = {"supercategory": "none", "id": cid, "name": cate}
            json_dict["categories"].append(cat)

        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        json_fp = open(json_file, "w")
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        json_fp.close()

    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    convert(xml_files, out_coco_anno)


def uiseegt_to_cocogt(uisee_anno, out_coco_anno, PRE_DEFINE_CATEGORIESl:dict):
    # PRE_DEFINE_CATEGORIESl:{
    #     "red":1,....
    # }
    START_BOUNDING_BOX_ID = 1
    UISEE_IMAGE_HEIGHT = 720
    UISEE_IMAGE_WIDTH = 1280
    uisee_anno_dict = parse_json(uisee_anno)
    json_dict = {"images": [], "annotations": [], "categories": []}
    bnd_id = START_BOUNDING_BOX_ID

    for image_anno in uisee_anno_dict:
        filename = image_anno['file_obj'].split('/')[-1]
        image = {
                    "file_name": filename,
                    "height": UISEE_IMAGE_HEIGHT,
                    "width": UISEE_IMAGE_WIDTH,
                    "id": image_anno['id'],
        }
        json_dict['images'].append(image)
        for bbox_anno in image_anno['result']:
            bbox = {
                "segmentation" : [],
                "area" : bbox_anno["data"][2] * bbox_anno["data"][3],
                "iscrowd" : 0,
                "image_id" : image_anno['id'],
                "id" : bnd_id,
                "category_id": PRE_DEFINE_CATEGORIESl[bbox_anno['tagtype']],
                "bbox": bbox_anno["data"][3],
                "ignore": 0
            }
            json_dict['annotations'].append(bbox)
            bnd_id += 1
    for key, result in PRE_DEFINE_CATEGORIESl.items():
        cat_item = {
            "supercategory" : "light",
            "id" : result,
            "name" : key
        }
        json_dict["categories"].append(cat_item)
    write_json(json_dict, out_coco_anno)


# def simplify_cocogt(coco_anno):
#     # PRE_DEFINE_CATEGORIESl:{
#     #     "red":1,....
#     # }
#     START_BOUNDING_BOX_ID = 1
#     UISEE_IMAGE_HEIGHT = 720
#     UISEE_IMAGE_WIDTH = 1280
#     coco_anno_dict = parse_json(coco_anno)
#     json_dict = {"images": [], "annotations": [], "categories": []}
#     bnd_id = START_BOUNDING_BOX_ID
#     import ipdb
#     ipdb.set_trace()

#     image_nums = len(coco_anno_dict['images'])

#     for i in range(image_nums):
#         image_anno = coco_anno_dict['images'][i]
#         filename = image_anno['file_obj'].split('/')[-1]
#         image = {
#                     "file_name": filename,
#                     "height": UISEE_IMAGE_HEIGHT,
#                     "width": UISEE_IMAGE_WIDTH,
#                     "id": image_anno['id'],
#         }

#     for image_anno in coco_anno_dict:
#         filename = image_anno['file_obj'].split('/')[-1]
#         image = {
#                     "file_name": filename,
#                     "height": UISEE_IMAGE_HEIGHT,
#                     "width": UISEE_IMAGE_WIDTH,
#                     "id": image_anno['id'],
#         }

#         bbox_anno = json_dict['annotations'][i]
#         json_dict['images'].append(image)
#         for bbox_anno in image_anno['result']:
#             bbox = {
#                 "segmentation" : [],
#                 "area" : bbox_anno["data"][2] * bbox_anno["data"][3],
#                 "iscrowd" : 0,
#                 "image_id" : image_anno['id'],
#                 "id" : bnd_id,
#                 "category_id": PRE_DEFINE_CATEGORIESl[bbox_anno['tagtype']],
#                 "bbox": bbox_anno["data"][3],
#                 "ignore": 0
#             }
#             json_dict['annotations'].append(bbox)
#             bnd_id += 1
#     for key, result in PRE_DEFINE_CATEGORIESl.items():
#         cat_item = {
#             "supercategory" : "light",
#             "id" : result,
#             "name" : key
#         }
#         json_dict["categories"].append(cat_item)
#     write_json(json_dict, out_coco_anno)

if __name__ == "__main__":
    # cocogt_to_cocodet("data/test_data.json")
    # obj_gt_path = "/home/uisee/dolly_test/ground_truth.json"
    # coco_gt_images_dict = parse_json(obj_gt_path)
    # uiseefloder = "/home/uisee/Downloads/chy_results"
    # outfile = "./2022-07-05-17-07-16.json"
    # uiseetxt_to_cocodet(uiseefloder, coco_gt_images_dict, outfile)
    # simplify_cocogt("/home/uisee/COCODetection/annotations/instances_train2017.json")
    gt_dict = parse_json('/home/uisee/MainDisk/DollyMonitor/train_check/dataset_0028/ground_truth.json')
    yolotxt_to_cocodet("/home/uisee/MainDisk/DollyMonitor/train_check/dataset_0028/labels", gt_dict, "ground_truth.json")
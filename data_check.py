import numpy as np
import os

import os
import json
import numpy as np
import cv2
from tqdm import tqdm

def get_bboxes_from_file(dir_, cls_id=[1],
        score_thres = 0.0):
    # Store bboxes results
    '''
        {
            'image_fname' : np.array(
                [x1 y1 x2 y2],
            )
        }
    '''
    ret = {}
    if os.path.isfile(dir_):
        # Assuming COCO format
        with open(dir_, 'r') as f:
            coco_data = json.load(f)
            # Create a mapping from image_id to file_name
            image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

            for annotation in coco_data['annotations']:
                if annotation['category_id'] in cls_id:
                    image_id = annotation['image_id']
                    file_name = image_id_to_filename[image_id]
                    file_name = os.path.splitext(file_name)[0]
                    bbox = annotation['bbox']  # COCO bbox format: [x, y, width, height]
                    x1, y1, w, h = bbox
                    x2 = x1 + w
                    y2 = y1 + h
                    if file_name not in ret:
                        ret[file_name] = []
                    ret[file_name].append([x1, y1, x2, y2])
    elif os.path.isdir(dir_):
        # Assuming YOLO format
        for fname in os.listdir(dir_):
            if fname.endswith('.txt'):
                image_id = os.path.splitext(fname)[0]
                with open(os.path.join(dir_, fname), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if int(parts[0]) in cls_id:
                            cls, x1, y1, w, h, score = map(float, parts)
                            if score < score_thres:continue
                            x1 = x1
                            y1 = y1
                            x2 = x1 + w
                            y2 = y1 + h
                            if image_id not in ret:
                                ret[image_id] = []
                            ret[image_id].append([x1, y1, x2, y2, score])
                # Convert list to numpy array
                if image_id in ret:
                    ret[image_id] = np.array(ret[image_id])

    return ret


def compute_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union != 0 else 0

def compute_area(box1):
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    return box1_area

def find_false_positives_and_negatives(predicted_boxes, ground_truth_boxes, iou_threshold=0.3):
    """找出误检和漏检目标"""
    false_positives = []
    false_negatives = []
    true_positives = []
    matched_gt_indices = set()

    for pred_box in predicted_boxes:
        best_iou = 0
        best_gt_index = -1
        for gt_index, gt_box in enumerate(ground_truth_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_iou >= iou_threshold:
            matched_gt_indices.add(best_gt_index)
            true_positives.append(pred_box)
        else:
            false_positives.append(pred_box)

    for gt_index, gt_box in enumerate(ground_truth_boxes):
        if gt_index not in matched_gt_indices:
            false_negatives.append(gt_box)

    return false_positives, false_negatives, true_positives


def draw_detect_box(img, bboxes, info="error"):
    color = {
        "error": (0, 0, 255),
        "miss" : (0, 255, 255),
        "gt" : (0, 255, 0),
        "det": (255, 0, 0)
    }

    scale = {
        "error" : 1,
        "miss"  : 1,
        "gt"    : 0.5,
        "det"   : 0.5,
    }

    for b in bboxes:
        cv2.rectangle(img,
            (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
            color[info], 2)

        msg = info
        if len(b) == 5:
            msg += ": %.3f"%b[-1]

        cv2.putText(img,
                msg,
                (int(b[0]), int(b[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, scale[info],
                color[info], 2)
    return img


def save_vis(img_fname, img, false_posi, fasle_navi):
    false_posi_path = "check/error"
    false_navi_path = "check/miss"

    if not os.path.exists(false_posi_path):
        os.makedirs(false_posi_path)

    if not os.path.exists(false_navi_path):
        os.makedirs(false_navi_path)

    # img_fname = key + ".jpg"
    if len(false_posi) != 0:
        img_path = os.path.join(false_posi_path, img_fname)
        cv2.imwrite(img_path, img)
    if len(fasle_navi) != 0:
        img_path = os.path.join(false_navi_path, img_fname)
        cv2.imwrite(img_path, img)


def evaluate_det_bbox_error(detection_dir, ground_truth_dir, score_thres = 0.5,
                        images_dir = "./merge_case", area_thres=1250):

    # 获取
    gt_dict = get_bboxes_from_file(ground_truth_dir, score_thres=score_thres)
    det_dict = get_bboxes_from_file(detection_dir, score_thres=score_thres)

    results = {
        # 正常
        "normal" : 0,
        # 假阳性 误检
        "false_positives": 0,
        # 假阳性 误检
        "false_positives_bboxes": 0,
        # 假阴性 漏检
        "false_negatives": 0,
        "false_score": [],
        "false_positives_database" : {

        },

        "false_negatives_database" : {

        },

        "true_positives_database" : {

        }
    }

    sort_image_dir = os.listdir(images_dir)
    sort_image_dir.sort()

    for img_fname in tqdm(sort_image_dir):
        img_key = os.path.splitext(img_fname)[0]
        false_positives = []
        false_negatives = []
        if (img_key not in gt_dict.keys()) and (img_key not in det_dict.keys()):
            results['normal'] += 1
        else:
            false_positives_tmp = []
            false_negatives_tmp = []
            ture_positives_tmp = []

            if img_key not in gt_dict.keys():
                false_positives_tmp = det_dict[img_key]
            elif img_key not in det_dict.keys():
                false_negatives_tmp = gt_dict[img_key]
            else:
                false_positives_tmp, false_negatives_tmp, ture_positives_tmp = find_false_positives_and_negatives(det_dict[img_key], gt_dict[img_key])

            false_positives = []
            false_negatives = []
            # import ipdb
            # ipdb.set_trace()
            for fp in false_positives_tmp:
                if compute_area(fp) > area_thres:
                    false_positives.append(fp)

            for fn in false_negatives_tmp:
                if compute_area(fn) > area_thres:
                    false_negatives.append(fn)

            if len(false_negatives) == 0 and len(false_positives) == 0:
                results['normal'] += 1

            if len(ture_positives_tmp) != 0:
                results['true_positives_database'][img_fname] = ture_positives_tmp

            if len(false_positives) != 0:
                results['false_positives'] += 1
                results['false_positives_bboxes'] += len(false_positives)
                results["false_positives_database"][img_fname] = false_positives
                for bb in false_positives:
                    results['false_score'].append(bb[-1])

            if len(false_negatives) != 0:
                results['false_negatives'] += 1
                results["false_negatives_database"][img_fname] = false_negatives

            if False:
                img_fpath = os.path.join(images_dir, img_fname)
                cvimg = cv2.imread(img_fpath)

                if img_key in det_dict.keys():
                    cvimg = draw_detect_box(cvimg, det_dict[img_key], info="det")

                if img_key in gt_dict.keys():
                    cvimg = draw_detect_box(cvimg, gt_dict[img_key], info="gt")

                cvimg = draw_detect_box(cvimg, false_positives, info="error")
                cvimg = draw_detect_box(cvimg, false_negatives, info="miss")
                save_vis(img_fname, cvimg, false_positives, false_negatives)

                # cv2.imshow("test", cvimg)
                # cv2.waitKey()
    return results
    # sample_num = len(sort_image_dir)

    # print(results)
    # print("sample num:", sample_num)

    # print("TPR: ", results['normal'] / sample_num)
    # print("FPR: ", results['false_positives'] / sample_num)
    # print("FPB: ", results['false_positives_bboxes'])

    # if len(results['false_score']) > 0:
    #     results['false_score'].sort()
    #     fp_length = len(results['false_score'])
    #     mFPS = sum(results['false_score']) / fp_length
    #     MaFPS = max(results['false_score'])
    #     MeFPS = results['false_score'][int(fp_length/2)]

    #     print('mFPS:', mFPS)
    #     print('MeFPS:', MeFPS)
    #     print('MaFPS:', MaFPS)
    # print("FNR: ", results['false_negatives'] / sample_num)


if __name__=="__main__":
    ret = evaluate_det_bbox_error("history/baseline-2024-06-13_18-38", "merge_case/ground_truth_merge.json", images_dir="merge_case/imgs")
    print(ret['false_positives'])

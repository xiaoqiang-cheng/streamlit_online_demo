import numpy as np
import os

import os
import json
import numpy as np
import cv2
from tqdm import tqdm

def get_bboxes_from_file(dir_, cls_id=[0, 1], score_thres = 0.0):
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



images_dir = "/home/uisee/MainDisk/image2"
detection_dir =  "/home/uisee/MainDisk/dl_multitask_prediction"

sort_image_dir  = os.listdir(images_dir)
sort_image_dir.sort()
need_save = True
det_dict = get_bboxes_from_file(detection_dir)

for img_fname in tqdm(sort_image_dir):
    img_key = os.path.splitext(img_fname)[0]
    img_fpath = os.path.join(images_dir, img_fname)
    cvimg = cv2.imread(img_fpath)
    img = draw_detect_box(cvimg, det_dict[img_key], info="det")
    cv2.imwrite(img_fpath, cvimg)
    # cv2.imshow("cvimg", cvimg)
    # cv2.waitKey(0)
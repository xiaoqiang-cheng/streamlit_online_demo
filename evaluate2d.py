from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils import *

def eval_mAP_by_coco(det_file, gt_file, catid = None):

    # origin_stdout = sys.stdout
    # sys.stdout = redirect()
    cocoGt = COCO(gt_file)
    try:
        cocoDt = cocoGt.loadRes(det_file)
    except:
        cocoDt = COCO()
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    if catid in cocoEval.params.catIds:
        cocoEval.params.catIds = catid
    cocoEval.evaluate()
    cocoEval.accumulate()

    origin_stdout = sys.stdout
    sys.stdout = redirect()
    cocoEval.summarize()

    content = sys.stdout.content
    sys.stdout = origin_stdout
    print(content)
    return content

    # catId = cocoGt.getAnnIds(catIds=cocoEval.params.catIds)
    # # for i, x in enumerate(cocoEval.stats):
    # #     ret_dict[coco_head[i]] = x
    # return round(cocoEval.stats[0] * 100, 1), round(cocoEval.stats[1] * 100, 1), len(catId)



def eval_mAP_for_uisee(det_dir, gt_file):
    basename = os.path.splitext(os.path.basename(det_dir))[0]
    target_det_coco_dir = os.path.join("tmp", basename + ".json")
    coco_gt_images_dict = parse_json(gt_file)
    uiseetxt_to_cocodet(det_dir, coco_gt_images_dict, target_det_coco_dir)
    return eval_mAP_by_coco(target_det_coco_dir, gt_file)


if __name__=="__main__":

    eval_mAP_for_uisee("/home/uisee/Downloads/traffic_light8155/light_label_dsp",
                "/home/uisee/MainDisk/TrafficLight/traffic_light_case/case_00_00_0000/ground_truth.json")
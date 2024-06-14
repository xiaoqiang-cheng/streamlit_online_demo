from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils import *
import matplotlib.pyplot as plt

def eval_mAP_by_coco(det_file, gt_file, catid = None):


    cocoGt = COCO(gt_file)
    try:
        cocoDt = cocoGt.loadRes(det_file)
    except:
        cocoDt = COCO()
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    # if catid in cocoEval.params.catIds:

    origin_stdout = sys.stdout
    sys.stdout = redirect()
    if catid is not None:
        print("============[%d] Cls mAP============"%catid)
        cocoEval.params.catIds = catid
        cocoEval.params.iouThrs = np.array([0.5])
        cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 1e5 ** 2]]
        cocoEval.params.areaRngLbl = ['all', 'small', 'large']
    else:
        print("============All Cls mAP============")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print("===================================")

    ret = {
        "mAP@0.5"      : cocoEval.stats[1],
        "recall@0.5"   : cocoEval.stats[8],
        "mAP@large"    : cocoEval.stats[5],
        "mAP@small"    : cocoEval.stats[3],
        "recall@large" : cocoEval.stats[11],
        "recall@small" : cocoEval.stats[9],
    }

    content = sys.stdout.content
    sys.stdout = origin_stdout
    print(content)
    return content, ret

    # catId = cocoGt.getAnnIds(catIds=cocoEval.params.catIds)
    # # for i, x in enumerate(cocoEval.stats):
    # #     ret_dict[coco_head[i]] = x
    # return round(cocoEval.stats[0] * 100, 1), round(cocoEval.stats[1] * 100, 1), len(catId)



def eval_mAP_for_uisee(det_dir, gt_file, score_thres = 0.5):
    basename = os.path.splitext(os.path.basename(det_dir))[0]
    target_det_coco_dir = os.path.join("tmp", basename + ".json")
    coco_gt_images_dict = parse_json(gt_file)
    uiseetxt_to_cocodet(det_dir, coco_gt_images_dict, target_det_coco_dir, score_thres)
    return eval_mAP_by_coco(target_det_coco_dir, gt_file, catid=1)


if __name__=="__main__":
    import sys
    det_dir = sys.argv[1]
    eval_mAP_for_uisee(det_dir,
                "tmp/ground_truth.json")

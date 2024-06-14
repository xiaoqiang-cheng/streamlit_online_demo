import streamlit as st
import os
from evaluate2d import eval_mAP_for_uisee
from utils import get_datatime_tail, serialize_data, deserialize_data
import pandas as pd
from data_check import evaluate_det_bbox_error
import copy
import utils

st.set_page_config(layout="wide",
                page_title="Evaluate Object Detection",
                page_icon="📡",
            )
utils.global_compare_items_var = None
if "global_metric_database" not in st.session_state.keys():
    st.session_state.global_metric_database = {}
if "curr_eval_df" not in st.session_state.keys():
    st.session_state.curr_eval_df = {}

# if "compare_commit_id" not in st.session_state.keys():
#     st.session_state.compare_commit_id = 0


if "bbox_metric_ret" not in st.session_state.keys():
    st.session_state.bbox_metric_ret = {}
# st.session_state.curr_eval_df = {}

def save_and_utgz_uploaded_file(uploadedfile):
    tgz_name = os.path.join("tmp", uploadedfile.name)
    with open(tgz_name, "wb") as f:
        f.write(uploadedfile.getbuffer())

    fname = uploadedfile.name.split(".tar")[0]

    exp_fname = os.path.join("tmp", fname)
    if os.path.exists(exp_fname):
        os.system("rm -rf %s"%exp_fname)

    os.system("tar -xf %s -C %s"%(tgz_name, "tmp"))
    return exp_fname

def save_coco_json(uploadedfile):
    tgz_name = os.path.join("tmp", uploadedfile.name)
    with open(tgz_name, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return tgz_name




def tmp_warn_log_msg(msg):
    st.sidebar.error(msg,  icon="ℹ️")



def sidebar_ui_layout():
    st.sidebar.markdown("## Dolly Person 测试集检测评估")
    st.sidebar.markdown("下载测试集")
    st.sidebar.code("sshpass -p 123456 rsync -avz --progress -e 'ssh -p 10490 -o StrictHostKeyChecking=no' root@10.9.100.43:/mnt/private_data/dolly_test_case . ")
    st.sidebar.markdown("上传检测结果 （以tar.gz保存）")
    status = True
    upload_file = st.sidebar.file_uploader("检测结果", "tar.gz", label_visibility = "collapsed")
    det_dir = None
    if upload_file is not None:
        det_dir = save_and_utgz_uploaded_file(upload_file)
    else:
        status = False

    gt_dir = "./merge_case/ground_truth_merge.json"

    score_thres = st.sidebar.text_input("置信度阈值", value=0.5)
    commit_title = st.sidebar.text_input("提交信息", placeholder = "like: yolo640")
    commit_button = st.sidebar.button("提交", use_container_width=True)

    if commit_button:
        commit_status = True
        if status:
            pass
        else:
            tmp_warn_log_msg("请上传检测结果再提交")
            commit_status = False

        if commit_title == "":
            commit_status = False
            tmp_warn_log_msg("请填写提交信息再提交")

        if commit_status:
            return det_dir, gt_dir, score_thres, commit_title

    return None

def dataset_to_pd_frame(dataset_dict):
    '''
        dataset_00xx:
        {
            "tag": xxx
            "xx" : xx
            ...
        }

    ------------
    |name |tag | |
    '''
    ret = {}
    ret["dataset"] = []
    dataset_list = list(dataset_dict.keys())
    dataset_list.sort()
    ret["enable"]  = [False] * len(dataset_list)
    for key in dataset_list:
        ret["dataset"].append(key)
        val = dataset_dict[key]
        for p, v in val.items():
            if p not in ret.keys():
                ret[p] = []
            ret[p].append(v)
    return pd.DataFrame(ret), len(dataset_list), sum(ret["train"]), sum(ret["val"])


def evaluate(det_dir, gt_dir, score_thres, commit_time, commit_title, database_dict = None):
    _, curr_eval = eval_mAP_for_uisee(det_dir, gt_dir, score_thres)
    bbox_metric_ret = evaluate_det_bbox_error(det_dir, gt_dir, score_thres=score_thres,images_dir= "./merge_case/imgs")

    eval_df = {
        "record"        : [False],
        "commit"        : [commit_title],
        "time"          : [commit_time],
        "score_thres"   : [score_thres],
        "mAP@0.5"       : [curr_eval["mAP@0.5"]],
        "recall@0.5"    : [curr_eval["recall@0.5" ]],
        "mAP@large"    : [curr_eval["mAP@large" ]],
        "mAP@small"    : [curr_eval["mAP@small"]],
        "recall@large" : [curr_eval["recall@large"]],
        "recall@small" : [curr_eval["recall@small" ]],
        "误检帧数"        : [bbox_metric_ret['false_positives']],
        "漏检帧数"        : [bbox_metric_ret['false_negatives']],
        "新增误检"      : [0],
        "新增漏检"      : [0],
        "公共误检"      : [0],
        "公共漏检"      : [0],
        "消除误检"      : [0],
        "消除漏检"      : [0],
    }

    if os.path.exists("bbox_compare_results/0.pkl"):
        baseline_bbox_metric_ret = deserialize_data("bbox_compare_results/0.pkl")
        print(baseline_bbox_metric_ret)
        baseline_error_frame = set(baseline_bbox_metric_ret['false_positives_database'].keys())
        baseline_miss_frame = set(baseline_bbox_metric_ret['false_negatives_database'].keys())

        curr_error_frame = set(bbox_metric_ret['false_positives_database'].keys())
        curr_miss_frame = set(bbox_metric_ret['false_negatives_database'].keys())

        new_error_frame = curr_error_frame - baseline_error_frame
        new_miss_frame = curr_miss_frame - baseline_miss_frame

        common_error_frame = curr_error_frame & baseline_error_frame
        common_miss_frame = curr_miss_frame & baseline_miss_frame

        reduce_error_frame = baseline_error_frame - curr_error_frame
        reduce_miss_frame = baseline_miss_frame - curr_miss_frame

        eval_df['新增误检'] = [len(new_error_frame)]
        eval_df['新增漏检'] = [len(new_miss_frame)]

        eval_df['公共误检'] = [len(common_error_frame)]
        eval_df['公共漏检'] = [len(common_miss_frame)]

        eval_df['消除误检'] = [len(reduce_error_frame)]
        eval_df['消除漏检'] = [len(reduce_miss_frame)]

    return eval_df, bbox_metric_ret


def main_ui_layout(slidebar_status):

    database_path = "./metric_database.pkl"
    if os.path.exists(database_path):
        st.session_state.global_metric_database = deserialize_data(database_path)

    st.markdown("### 当前评估结果")
    first_col, second_col = st.columns([1,4], gap="small")
    eval_df = {
        "record"        : [],
        "commit"        : [],
        "time"          : [],
        "score_thres"   : [],
        "mAP@0.5"       : [],
        "recall@0.5"    : [],
        "mAP@large"    : [],
        "mAP@small"    : [],
        "recall@large" : [],
        "recall@small" : [],
        "误检框数"        : [],
        "漏检框数"        : [],
        "新增误检"      : [],
        "新增漏检"      : [],
        "公共误检"      : [],
        "公共漏检"      : [],
        "消除误检"      : [],
        "消除漏检"      : [],
    }

    if slidebar_status is not None:
        det_dir, gt_dir, score_thres, commit_title = slidebar_status
        commit_time = get_datatime_tail()

        with first_col:
            with st.spinner('evaluate ing (about 1 min)...'):
                summary_info = '''
                            ##### summary: \n
                                title : %s;\n
                                score : %s;\n
                                upload: %s;\n
                                time  : %s;\n
                        '''%(
                            commit_title, score_thres, det_dir, commit_time
                        )
                st.info(summary_info)
                st.session_state.det_dir = det_dir
                st.session_state.commit_title = commit_title
                st.session_state.commit_time = commit_time
                st.session_state.curr_eval_df, st.session_state.bbox_metric_ret = evaluate(det_dir, gt_dir, float(score_thres), commit_time, commit_title, st.session_state.global_metric_database)


    with second_col:
        curr_eval_df = pd.DataFrame(copy.deepcopy(st.session_state.curr_eval_df))
        st.data_editor(
            curr_eval_df,
            # width = 600,
            # height = 50,
            disabled=list(curr_eval_df.keys()),
            hide_index=True,
        )
        if st.button("保存当前记录"):
            # record curr result to database
            if len(st.session_state.global_metric_database) == 0:
                st.session_state.global_metric_database = copy.deepcopy(st.session_state.curr_eval_df)
                st.session_state.global_metric_database["enable"] = st.session_state.global_metric_database.pop("record")
            else:
                for key, res in st.session_state.curr_eval_df.items():
                    if key == "record":
                        st.session_state.global_metric_database["enable"] += [False]
                    else:
                        st.session_state.global_metric_database[key] += res

            items_length = len(st.session_state.global_metric_database["commit"])
            bbox_metric_dump_fpath = os.path.join("bbox_compare_results", str(items_length - 1) + ".pkl")

            print(st.session_state.global_metric_database)
            time_tail = st.session_state.commit_time.replace(":", "-").replace(" ", "_")
            os.system("mv  %s %s"%(
                st.session_state.det_dir,
                os.path.join("history", st.session_state.commit_title + "-" + time_tail)
            ))
            serialize_data(st.session_state.bbox_metric_ret, bbox_metric_dump_fpath)
            serialize_data(st.session_state.global_metric_database, database_path)
    st.markdown("-"*1000)

    st.markdown("### 历史记录")
    if len(st.session_state.global_metric_database) > 0:
        summary_history_df = pd.DataFrame(st.session_state.global_metric_database)

        summary_history_df_info = st.data_editor(
            summary_history_df,
            column_config={
                "enable": st.column_config.CheckboxColumn(
                    "enable",
                    default=False,
                )
            },
            disabled=[
                "commit"        ,
                "time"          ,
                "score_thres"   ,
                "mAP@0.5"       ,
                "recall@0.5"    ,
                "mAP@large"    ,
                "mAP@small"    ,
                "recall@large" ,
                "recall@small" ,
                "误检框数"      ,
                "漏检框数"      ,
                "新增误检"      ,
                "新增漏检"      ,
            ],
            hide_index=True,
        )

        if len(summary_history_df_info) > 0:
            mask = summary_history_df_info.enable == True
            utils.global_compare_items_var = {}
            for i, status in enumerate(mask):
                if status:
                    utils.global_compare_items_var[i] = st.session_state.global_metric_database['commit'][i]
            if st.button("compare"):
                print(  utils.global_compare_items_var, "====main=====" )
                st.markdown("[%s](http://%s)"%("compare",
                    ("10.11.1.91" + ":9090/可视化")),
                                unsafe_allow_html=True)


def main():
    slidebar_status = sidebar_ui_layout()

    main_ui_layout(slidebar_status)



main()

import streamlit as st
import os
import numpy as np
import  utils
from utils import deserialize_data
import cv2
import copy
import matplotlib.pyplot as plt

st.set_page_config(layout="wide",
            page_title="Dolly View",
            page_icon="📡",
            )

if "load_database_from_disk" not in st.session_state.keys():
    st.session_state.load_database_from_disk = {}

if "slider_index" not in st.session_state:
    st.session_state.slider_index = 0


def draw_detect_box(img, bboxes,info="error"):
    color = {
        "error": (0, 0, 255),
        "miss" : (0, 255, 255),
        "gt" : (255, 0, 0),
        "det": (0, 255, 0)
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
        print(b)
    return img


def draw_conf_histogram(detection_results, info = "正检"):
    # 提取所有置信度分数
    all_scores = []
    for detections in detection_results.values():
        all_scores.extend([detection[4] for detection in detections])

    # 将所有置信度分数转换为 NumPy 数组
    scores_array = np.array(all_scores)

    # 定义置信度区间
    bins = np.arange(0, 1.1, 0.1)

    # # 计算每个区间的框数
    counts, _ = np.histogram(scores_array, bins=bins)

    # 准备区间标签
    interval_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

    fig, ax = plt.subplots()
    ax.bar(interval_labels, counts, width=0.8, align='center')
    ax.set_xlabel('Score Intervals')
    ax.set_ylabel('Number of Boxes')
    ax.set_title('Histogram of %s'%info)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # st.pyplot(fig)
    return fig



def show_image():

    image_fname = st.session_state.compare_dataframe['image'][st.session_state.slider_index]
    image_fpath = os.path.join("merge_case", "imgs", image_fname)
    cv_img = cv2.imread(image_fpath)
    cv2.putText(cv_img,
        image_fname,
        (0,40),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (255, 0, 255), 2)

    image_group_num = len(st.session_state.compare_dataframe) - 1
    group_list = st.columns([1] * image_group_num, gap="small")
    commit_list = list(st.session_state.compare_dataframe.keys())
    commit_list.remove("image")

    for idx, gl in enumerate(group_list):
        curr_commit = commit_list[idx]
        show_img = copy.copy(cv_img)
        if image_fname in st.session_state.load_database_from_disk[curr_commit]['false_positives_database'].keys():
            fp_box = st.session_state.load_database_from_disk[curr_commit]['false_positives_database'][image_fname]
            show_img = draw_detect_box(show_img, fp_box, "error")

        if image_fname in st.session_state.load_database_from_disk[curr_commit]['false_negatives_database'].keys():
            fn_box = st.session_state.load_database_from_disk[curr_commit]['false_negatives_database'][image_fname]
            show_img = draw_detect_box(show_img, fn_box, "miss")

        if image_fname in st.session_state.load_database_from_disk[curr_commit]['true_positives_database'].keys():
            tp_box = st.session_state.load_database_from_disk[curr_commit]['true_positives_database'][image_fname]
            show_img = draw_detect_box(show_img, tp_box, "det")

        with gl:
            st.markdown("#### %s"%curr_commit)
            show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
            st.image(show_img)
            st.markdown("-"*1000)

            if "first_show_image" not in st.session_state.keys():
                st.markdown("##### 误检样本")
                fp_key = curr_commit + "false_positives"

                if fp_key not in st.session_state.keys():
                    st.session_state[fp_key] = draw_conf_histogram(st.session_state.load_database_from_disk[curr_commit]['false_positives_database'], "FP")
                st.pyplot(st.session_state[fp_key] )

                st.markdown("##### 正检样本")
                tp_key = curr_commit + "true_positives"
                if tp_key not in st.session_state.keys():
                    st.session_state[tp_key] = draw_conf_histogram(st.session_state.load_database_from_disk[curr_commit]['true_positives_database'], "TP")
                st.pyplot(st.session_state[tp_key])
    st.session_state.first_show_image = False

def main():
    if utils.global_compare_items_var is None:
        st.sidebar.info("未选择要比较的项")
    else:
        keys_image_name_list  = set()
        for index, commit_title in utils.global_compare_items_var.items():
            st.session_state.load_database_from_disk[commit_title] =  \
                deserialize_data(os.path.join("bbox_compare_results", str(index) + ".pkl"))
            keys_image_name_list |= set(st.session_state.load_database_from_disk[commit_title]['false_positives_database'].keys())
            keys_image_name_list |= set(st.session_state.load_database_from_disk[commit_title]['false_negatives_database'].keys())
        if len(keys_image_name_list) > 0:
            roi_image_fname = list(keys_image_name_list)
            roi_image_fname.sort()

            # build data frame
            col1_title = "image"
            st.session_state.compare_dataframe = {
                col1_title: roi_image_fname,
            }

            for img in roi_image_fname:
                for commit in st.session_state.load_database_from_disk.keys():
                    if commit  not in st.session_state.compare_dataframe.keys():
                        st.session_state.compare_dataframe[commit] = []

                    status = "/ "

                    if img in st.session_state.load_database_from_disk[commit]['false_positives_database'].keys():
                        status += "💩 /"

                    if img in st.session_state.load_database_from_disk[commit]['false_negatives_database'].keys():
                        status += " 🤕 /"

                    st.session_state.compare_dataframe[commit].append(status)




            st.markdown("##### compare list")
            st.session_state.select = st.dataframe(
                    st.session_state.compare_dataframe,
                    # width = 1200,
                    height = 300,
                    use_container_width=True,
                    hide_index=False,
                    on_select="rerun",
                    selection_mode="single-row")

            if "select" in st.session_state and len(st.session_state.select['selection']['rows']) > 0:
                st.session_state.slider_index = st.session_state.select['selection']['rows'][0]

            max_img_length = len(st.session_state.compare_dataframe['image'])
            st.session_state.slider_index = st.slider("选择", min_value=0,
                                max_value=max_img_length,
                                value=st.session_state.slider_index,
                                step=1)


            print(st.session_state.select)
            show_image()




            # st.session_state.compare_table_view =
            # st.data_editor(
            #     st.session_state.compare_dataframe,
            #     width = 1000,
            #     height = 1200,
            #     disabled=list(st.session_state.compare_dataframe.keys()),
            #     hide_index=True,
            #     on_change=table_click_callback
            # )
        else:
            st.sidebar.info("未选择要比较的项")


main()
import streamlit as st
import os
from evaluate2d import eval_mAP_for_uisee

st.set_page_config(layout="wide",
                page_title="Evaluate Object Detection",
                page_icon="📡",
            )

def save_and_utgz_uploaded_file(uploadedfile):
    tgz_name = os.path.join("tmp", uploadedfile.name)
    with open(tgz_name, "wb") as f:
        f.write(uploadedfile.getbuffer())

    os.system("tar -xf %s -C %s"%(tgz_name, "tmp"))
    fname = uploadedfile.name.split(".tar")[0]
    return os.path.join("tmp", fname)

def save_coco_json(uploadedfile):
    tgz_name = os.path.join("tmp", uploadedfile.name)
    with open(tgz_name, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return tgz_name


first, second, third = st.columns([2,3,1], gap='small')

with first:
    st.markdown("## 2D检测评估")
    st.markdown("#### 上传检测结果 （以tar.gz保存）")

    status = True
    upload_file = st.file_uploader("检测结果", "tar.gz", label_visibility = "collapsed")
    det_dir = None
    if upload_file is not None:
        det_dir = save_and_utgz_uploaded_file(upload_file)
    else:
        status = False

    st.markdown("#### 上传COCO格式真值")
    upload_file = st.file_uploader("真值", "json", label_visibility = "collapsed")


    gt_dir = None
    if upload_file is not None:
        gt_dir = save_coco_json(upload_file)
    else:
        status = False


eval_result = None
if st.button("提交评估"):
    with second:
        st.markdown("## 评估结果")
        with st.code("console"):
            taskinfo = st.code("start evaluate:")

            if status:
                eval_result = eval_mAP_for_uisee(det_dir, gt_dir)
                taskinfo.text(eval_result)

            else:
                taskinfo.text("请检查上传文件是否齐全")



with third:
    if eval_result is not None:
        fname = os.path.basename(det_dir) + ".json"
        st.markdown("## 下载")
        with open(det_dir + ".json", 'rb') as f:
            st.download_button(fname, f, file_name=fname)



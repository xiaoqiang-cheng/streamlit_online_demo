import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# 示例目标检测结果字典
detection_results = {
    "xxx.jpg": [[10, 20, 30, 40, 0.15], [25, 35, 45, 55, 0.55]],
    "yyy.jpg": [[15, 25, 35, 45, 0.45], [20, 30, 40, 50, 0.75]],
    # 其他图片的结果
}

# 提取所有置信度分数
all_scores = []
for detections in detection_results.values():
    all_scores.extend([detection[4] for detection in detections])

# 将所有置信度分数转换为 NumPy 数组
scores_array = np.array(all_scores)

# 定义置信度区间
bins = np.arange(0, 1.1, 0.1)

import ipdb
ipdb.set_trace()

# # 计算每个区间的框数
# counts, _ = np.histogram(scores_array, bins=bins)

# # 准备区间标签
# interval_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

# # 在 Streamlit 中显示直方图
# st.title("Detection Confidence Score Histogram")
# fig, ax = plt.subplots()
# ax.bar(interval_labels, counts, width=0.8, align='center')
# ax.set_xlabel('Score Intervals')
# ax.set_ylabel('Number of Boxes')
# ax.set_title('Histogram of Detection Confidence Scores')
# plt.xticks(rotation=45)
# plt.tight_layout()

# st.pyplot(fig)

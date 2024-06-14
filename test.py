import streamlit as st
import pandas as pd

# 创建示例数据框
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9],
    "selected" : [False, False, False]
}
df = pd.DataFrame(data)

# selection = st.dataframe(df)

selection = st.dataframe(df, on_select="rerun", selection_mode="single-row")
print(selection)
import streamlit as st  # 导入 Streamlit 库用于创建 Web 应用
import joblib  # 导入 joblib 库用于加载模型
import numpy as np  # 导入 NumPy 库用于数值计算
import pandas as pd  # 导入 Pandas 库用于数据处理
import shap  # 导入 SHAP 库用于模型解释
import matplotlib.pyplot as plt  # 导入 Matplotlib 库用于绘图
from lime.lime_tabular import LimeTabularExplainer  # 导入 LIME 库用于局部可解释模型

# 加载训练好的随机森林模型
model = joblib.load('RF.pkl')

# 从 X_test.csv 加载测试数据以创建 LIME 解释器
X_test = pd.read_csv('X_test.csv')

# 定义特征名称
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Streamlit 用户界面
st.title("心脏病预测器")

# 用户输入
age = st.number_input("年龄:", min_value=0, max_value=120, value=41)  # 年龄输入
sex = st.selectbox("性别:", options=[0, 1], format_func=lambda x: "男性" if x == 1 else "女性")  # 性别选择
cp = st.selectbox("胸痛类型 (CP):", options=[0, 1, 2, 3])  # 胸痛类型选择
trestbps = st.number_input("静息血压 (trestbps):", min_value=50, max_value=200, value=120)  # 静息血压输入
chol = st.number_input("胆固醇 (chol):", min_value=100, max_value=600, value=157)  # 胆固醇输入
fbs = st.selectbox("空腹血糖 > 120 mg/dl (FBS):", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")  # 空腹血糖选择
restecg = st.selectbox("静息心电图 (restecg):", options=[0, 1, 2])  # 静息心电图选择
thalach = st.number_input("最大心率 (thalach):", min_value=60, max_value=220, value=182)  # 最大心率输入
exang = st.selectbox("运动诱发心绞痛 (exang):", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")  # 心绞痛选择
oldpeak = st.number_input("运动诱发的 ST 抑制 (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)  # ST 抑制输入
slope = st.selectbox("峰值运动 ST 段的坡度 (slope):", options=[0, 1, 2])  # 坡度选择
ca = st.selectbox("荧光透视下的主要血管数量 (ca):", options=[0, 1, 2, 3, 4])  # 血管数量选择
thal = st.selectbox("地中海贫血 (thal):", options=[0, 1, 2, 3])  # 地中海贫血选择

# 处理输入并进行预测
feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
features = np.array([feature_values])  # 将特征值转换为 NumPy 数组

# 当用户点击预测按钮时
if st.button("预测"):
    # 进行类别预测和概率预测
    predicted_class = model.predict(features)[0]  # 预测类别
    predicted_proba = model.predict_proba(features)[0]  # 预测概率

    # 显示预测结果
    st.write(f"**预测类别:** {predicted_class} (1: 有疾病, 0: 无疾病)")
    st.write(f"**预测概率:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100  # 计算概率
    if predicted_class == 1:
        advice = (
            f"根据我们的模型，您有高风险心脏病。"
            f"模型预测您患心脏病的概率为 {probability:.1f}%。"
            "建议您咨询医疗服务提供者以进一步评估和可能的干预。"
        )
    else:
        advice = (
            f"根据我们的模型，您有低风险心脏病。"
            f"模型预测您没有心脏病的概率为 {probability:.1f}%。"
            "但是，保持健康的生活方式仍然很重要。请继续定期检查。"
        )
    st.write(advice)

    # SHAP 解释
    st.subheader("SHAP 力量图解释")
    explainer_shap = shap.TreeExplainer(model)  # 创建 SHAP 解释器
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))  # 计算 SHAP 值

    # 显示预测类别的 SHAP 力量图
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)  # 保存 SHAP 力量图
    st.image("shap_force_plot.png", caption='SHAP 力量图解释')  # 显示 SHAP 力量图

    # LIME 解释
    st.subheader("LIME 解释")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,  # 使用测试数据创建 LIME 解释器
        feature_names=X_test.columns.tolist(),  # 特征名称
        class_names=['无病', '有病'],  # 类别名称
        mode='classification'  # 分类模式
    )

    # 解释实例
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),  # 展平特征数组
        predict_fn=model.predict_proba  # 预测函数
    )

    # 显示 LIME 解释，不显示特征值表
    lime_html = lime_exp.as_html(show_table=False)  # 禁用特征值表
    st.components.v1.html(lime_html, height=800, scrolling=True)  # 显示 LIME 解释
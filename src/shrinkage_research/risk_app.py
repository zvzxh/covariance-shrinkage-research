"""Local browser workbench; uploads stay in this Python process."""

import hashlib
import json

import streamlit as st

from shrinkage_research.risk_analysis import validate_risk_config
from shrinkage_research.risk_cli import analysis_fingerprint, run_analysis
from shrinkage_research.risk_data import read_risk_input
from shrinkage_research.risk_demo import synthetic_csv
from shrinkage_research.risk_export import zip_exports

st.set_page_config(page_title="Portfolio Risk Workbench", page_icon="📊", layout="wide")
st.title("组合风险工作台")
st.caption("上传数据，比较组合风险、查找风险来源，再用后续样本检验。所有计算在本机完成。")
source = st.radio("数据来源", ["Synthetic demo", "Upload CSV"], format_func=lambda x: {"Synthetic demo": "合成样例", "Upload CSV": "上传 CSV"}[x], horizontal=True, key="source")
upload = st.file_uploader("上传收益率或复权价格 CSV", type=["csv"], key="input_file") if source == "Upload CSV" else None
with st.expander("高级：导入已保存的配置（可选）"):
    imported = st.file_uploader("分析配置 JSON", type=["json"], key="config_file")

if imported is not None:
    try:
        config = json.loads(imported.getvalue())
        validate_risk_config(config)
    except Exception as error:
        st.session_state.pop("risk_result", None)
        st.error(f"配置校验失败：{error}")
        st.stop()
    st.info("正在使用导入的配置。移除 JSON 文件后可编辑参数；数据来源标签以当前选择为准。")
    config["data_kind"] = "synthetic" if source == "Synthetic demo" else "user_supplied"
    st.json(config, expanded=False)
else:
    left, right = st.columns(2)
    with left:
        input_kind = st.selectbox("数据类型", ["returns", "adjusted_prices"], format_func=lambda x: {"returns": "简单收益率（0.01 = 1%）", "adjusted_prices": "复权价格（必须为正）"}[x], key="kind",
                                  help="Returns are decimal simple returns. Prices must be adjusted and strictly positive; the app cannot verify adjustment.")
        start = st.text_input("收益截止日从（可选，YYYY-MM-DD）", key="start")
        end = st.text_input("收益截止日到（可选，YYYY-MM-DD）", key="end")
        last_n = st.number_input("仅用最后 N 个收益期（0 = 全部）", min_value=0, value=0, step=1, key="last_n")
    with right:
        annualization = st.number_input("每年观察期数（日频常用 252）", min_value=1, value=252, step=1, key="annualization")
        explicit = st.checkbox("自定义资产权重（默认等权）", key="explicit_weights")
        rolling_enabled = st.checkbox("用后续观察检验风险估计", value=True, key="rolling")
        train = st.number_input("滚动训练期数", min_value=2, value=60, step=1, key="train")
        test = st.number_input("每段检验期数（至少 2）", min_value=2, value=20, step=1, key="test")
        partial = st.checkbox("末段不足完整窗口时仍检验（至少 2 期）", value=True, key="partial")
    config = {"schema_version": 2, "study_id": "browser-risk-analysis", "data_kind": "synthetic" if source == "Synthetic demo" else "user_supplied",
              "input_kind": input_kind, "start": start or None, "end": end or None, "last_n": int(last_n) or None,
              "periods_per_year": int(annualization), "weights": None,
              "rolling": {"enabled": rolling_enabled, "train_window": int(train), "test_window": int(test), "include_partial": partial}}

if source == "Synthetic demo":
    content = synthetic_csv(config["input_kind"])
    filename = "synthetic_" + config["input_kind"] + ".csv"
    st.warning("合成样例：由程序生成，不是真实行情，也不能证明投资效果。")
else:
    content = upload.getvalue() if upload is not None else None
    filename = upload.name if upload is not None else ""

if content is not None:
    try:
        preview = read_risk_input(content, input_kind=config["input_kind"])
        st.caption(f"数据可用范围：{preview.dates[0]} 至 {preview.dates[-1]}；{len(preview.dates)} 个收益期，{len(preview.assets)} 个资产。")
        if imported is None and explicit:
            st.subheader("资产权重")
            st.caption("输入百分比，总和须为 100%；不允许负权重。")
            weights = {}
            columns = st.columns(2)
            for i, asset in enumerate(preview.assets):
                value = columns[i % 2].number_input(asset + " (%)", min_value=0.0, max_value=100.0,
                                                     value=100.0 / len(preview.assets), step=1.0, format="%.6f",
                                                     key="asset_weight_" + hashlib.sha256(("|".join(preview.assets) + asset).encode()).hexdigest()[:16])
                weights[asset] = value / 100.0
            config["weights"] = weights
            st.caption(f"当前权重合计：{sum(weights.values()):.6%}")
    except Exception as error:
        st.session_state.pop("risk_result", None)
        st.error(f"数据校验失败：{error}")
        st.stop()

st.caption("三个方法均去均值后除以 n。风险贡献可为负；后续样本方差是含噪声的检验指标，不是买卖信号。")
st.caption("价格先转收益，再按收益截止日筛选；首个入选收益可能使用起始日前一行价格，报告会记录。")

fingerprint = analysis_fingerprint(content or b"", config, filename)
if st.session_state.get("risk_fingerprint") != fingerprint:
    had_result = "risk_result" in st.session_state
    st.session_state.pop("risk_result", None)
    st.session_state["risk_fingerprint"] = fingerprint
    if had_result:
        st.info("输入或设置已改变，旧结果已清空，请重新分析。")

if content is None:
    st.info("请上传 CSV，或选择合成样例。")
if st.button("开始分析", type="primary", disabled=content is None, key="analyze"):
    st.session_state.pop("risk_result", None)
    try:
        with st.spinner("正在校验数据并计算风险……"):
            analysis, files = run_analysis(content, config, filename=filename)
        st.session_state["risk_result"] = (analysis, files)
    except Exception as error:
        st.error(f"分析未完成：{error}")

if "risk_result" in st.session_state:
    analysis, files = st.session_state["risk_result"]
    st.success(f"已分析 {analysis['observations']} 个收益期、{len(analysis['assets'])} 个资产。")
    names = {"sample_mle": "样本协方差", "ledoit_wolf": "Ledoit–Wolf 收缩", "diagonal_sample_mle": "仅保留单资产方差"}
    summary = []
    for name, values in analysis["methods"].items():
        contributions = values["volatility_components"]
        largest = analysis["assets"][max(range(len(contributions)), key=lambda i: contributions[i])] if contributions is not None else "未定义"
        summary.append({"方法": names[name], "组合年化波动": f"{values['annualized_volatility']:.2%}", "最大风险贡献资产": largest,
                        "提示": " / ".join(values["warnings"]) or "无数值警示"})
    st.dataframe(summary, hide_index=True)
    st.image(files["risk_diagnostics.png"], caption="各资产波动贡献与协方差特征值")
    warning_messages = sorted({warning for values in analysis["methods"].values() for warning in values["warnings"]})
    for warning in warning_messages:
        st.warning(warning)
    with st.expander("输入与转换记录"):
        st.json(analysis["input_audit"])
    if analysis["rolling_summary"]:
        st.subheader("后续样本风险检验")
        st.caption("实际波动由后续样本方差计算，是含噪声的检验指标；使用固定暴露，不模拟持仓漂移。下表波动均按所选期数年化。")
        factor = config["periods_per_year"]
        st.dataframe([{"方法": names[row["method"]], "检验起日": row["test_start"], "检验止日": row["test_end"],
                       "预测波动": f"{(factor * row['predicted_period_variance']) ** 0.5:.2%}",
                       "实际波动": f"{(factor * row['realized_period_variance_ddof1']) ** 0.5:.2%}"}
                      for row in analysis["rolling_rows"]], hide_index=True)
        with st.expander("完整滚动记录与统计口径"):
            st.dataframe(analysis["rolling_rows"], hide_index=True)
            st.json(analysis["rolling_summary"])
    st.subheader("下载与复用")
    buttons = st.columns(4)
    buttons[0].download_button("完整 HTML 报告", files["report.html"], "risk_report.html", "text/html", key="download_html")
    buttons[1].download_button("可复用配置", files["config.json"], "risk_config.json", "application/json", key="download_config")
    buttons[2].download_button("选中收益 CSV", files["selected_returns.csv"], "selected_returns.csv", "text/csv", key="download_returns")
    buttons[3].download_button("全部记录 ZIP", zip_exports(files), "risk_analysis.zip", "application/zip", key="download_zip")
    st.caption("选中收益 CSV 可直接交给 Portfolio Walk-Forward Lab。要重复本次风险分析，请使用原始输入文件与下载的配置。")

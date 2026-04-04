from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

PROJECT_TITLE = "Hệ thống lọc bình luận độc hại và quảng cáo rác cho Fanpage lớn"
STUDENT_NAME = "Điền họ tên của bạn"
STUDENT_ID = "Điền MSSV của bạn"

REMOTE_URLS = {
    "train": "https://raw.githubusercontent.com/tarudesu/ViCTSD/main/ViCTSD_train.csv",
    "valid": "https://raw.githubusercontent.com/tarudesu/ViCTSD/main/ViCTSD_valid.csv",
    "test": "https://raw.githubusercontent.com/tarudesu/ViCTSD/main/ViCTSD_test.csv",
}

SPAM_PATTERNS: Dict[str, List[str]] = {
    "link": [r"https?://", r"www\.", r"bit\.ly", r"t\.me/"],
    "contact": [r"\b0\d{8,10}\b", r"zalo", r"telegram", r"inbox", r"ib\b", r"liên hệ"],
    "sales": [r"giá sỉ", r"giảm giá", r"khuyến mãi", r"sale", r"mua hàng", r"đặt hàng"],
    "recruit": [r"cộng tác viên", r"việc nhẹ lương cao", r"tuyển dụng", r"tuyển ctv"],
    "finance": [r"vay vốn", r"mở thẻ", r"bảo hiểm", r"kiếm tiền online"],
    "growth": [r"tăng follow", r"tăng tương tác", r"seeding", r"bán acc"],
}

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def apply_custom_styles() -> None:
    st.markdown(
        '''
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            font-size: 17px !important;
        }
        .stApp, .stApp p, .stApp li, .stApp label, .stApp div, .stApp span, .stMarkdown, .stText, .stCaption {
            font-size: 17px !important;
            line-height: 1.45 !important;
        }
        .stApp h1 { font-size: 2.05rem !important; }
        .stApp h2 { font-size: 1.55rem !important; }
        .stApp h3 { font-size: 1.15rem !important; }
        .stTextInput input, .stTextArea textarea, .stSelectbox div, .stSlider, .stRadio, .stFileUploader {
            font-size: 16px !important;
        }
        .stButton button {
            font-size: 16px !important;
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
        }
        .stDataFrame, .stTable { font-size: 15px !important; }
        [data-testid="stSidebar"] * { font-size: 16px !important; }
        div[data-testid="metric-container"] {
            padding: 0.7rem 0.85rem 0.65rem 0.85rem !important;
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 0.65rem;
            background: rgba(255,255,255,0.35);
        }
        div[data-testid="metric-container"] label {
            font-size: 0.95rem !important;
        }
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
        </style>
        ''',
        unsafe_allow_html=True,
    )


def normalize_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"\S+@\S+", " <email> ", text)
    text = re.sub(r"@[\w_]+", " <user> ", text)
    text = re.sub(r"\d{8,}", " <number> ", text)
    text = re.sub(r"([!?.]){2,}", r" \1 ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def spam_rule_score(text: str) -> Tuple[int, List[str]]:
    raw = normalize_text(text)
    score = 0
    reasons: List[str] = []

    for rule_name, patterns in SPAM_PATTERNS.items():
        if any(re.search(p, raw) for p in patterns):
            score += 1
            reasons.append(rule_name)

    if raw.count("<url>") >= 1:
        score += 1
        reasons.append("url_count")

    if len(re.findall(r"<number>", raw)) >= 1:
        score += 1
        reasons.append("phone_like_number")

    return score, sorted(set(reasons))


@st.cache_data(show_spinner=False)
def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dfs = {}
    for split in ("train", "valid", "test"):
        local_path = DATA_DIR / f"ViCTSD_{split}.csv"
        if local_path.exists():
            df = pd.read_csv(local_path)
        else:
            df = pd.read_csv(REMOTE_URLS[split])

        df = df.rename(columns={"Comment": "comment", "Toxicity": "label", "Title": "title", "Topic": "topic"})
        df["comment"] = df["comment"].astype(str)
        df["label"] = df["label"].astype(int)
        df["comment_length"] = df["comment"].astype(str).str.len()
        dfs[split] = df

    return dfs["train"], dfs["valid"], dfs["test"]


@st.cache_data(show_spinner=False)
def load_demo_comments() -> pd.DataFrame:
    path = DATA_DIR / "demo_comments.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame({"comment": ["cảm ơn admin", "đồ mất dạy", "ib em nhận báo giá sỉ"]})


def _get_final_estimator(model):
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    return model


def _patch_logreg_compat(model):
    est = _get_final_estimator(model)
    defaults = {
        "multi_class": "auto",
        "n_jobs": None,
        "l1_ratio": None,
        "class_weight": None,
        "warm_start": False,
        "verbose": 0,
    }
    for key, value in defaults.items():
        if not hasattr(est, key):
            setattr(est, key, value)
    return model


@st.cache_resource(show_spinner=False)
def load_logreg_model():
    path = MODELS_DIR / "logreg_toxic_pipeline.joblib"
    if path.exists():
        model = joblib.load(path)
        return _patch_logreg_compat(model)
    return None


@st.cache_resource(show_spinner=False)
def load_phobert_model():
    model_dir = MODELS_DIR / "phobert_toxic_model"
    if not model_dir.exists():
        return None, None, None

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_logreg(text: str) -> Tuple[int, float]:
    model = load_logreg_model()
    if model is None:
        raise FileNotFoundError("Chưa tìm thấy models/logreg_toxic_pipeline.joblib")
    clean_text = normalize_text(text)
    try:
        proba = float(model.predict_proba([clean_text])[0][1])
    except AttributeError:
        model = _patch_logreg_compat(model)
        proba = float(model.predict_proba([clean_text])[0][1])
    pred = int(proba >= 0.5)
    return pred, proba


def predict_phobert(text: str) -> Tuple[int, float]:
    tokenizer, model, device = load_phobert_model()
    if tokenizer is None or model is None:
        raise FileNotFoundError("Chưa tìm thấy thư mục models/phobert_toxic_model")

    import torch

    encoded = tokenizer(
        normalize_text(text),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, float(probs[1])


def moderation_pipeline(text: str, model_name: str, toxic_threshold: float = 0.7) -> Dict[str, object]:
    spam_score, spam_reasons = spam_rule_score(text)
    if spam_score >= 2:
        return {
            "final_label": "spam",
            "toxic_prob": 0.0,
            "spam_score": spam_score,
            "spam_reasons": spam_reasons,
            "decision": "Ẩn tự động",
        }

    if model_name == "PhoBERT":
        pred, toxic_prob = predict_phobert(text)
    else:
        pred, toxic_prob = predict_logreg(text)

    if toxic_prob >= 0.85:
        final_label = "toxic"
        decision = "Ẩn tự động"
    elif toxic_prob >= toxic_threshold:
        final_label = "toxic"
        decision = "Chuyển kiểm duyệt thủ công"
    else:
        final_label = "clean"
        decision = "Giữ bình luận"

    return {
        "final_label": final_label,
        "toxic_prob": toxic_prob,
        "spam_score": spam_score,
        "spam_reasons": spam_reasons,
        "decision": decision,
        "model_pred": pred,
    }


@st.cache_data(show_spinner=False)
def load_metrics_json(name: str) -> dict | None:
    path = REPORTS_DIR / name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_predictions_csv(name: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return None


def report_to_df(report: dict) -> pd.DataFrame:
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append({"label": label, **values})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("label")


def render_dataset_comment(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    total = len(train_df) + len(valid_df) + len(test_df)
    toxic_ratio = pd.concat([train_df["label"], valid_df["label"], test_df["label"]]).mean() * 100
    st.write(
        f"Bộ dữ liệu ViCTSD đang dùng cho app có {total:,} bình luận; trong đó train/valid/test tương ứng là "
        f"{len(train_df):,}/{len(valid_df):,}/{len(test_df):,}. Tỷ lệ nhãn toxic toàn bộ khoảng {toxic_ratio:.2f}% nên dữ liệu bị lệch lớp."
    )


def _style_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.25)
    ax.spines["bottom"].set_alpha(0.25)
    ax.grid(axis=grid_axis, alpha=0.15, linewidth=0.8)


CHART_SIZE = (5.2, 3.4)
CHART_DPI = 150


def _plot_label_distribution(full_df: pd.DataFrame):
    counts = full_df["label"].map({0: "clean", 1: "toxic"}).value_counts().reindex(["clean", "toxic"]).fillna(0)
    fig, ax = plt.subplots(figsize=CHART_SIZE, dpi=CHART_DPI)
    bars = ax.bar(counts.index, counts.values, width=0.58)
    ax.set_xlabel("Nhãn")
    ax.set_ylabel("Số bình luận")
    _style_axes(ax, grid_axis="y")
    ymax = counts.max() * 1.14 if counts.max() > 0 else 1
    ax.set_ylim(0, ymax)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + counts.max() * 0.02, f"{int(value):,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def _plot_comment_length(full_df: pd.DataFrame):
    comment_lengths = full_df["comment_length"].fillna(0)
    fig, ax = plt.subplots(figsize=CHART_SIZE, dpi=CHART_DPI)
    ax.hist(comment_lengths, bins=24, edgecolor="white", linewidth=0.45)
    ax.set_xlabel("Số ký tự")
    ax.set_ylabel("Số lượng")
    _style_axes(ax, grid_axis="y")
    ax.margins(x=0.02)
    fig.tight_layout()
    return fig


def _plot_topic_distribution(full_df: pd.DataFrame):
    topic_counts = full_df["topic"].fillna("Khác").value_counts().head(10).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=CHART_SIZE, dpi=CHART_DPI)
    bars = ax.barh(topic_counts.index, topic_counts.values, height=0.62)
    ax.set_xlabel("Số bình luận")
    ax.set_ylabel("Chủ đề")
    _style_axes(ax, grid_axis="x")
    xmax = topic_counts.max() * 1.12 if topic_counts.max() > 0 else 1
    ax.set_xlim(0, xmax)
    for bar, value in zip(bars, topic_counts.values):
        ax.text(value + max(topic_counts.max() * 0.01, 1), bar.get_y() + bar.get_height() / 2, f"{int(value):,}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def page_eda(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    st.title(PROJECT_TITLE)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Họ tên SV:** {STUDENT_NAME}")
        st.markdown(f"**MSSV:** {STUDENT_ID}")
        st.markdown(
            "Đề tài xây dựng hệ thống lọc bình luận độc hại và quảng cáo rác cho Fanpage lớn. "
            "Mục tiêu là giảm tải kiểm duyệt thủ công, hỗ trợ ẩn tự động comment spam hoặc toxic, "
            "và giữ môi trường tương tác lành mạnh hơn."
        )
    with col2:
        st.info("Phù hợp yêu cầu trang 1: Giới thiệu đề tài, giá trị thực tiễn, dữ liệu thô và EDA.")

    st.subheader("Một phần dữ liệu thô")
    preview = pd.concat([train_df, valid_df, test_df], ignore_index=True)[["comment", "label", "topic", "title"]].head(15)
    st.dataframe(preview, use_container_width=True)

    render_dataset_comment(train_df, valid_df, test_df)
    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    st.subheader("Biểu đồ mô tả dữ liệu")
    top_left, top_right = st.columns(2, gap="large")
    with top_left:
        st.markdown("**Phân phối nhãn**")
        st.pyplot(_plot_label_distribution(full_df), use_container_width=False)
    with top_right:
        st.markdown("**Độ dài bình luận**")
        st.pyplot(_plot_comment_length(full_df), use_container_width=False)

    left_space, center_chart, right_space = st.columns([1, 2, 1])
    with center_chart:
        st.markdown("**Phân bố theo chủ đề**")
        st.pyplot(_plot_topic_distribution(full_df), use_container_width=False)

    toxic_by_topic = full_df.groupby("topic")["label"].mean().sort_values(ascending=False).head(8).mul(100).round(2)
    st.markdown("**Nhận xét dữ liệu**")
    st.write(
        "Biểu đồ trên giữ nguyên số liệu gốc của bộ dữ liệu. Với ViCTSD, số bình luận giữa các chủ đề khá cân bằng, "
        "nên biểu đồ chủ đề có thể trông gần giống nhau. Điều này phản ánh đặc điểm dữ liệu, không phải lỗi hiển thị."
    )
    st.dataframe(toxic_by_topic.rename("Tỷ lệ toxic (%)"), use_container_width=True)


def page_inference():
    st.title("Triển khai mô hình")
    st.write("Trang này cho phép nhập bình luận đơn hoặc tải lên CSV để dự đoán hàng loạt.")

    available_logreg = (MODELS_DIR / "logreg_toxic_pipeline.joblib").exists()
    available_phobert = (MODELS_DIR / "phobert_toxic_model").exists()

    model_options = []
    if available_logreg:
        model_options.append("Logistic Regression")
    if available_phobert:
        model_options.append("PhoBERT")
    if not model_options:
        model_options = ["Logistic Regression"]
        st.warning("Chưa tìm thấy model trong thư mục models/. Hãy chạy notebook trước để sinh model.")

    model_name = st.selectbox("Chọn mô hình", model_options)
    toxic_threshold = st.slider("Ngưỡng toxic để chuyển kiểm duyệt", min_value=0.50, max_value=0.95, value=0.70, step=0.01)

    default_text = "đúng là lũ mất dạy"
    text = st.text_area("Nhập bình luận cần kiểm tra", value=default_text, height=120)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Dự đoán bình luận", use_container_width=True):
            try:
                result = moderation_pipeline(text, model_name=model_name, toxic_threshold=toxic_threshold)

                if result["final_label"] == "spam":
                    st.error("Kết quả: SPAM / QUẢNG CÁO RÁC")
                elif result["final_label"] == "toxic":
                    if result["decision"] == "Ẩn tự động":
                        st.error("Kết quả: TOXIC / BÌNH LUẬN ĐỘC HẠI")
                    else:
                        st.warning("Kết quả: TOXIC / CẦN KIỂM DUYỆT THỦ CÔNG")
                else:
                    st.success("Kết quả: CLEAN / BÌNH LUẬN BÌNH THƯỜNG")

                st.write(f"**Quyết định gợi ý:** {result['decision']}")
                st.write(f"**Xác suất toxic:** {result['toxic_prob']:.4f}")
                st.write(f"**Spam score:** {result['spam_score']}")
                if result["spam_reasons"]:
                    st.write("**Lý do spam rule kích hoạt:**", ", ".join(result["spam_reasons"]))
            except Exception as exc:
                st.exception(exc)

    with col2:
        st.markdown("**Một số bình luận mẫu để thử nhanh**")
        st.dataframe(load_demo_comments(), use_container_width=True)

    st.divider()
    st.subheader("Dự đoán hàng loạt từ file CSV")
    uploaded = st.file_uploader("Tải file CSV có cột `comment`", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        if "comment" not in batch_df.columns:
            batch_df = batch_df.rename(columns={batch_df.columns[0]: "comment"})

        outputs = []
        for c in batch_df["comment"].astype(str).tolist():
            try:
                result = moderation_pipeline(c, model_name=model_name, toxic_threshold=toxic_threshold)
                outputs.append({
                    "comment": c,
                    "final_label": result["final_label"],
                    "decision": result["decision"],
                    "toxic_prob": result["toxic_prob"],
                    "spam_score": result["spam_score"],
                    "spam_reasons": ", ".join(result["spam_reasons"]),
                })
            except Exception as exc:
                outputs.append({"comment": c, "final_label": "error", "decision": str(exc)})

        out_df = pd.DataFrame(outputs)
        st.dataframe(out_df, use_container_width=True)
        st.download_button(
            label="Tải kết quả CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )


def page_evaluation():
    st.title("Đánh giá & Hiệu năng")
    st.write("Trang này hiển thị các chỉ số, confusion matrix và phân tích lỗi.")

    metric_files = []
    if (REPORTS_DIR / "logreg_metrics.json").exists():
        metric_files.append(("Logistic Regression", "logreg_metrics.json", "logreg_predictions.csv", "logreg_confusion_matrix.png"))
    if (REPORTS_DIR / "phobert_metrics.json").exists():
        metric_files.append(("PhoBERT", "phobert_metrics.json", "phobert_predictions.csv", "phobert_confusion_matrix.png"))

    if not metric_files:
        st.warning("Chưa có file đánh giá trong thư mục reports/. Hãy chạy notebook để tạo metrics và confusion matrix.")
        return

    model_label = st.selectbox("Chọn mô hình để xem đánh giá", [x[0] for x in metric_files])
    item = next(x for x in metric_files if x[0] == model_label)
    metrics = load_metrics_json(item[1])
    preds = load_predictions_csv(item[2])
    image_path = REPORTS_DIR / item[3]

    if metrics is None:
        st.warning("Không đọc được file metrics.")
        return

    st.subheader(f"Kết quả đánh giá: {model_label}")
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    m2.metric("Precision toxic", f"{metrics.get('precision_toxic', 0):.4f}")
    m3.metric("Recall toxic", f"{metrics.get('recall_toxic', 0):.4f}")
    m4.metric("F1 toxic", f"{metrics.get('f1_toxic', 0):.4f}")

    if image_path.exists():
        st.markdown("**Confusion Matrix**")
        left, center, right = st.columns([1, 1.55, 1])
        with center:
            st.image(str(image_path), caption=f"{model_label} - Confusion Matrix", width=520)

    report_df = report_to_df(metrics.get("classification_report", {}))
    if not report_df.empty:
        st.subheader("Classification report")
        st.dataframe(report_df.round(4), use_container_width=True)

    if preds is not None:
        st.subheader("Phân tích sai số")
        error_df = preds[preds["true_label"] != preds["pred_label"]].copy()
        if not error_df.empty:
            if "prob_toxic" in error_df.columns:
                error_df = error_df.sort_values("prob_toxic", ascending=False)

            st.write(
                "Mô hình thường sai ở các trường hợp mỉa mai, nói giảm nói tránh, hoặc bình luận toxic nhưng không dùng từ tục rõ ràng. "
                "Đây là lý do nên giữ rule-based cho spam và cân nhắc dùng PhoBERT cho phần toxic khó."
            )

            st.markdown("""
### 🔍 Nhận định các trường hợp mô hình dễ dự đoán sai

**1. Bình luận mang tính mỉa mai hoặc ẩn ý**
- Ví dụ: câu không chứa từ tục rõ ràng nhưng vẫn mang ý công kích.
- Đây là nhóm mà Logistic Regression dễ bỏ sót vì mô hình chủ yếu dựa vào từ khóa.

**2. Bình luận có từ nhạy cảm nhưng ngữ cảnh không độc hại**
- Một số comment chứa từ mạnh nhưng mục đích là trích dẫn, giải thích hoặc phản biện.
- Trường hợp này dễ tạo ra lỗi false positive.

**3. Bình luận spam ngụy trang**
- Có nội dung kiểu “ib để biết thêm”, “liên hệ mình”, “xem chi tiết inbox” nhưng không chứa link rõ ràng.
- Rule-based có thể bỏ sót nếu người dùng cố tình viết biến thể để né luật.

**4. Bình luận dài hoặc ngữ cảnh phức tạp**
- Các câu dài, có nhiều vế hoặc sắc thái mỉa mai thường khó với mô hình tuyến tính.
- PhoBERT có xu hướng xử lý tốt hơn nhưng vẫn có thể sai nếu dữ liệu huấn luyện chưa đủ đa dạng.

### ⚠️ Nguyên nhân chính
- Dữ liệu toxic ít hơn clean nên bài toán bị lệch lớp.
- Logistic Regression không hiểu ngữ cảnh sâu.
- Spam thực tế biến đổi linh hoạt và không phải lúc nào cũng có mẫu cố định.

### 🚀 Hướng cải thiện
- Bổ sung thêm dữ liệu toxic khó như mỉa mai, châm biếm, nói giảm nói tránh.
- Thu thập thêm dữ liệu spam tiếng Việt chuyên biệt.
- Ưu tiên PhoBERT cho các trường hợp cần hiểu ngữ cảnh.
- Tối ưu lại ngưỡng dự đoán để cân bằng precision và recall.
- Giữ kiến trúc hybrid: rule-based + mô hình học máy + mô hình ngữ cảnh.

### 📌 Kết luận
Hệ thống hoạt động tốt với các bình luận rõ ràng, nhưng vẫn còn hạn chế với:
- ngữ cảnh phức tạp,
- spam tinh vi,
- và các câu mang tính ẩn ý.
""")

            show_cols = [c for c in ["Comment", "Title", "Topic", "true_label", "pred_label", "prob_toxic"] if c in error_df.columns]
            if not show_cols:
                show_cols = [c for c in ["comment", "title", "topic", "true_label", "pred_label", "prob_toxic"] if c in error_df.columns]

            st.markdown("### 📉 Ví dụ các trường hợp dự đoán sai")
            st.dataframe(error_df[show_cols].head(20), use_container_width=True)
        else:
            st.success("Không có lỗi dự đoán trong file đang chọn.")

def main():
    st.set_page_config(page_title="Toxic Comment Filtering", layout="wide")
    apply_custom_styles()
    train_df, valid_df, test_df = load_splits()

    with st.sidebar:
        st.title("Điều hướng")
        page = st.radio("Chọn trang", ["1. Giới thiệu & EDA", "2. Triển khai mô hình", "3. Đánh giá & Hiệu năng"])
        st.caption("Ứng dụng đáp ứng yêu cầu tối thiểu 3 trang, có cache, EDA, triển khai và đánh giá.")
        st.write("**Tình trạng model**")
        st.write("Logistic Regression:", "Có" if (MODELS_DIR / "logreg_toxic_pipeline.joblib").exists() else "Chưa có")
        st.write("PhoBERT:", "Có" if (MODELS_DIR / "phobert_toxic_model").exists() else "Chưa có")

    if page.startswith("1"):
        page_eda(train_df, valid_df, test_df)
    elif page.startswith("2"):
        page_inference()
    else:
        page_evaluation()


if __name__ == "__main__":
    main()

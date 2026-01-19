import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# ============================
# FUNGSI UTILITAS & PIPELINE
# ============================

@st.cache_data
def load_data(path: str = "heart.csv") -> pd.DataFrame:
    """
    Load dataset dari file CSV.
    Fungsi ini di-cache agar tidak perlu membaca ulang file setiap kali terjadi interaksi di Streamlit.
    """
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Tahap Preprocessing:
    - Memisahkan fitur (X) dan target (y)
    - Menangani missing values (cek & drop jika ada)
    - Menghapus duplikat data jika ada
    - Mengembalikan dataset bersih (X, y) + info ringkas
    """
    info = {}

    # Copy agar tidak mengubah dataframe asli
    data = df.copy()

    # Cek dan tangani missing values
    missing_per_col = data.isna().sum()
    total_missing = missing_per_col.sum()
    info["missing_per_col"] = missing_per_col
    info["total_missing"] = int(total_missing)

    if total_missing > 0:
        # Untuk kesederhanaan tugas besar, kita drop baris yang memiliki missing
        data = data.dropna()
        info["after_missing_shape"] = data.shape

    # Cek dan hapus duplikat
    duplicate_count = data.duplicated().sum()
    info["duplicate_count"] = int(duplicate_count)
    if duplicate_count > 0:
        data = data.drop_duplicates()
        info["after_duplicates_shape"] = data.shape

    # Pisahkan fitur dan target
    if "target" not in data.columns:
        raise ValueError("Kolom 'target' tidak ditemukan pada dataset.")

    X = data.drop("target", axis=1)
    y = data["target"]

    info["final_shape"] = data.shape
    info["n_features"] = X.shape[1]

    return X, y, info


@st.cache_resource
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Logistic Regression",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Tahap Modeling:
    - Train-test split
    - Standardisasi fitur menggunakan StandardScaler
    - Melatih model (Logistic Regression / Random Forest)
    - Mengembalikan model terlatih, scaler, serta metrik evaluasi
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # StandardScaler dipakai di kedua model agar konsisten
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
        )

    model.fit(X_train_scaled, y_train)

    # Prediksi pada data test
    y_pred = model.predict(X_test_scaled)

    # Hitung metrik wajib
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "y_pred": y_pred,
    }

    return model, scaler, metrics


def build_user_input_form(feature_names):
    """
    Membangun form input pengguna untuk prediksi baru.
    Karena semua fitur di dataset ini numerik, kita buat beberapa asumsi rentang nilai yang wajar.
    """
    input_data = {}

    # Penjelasan singkat nama fitur berdasarkan dataset heart (UCI-like)
    feature_desc = {
        "age": "Usia (tahun)",
        "sex": "Jenis kelamin (1 = pria, 0 = wanita)",
        "cp": "Chest pain type (0–3)",
        "trestbps": "Resting blood pressure (mm Hg)",
        "chol": "Serum cholesterol (mg/dl)",
        "fbs": "Fasting blood sugar > 120 mg/dl (1 = ya, 0 = tidak)",
        "restecg": "Resting ECG results (0–2)",
        "thalach": "Maximum heart rate achieved",
        "exang": "Exercise induced angina (1 = ya, 0 = tidak)",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "slope": "Slope of the peak exercise ST segment (0–2)",
        "ca": "Number of major vessels (0–3)",
        "thal": "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)",
    }

    for col in feature_names:
        desc = feature_desc.get(col, col)

        # Kita bedakan integer vs float dengan melihat tipe pada dataset
        if col in ["oldpeak"]:
            value = st.number_input(desc, value=0.0, step=0.1, format="%.2f")
        else:
            value = st.number_input(desc, value=0, step=1)

        input_data[col] = value

    # Mengembalikan sebagai DataFrame 1 baris agar mudah dipakai pada scaler & model
    return pd.DataFrame([input_data])


# ============================
# KONFIGURASI HALAMAN
# ============================

st.set_page_config(
    page_title="Heart Disease Prediction ML App",
    page_icon="❤️",
    layout="wide",
)

st.title("Aplikasi Machine Learning – Prediksi Penyakit Jantung")
st.write(
    "Capstone Project / Tugas Besar Machine Learning – Pipeline end-to-end dari EDA, "
    "training model, hingga prediksi real-time."
)


# ============================
# LOAD & PREPROCESS DATA
# ============================

df = load_data("heart.csv")
X, y, prep_info = preprocess_data(df)


# ============================
# SIDEBAR – PENGATURAN MODEL
# ============================

st.sidebar.header("Pengaturan Model")

model_choice = st.sidebar.selectbox(
    "Pilih Algoritma",
    ("Logistic Regression", "Random Forest"),
)

test_size = st.sidebar.slider(
    "Proporsi Data Test",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=999,
    value=42,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.write("Dataset: `heart.csv`")
st.sidebar.write(f"Jumlah sampel: **{len(df)}**")
st.sidebar.write(f"Jumlah fitur: **{X.shape[1]}**")


# ============================
# LATIH MODEL BERDASARKAN PILIHAN
# ============================

model, scaler, metrics = train_model(
    X, y, model_name=model_choice, test_size=test_size, random_state=random_state
)


# ============================
# TAB-TAB UTAMA
# ============================

tab1, tab2, tab3 = st.tabs(
    ["📊 Ringkasan Dataset (EDA)", "🤖 Pelatihan Model & Evaluasi", "🩺 Prediksi Pasien Baru"]
)


# ============================
# TAB 1 – RINGKASAN DATASET
# ============================

with tab1:
    st.subheader("Ringkasan Dataset")

    st.markdown("**Cuplikan Data (5 baris pertama)**")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Informasi Dasar Dataset**")
        st.write(f"Jumlah baris awal: **{df.shape[0]}**")
        st.write(f"Jumlah kolom: **{df.shape[1]}**")
        st.write(f"Total missing values: **{prep_info['total_missing']}**")
        st.write(f"Jumlah duplikat: **{prep_info['duplicate_count']}**")
        st.write(f"Dimensi akhir setelah pembersihan: **{prep_info['final_shape']}**")

    with col2:
        st.markdown("**Distribusi Target**")
        target_counts = y.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax, palette="viridis")
        ax.set_xlabel("Target (1 = Penyakit Jantung, 0 = Sehat)")
        ax.set_ylabel("Jumlah")
        ax.set_title("Distribusi Kelas Target")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Korelasi Antar Fitur (Heatmap)")

    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Heatmap Korelasi Fitur")
    st.pyplot(fig_corr)


# ============================
# TAB 2 – PELATIHAN & EVALUASI
# ============================

with tab2:
    st.subheader("Pelatihan Model & Komparasi Performa")

    st.markdown(
        f"Model yang sedang dievaluasi: **{model_choice}** dengan test size **{test_size}** "
        f"dan random state **{random_state}**."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Metrik Utama**")
        st.metric("Akurasi", f"{metrics['accuracy']:.3f}")
        st.metric("F1-Score", f"{metrics['f1']:.3f}")

    with col2:
        st.markdown("**Ringkasan Classification Report (macro avg)**")
        report = metrics["report"]
        macro = report.get("macro avg", {})
        st.write(
            {
                "precision (macro)": round(macro.get("precision", np.nan), 3),
                "recall (macro)": round(macro.get("recall", np.nan), 3),
                "f1-score (macro)": round(macro.get("f1-score", np.nan), 3),
            }
        )

    st.markdown("---")
    st.subheader("Confusion Matrix")

    cm = metrics["confusion_matrix"]
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_cm,
        xticklabels=["Sehat (0)", "Penyakit (1)"],
        yticklabels=["Sehat (0)", "Penyakit (1)"],
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"Confusion Matrix – {model_choice}")
    st.pyplot(fig_cm)


# ============================
# TAB 3 – PREDIKSI DATA BARU
# ============================

with tab3:
    st.subheader("Prediksi Pasien Baru")
    st.write(
        "Isi form berikut dengan data kesehatan pasien, kemudian klik tombol **Prediksi** "
        "untuk mengetahui apakah model memprediksi **Terkena Penyakit Jantung** atau **Sehat**."
    )

    with st.form("prediction_form"):
        user_input_df = build_user_input_form(X.columns)
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        # Lakukan scaling menggunakan scaler yang sudah dilatih di pipeline
        user_input_scaled = scaler.transform(user_input_df)

        # Prediksi kelas dan probabilitas
        pred_class = model.predict(user_input_scaled)[0]
        pred_proba = model.predict_proba(user_input_scaled)[0][1]

        if pred_class == 1:
            st.error(
                f"Model memprediksi: **Terkena Penyakit Jantung** "
                f"(Probabilitas positif: {pred_proba:.2%})"
            )
        else:
            st.success(
                f"Model memprediksi: **Sehat** "
                f"(Probabilitas positif: {pred_proba:.2%})"
            )

        st.markdown("**Data yang Anda masukkan:**")
        st.dataframe(user_input_df)



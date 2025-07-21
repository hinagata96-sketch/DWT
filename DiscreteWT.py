
import streamlit as st
import os
import numpy as np
import pandas as pd
import pywt
import mne
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import MinMaxScaler
import zipfile
import io

echo "python-3.12" > runtime.txt
git add runtime.txt
git commit -m "fix: downgrade to Python 3.12 for PyWavelets compatibility"
git push origin main
st.title("EEG DWT Batch Feature Extractor")
st.markdown("""
Upload a ZIP file containing EEG CSVs organized in class/emotion folders. Select DWT parameters and extract features in batch. Download the results as CSV.
""")

uploaded_zip = st.file_uploader("Upload ZIP file of EEG data (folders by class)", type=["zip"])
wavelet_name = st.selectbox("Wavelet Name", ["db4", "db6", "sym5", "sym8", "coif5", "coif3"], index=0)
dwt_level = st.number_input("DWT Level", min_value=1, value=4)
fs = st.number_input("Sampling Frequency (Hz)", min_value=1, value=128)
n_segments = st.number_input("Number of Segments", min_value=1, value=5)

def extract_dwt_features_from_zip(zip_file, wavelet_name, dwt_level, fs, n_segments):
    all_dwt_features = []
    error_files = []
    with zipfile.ZipFile(zip_file) as z:
        class_folders = set([os.path.dirname(f) for f in z.namelist() if f.lower().endswith('.csv')])
        class_folders = [f for f in class_folders if f and not f.startswith('__MACOSX')]
        for class_folder in class_folders:
            class_label = os.path.basename(class_folder)
            csv_files = [
                f for f in z.namelist()
                if f.startswith(class_folder + '/')
                and f.lower().endswith('.csv')
                and not f.startswith('__MACOSX/')
                and not os.path.basename(f).startswith('._')
            ]
            for csv_name in csv_files:
                try:
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f)
                        ch_names = df.columns.tolist()
                        data = df.values.T
                        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
                        raw = mne.io.RawArray(data, info)
                        # ICA
                        ica = mne.preprocessing.ICA(n_components=min(15, len(ch_names)), random_state=42, method='fastica')
                        ica.fit(raw)
                        sources = ica.get_sources(raw).get_data()
                        stds = np.std(sources, axis=1)
                        threshold = np.percentile(stds, 90)
                        ica.exclude = [i for i, s in enumerate(stds) if s > threshold]
                        raw_clean = ica.apply(raw.copy())
                        # Segmentation
                        total_samples = raw_clean.n_times
                        seg_len = total_samples // n_segments
                        for seg_idx in range(n_segments):
                            start = seg_idx * seg_len
                            stop = start + seg_len
                            seg_data = raw_clean.get_data()[:, start:stop]
                            for ch_idx, ch_name in enumerate(ch_names):
                                signal = seg_data[ch_idx]
                                coeffs = pywt.wavedec(signal, wavelet=wavelet_name, level=dwt_level)
                                for lvl, coef in enumerate(coeffs):
                                    coef_abs = np.abs(coef)
                                    energy = np.sum(coef_abs ** 2)
                                    mean_val = np.mean(coef)
                                    std_val = np.std(coef)
                                    ent_val = entropy(coef_abs / (np.sum(coef_abs) + 1e-12))
                                    skew_val = skew(coef)
                                    kurt_val = kurtosis(coef)
                                    all_dwt_features.append({
                                        "class": class_label,
                                        "trial": os.path.splitext(os.path.basename(csv_name))[0],
                                        "segment": seg_idx + 1,
                                        "channel": ch_name,
                                        "level": lvl,
                                        "energy": energy,
                                        "mean": mean_val,
                                        "entropy": ent_val,
                                        "std": std_val,
                                        "skewness": skew_val,
                                        "kurtosis": kurt_val
                                    })
                except Exception as e:
                    error_files.append(f"{csv_name} ({e})")
    df_dwt = pd.DataFrame(all_dwt_features)
    # One-hot encode class labels
    if not df_dwt.empty:
        for class_name in df_dwt['class'].unique():
            df_dwt[f"label_{class_name.lower()}"] = (df_dwt["class"].str.lower() == class_name.lower()).astype(int)
        # Normalize selected features
        feat_cols = ["energy", "mean", "entropy", "std", "skewness", "kurtosis"]
        scaler = MinMaxScaler()
        df_dwt[feat_cols] = scaler.fit_transform(df_dwt[feat_cols])
    return df_dwt, error_files

if uploaded_zip:
    with st.spinner("Extracting DWT features from uploaded data..."):
        df_dwt, error_files = extract_dwt_features_from_zip(uploaded_zip, wavelet_name, dwt_level, fs, n_segments)
    if not df_dwt.empty:
        st.success(f"Extracted {len(df_dwt)} DWT feature rows.")
        st.dataframe(df_dwt.head(100))
        csv = df_dwt.to_csv(index=False).encode('utf-8')
        st.download_button("Download All DWT Features CSV", data=csv, file_name="dwt_features.csv", mime="text/csv")
        # Download per-level CSVs
        levels = sorted(df_dwt['level'].unique())
        for lvl in levels:
            df_level = df_dwt[df_dwt["level"] == lvl]
            csv_level = df_level.to_csv(index=False).encode('utf-8')
            st.download_button(f"Download DWT Features for Level {lvl}", data=csv_level, file_name=f"dwt_features_level_{lvl}.csv", mime="text/csv")
    else:
        st.warning("No DWT features found. Please check your data format.")
    if error_files:
        st.warning("Some files were skipped due to errors:\n" + '\n'.join(error_files[:10]))


import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def plot_feature_mi_streamlit(df_dwt, level=None):
    """
    Plots mutual information (MI) between DWT features and class labels in Streamlit.
    """
    if df_dwt is None or df_dwt.empty:
        st.warning("No feature data found. Please run feature extraction first.")
        return

    # Filter by level (optional)
    if level is not None:
        df = df_dwt[df_dwt['level'] == level]
        st.info(f"Viewing features at DWT level {level}")
    else:
        df = df_dwt.copy()
        st.info("Viewing features across all DWT levels")

    # Encode class labels as integers
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df['class'])

    # Feature columns
    feature_cols = ["energy", "mean", "entropy", "std", "skewness", "kurtosis"]
    X = df[feature_cols]

    # Compute mutual information
    mi_scores = mutual_info_classif(X, y, discrete_features=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(feature_cols, mi_scores, color="skyblue")
    ax.set_title(f"Mutual Information of DWT Features{' (Level ' + str(level) + ')' if level is not None else ''}")
    ax.set_ylabel("Mutual Information")
    ax.set_xlabel("DWT Feature")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)

# === Streamlit MI Plot Controls ===
if 'df_dwt' in locals() and df_dwt is not None and not df_dwt.empty:
    st.header("Mutual Information (MI) Plot")
    levels = sorted(df_dwt['level'].unique())
    selected_level = st.selectbox("Select DWT Level for MI Plot", options=[None] + levels, format_func=lambda x: "All Levels" if x is None else f"Level {x}")
    plot_feature_mi_streamlit(df_dwt, level=selected_level)

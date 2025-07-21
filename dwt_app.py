import streamlit as st
import pandas as pd
import numpy as np
import mne
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import MinMaxScaler
import zipfile
import os

def run_dwt_app():
    st.title("EEG Discrete Wavelet Transform (DWT) Feature Extractor")

    st.markdown("""
**How segmentation and feature extraction works:**
""")

    # Try to load the image and show a friendly error if it fails
    try:
        st.image("/Users/wankhai/Documents/EEGDashboard/DWT.png", caption="DWT Domain EEG Signals", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load image: {e}")
    uploaded_zip = st.file_uploader("Upload ZIP file containing EEG CSVs (organized by class folders)", type=["zip"])

    fs = st.number_input("Sampling Frequency (Hz)", min_value=1, value=128)
    n_segments = st.number_input("Number of Segments", min_value=1, value=5)
    wavelet_options = ["db4", "db6", "sym5", "sym8", "coif5", "coif3"]
    wavelet_name = st.selectbox("Wavelet Name", options=wavelet_options, index=0)
    dwt_level = st.number_input("DWT Level", min_value=1, value=4)

    all_dwt_features = []
    error_files = []

    if uploaded_zip:
        with zipfile.ZipFile(uploaded_zip) as z:
            class_folders = set([os.path.dirname(f) for f in z.namelist() if f.lower().endswith('.csv')])
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
                            data_columns = [col for col in ch_names if col.lower() not in ['label', 'target']]
                            if len(data_columns) < 2:
                                error_files.append(f"{csv_name} (not enough channels)")
                                continue
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
                                    try:
                                        import pywt
                                        signal = seg_data[ch_idx]
                                        coeffs = pywt.wavedec(signal, wavelet_name, level=dwt_level)
                                        for level, coef in enumerate(coeffs):
                                            abs_coef = np.abs(coef)
                                            energy = np.sum(abs_coef ** 2)
                                            coef_mean = np.mean(abs_coef)
                                            coef_entropy = entropy(abs_coef / (np.sum(abs_coef) + 1e-12))
                                            coef_std = np.std(abs_coef)
                                            coef_kurtosis = kurtosis(abs_coef)
                                            coef_skewness = skew(abs_coef)
                                            all_dwt_features.append({
                                                "class": class_label,
                                                "trial": os.path.splitext(os.path.basename(csv_name))[0],
                                                "segment": seg_idx + 1,
                                                "channel": ch_name,
                                                "level": level,
                                                "energy": energy,
                                                "mean": coef_mean,
                                                "entropy": coef_entropy,
                                                "std": coef_std,
                                                "skewness": coef_skewness,
                                                "kurtosis": coef_kurtosis
                                            })
                                    except Exception as e:
                                        error_files.append(f"{csv_name} (DWT error: {e})")
                    except Exception as e:
                        error_files.append(f"{csv_name} ({e})")
    if error_files:
        st.warning("Some files were skipped due to errors or insufficient channels:\n" + '\n'.join(error_files))
    # DataFrame
    df_dwt = pd.DataFrame(all_dwt_features)
    if not df_dwt.empty:
        # One-hot encode labels as 0/1 integers
        class_dummies = pd.get_dummies(df_dwt['class']).astype(int)
        df_dwt = pd.concat([df_dwt, class_dummies], axis=1)
        # Add 'label' column to represent class as one-hot [0,1] (first class column with 1)
        df_dwt['label'] = df_dwt[class_dummies.columns].idxmax(axis=1)
        # Normalize features [0,1]
        feat_cols = ["energy", "mean", "entropy", "std", "skewness", "kurtosis"]
        scaler = MinMaxScaler()
        df_dwt[feat_cols] = scaler.fit_transform(df_dwt[feat_cols])
        st.write("DWT Features (with one-hot class labels):", df_dwt)
        # Download
        csv = df_dwt.to_csv(index=False).encode('utf-8')
        st.download_button("Download DWT Features CSV", data=csv, file_name="dwt_features.csv", mime="text/csv")

        # MI Viewer
        st.header("Mutual Information (MI) Scores: Features vs Class (per DWT Level)")
        from sklearn.feature_selection import mutual_info_classif
        levels = sorted(df_dwt['level'].unique())
        selected_level = st.selectbox("Select DWT Level for MI Analysis", levels)
        df_level = df_dwt[df_dwt['level'] == selected_level]
        if not df_level.empty:
            feature_cols = [col for col in feat_cols]
            class_names = df_level['class'].unique()
            mi_results = []
            X = df_level[feature_cols].values
            for class_name in class_names:
                y_binary = (df_level['class'] == class_name).astype(int)
                mi_scores = mutual_info_classif(X, y_binary, discrete_features=False)
                for feat, score in zip(feature_cols, mi_scores):
                    mi_results.append({
                        "Class": class_name,
                        "Feature": feat,
                        "MI Score": score
                    })
            mi_df = pd.DataFrame(mi_results).sort_values(["Class", "MI Score"], ascending=[True, False])
            st.write(f"MI Scores for DWT Level {selected_level}", mi_df)
            # Download button for MI scores (level vs class)
            mi_csv = mi_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download MI Scores for DWT Level {selected_level}",
                data=mi_csv,
                file_name=f"dwt_level_{selected_level}_mi_scores.csv",
                mime="text/csv"
            )
            import matplotlib.pyplot as plt
            for class_name in class_names:
                class_mi = mi_df[mi_df["Class"] == class_name]
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(class_mi["Feature"], class_mi["MI Score"], color='mediumseagreen')
                ax.set_xlabel("Mutual Information Score")
                ax.set_title(f"MI Scores for DWT Level {selected_level} vs {class_name}")
                ax.invert_yaxis()
                st.pyplot(fig)
        else:
            st.info("No features found for selected DWT level.")
    else:
        st.info("No valid features extracted.")
if __name__ == "__main__":
    run_dwt_app()

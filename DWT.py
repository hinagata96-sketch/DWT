# === Step 1: Batch DWT Feature Extraction ===
from ipywidgets import IntSlider, Button, Output, VBox, Label
import os, numpy as np, pandas as pd, mne
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import MinMaxScaler

# === GUI ===
dwt_segment_slider = IntSlider(value=5, min=1, max=10, step=1, description="Segments:")
dwt_run_button = Button(description="Run Batch DWT Feature Extraction", button_style="info")
dwt_status_label = Label()
dwt_output_log = Output()

# === Parameters ===
base_path = "/content/drive/MyDrive/Temporary Files HERE/Hafiz data/Data Collection"  # ✅ Adjust path if needed
wavelet_name = "db4"
dwt_level = 4
fs = 128
df_dwt = None  # Global DataFrame

# === Main Callback ===
def run_batch_dwt(b):
    global df_dwt
    dwt_output_log.clear_output()
    dwt_status_label.value = "⏳ Processing..."
    all_dwt_features = []
    n_segments = dwt_segment_slider.value

    emotions = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    with dwt_output_log:
        print("🎯 Detected emotion folders:", emotions)

    for emotion in emotions:
        emotion_path = os.path.join(base_path, emotion)

        for trial_file in os.listdir(emotion_path):
            if trial_file.lower().endswith(".csv"):
                trial_path = os.path.join(emotion_path, trial_file)
                trial_name = os.path.splitext(trial_file)[0]
                subject = trial_name.split("_")[0]

                try:
                    df = pd.read_csv(trial_path)
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
                            # pywt removed: DWT feature extraction is disabled due to lack of pywt support on Python 3.13
                            pass  # Feature extraction for DWT is currently disabled
                                    "subject": subject,
                                    "trial": trial_name,
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

                    with dwt_output_log:
                        print(f"✅ {emotion}/{trial_file} processed")

                except Exception as e:
                    with dwt_output_log:
                        print(f"❌ Error in {trial_file}: {e}")

    # DataFrame
    df_dwt = pd.DataFrame(all_dwt_features)

    # One-hot encode labels
    for emo in emotions:
        df_dwt[f"label_{emo.lower()}"] = (df_dwt["emotion"].str.lower() == emo.lower()).astype(int)

    # Normalize features [0,1]
    feat_cols = ["energy", "mean", "entropy", "std", "skewness", "kurtosis"]
    scaler = MinMaxScaler()
    df_dwt[feat_cols] = scaler.fit_transform(df_dwt[feat_cols])

    dwt_status_label.value = "✅ DWT features extracted and ready for download."

# === Display GUI ===
dwt_run_button.on_click(run_batch_dwt)

display(VBox([
    Label("📊 Batch DWT Feature Extraction (ICA, Segment-wise, Normalized)"),
    dwt_segment_slider,
    dwt_run_button,
    dwt_status_label,
    dwt_output_log
]))


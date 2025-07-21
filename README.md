
# EEG DWT Batch Feature Extractor

This Streamlit app extracts Discrete Wavelet Transform (DWT) features from EEG CSV files organized in class/emotion folders. Upload a ZIP, select parameters, and download results as CSV.

## How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   streamlit run DiscreteWT.py
   ```

## Usage
- Upload a ZIP file containing EEG CSVs organized by class folders.
- Select DWT parameters (wavelet, level, segments, etc.).
- Extract features, view results, and download CSVs (all features and per-level).
- View Mutual Information (MI) plots for feature importance.

## Project Files
- `DiscreteWT.py`: Main Streamlit app for batch DWT feature extraction.
- `requirements.txt`: Python dependencies.
- `runtime.txt`: (Optional) Ensures Python 3.12 for Streamlit Cloud compatibility.

## License
MIT

## How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the dashboard:
   ```bash
   streamlit run main_dashboard.py
   ```

## Project Structure
- `main_dashboard.py`: Main entry point for the dashboard
- `TDMI_app.py`, `fdmi_app.py`, `cwt_app.py`, `dwt_app.py`: Individual analysis modules
- `TDMI.py`, `FDMI.py`, `CWT.py`, `DWT.py`: Feature extraction logic
- `requirements.txt`: Python dependencies

## Usage
- Upload ZIP files containing EEG CSVs organized by class folders
- Select analysis domain from the sidebar
- Extract features, view results, and download CSVs

## License
MIT

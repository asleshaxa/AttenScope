ATTENSCOPE — FINAL UPDATED VERSION

Contents
- app.py: Streamlit dashboard. The attention heatmap occupies the full page width; the six-step encoder flowchart is below it.
- transformer.py: short NumPy implementation of positional encoding, multi-head attention, residual connections, LayerNorm, and feed-forward network.
- requirements.txt: dependencies.

INSTALL / RUN (Windows)
1. Extract all files into the same AttenScope folder, replacing both old app.py and transformer.py.
2. In VS Code, stop the old Streamlit server with Ctrl+C.
3. Open a terminal in the folder containing app.py and transformer.py.
4. If using the existing virtual environment at C:\Users\asles\AttenScope\.venv, run:
   C:\Users\asles\AttenScope\.venv\Scripts\python.exe -m streamlit run "C:\Users\asles\OneDrive\Desktop\AttenScope\app.py"
   (Adjust the app.py path if your project folder is elsewhere.)

The UI provides an input sentence and experiment controls in the sidebar. It annotates attention values in the heatmap for sentences up to 12 tokens and uses a vivid turbo colour map scaled to the observed min/max values. The colour bar explicitly states that scale.

Important limitation: model parameters and embeddings are randomly initialized for education; this is not a trained language model.

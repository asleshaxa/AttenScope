import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from transformer import run_encoder


# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="AttenScope | Transformer Attention Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- DESIGN ----------------
st.markdown(
    """
    <style>
    .stApp { background: #0b1020; color: #edf2ff; }
    .block-container { max-width: 1450px; padding-top: 1.4rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { background: #111a30; }
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 18px;
        background: linear-gradient(120deg, #172554, #312e81 58%, #164e63);
        border: 1px solid #34446d; margin-bottom: 1.5rem;
    }
    .eyebrow { color: #bfdbfe; font-size: .76rem; letter-spacing: .19em; font-weight: 800; }
    .hero h1 { color: #fff; font-size: 2.7rem; margin: .35rem 0; }
    .hero p { color: #e0e7ff; font-size: 1rem; margin: 0; }
    .metric-card {
        background: #121b30; border: 1px solid #2b3a58;
        border-radius: 14px; padding: 1rem 1.1rem; min-height: 100px;
    }
    .metric-label { color: #9fb0ce; font-size: .75rem; letter-spacing: .1em; font-weight: 700; }
    .metric-value { color: #f8fafc; font-size: 1.65rem; font-weight: 800; margin-top: .35rem; }
    .section-label { color: #93c5fd; font-size: .76rem; letter-spacing: .15em; font-weight: 800; }
    .step-card {
        background: #121b30; border: 1px solid #2b3a58;
        border-radius: 13px; padding: 1rem; min-height: 122px;
    }
    .step-number { color: #60a5fa; font-size: .82rem; font-weight: 900; letter-spacing: .08em; }
    .step-title { color: #f8fafc; font-size: 1rem; font-weight: 750; margin-top: .4rem; }
    .step-detail { color: #a8b7d2; font-size: .85rem; margin-top: .35rem; }
    .subtle { color: #a8b7d2; font-size: .9rem; }
    div[data-testid="stAlert"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- SIDEBAR INPUT ----------------
with st.sidebar:
    st.markdown("## 🧠 AttenScope")
    st.caption("TRANSFORMER ATTENTION STUDIO")
    st.divider()

    with st.form("encoder_settings"):
        sentence = st.text_area(
            "Input sentence",
            value="The cat sat on the mat",
            height=105,
            help="Use a short sentence. Words separated by spaces are treated as tokens.",
        )
        num_heads = st.selectbox("Attention heads", options=[1, 2, 4], index=1)
        embedding_dim = st.selectbox("Embedding dimension", options=[4, 8, 12, 16], index=1)
        run_button = st.form_submit_button("▶  Run Experiment", use_container_width=True)

    st.divider()
    st.markdown("**Model setup**")
    st.caption("Scaled dot-product attention · NumPy implementation")
    st.info("This is an educational encoder with randomly initialized weights, not a trained language model.")


# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">TRANSFORMER ATTENTION STUDIO</div>
      <h1>AttenScope</h1>
      <p>Explore how attention scores connect tokens, then inspect the encoder computation step by step.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- DEFAULT / FORM SUBMISSION ----------------
if "experiment" not in st.session_state:
    default_sentence = "The cat sat on the mat"
    default_result = run_encoder(default_sentence, num_heads=2, embedding_dim=8)
    st.session_state.experiment = {
        "result": default_result,
        "sentence": default_sentence,
        "heads": 2,
        "dimension": 8,
    }

if run_button:
    try:
        result = run_encoder(
            sentence,
            num_heads=int(num_heads),
            embedding_dim=int(embedding_dim),
        )
        st.session_state.experiment = {
            "result": result,
            "sentence": sentence,
            "heads": int(num_heads),
            "dimension": int(embedding_dim),
        }
    except ValueError as exc:
        st.error(str(exc))

experiment = st.session_state.experiment
result = experiment["result"]
words = result["words"]
attention = result["attention"]
encoder_input = result["input"]
encoder_output = result["output"]
token_count = len(words)
heads = experiment["heads"]
embedding_dim = experiment["dimension"]


# ---------------- METRICS ----------------
st.markdown('<div class="section-label">EXPERIMENT OVERVIEW</div>', unsafe_allow_html=True)
st.write("")
metric_columns = st.columns(4)
metric_values = [
    ("TOKENS", str(token_count)),
    ("ATTENTION HEADS", str(heads)),
    ("EMBEDDING SIZE", str(embedding_dim)),
    ("OUTPUT SHAPE", f"{token_count} × {embedding_dim}"),
]
for column, (label, value) in zip(metric_columns, metric_values):
    with column:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
st.divider()


# ---------------- LARGE FULL-WIDTH HEATMAP ----------------
st.markdown('<div class="section-label">01 / ATTENTION MAP</div>', unsafe_allow_html=True)
st.subheader("Where does each token look?")
st.markdown(
    '<div class="subtle">Rows are the current token; columns are the tokens receiving attention. '
    'Each cell includes its calculated weight.</div>',
    unsafe_allow_html=True,
)
st.write("")

# Scale colours to the actual observed range so small differences are visible.
low = float(np.min(attention))
high = float(np.max(attention))
if np.isclose(low, high):
    low = max(0.0, low - 0.01)
    high = high + 0.01

# Make the heatmap large and square-ish. Streamlit expands it to the full page width.
fig_height = max(7.5, min(12.0, token_count * 0.95))
fig, ax = plt.subplots(figsize=(13.5, fig_height), constrained_layout=True)
fig.patch.set_facecolor("#0b1020")
ax.set_facecolor("#121b30")
cmap = plt.get_cmap("turbo")
image = ax.imshow(
    attention,
    cmap=cmap,
    vmin=low,
    vmax=high,
    aspect="equal",
    interpolation="nearest",
)

ax.set_xticks(range(token_count))
ax.set_yticks(range(token_count))
ax.set_xticklabels(words, rotation=35, ha="right", fontsize=11, color="#e5edff")
ax.set_yticklabels(words, fontsize=11, color="#e5edff")
ax.set_xlabel("Token receiving attention", fontsize=12, color="#dbeafe", labelpad=12)
ax.set_ylabel("Current token", fontsize=12, color="#dbeafe", labelpad=12)
ax.set_title("Self-Attention Weight Matrix", fontsize=16, color="#f8fafc", pad=18, fontweight="bold")
ax.tick_params(length=0, colors="#e5edff")

# Fine white separators keep adjacent cells visually distinct.
ax.set_xticks(np.arange(-0.5, token_count, 1), minor=True)
ax.set_yticks(np.arange(-0.5, token_count, 1), minor=True)
ax.grid(which="minor", color="#f8fafc", linestyle="-", linewidth=0.85, alpha=0.65)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_color("#7184a6")

# Annotate values for normal short sentences; choose text contrast from the cell colour.
if token_count <= 12:
    for row in range(token_count):
        for col in range(token_count):
            value = float(attention[row, col])
            position = (value - low) / (high - low)
            red, green, blue, _ = cmap(position)
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            text_color = "#08111f" if luminance > 0.62 else "#ffffff"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
colorbar.set_label("Attention weight (observed min → max)", color="#dbeafe", fontsize=10, labelpad=10)
colorbar.ax.tick_params(colors="#e5edff", labelsize=9)
colorbar.outline.set_edgecolor("#7184a6")

st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.caption(
    f"Matrix shape: {attention.shape}  ·  "
    f"Row sums: {np.round(attention.sum(axis=1), 3)}  ·  "
    "Colour scale uses the observed minimum and maximum for better contrast."
)

st.divider()


# ---------------- FLOWCHART BELOW THE HEATMAP ----------------
st.markdown('<div class="section-label">02 / ENCODER PIPELINE</div>', unsafe_allow_html=True)
st.subheader("How the sentence moves through the encoder")
st.caption("The cards below follow the same order as the implemented computation.")

steps = [
    ("01", "Tokenization", f"{token_count} word tokens"),
    ("02", "Embedding + position", f"Tensor: {token_count} × {embedding_dim}"),
    ("03", "Multi-head attention", f"{heads} head(s), scaled dot-product"),
    ("04", "Residual + LayerNorm", "Combines input and attention output"),
    ("05", "Feed-forward network", "Dense → ReLU → Dense"),
    ("06", "Contextual output", f"Tensor: {token_count} × {embedding_dim}"),
]
for row_start in (0, 3):
    columns = st.columns(3, gap="medium")
    for column, (number, title, detail) in zip(columns, steps[row_start:row_start + 3]):
        with column:
            st.markdown(
                f'<div class="step-card"><div class="step-number">STEP {number}</div>'
                f'<div class="step-title">{title}</div>'
                f'<div class="step-detail">{detail}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")

st.divider()


# ---------------- INSIGHTS ----------------
st.markdown('<div class="section-label">03 / ATTENTION INSIGHTS</div>', unsafe_allow_html=True)
st.subheader("Strongest attention target for each token")
st.caption("This table reports the highest value in each row of the displayed matrix.")
strongest_indices = np.argmax(attention, axis=1)
insights = {
    "Current token": words,
    "Highest-attention target": [words[int(i)] for i in strongest_indices],
    "Highest weight": [round(float(attention[row, idx]), 3) for row, idx in enumerate(strongest_indices)],
    "Row sum": [round(float(attention[row].sum()), 3) for row in range(token_count)],
}
st.dataframe(insights, use_container_width=True, hide_index=True)

with st.expander("View complete attention matrix"):
    st.dataframe(np.round(attention, 4), use_container_width=True)

with st.expander("View encoder output vectors"):
    st.caption(f"Input representation shape: {encoder_input.shape} · Output shape: {encoder_output.shape}")
    st.dataframe(np.round(encoder_output, 3), use_container_width=True, hide_index=True)

st.info(
    "Interpretation note: weights and embeddings are randomly initialized, not trained. "
    "This app illustrates the math and data flow of a Transformer encoder; it does not demonstrate learned language understanding."
)


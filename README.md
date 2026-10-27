# 🧠 AttenScope

### An Interactive Transformer Attention Visualizer

AttenScope is an interactive learning lab that helps you **see how
self-attention works inside a Transformer encoder**. Enter your own
sentence, run an experiment, and explore the attention heatmap, encoder
pipeline, and tensor dimensions.

> Built for learning and demonstration --- not as a trained language
> model.

## ✨ What You Can Explore

-   **Your own sentence:** type text directly in the sidebar and run an
    experiment.
-   **Attention heatmap:** compare how strongly each token attends to
    other tokens.
-   **Encoder pipeline:** follow tokenization, embeddings + position,
    multi-head attention, residual connections + LayerNorm, feed-forward
    network, and contextual output.
-   **Tensor shapes:** inspect the input representation, attention
    matrix, and encoder output.
-   **Adjustable settings:** experiment with attention-head count and
    embedding dimension.

## 🛠️ Tech Stack

Python · NumPy · Matplotlib · Streamlit

## 🚀 Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/asleshaxa/AttenScope.git
cd AttenScope
```

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install numpy matplotlib streamlit
```

### 4. Launch the app

``` bash
streamlit run app.py
```

The app should open in your browser. Enter a sentence in the sidebar and
select **Run Experiment**.

## 🔎 How the Visualization Works

For a sentence with **n tokens** and an embedding dimension of **d**:

  Component              Shape      What it represents
  ---------------------- ---------- ----------------------------------
  Input representation   `(n, d)`   One vector per token
  Attention matrix       `(n, n)`   Token-to-token attention weights
  Encoder output         `(n, d)`   Updated token representations

Self-attention uses scaled dot-product attention:

`Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V`

-   **Q (Query):** what a token is looking for.
-   **K (Key):** what each token offers for matching.
-   **V (Value):** information combined using attention weights.
-   **Heatmap:** brighter cells indicate higher attention weights. Each
    row should sum to approximately **1**.

## 🧪 Try These Examples

-   `The cat sat on the mat`
-   `The dog chased the ball`
-   `Deep learning models process text using attention mechanisms`

Try sentences with different numbers of words and observe how the
heatmap and tensor shapes change.

## ⚠️ Important Limitations

AttenScope is an **educational, randomly initialized implementation**.
Its embeddings and model weights are not trained on a language dataset,
so the heatmap demonstrates the attention calculation---not meaningful
language understanding. Attention patterns should not be treated as
proof that the model understands a sentence.

## 🎯 Why AttenScope?

Transformers can feel abstract when explained only through equations.
AttenScope makes the key steps more tangible through visual feedback,
helping learners connect **the math, tensor shapes, and encoder
pipeline**.

## 👩‍💻 Author

**Aslesha Azad** · [@asleshaxa](https://github.com/asleshaxa)

------------------------------------------------------------------------

If you find this useful for learning, feel free to ⭐ the repository!

import numpy as np


def softmax(values):
    """Convert each row of scores into probabilities that sum to 1."""
    values = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def positional_encoding(token_count, embedding_dim):
    """Add a simple sinusoidal position signal to each token vector."""
    positions = np.arange(token_count)[:, None]
    dimensions = np.arange(embedding_dim)[None, :]
    angles = positions / np.power(
        10000, (2 * (dimensions // 2)) / embedding_dim
    )

    encoding = np.zeros((token_count, embedding_dim))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding


def layer_norm(values, epsilon=1e-6):
    """Normalize the features of each token representation."""
    mean = np.mean(values, axis=1, keepdims=True)
    variance = np.var(values, axis=1, keepdims=True)
    return (values - mean) / np.sqrt(variance + epsilon)


def multi_head_attention(values, num_heads=2, seed=42):
    """Run multi-head scaled dot-product attention using NumPy."""
    token_count, embedding_dim = values.shape
    if embedding_dim % num_heads != 0:
        raise ValueError("Embedding dimension must be divisible by attention heads.")

    head_dim = embedding_dim // num_heads
    rng = np.random.default_rng(seed)
    scale = 1 / np.sqrt(embedding_dim)

    # Query, Key, and Value projections (randomly initialized for this demo).
    w_query = rng.normal(0, scale, (embedding_dim, embedding_dim))
    w_key = rng.normal(0, scale, (embedding_dim, embedding_dim))
    w_value = rng.normal(0, scale, (embedding_dim, embedding_dim))
    w_output = rng.normal(0, scale, (embedding_dim, embedding_dim))

    query = values @ w_query
    key = values @ w_key
    value = values @ w_value

    head_outputs = []
    head_weights = []

    for head in range(num_heads):
        start = head * head_dim
        end = start + head_dim

        q = query[:, start:end]
        k = key[:, start:end]
        v = value[:, start:end]

        # Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        scores = (q @ k.T) / np.sqrt(head_dim)
        weights = softmax(scores)

        head_outputs.append(weights @ v)
        head_weights.append(weights)

    combined_heads = np.concatenate(head_outputs, axis=1)
    attention_output = combined_heads @ w_output
    average_weights = np.mean(np.stack(head_weights, axis=0), axis=0)

    return attention_output, average_weights


def run_encoder(sentence, num_heads=2, embedding_dim=8, seed=42):
    """Run one small, educational Transformer encoder block."""
    words = sentence.strip().split()

    if not words:
        raise ValueError("Please enter a sentence with at least one word.")
    if len(words) > 16:
        raise ValueError("Please use 16 words or fewer for a readable heatmap.")
    if embedding_dim % num_heads != 0:
        raise ValueError("Embedding dimension must be divisible by attention heads.")

    rng = np.random.default_rng(seed)

    # Random token embeddings + positional information.
    token_embeddings = rng.normal(0, 0.5, (len(words), embedding_dim))
    encoder_input = token_embeddings + positional_encoding(len(words), embedding_dim)

    # Multi-head attention, followed by a residual connection and LayerNorm.
    attention_output, attention_weights = multi_head_attention(
        encoder_input, num_heads=num_heads, seed=seed + 1
    )
    normalized = layer_norm(encoder_input + attention_output)

    # Small feed-forward network: Dense -> ReLU -> Dense.
    w1 = rng.normal(0, 1 / np.sqrt(embedding_dim), (embedding_dim, embedding_dim * 2))
    w2 = rng.normal(0, 1 / np.sqrt(embedding_dim * 2), (embedding_dim * 2, embedding_dim))
    hidden = np.maximum(0, normalized @ w1)
    feed_forward_output = hidden @ w2

    # Second residual connection and normalization.
    encoder_output = layer_norm(normalized + feed_forward_output)

    return {
        "words": words,
        "input": encoder_input,
        "attention": attention_weights,
        "output": encoder_output,
    }

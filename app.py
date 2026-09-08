"""
Linear Attention Explorer
=========================
DataForge 2026 — Pathway Track (Solo submission)

An educational Streamlit app that teaches the basic idea behind
linear-attention-style computation, and conceptually connects it to
Pathway's Dragon Hatchling (BDH / BDH-CQ) architecture.

Tech stack: Python, Streamlit, NumPy, Plotly only (no PyTorch/TF/JAX).

IMPORTANT SCOPE NOTE (read this before judging the math):
- This app does NOT implement, reproduce, or train Pathway's BDH/BDH-CQ.
- It does NOT claim "BDH = linear attention" or "BDH is an SSM".
- It does NOT claim linear attention "always" uses O(N) memory — see
  Tab 4 for the precise, caveated version of that claim.
- All "10,000 token" scenarios are computed THEORETICALLY (just N and N²
  as numbers). The app never builds a real matrix bigger than ~8x8.

Run with:  streamlit run app.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# -----------------------------------------------------------------------
# Global config — fixed seed so every rerun of the toy math is IDENTICAL.
# -----------------------------------------------------------------------
SEED = 42
EMBED_DIM = 8  # "d" — the embedding dimension used throughout this app

st.set_page_config(page_title="Linear Attention Explorer", layout="wide")


# =========================================================================
# CORE MATH (kept intentionally small — every function does ONE thing)
# =========================================================================

def softmax(x, axis=-1):
    """
    Numerically stable softmax.

    softmax turns a row of raw similarity scores into a probability
    distribution: every value becomes non-negative and each row sums to 1.
    This is what makes standard attention weights interpretable as
    "how much attention token i pays to token j".
    """
    x = x - np.max(x, axis=axis, keepdims=True)  # prevents overflow in exp()
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def feature_map(x):
    """
    phi(x) = ELU(x) + 1

    Standard attention gets its non-negative, "attention-like" weights
    from softmax(QK^T). Linear attention instead applies a feature map
    phi() separately to Q and K *before* any multiplication happens.

    We use phi(x) = ELU(x) + 1 (a common, simple choice from the linear
    attention literature) because it keeps every value positive — similar
    in spirit to softmax weights — while being cheap to compute:

        phi(x) = x + 1        if x > 0
        phi(x) = exp(x)       if x <= 0   (this is ELU(x)+1 when alpha=1)

    The KEY property we actually need is that phi(Q) and phi(K) can be
    combined in a different order than QK^T (see linear attention notes
    in Tab 2) — that's what avoids ever forming an N x N matrix.
    """
    return np.where(x > 0, x + 1.0, np.exp(x))


def make_toy_qkv(tokens, dim=EMBED_DIM, seed=SEED):
    """
    Build small, DETERMINISTIC, toy Q/K/V matrices for a short list of
    tokens. This is a fixed random projection, not a trained embedding
    model — it exists purely to make the attention math concrete and
    reproducible for teaching purposes.
    """
    rng = np.random.RandomState(seed)
    n = len(tokens)
    X = rng.randn(n, dim)        # toy "embeddings", one row per token
    Wq = rng.randn(dim, dim)     # toy query projection
    Wk = rng.randn(dim, dim)     # toy key projection
    Wv = rng.randn(dim, dim)     # toy value projection
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    return Q, K, V


def standard_attention(Q, K, V):
    """
    Standard (softmax) self-attention.

        scores = Q K^T / sqrt(d)      -> shape (N, N)  <-- the expensive part
        attn   = softmax(scores)      -> each row sums to 1
        output = attn @ V

    Q K^T explicitly compares EVERY query token against EVERY key token,
    producing an N x N matrix of pairwise interaction scores. That matrix
    is the thing that grows quadratically with sequence length N.
    """
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)   # scaling by sqrt(d) keeps scores well-behaved
    attn = softmax(scores, axis=-1)
    output = attn @ V
    return attn, output


def causal_linear_attention_steps(Q, K, V):
    """
    Token-by-token linear-attention-style state accumulation.

    Instead of comparing every pair of tokens, we walk through the
    sequence ONCE and keep two running totals ("the state"):

        S    (d x d matrix) = sum of phi(k_i) outer_product v_i, for i <= t
        Zvec (d vector)     = sum of phi(k_i),                   for i <= t

    At each step t, the output is:

        output_t = ( phi(q_t) @ S ) / ( phi(q_t) . Zvec )

    Why this is equivalent to the "phi(Q)(phi(K)^T V)" idea:
    Matrix multiplication is associative, so instead of computing
    (phi(Q) phi(K)^T) V  — which needs the N x N matrix phi(Q)phi(K)^T —
    we can compute phi(Q) (phi(K)^T V) — which only ever needs the much
    smaller d x d matrix (phi(K)^T V). The running "state" S above is
    exactly that d x d matrix, built up incrementally one token at a time.

    Crucially: S and Zvec have a FIXED size (they depend only on the
    embedding dimension d, not on how many tokens we've seen). Only their
    VALUES change as more tokens arrive — their SIZE never grows.
    """
    phi_Q = feature_map(Q)
    phi_K = feature_map(K)
    n, d = Q.shape
    S = np.zeros((d, d))
    Zvec = np.zeros(d)
    steps = []
    for t in range(n):
        S = S + np.outer(phi_K[t], V[t])      # fold token t's key/value into the state
        Zvec = Zvec + phi_K[t]                # update the normalizer
        denom = max(phi_Q[t] @ Zvec, 1e-6)    # avoid divide-by-zero
        out_t = (phi_Q[t] @ S) / denom
        steps.append({"state": S.copy(), "output": out_t.copy()})
    return steps


# =========================================================================
# STREAMLIT UI
# =========================================================================

st.title("Linear Attention Explorer")
st.write(
    "See why comparing every token with every other token becomes "
    "expensive — and how state-based attention can change the computation."
)

with st.expander("How to use this"):
    st.markdown(
        """
        1. Start with **Big Picture**
        2. Try increasing the sequence length
        3. Open **See It Step by Step**
        4. Watch the state update, token by token
        5. See how this connects to BDH
        """
    )

with st.sidebar:
    st.write(
        "Explore how attention behaves as sequences get longer, then see "
        "a small step-by-step toy example."
    )
    st.caption("Seed fixed at 42, so results are reproducible.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Big Picture", "See It Step by Step", "Connecting to BDH", "Trade-offs"]
)

# -------------------------------------------------------------------------
# TAB 1 — The Big Picture
# -------------------------------------------------------------------------
with tab1:
    st.write(
        "Imagine 100 people in a room, and everyone has to compare "
        "themselves with everyone else. The number of comparisons grows "
        "fast — really fast."
    )
    st.write(
        "With **N** tokens, standard attention does something similar: "
        "it looks at an **N × N** set of token-to-token interactions."
    )

    n_val = st.slider(
        "Try increasing the sequence length (N):", min_value=100, max_value=10000, value=2000, step=100
    )

    n_range = np.arange(100, 10001, 100)
    pairwise = n_range.astype(np.int64) ** 2

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=n_range, y=pairwise, mode="lines", name="N² interactions",
                   line=dict(width=3))
    )
    fig.add_trace(
        go.Scatter(
            x=[n_val], y=[n_val ** 2], mode="markers",
            marker=dict(size=14, color="red"), name="Selected N"
        )
    )
    fig.update_layout(
        xaxis_title="N (sequence length)",
        yaxis_title="N² interactions",
        height=420,
        margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"At N = {n_val:,}, that's {n_val ** 2:,} interactions. "
        "This is theoretical scaling, not a measurement of real GPU runtime."
    )

    with st.expander("Want the math?"):
        st.markdown(
            """
            Each token gets three vectors: a **Query (Q)**, a **Key (K)**,
            and a **Value (V)**.

            To find out how much each token should "attend to" every
            other token, we compute **Q Kᵀ**, scaled by `1/√d` (where
            `d` is the embedding size). This compares *every* query
            against *every* key — for N tokens, that's N × N comparisons,
            which is exactly the quadratic growth shown in the chart above.
            """
        )

# -------------------------------------------------------------------------
# TAB 2 — See the Difference
# -------------------------------------------------------------------------
with tab2:
    sentence = st.text_input(
        "Try your own sentence (5–8 words work best):",
        value="AI research moves very fast",
    )
    tokens = sentence.strip().split()

    if len(tokens) < 2:
        st.warning("Please enter at least 2 words to continue.")
        st.stop()
    if len(tokens) > 8:
        tokens = tokens[:8]
        st.caption("Sentence trimmed to the first 8 tokens for this toy demo.")

    Q, K, V = make_toy_qkv(tokens)
    attn, std_out = standard_attention(Q, K, V)

    st.write(
        "Here, every token can interact with every other token. The "
        "square below shows those interactions."
    )
    fig_heat = go.Figure(
        data=go.Heatmap(z=attn, x=tokens, y=tokens, colorscale="Blues", colorbar=dict(title="weight"))
    )
    fig_heat.update_layout(height=400, margin=dict(t=10))
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    st.write(
        "Now watch a different idea: instead of comparing every pair, "
        "each token folds its information into a small **state** as we "
        "move through the sentence, one token at a time."
    )

    steps = causal_linear_attention_steps(Q, K, V)
    step_idx = st.slider(
        "Step through the sentence, token by token:", 0, len(tokens) - 1, 0
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(f"**Current token:** '{tokens[step_idx]}'")
        st.write(f"**State size:** {Q.shape[1]} × {Q.shape[1]} (fixed)")
        st.write("**So far:** " + " → ".join(tokens[: step_idx + 1]))
    with col2:
        state = steps[step_idx]["state"]
        fig_state = go.Figure(data=go.Heatmap(z=state, colorscale="Oranges"))
        fig_state.update_layout(height=380, margin=dict(t=10))
        st.plotly_chart(fig_state, use_container_width=True)

    st.write(
        "Notice: the state's *values* keep changing, but its *size* "
        "never does — unlike the square above, which grows with every "
        "extra token."
    )

    with st.expander("Want to see what's happening mathematically?"):
        st.markdown(
            """
            This is a small educational toy demo — not an implementation
            of Pathway's BDH.

            Each token's key and value are folded into a running state
            matrix `S`, and a running total `Z` keeps everything on a
            sensible scale:

            `S = S + φ(k) ⊗ v`   and   `Z = Z + φ(k)`

            where **φ** ("phi") is a simple feature map that keeps values
            positive. This works because matrix multiplication is
            associative — `(φ(Q) φ(K)ᵀ) V` (needs an N×N matrix) equals
            `φ(Q) (φ(K)ᵀ V)` (only ever needs a small d×d matrix). The
            state above *is* that small matrix, built up one token at a time.
            """
        )

# -------------------------------------------------------------------------
# TAB 3 — Connecting to BDH
# -------------------------------------------------------------------------
with tab3:
    st.write("So where does BDH fit?")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Standard Attention**")
        st.write("Tokens → pairwise interactions")
    with c2:
        st.markdown("**Linear-attention-style**")
        st.write("Tokens → accumulated state")
    with c3:
        st.markdown("**BDH / BDH-CQ**")
        st.write("Context → synaptic/state updates")

    st.write(
        "These are related ideas about handling context through "
        "different forms of computation and state, rather than only "
        "comparing every token pair explicitly."
    )

    st.markdown("### A concrete BDH-CQ memory update")
    st.markdown(
        "The published BDH-CQ paper describes its high-level contextual memory update as:"
    )
    st.latex(r"S_t = U_\theta(S_{t-1}, D_t)")
    st.markdown(
        "Here, `D_t` is the current demonstration and `S_t` is the updated recurrent "
        "memory. The update function is part of the published architecture; this app "
        "does **not** implement `U_θ` or reproduce the BDH-CQ model."
    )
    st.caption("Source: Engdahl et al., *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning* (2026), Eq. 1.")

    with st.expander("Want more detail?"):
        st.markdown(
            """
            Pathway's Dragon Hatchling (BDH) is a brain-inspired,
            post-Transformer architecture where contextual information is
            accumulated through synaptic/state-oriented updates.
            BDH-CQ additionally uses evolving recurrent memory and latent
            reasoning; its paper describes the high-level memory update shown above.

            These connections are **conceptual analogies**, not claims of
            mathematical equivalence:

            - This app's linear-attention demo is a simplified educational
              toy — it is **not** BDH.
            - We are **not** claiming "BDH = linear attention."
            - We are **not** claiming "BDH is a State Space Model (SSM)."
            - This app does not implement, reproduce, or train any part of
              BDH or BDH-CQ.
            """
        )

# -------------------------------------------------------------------------
# TAB 4 — Trade-offs
# -------------------------------------------------------------------------
with tab4:
    st.write("What do we gain, and what do we lose?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Gain**")
        st.markdown(
            """
            - Fewer explicit pairwise interactions
            - Useful for long sequences
            - State can be updated incrementally
            """
        )
    with col2:
        st.markdown("**Trade-off**")
        st.markdown(
            """
            - Not identical to standard softmax attention
            - Retrieval behavior can differ
            - Exact efficiency depends on the formulation
            - Theoretical scaling ≠ real-world runtime
            """
        )

    with st.expander("Want more detail?"):
        st.markdown(
            """
            | Quantity | Standard Attention | Linear-Attention-Style (this app) |
            |---|---|---|
            | Pairwise interactions considered | O(N²) | Not materialized pairwise |
            | Attention matrix storage | O(N²) | Not required |
            | State size | — | O(d²) — depends on embedding dim **d**, not on N |
            | Compute (this app's formulation) | O(N² · d) | O(N · d²) |

            The state size doesn't grow with N *for a fixed embedding
            dimension d* — that's a specific, correct claim, not the
            same as the broader claim "linear attention always uses
            O(N) memory." The real picture depends on N, on d, and on
            exactly how the formulation is implemented.
            """
        )

st.divider()
st.caption("Seed fixed at 42 for reproducibility.")

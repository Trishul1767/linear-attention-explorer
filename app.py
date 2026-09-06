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

st.title("🔗 Linear Attention Explorer")
st.caption(
    "DataForge 2026 — Pathway Track · Educational demo built with "
    "Python, Streamlit, NumPy & Plotly (no ML training, no PyTorch/TF/JAX)."
)

with st.sidebar:
    st.header("About this app")
    st.markdown(
        """
        This app teaches the **core idea** behind linear-attention-style
        computation and connects it — conceptually only — to Pathway's
        **Dragon Hatchling (BDH / BDH-CQ)**.

        **What this app is NOT:**
        - Not an implementation of BDH/BDH-CQ
        - Not a trained neural network
        - Not a runtime/GPU benchmark

        Walk through the tabs in order for the intended learning path.
        """
    )
    st.divider()
    st.caption("Fixed random seed = 42 → all toy math is fully reproducible.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. The Big Picture", "2. See the Difference", "3. Connecting to BDH", "4. Trade-offs"]
)

# -------------------------------------------------------------------------
# TAB 1 — The Big Picture
# -------------------------------------------------------------------------
with tab1:
    st.header("Why does standard attention get expensive?")

    st.markdown(
        """
        In standard self-attention, three matrices are computed from each
        token's embedding: a **Query (Q)**, a **Key (K)**, and a **Value (V)**.

        To decide how much each token should "attend to" every other
        token, we compute **Q Kᵀ**. This single operation compares
        *every* query against *every* key — for a sequence of length
        **N**, that's **N × N** pairwise comparisons.

        A sequence of just 10,000 tokens therefore produces
        **10,000 × 10,000 = 100,000,000** pairwise interactions — and
        that count grows *quadratically*: doubling N roughly
        **quadruples** the work.
        """
    )

    n_val = st.slider(
        "Sequence length N", min_value=100, max_value=10000, value=2000, step=100
    )

    n_range = np.arange(100, 10001, 100)
    pairwise = n_range.astype(np.int64) ** 2

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=n_range, y=pairwise, mode="lines", name="N² pairwise interactions",
                   line=dict(width=3))
    )
    fig.add_trace(
        go.Scatter(
            x=[n_val], y=[n_val ** 2], mode="markers",
            marker=dict(size=14, color="red"), name="Selected N"
        )
    )
    fig.update_layout(
        title="Pairwise interactions vs. sequence length (theoretical, not measured runtime)",
        xaxis_title="N (sequence length)",
        yaxis_title="N² pairwise interactions",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.metric(f"Pairwise interactions at N = {n_val:,}", f"{n_val ** 2:,}")

    st.info(
        "📌 This chart shows a **theoretical calculation** of N² — it does "
        "NOT measure actual GPU runtime, wall-clock time, or memory usage "
        "on real hardware. Real performance depends on hardware, "
        "implementation, and many other factors."
    )

# -------------------------------------------------------------------------
# TAB 2 — See the Difference
# -------------------------------------------------------------------------
with tab2:
    st.header("Standard attention vs. linear-attention-style state accumulation")

    sentence = st.text_input(
        "Enter a short sentence (5–8 words work best):",
        value="AI research moves very fast",
    )
    tokens = sentence.strip().split()

    if len(tokens) < 2:
        st.warning("Please enter at least 2 words to continue.")
        st.stop()
    if len(tokens) > 8:
        tokens = tokens[:8]
        st.caption("ℹ️ Sentence truncated to the first 8 tokens for this toy demo.")

    Q, K, V = make_toy_qkv(tokens)
    attn, std_out = standard_attention(Q, K, V)

    st.subheader("① Standard attention: the N × N matrix")
    st.markdown(
        "Every token (row) attends to every other token (column). "
        "This is the **attention matrix** — the thing that must be "
        "explicitly built and stored in standard attention."
    )
    fig_heat = go.Figure(
        data=go.Heatmap(z=attn, x=tokens, y=tokens, colorscale="Blues", colorbar=dict(title="weight"))
    )
    fig_heat.update_layout(
        title=f"Attention matrix ({len(tokens)} × {len(tokens)} = {len(tokens)**2} stored numbers)",
        height=420,
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption(
        f"This toy example stores {len(tokens)**2} numbers. "
        f"If this sentence had 10,000 tokens instead of {len(tokens)}, "
        f"the SAME idea would require 100,000,000 stored numbers — "
        f"which is exactly why we never build that matrix in this app."
    )

    st.divider()

    st.subheader("② Linear-attention-style: accumulating a state, token by token")
    st.markdown(
        """
        **🧪 Educational toy demonstration — this is *not* an
        implementation of Pathway's BDH.**

        Instead of comparing every pair of tokens, we walk through the
        sentence **once**, left to right, and keep updating a small,
        fixed-size **state matrix**. This works because matrix
        multiplication is associative:

        `(φ(Q) φ(K)ᵀ) V`  (needs an N×N matrix)  is mathematically equal to
        `φ(Q) (φ(K)ᵀ V)`  (only ever needs a small d×d matrix)

        where **φ** ("phi") is a simple feature map (see sidebar/code for
        the exact formula). The state below *is* that small d×d matrix,
        built up one token at a time.
        """
    )

    steps = causal_linear_attention_steps(Q, K, V)
    step_idx = st.slider(
        "Step through the sentence, token by token:", 0, len(tokens) - 1, 0
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Current token", f"'{tokens[step_idx]}'")
        st.metric("State size (fixed)", f"{Q.shape[1]} × {Q.shape[1]}")
        st.metric("Tokens processed so far", step_idx + 1)
        st.write("**Sequence so far:**")
        st.write(" → ".join(tokens[: step_idx + 1]))
    with col2:
        state = steps[step_idx]["state"]
        fig_state = go.Figure(data=go.Heatmap(z=state, colorscale="Oranges"))
        fig_state.update_layout(
            title=f"Accumulated state after token {step_idx + 1} ('{tokens[step_idx]}')",
            height=420,
        )
        st.plotly_chart(fig_state, use_container_width=True)

    st.success(
        "💡 **The 'aha' moment:** As you move the slider, the state's "
        "*values* keep changing — but its *size* never does. Compare that "
        "to the N×N matrix above, which would grow every time we add a "
        "new token. That's the core reorganization idea behind "
        "linear-attention-style computation."
    )

# -------------------------------------------------------------------------
# TAB 3 — Connecting to BDH
# -------------------------------------------------------------------------
with tab3:
    st.header("Conceptual connection to Pathway's Dragon Hatchling (BDH / BDH-CQ)")

    st.warning(
        "⚠️ The connections below are **conceptual analogies** meant to "
        "build intuition. They are **not** claims that these architectures "
        "are mathematically identical. This app does not implement, "
        "reproduce, or train any part of BDH."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Standard Attention")
        st.markdown("**Tokens → Pairwise interactions**")
        st.caption(
            "Every pair of tokens is compared explicitly via QKᵀ, "
            "producing an N×N matrix (Tab 1 & Tab 2)."
        )
    with c2:
        st.markdown("#### Linear-Attention-Style")
        st.markdown("**Tokens → Accumulated state**")
        st.caption(
            "Tokens are folded one-by-one into a small, fixed-size "
            "state instead of being compared pairwise (Tab 2)."
        )
    with c3:
        st.markdown("#### BDH / BDH-CQ (Pathway)")
        st.markdown("**Tokens/demonstrations → Synaptic/state updates**")
        st.caption(
            "A brain-inspired, post-Transformer architecture where "
            "contextual information is accumulated through "
            "synaptic/state-oriented updates. BDH-CQ additionally "
            "introduces a compressed recurrent/synaptic state."
        )

    st.divider()

    st.markdown(
        """
        ### What's actually similar?
        - All three ideas move away from relying *only* on explicit,
          pairwise token comparison.
        - Both the linear-attention toy demo and BDH rely on some form of
          **accumulated / stateful representation** rather than
          materializing every pairwise interaction.

        ### What's NOT the same — please read before your demo/pitch
        - This app's linear-attention demo is a **simplified educational
          toy**. It is **not** BDH, and it does not claim to be.
        - BDH is a distinct, brain-inspired architecture from Pathway with
          its own design. **We are not claiming "BDH = linear attention."**
        - **We are not claiming "BDH is a State Space Model (SSM)"** —
          that claim is only appropriate if Pathway's own materials state
          it directly, which this app does not assume.
        - This app does not implement, reproduce, or train any part of
          BDH or BDH-CQ.
        """
    )

# -------------------------------------------------------------------------
# TAB 4 — Trade-offs
# -------------------------------------------------------------------------
with tab4:
    st.header("Trade-offs: what is gained, and what is lost?")

    col1, col2 = st.columns(2)
    with col1:
        st.success("**Advantages**")
        st.markdown(
            """
            - Avoids explicitly building the full N×N attention matrix
            - Can support efficient, state-based (streaming) processing
            - Attractive for long sequences, since the state's *size*
              doesn't grow with N
            """
        )
    with col2:
        st.warning("**Trade-offs**")
        st.markdown(
            """
            - Not identical to standard softmax attention — the math is
              genuinely different, not just a faster version of the same thing
            - Retrieval behavior can differ; reproducing very sharp,
              highly selective attention patterns can be harder
            - Exact complexity depends on the specific formulation and
              on the embedding dimension, not just on N
            - Theoretical complexity ≠ measured wall-clock performance
              on real hardware
            """
        )

    st.divider()
    st.markdown("### A precise way to think about complexity")
    st.markdown(
        """
        | Quantity | Standard Attention | Linear-Attention-Style (this app) |
        |---|---|---|
        | Pairwise interactions considered | O(N²) | Not materialized pairwise |
        | Attention matrix storage | O(N²) | Not required |
        | State size | — | O(d²) — depends on embedding dim **d**, not on N |
        | Compute (this app's formulation) | O(N² · d) | O(N · d²) |
        """
    )
    st.info(
        "📌 **Important caveat:** the state size doesn't grow with N "
        "*for a fixed embedding dimension d*. That is a specific, "
        "correct claim — not the same as the overly broad claim "
        "**\"linear attention always uses O(N) memory.\"** The real "
        "picture depends on N, on d, and on exactly how the formulation "
        "is implemented."
    )

st.divider()
st.caption(
    "Educational toy project for DataForge 2026 (Pathway Track). "
    "All computations use fixed random seeds for full reproducibility."
)

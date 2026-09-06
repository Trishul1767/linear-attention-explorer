# linear-attention-explorer
# 🔗 Linear Attention Explorer

**DataForge 2026 — Pathway Track (Solo submission)**

An educational Streamlit app that teaches the core idea behind
linear-attention-style computation, and conceptually connects it to
Pathway's Dragon Hatchling (BDH / BDH-CQ) architecture.

🔴 **Live demo:** _add your Streamlit Community Cloud link here after deploying_

---

## What this app is

A single-page, 4-tab interactive teaching tool built with **Python, Streamlit,
NumPy, and Plotly only** — no PyTorch/TensorFlow/JAX, no model training, no
LLM calls.

1. **The Big Picture** — why standard attention's pairwise comparisons grow
   as O(N²) with sequence length N.
2. **See the Difference** — a real, tiny (≤8-token) standard-attention
   heatmap, followed by a step-by-step toy demo of linear-attention-style
   state accumulation (token by token).
3. **Connecting to BDH** — a purely conceptual comparison between standard
   attention, linear-attention-style state accumulation, and Pathway's BDH.
4. **Trade-offs** — what's gained and lost, with a precise (not oversimplified)
   complexity comparison.

## What this app explicitly does **NOT** claim

- It does **not** implement, reproduce, or train Pathway's BDH / BDH-CQ.
- It does **not** claim "BDH = linear attention."
- It does **not** claim "BDH is a State Space Model (SSM)."
- It does **not** claim linear attention "always" uses O(N) memory — Tab 4
  gives the precise, caveated version of that claim.
- The N=10,000 scenarios in Tab 1 are **theoretical calculations only**
  (just plotting N²) — the app never builds a real matrix larger than 8×8,
  and none of this is a GPU/runtime benchmark.

## Running locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

- Python 3
- Streamlit — UI
- NumPy — all math (fixed random seed = 42 for full reproducibility)
- Plotly — heatmaps and charts

## Reproducibility

All toy computations use `numpy.random.RandomState(42)`. Re-running the app,
or reloading the page, produces identical numbers for the same input
sentence — nothing here is randomly regenerated on each rerun.

## AI-assistance disclosure

AI assistance (Claude, Anthropic) was used to help draft the initial code
structure for `app.py`, including the NumPy implementation of standard and
linear attention, the feature-map function, the Plotly visualizations, and
the explanatory text across tabs.

I reviewed, tested, and understand every formula used in this app
(Q/K/V, QKᵀ, softmax, the φ(x)=ELU(x)+1 feature map, and the state-accumulation
update rule). I verified the causal state-accumulation logic numerically
against a brute-force calculation before including it, and made the final
decisions on scope and framing — in particular, what claims to make and
avoid regarding Pathway's BDH/BDH-CQ.

## Author

Trishul — 2nd-year CSE — Solo submission, DataForge 2026, Pathway Track.

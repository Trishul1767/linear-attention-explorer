# 🔗 Linear Attention Explorer

**DataForge 2026 — Pathway Track**
**Solo submission**

An interactive educational app that makes the basic idea behind **linear-attention-style computation** easier to see and understand.

🔴 **Live demo:** https://linear-attention-explorer-trishul.streamlit.app/

---

## What is this?

Attention can become expensive when a sequence gets longer because tokens can interact with many other tokens.

This app lets you **see that idea instead of only reading about it**.

The learning path is simple:

1. See how token-to-token interactions grow.
2. Try a small attention example yourself.
3. Watch information accumulate into a state.
4. See how this connects conceptually to Pathway's BDH / BDH-CQ.
5. Explore the trade-offs.

The app is designed to be understandable even if you haven't studied Transformers or modern AI architectures in depth.

---

## 🔎 Explore the App

### 1. The Big Picture

Change the sequence length and see how the number of possible token-to-token interactions grows.

For `N` tokens, standard attention creates an `N × N` interaction structure.

The visualization makes the rapid growth easy to see.

> **Note:** The large sequence-length examples are theoretical scaling calculations, not measurements of actual GPU runtime.

---

### 2. See the Difference

Enter a short sentence such as:

> **AI research moves very fast**

The app creates a small, deterministic toy example and lets you explore:

* Standard attention
* A linear-attention-style computation

You can see the standard attention matrix and then follow how information can be accumulated into a state one token at a time.

The goal is to make the difference **visual and intuitive**, rather than requiring the learner to understand the mathematics first.

---

### 3. Connecting to BDH

The app then introduces **Pathway's Dragon Hatchling (BDH / BDH-CQ)**.

The connection is conceptual:

```text
Standard Attention
        ↓
Pairwise token interactions

Linear-attention-style computation
        ↓
Accumulated state

BDH / BDH-CQ
        ↓
Synaptic / recurrent state mechanisms
```

These are **not the same architecture**.

This project does not implement or reproduce BDH / BDH-CQ. The comparison is intended to help learners understand the broader idea of processing context through state or memory mechanisms.

---

### 4. Trade-offs

Linear-attention-style approaches are not simply "better attention."

The app also explores some of the trade-offs, including:

* avoiding explicit full pairwise interaction matrices in suitable formulations
* differences from standard softmax attention
* different retrieval behavior
* the difference between theoretical complexity and real-world runtime

---

## 🧠 The Main Idea

A simple way to think about the concepts explored in this app is:

### Standard attention

```text
Token ──┐
Token ──┼──→ Token-to-token interactions
Token ──┤
Token ──┘
              ↓
           N × N
```

As `N` grows, the number of possible interactions grows roughly as `N²`.

### Linear-attention-style idea

```text
Token ──┐
Token ──┤
Token ──┼──→ Accumulated state
Token ──┤
Token ──┘
```

The computation can be reorganized in some formulations so that information is accumulated into a state instead of explicitly materializing the complete `N × N` interaction matrix.

The exact complexity and memory behavior depend on the particular formulation and dimensions involved.

---

## 🛠️ Built With

* **Python**
* **Streamlit**
* **NumPy**
* **Plotly**

No PyTorch, TensorFlow, or JAX is required.

The toy demonstrations use small matrices so that the underlying computation can be visualized directly.

---

## 🚀 Run Locally

Clone the repository:

```bash
git clone https://github.com/Trishul1767/linear-attention-explorer.git
cd linear-attention-explorer
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
linear-attention-explorer/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🔁 Reproducibility

The toy calculations use a fixed random seed, so the same input produces the same results when the application is rerun.

The application also keeps the actual matrix demonstrations small. Large sequence lengths are handled through theoretical calculations rather than by creating enormous attention matrices.

---

## 🤖 AI Assistance

AI tools were used during development to assist with the initial code structure, mathematical implementation, visualizations, and wording.

The final application was reviewed, tested, and simplified by the author, with particular attention to keeping the project understandable, reproducible, and technically scoped.

---

## 🎓 Hackathon

Built as a solo submission for:

**DataForge 2026 — Pathway Track**

The project focuses on turning a difficult AI concept into a small interactive learning experience that students can explore directly.

---

## 👤 Author

**Trishul**
2nd-year CSE student
Solo participant — DataForge 2026

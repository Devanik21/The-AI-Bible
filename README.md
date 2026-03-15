# The AI Bible

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/The-AI-Bible?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/The-AI-Bible?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> The definitive AI knowledge compendium — a structured, curated, and continuously evolving reference for AI/ML concepts, architectures, and research.

---

**Topics:** `deep-learning` · `education` · `generative-ai` · `knowledge-base` · `large-language-models` · `machine-learning` · `neural-networks` · `research` · `tutorial` · `comprehensive-ai-reference`

## Overview

The AI Bible is a structured knowledge base and reference repository for artificial intelligence and
machine learning — organised as a comprehensive, navigable compendium rather than a linear tutorial.
It covers the full intellectual landscape of modern AI: mathematical foundations (linear algebra,
probability, information theory, optimisation), classical ML algorithms, deep learning architectures,
natural language processing, computer vision, reinforcement learning, AI safety, and the frontier of
large language models and foundation models.

Each topic entry follows a consistent structure: a precise definition, the mathematical formulation
(where applicable), intuitive explanation, concrete examples, connections to related concepts,
implementation notes, and curated references to key papers and textbooks. This consistency makes
the compendium both readable from cover to cover and efficient as a reference for targeted lookup.

The repository is organised as an interconnected knowledge graph rather than a linear sequence:
each concept links to its prerequisites, its applications, and related concepts — making it possible
to navigate from a surface-level question ('what is a transformer?') through the dependency chain
(attention mechanism → softmax → matrix multiplication) to the mathematical foundations, or to
explore a concept's applications across different domains.

---

## Motivation

The field of AI is advancing faster than any single textbook can track. Course notes become outdated,
blog posts lack rigour, and research papers assume substantial prior knowledge. The AI Bible was
built to fill the gap: a living document that combines textbook rigour with research currency,
maintained as a version-controlled repository that grows with the field.

---

## Architecture

```
The AI Bible — Knowledge Structure
        │
  ┌─────────────────────────────────────────────┐
  │  /01_foundations/                           │
  │  ├── linear_algebra.md                      │
  │  ├── probability_statistics.md              │
  │  ├── information_theory.md                  │
  │  └── optimisation.md                        │
  ├── /02_classical_ml/                         │
  │  ├── supervised_learning.md                 │
  │  ├── unsupervised_learning.md               │
  │  └── ensemble_methods.md                    │
  ├── /03_deep_learning/                        │
  ├── /04_nlp_llm/                              │
  ├── /05_computer_vision/                      │
  ├── /06_reinforcement_learning/               │
  └── /07_ai_safety_alignment/                 │
  └─────────────────────────────────────────────┘
```

---

## Features

### Mathematical Foundations Module
Rigorous coverage of the mathematics underlying ML: linear algebra (eigendecomposition, SVD), probability (Bayesian inference, conjugate priors), information theory (entropy, KL divergence, mutual information), and optimisation (gradient descent variants, convexity).

### Classical ML Reference
Comprehensive entries for all major classical algorithms: linear/logistic regression, SVMs, decision trees, random forests, k-means, DBSCAN, PCA, ICA — each with derivation, implementation notes, and hyperparameter guidance.

### Deep Learning Architecture Guide
Detailed coverage of neural network architectures: feedforward, CNN, RNN, LSTM, GRU, ResNet, Transformer, BERT, GPT, diffusion models, VAE, GAN — with architectural diagrams and key equations.

### LLM and Foundation Model Section
Up-to-date coverage of large language models: attention mechanisms, tokenisation, RLHF, instruction tuning, in-context learning, chain-of-thought, RAG, tool use, and model evaluation.

### Interconnected Concept Graph
Each concept links bidirectionally to prerequisites, applications, and related topics — enabling navigation of the knowledge graph rather than linear reading.

### Paper Reference Index
Curated index of seminal and recent papers per topic, with one-sentence summaries and citation counts — distinguishing foundational work from recent advances.

### Implementation Notes
Practical implementation guidance per algorithm: common pitfalls, numerical stability considerations, hyperparameter sensitivity, and library recommendations.

### AI Safety and Alignment Chapter
Coverage of AI safety concepts: alignment problem, reward hacking, distributional shift, RLHF, Constitutional AI, interpretability, and current research directions.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **Markdown** | Content format | Human-readable, version-controllable, widely renderable |
| **MathJax / KaTeX** | Mathematical notation | LaTeX equation rendering in GitHub and documentation sites |
| **MkDocs / Docusaurus** | Site generation | Build a searchable, navigable website from Markdown files |
| **GitHub Actions** | CI/CD | Automated link checking, spelling validation, site deployment |

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JavaScript projects)
- A virtual environment manager (`venv`, `conda`, or equivalent)
- API keys as listed in the Configuration section

### Installation

```bash
git clone https://github.com/Devanik21/The-AI-Bible.git
cd The-AI-Bible

# Read locally
# Open any .md file in your preferred Markdown reader

# Build searchable documentation site
pip install mkdocs mkdocs-material
mkdocs serve
# Open http://localhost:8000
```

---

## Usage

```bash
# Browse the knowledge base
ls -la chapters/

# Search for a concept
grep -r 'attention mechanism' . --include='*.md'

# Build documentation site
mkdocs build --site-dir docs/

# Deploy to GitHub Pages
mkdocs gh-deploy
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `mkdocs.yml` | `Site config` | Navigation structure, theme, plugins |
| `MATH_RENDERER` | `mathjax` | Equation rendering: mathjax or katex |
| `SEARCH_ENABLED` | `True` | Full-text search across all entries |

> Copy `.env.example` to `.env` and populate required values before running.

---

## Project Structure

```
The-AI-Bible/
├── README.md
├── requirements.txt
├── app.py
└── ...
```

---

## Roadmap

- [ ] Interactive concept map: visualise the full knowledge graph as a navigable force-directed graph
- [ ] Spaced repetition quiz mode: auto-generated questions from concept definitions for active recall practice
- [ ] Video companion: match each entry to the most relevant YouTube lecture (3Blue1Brown, Andrej Karpathy)
- [ ] Community contributions: pull request workflow for community-submitted entries and corrections
- [ ] LLM-powered search: semantic search across the knowledge base for natural language queries

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push to your branch: `git push origin feature/your-idea`
5. Open a Pull Request with a clear description

Please follow conventional commit messages and add documentation for new features.

---

## Notes

This is a living document maintained by one author — coverage is intentionally comprehensive but not exhaustive. Contributions and corrections via pull request are welcome. Mathematical formulations follow standard notation where possible; deviations are noted.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with curiosity, depth, and care — because good projects deserve good documentation.*

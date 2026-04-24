# 📚 AI Book Explorer — Semantic Book Map

An interactive AI-assisted book exploration system that maps thousands of books into a semantic 2D space using embeddings, dimensionality reduction, clustering, and local LLM-generated cluster labels.

The project creates a frontend-ready semantic map where users can explore books by topic, cluster, and dominant emotional tone.

---

## Project Description

This project builds an **AI Book Explorer** that helps users understand large book collections visually.

Instead of browsing books only by title, author, or category, this tool represents each book as a point in a semantic space. Books with similar descriptions appear closer together, forming meaningful clusters such as:

- Sports Stories
- Home Improvement
- Disney Fairy Tales
- Romantic Fiction
- War & Military
- Sci-Fi & Time
- Inspiring Faith

The system combines:

- Sentence embeddings
- UMAP dimensionality reduction
- HDBSCAN clustering
- Approximate tone labeling
- Local LLM cluster naming with Ollama
- Interactive Plotly visualizations

The goal is to support **human interpretation**, not replace it. The AI helps organize and label large-scale textual data so users can explore patterns more easily.

---

## Current Demo Features

The main interactive demo is:

```text
frontend_data/book_cluster_map_views.html


## Contributors

We want to thank to Armando Beltran for being the advisor of the project:

| Name | GitHub Username |
|---|---|
| Jesus A. Beltran - Advisor | [3eltran23](https://github.com/3eltran23) |


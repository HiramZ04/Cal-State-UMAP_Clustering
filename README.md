# 📚 AI Book Explorer — Semantic Book Map

Interactive semantic map for exploring books by cluster, theme, and dominant tone.

The important frontend-ready files are inside:

```text
frontend_data/
```

---

## Main Files

### `frontend_data/book_cluster_map_views.html`

Main interactive demo and recommended version for the frontend team.

Includes:

- Explore view
- Labeled overview view
- Tone view
- Slider for books per cluster
- Tone filter
- Click-to-highlight cluster
- Side panel with book and cluster information

Open this file directly in the browser.

---

### `frontend_data/book_cluster_map_slider.html`

Simpler interactive demo.

Includes:

- Book map
- Slider for books per cluster
- Hover
- Side panel

This file is useful as a simpler reference, but the main version is:

```text
frontend_data/book_cluster_map_views.html
```

---

### `frontend_data/book_map_points.json`

Point data for the map.

Contains each visible book with fields like:

```text
title
authors
category
publisher
description
tone
cluster_id
x
y
cluster_color
dominant_tone
dominant_tone_color
```

Frontend should use this file to render the book points.

---

### `frontend_data/cluster_summary.json`

Cluster metadata.

Contains information for each cluster, including:

```text
cluster_name
short_label
cluster_description
top_tone
top_category
tone_composition
category_composition
cluster_color
```

Frontend should use this file for side panels, labels, filters, and cluster summaries.

---

## Notebook

### `Final_Notebook.ipynb`

Main notebook used to generate the frontend files:

```text
frontend_data/book_map_points.json
frontend_data/cluster_summary.json
frontend_data/book_cluster_map_views.html
frontend_data/book_cluster_map_slider.html
```

---

## Local Artifacts

Large generated files are stored in:

```text
artifacts/
```

These files are ignored by Git and are not required for the frontend.

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the main demo:

```text
frontend_data/book_cluster_map_views.html
```

---

## Contributors
Special thanks to our Advisor of the project: Armando Beltran

| Name | GitHub Username |
|---|---|
| Jesus A. Beltran - Advisor | [3eltran23](https://github.com/3eltran23) |


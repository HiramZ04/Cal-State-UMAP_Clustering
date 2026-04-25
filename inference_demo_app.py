from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FRONTEND_DIR = ROOT_DIR / "frontend_data"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)




@st.cache_data
def load_data():
    book_embeddings = np.load(ARTIFACTS_DIR / "book_embeddings.npy").astype("float32")

    df_books = pd.read_parquet(
        ARTIFACTS_DIR / "books_with_hdbscan_clusters_umap_tone.parquet"
    )

    df_points = pd.read_json(FRONTEND_DIR / "book_map_points.json")
    df_summary = pd.read_json(FRONTEND_DIR / "cluster_summary.json")

    # ------------------------------------------------------------
    # Ensure tone-view fields exist in df_points
    # ------------------------------------------------------------

    tone_palette = {
        "Dark": "#1F2937",
        "Inspiring": "#2563EB",
        "Informative": "#06B6D4",
        "Adventurous": "#10B981",
        "Mysterious": "#7C3AED",
        "Serious": "#6B7280",
        "Emotional": "#EC4899",
        "Romantic": "#F97316",
        "Uplifting": "#EAB308",
        "Humorous": "#D97706",
        "Whimsical": "#14B8A6",
        "Melancholic": "#475569",
        "Suspenseful": "#DC2626",
        "Reflective": "#8B5CF6",
        "Neutral": "#94A3B8",
    }

    df_points["cluster_id"] = df_points["cluster_id"].astype(int)
    df_summary["cluster_id"] = df_summary["cluster_id"].astype(int)

    df_summary["dominant_tone"] = (
        df_summary["top_tone"]
        .fillna("Unknown")
        .astype(str)
    )

    df_summary["dominant_tone_color"] = (
        df_summary["dominant_tone"]
        .map(tone_palette)
        .fillna("#94A3B8")
    )

    tone_lookup = df_summary[
        [
            "cluster_id",
            "dominant_tone",
            "dominant_tone_color",
            "short_label",
            "cluster_name",
            "cluster_color",
        ]
    ].copy()

    df_points = df_points.drop(
        columns=[
            "dominant_tone",
            "dominant_tone_color",
            "short_label",
            "cluster_name",
        ],
        errors="ignore",
    )

    df_points = df_points.merge(
        tone_lookup[
            [
                "cluster_id",
                "dominant_tone",
                "dominant_tone_color",
                "short_label",
                "cluster_name",
            ]
        ],
        on="cluster_id",
        how="left",
    )

    df_points["dominant_tone"] = df_points["dominant_tone"].fillna("Unknown")
    df_points["dominant_tone_color"] = df_points["dominant_tone_color"].fillna("#94A3B8")
    df_points["short_label"] = df_points["short_label"].fillna(
        "Cluster " + df_points["cluster_id"].astype(str)
    )
    df_points["cluster_name"] = df_points["cluster_name"].fillna(
        "Cluster " + df_points["cluster_id"].astype(str)
    )


    norms = np.linalg.norm(book_embeddings, axis=1)
    embeddings_are_normalized = bool(np.median(norms) > 0.95 and np.median(norms) < 1.05)

    embeddings_norm = book_embeddings / np.clip(
        norms.reshape(-1, 1),
        1e-12,
        None
    )

    return {
        "book_embeddings": book_embeddings,
        "embeddings_norm": embeddings_norm,
        "embeddings_are_normalized": embeddings_are_normalized,
        "df_books": df_books,
        "df_points": df_points,
        "df_summary": df_summary,
    }



def show_tone_legend(df_points, books_per_cluster):
    df_visible = df_points[df_points["cluster_rank"] <= books_per_cluster].copy()

    if "dominant_tone" not in df_visible.columns or "dominant_tone_color" not in df_visible.columns:
        return

    tone_counts = (
        df_visible
        .groupby(["dominant_tone", "dominant_tone_color"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    html_parts = []

    html_parts.append(
        '<div style="'
        'width:100%;'
        'display:flex;'
        'justify-content:center;'
        'align-items:center;'
        'flex-wrap:wrap;'
        'gap:8px;'
        'margin-top:-8px;'
        'margin-bottom:12px;'
        '">'
    )

    html_parts.append(
        '<span style="'
        'font-size:14px;'
        'font-weight:800;'
        'color:#1f3556;'
        'margin-right:8px;'
        'white-space:nowrap;'
        '">'
        'Dominant tone by cluster:'
        '</span>'
    )

    for _, row in tone_counts.iterrows():
        tone = row["dominant_tone"]
        color = row["dominant_tone_color"]
        count = int(row["count"])

        html_parts.append(
            f'<span style="'
            f'display:inline-flex;'
            f'align-items:center;'
            f'gap:6px;'
            f'padding:6px 10px;'
            f'border-radius:999px;'
            f'background:#f1f5f9;'
            f'color:#1f3556;'
            f'font-size:12px;'
            f'font-weight:700;'
            f'white-space:nowrap;'
            f'">'
            f'<span style="'
            f'width:10px;'
            f'height:10px;'
            f'border-radius:999px;'
            f'background:{color};'
            f'display:inline-block;'
            f'"></span>'
            f'{tone} · {count:,}'
            f'</span>'
        )

    html_parts.append("</div>")

    chips_html = "".join(html_parts)

    st.markdown(chips_html, unsafe_allow_html=True)





def clean_value(x, fallback="Unknown"):
    if pd.isna(x):
        return fallback

    x = str(x).strip()

    if x == "" or x.lower() in ["nan", "none", "null", "unknown", "n/a", "na"]:
        return fallback

    return x

def clear_query_state():
    st.session_state.query_prompt = ""
    st.session_state.last_result = None


def get_cluster_summary(df_summary, cluster_id):
    if cluster_id is None:
        return None

    row = df_summary[df_summary["cluster_id"].astype(int) == int(cluster_id)]

    if row.empty:
        return None

    return row.iloc[0].to_dict()


def recommend_books(prompt, top_k, model, data):
    df_books = data["df_books"]
    embeddings_norm = data["embeddings_norm"]

    query_norm = model.encode(
        [prompt],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    similarities = embeddings_norm @ query_norm[0]

    # Solo recomendar libros que sí pertenecen a un cluster válido
    valid_candidate_mask = (
        df_books["cluster_id"].notna()
        & (df_books["cluster_id"] != -1)
        & df_books["umap_x"].notna()
        & df_books["umap_y"].notna()
    )

    candidate_indices = np.where(valid_candidate_mask.to_numpy())[0]

    top_candidate_order = np.argsort(similarities[candidate_indices])[::-1]
    top_indices = candidate_indices[top_candidate_order[:top_k]]

    recommendations = df_books.iloc[top_indices].copy()
    recommendations["similarity"] = similarities[top_indices]

    cluster_scores = (
        recommendations
        .groupby("cluster_id")["similarity"]
        .sum()
        .sort_values(ascending=False)
    )

    predicted_cluster = int(cluster_scores.index[0])

    # Colocar el prompt cerca de sus libros recomendados
    weights = recommendations["similarity"].to_numpy()
    weights = weights - weights.min() + 1e-6

    query_x = np.average(recommendations["umap_x"].to_numpy(), weights=weights)
    query_y = np.average(recommendations["umap_y"].to_numpy(), weights=weights)

    query_point = {
        "x": float(query_x),
        "y": float(query_y),
        "prompt": prompt,
        "predicted_cluster": predicted_cluster,
    }

    return recommendations.reset_index(drop=True), query_point, cluster_scores


def build_map(df_points, recommendations=None, query_point=None, predicted_cluster=None, books_per_cluster=150, color_mode="Semantic cluster"):
    df_visible = df_points[df_points["cluster_rank"] <= books_per_cluster].copy()

    if color_mode == "Dominant tone":
        base_colors = df_visible["dominant_tone_color"].fillna("#94A3B8").tolist()
    else:
        base_colors = df_visible["cluster_color"].fillna("#94A3B8").tolist()

    colors = []
    sizes = []

    for color, cluster_id in zip(base_colors, df_visible["cluster_id"]):
        if color_mode == "Labeled overview":
            base_size = 3.3
        else:
            base_size = 4.5

        if predicted_cluster is None:
            colors.append(color)
            sizes.append(base_size)
        elif int(cluster_id) == int(predicted_cluster):
            colors.append(color)
            sizes.append(base_size + 1.4)
        else:
            colors.append("rgba(120, 132, 150, 0.42)")
            sizes.append(base_size - 0.2)

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=df_visible["x"],
            y=df_visible["y"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.42 if color_mode == "Labeled overview" else 0.88,
                line=dict(width=0),
            ),
            customdata=np.stack(
                [
                    df_visible["title"].fillna("Untitled").astype(str),
                    df_visible["category"].fillna("Unknown category").astype(str),
                    df_visible["tone"].fillna("Unknown tone").astype(str),
                    df_visible["cluster_id"].astype(str),
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Category: %{customdata[1]}<br>"
                "Book tone: %{customdata[2]}<br>"
                "Cluster: %{customdata[3]}"
                "<extra></extra>"
            ),
            name="Books",
        )
    )

    if recommendations is not None and len(recommendations) > 0:
        recs_on_map = recommendations[
            recommendations["umap_x"].notna()
            & recommendations["umap_y"].notna()
        ].head(10)

        fig.add_trace(
            go.Scatter(
                x=recs_on_map["umap_x"],
                y=recs_on_map["umap_y"],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=10,
                    color="#111827",
                    line=dict(width=1, color="white"),
                ),
                text=recs_on_map["title_clean"].fillna("Recommended book"),
                hovertemplate="<b>%{text}</b><br>Recommended book<extra></extra>",
                name="Top recommendations",
            )
        )

    if query_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[query_point["x"]],
                y=[query_point["y"]],
                mode="markers+text",
                marker=dict(
                    symbol="star",
                    size=22,
                    color="#FACC15",
                    line=dict(width=2, color="#111827"),
                ),
                text=["Your Query"],
                textposition="top center",
                hovertemplate=(
                    "<b>Your Query</b><br>"
                    f"{query_point['prompt'][:160]}"
                    "<extra></extra>"
                ),
                name="Your Query",
            )
        )

        # ------------------------------------------------------------
    # Labeled overview annotations
    # ------------------------------------------------------------

    if color_mode == "Labeled overview":
        max_labels = 12

        label_df = (
            df_visible
            .groupby("cluster_id")
            .agg(
                x=("x", "median"),
                y=("y", "median"),
                count=("title", "count"),
                short_label=("short_label", "first"),
                cluster_color=("cluster_color", "first"),
            )
            .reset_index()
            .sort_values("count", ascending=False)
            .head(max_labels)
        )

        x_min, x_max = df_visible["x"].min(), df_visible["x"].max()
        y_min, y_max = df_visible["y"].min(), df_visible["y"].max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        annotations = []

        for idx, row in label_df.reset_index(drop=True).iterrows():
            dx = row["x"] - center_x
            dy = row["y"] - center_y

            norm = np.sqrt(dx**2 + dy**2)

            if norm == 0:
                angle = (2 * np.pi * idx) / max_labels
                ux = np.cos(angle)
                uy = np.sin(angle)
            else:
                ux = dx / norm
                uy = dy / norm

            # Offset local: empuja el label cerca del cluster,
            # no hasta las esquinas del plot.
            label_x = row["x"] + ux * x_range * 0.105
            label_y = row["y"] + uy * y_range * 0.105

            # Pequeño stagger para evitar que labels cercanos caigan igual
            label_y += ((idx % 3) - 1) * y_range * 0.018

            annotations.append(
                dict(
                    x=row["x"],
                    y=row["y"],
                    ax=label_x,
                    ay=label_y,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    text=f"<b>{row['short_label']}</b>",
                    showarrow=True,
                    arrowhead=0,
                    arrowwidth=1,
                    arrowcolor="rgba(60, 70, 90, 0.55)",
                    font=dict(
                        size=13,
                        color=row["cluster_color"],
                    ),
                    bgcolor="rgba(255,255,255,0.80)",
                    bordercolor="rgba(255,255,255,0)",
                    borderpad=2,
                )
            )

        fig.update_layout(annotations=annotations)

    fig.update_layout(
        height=780,
        template="plotly_white",
        margin=dict(l=10, r=10, t=25, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
    )

    return fig


st.set_page_config(
    page_title="AI Book Explorer — Query Demo",
    layout="wide"
)

st.title("📚 AI Book Explorer — Local Query Demo")

st.caption(
    "Type a prompt, embed it locally, retrieve similar books, infer the nearest semantic cluster, "
    "and place the query on the UMAP book map."
)

data = load_data()
model = load_embedding_model()

df_summary = data["df_summary"]
df_points = data["df_points"]

DEFAULT_PROMPT = (
    "I want a dark fantasy book with adventure, mystery, "
    "and a story that feels emotional and magical."
)

if "query_prompt" not in st.session_state:
    st.session_state.query_prompt = DEFAULT_PROMPT

with st.sidebar:
    st.header("Query Controls")

    prompt = st.text_area(
        "User prompt",
        key="query_prompt",
        height=130,
    )

    top_k = st.slider("Top recommendations", min_value=5, max_value=30, value=15, step=5)
    books_per_cluster = st.slider("Books per cluster on map", min_value=20, max_value=500, value=150, step=10)

    color_mode = st.selectbox(
        "Map view",
        options=[
            "Semantic cluster",
            "Labeled overview",
            "Dominant tone",
        ],
        index=0,
    )


    run_query = st.button("Run query", type="primary")

    st.button(
        "Clear query / reset map",
        on_click=clear_query_state
    )

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if run_query:
    prompt = st.session_state.query_prompt

    if prompt.strip() == "":
        st.warning("Write a prompt first.")
        st.session_state.last_result = None
    else:
        with st.spinner("Embedding prompt and retrieving books..."):
            recommendations, query_point, cluster_scores = recommend_books(
                prompt=prompt,
                top_k=top_k,
                model=model,
                data=data,
            )

            st.session_state.last_result = {
                "recommendations": recommendations,
                "query_point": query_point,
                "cluster_scores": cluster_scores,
            }

if st.session_state.last_result is None:
    st.info("Write a prompt in the sidebar and click **Run query**.")

    fig = build_map(
        df_points=df_points,
        recommendations=None,
        query_point=None,
        predicted_cluster=None,
        books_per_cluster=books_per_cluster,
        color_mode=color_mode,
    )

    st.subheader("Semantic Book Map")
    st.plotly_chart(fig, width="stretch")

    if color_mode == "Dominant tone":
        show_tone_legend(df_points, books_per_cluster)

    st.stop()

recommendations = st.session_state.last_result["recommendations"]
query_point = st.session_state.last_result["query_point"]
cluster_scores = st.session_state.last_result["cluster_scores"]

predicted_cluster = query_point["predicted_cluster"]
cluster_info = get_cluster_summary(df_summary, predicted_cluster)

left_col, right_col = st.columns([2.2, 1])

with right_col:
    st.subheader("Predicted Cluster")

    if cluster_info is None:
        st.warning("No non-noise cluster was confidently assigned.")
    else:
        st.markdown(f"### {cluster_info.get('cluster_name', f'Cluster {predicted_cluster}')}")
        st.write(cluster_info.get("cluster_description", ""))

        st.markdown(f"**Cluster ID:** `{predicted_cluster}`")
        st.markdown(f"**Top tone:** {cluster_info.get('top_tone', 'Unknown')}")
        st.markdown(f"**Top category:** {cluster_info.get('top_category', 'Unknown')}")

        with st.expander("Tone composition"):
            st.json(cluster_info.get("tone_composition", []))

        with st.expander("Category composition"):
            st.json(cluster_info.get("category_composition", []))

    st.subheader("Top Recommendations")

    for i, row in recommendations.head(top_k).iterrows():
        title = clean_value(row.get("title_clean"), "Untitled")
        authors = clean_value(row.get("authors_clean"), "Unknown author")
        category = clean_value(row.get("category_clean"), "Unknown category")
        tone = clean_value(row.get("tone_label"), "Unknown tone")
        cluster_id = clean_value(row.get("cluster_id"), "Unknown cluster")
        similarity = float(row.get("similarity", 0.0))

        description = clean_value(
            row.get("description_original_clean"),
            clean_value(row.get("description_for_embedding"), "No description available.")
        )

        with st.expander(f"{i + 1}. {title}  ·  {similarity:.3f}"):
            st.markdown(f"**Authors:** {authors}")
            st.markdown(f"**Category:** {category}")
            st.markdown(f"**Tone:** {tone}")
            st.markdown(f"**Cluster:** `{cluster_id}`")
            st.markdown("**Description:**")
            st.write(description)

with left_col:
    st.subheader("Query Position on Semantic Map")

    fig = build_map(
        df_points=df_points,
        recommendations=recommendations,
        query_point=query_point,
        predicted_cluster=predicted_cluster,
        books_per_cluster=books_per_cluster,
        color_mode=color_mode,
    )

    st.plotly_chart(fig, use_container_width=True)
    if color_mode == "Dominant tone":
        show_tone_legend(df_points, books_per_cluster)
# app.py — Streamlit web UI for the News Classifier.
# Run with: streamlit run app.py
# Loads the fine-tuned model once at startup (cached) and predicts on demand.

import os
import streamlit as st
import pandas as pd
import plotly.express as px

# ------------------------------------------------------------------ #
# Page configuration — must be the first Streamlit call
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="News Classifier",
    page_icon="📰",
    layout="centered",
)

# ------------------------------------------------------------------ #
# Model loading — @st.cache_resource keeps the model in memory across
# reruns so we don't reload weights on every button click.
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner="Loading model…")
def load_classifier():
    """Load the fine-tuned NewsClassifier; raises if ./model doesn't exist."""
    # Import here so the rest of the UI can render even if predict.py has a
    # syntax error — gives a cleaner error message to the user.
    from predict import NewsClassifier
    return NewsClassifier()


# ------------------------------------------------------------------ #
# Header
# ------------------------------------------------------------------ #
st.title("📰 News Article Classifier")
st.markdown(
    "Paste a news headline or article excerpt below, then click **Classify**. "
    "The model will identify which of the four AG News categories it belongs to."
)

# Category descriptions help users understand what each label covers
st.markdown(
    """
| Category | What it covers |
|---|---|
| 🌍 World | International politics, diplomacy, conflicts |
| 🏆 Sports | Games, athletes, tournaments, scores |
| 💼 Business | Markets, economy, companies, finance |
| 💻 Sci/Tech | Science, technology, research, gadgets |
"""
)

st.divider()

# ------------------------------------------------------------------ #
# Input area
# ------------------------------------------------------------------ #
example_texts = {
    "— pick an example —": "",
    "🌍 World": (
        "United Nations Security Council holds emergency session as tensions escalate "
        "on the border between two neighbouring nations, calling for an immediate ceasefire."
    ),
    "🏆 Sports": (
        "The Golden State Warriors defeated the Boston Celtics 112-98 in Game 5 of the "
        "NBA Finals, with Stephen Curry scoring 38 points to seal the championship."
    ),
    "💼 Business": (
        "Federal Reserve raises interest rates by 25 basis points amid ongoing inflation "
        "concerns, signalling further hikes may come later this year to cool the economy."
    ),
    "💻 Sci/Tech": (
        "NASA's James Webb Space Telescope has captured stunning new images of distant "
        "galaxies, revealing unprecedented detail about the early universe and dark matter."
    ),
}

# Quick-fill selector — choosing an example overwrites the text area
selected_example = st.selectbox("Try an example:", list(example_texts.keys()))

user_input = st.text_area(
    "News article text",
    value=example_texts[selected_example],
    height=160,
    placeholder="Paste a headline or short article excerpt here…",
    label_visibility="collapsed",
)

# Character counter — helpful feedback for very short or very long inputs
char_count = len(user_input.strip())
st.caption(f"{char_count} characters")

classify_btn = st.button("🔍 Classify", type="primary", disabled=(char_count < 10))

# ------------------------------------------------------------------ #
# Prediction
# ------------------------------------------------------------------ #
if classify_btn:
    # Try to load the model; surface a friendly error if it isn't trained yet
    try:
        classifier = load_classifier()
    except FileNotFoundError as e:
        st.error(
            f"**Model not found.** Run `python train.py` first to train and save the model.\n\n"
            f"Details: {e}"
        )
        st.stop()

    with st.spinner("Classifying…"):
        result = classifier.predict(user_input)

    # ------------------------------------------------------------------ #
    # Result display
    # ------------------------------------------------------------------ #
    label = result["predicted_label"]
    confidence = result["confidence"]

    # Emoji map for visual flair
    emoji = {"World": "🌍", "Sports": "🏆", "Business": "💼", "Sci/Tech": "💻"}.get(label, "📰")

    st.success(f"### {emoji} {label}")
    st.metric("Confidence", f"{confidence:.1%}")

    st.divider()

    # ------------------------------------------------------------------ #
    # Confidence bar chart using Plotly for interactivity
    # ------------------------------------------------------------------ #
    scores = result["confidence_scores"]
    df = pd.DataFrame(
        {"Category": list(scores.keys()), "Confidence": list(scores.values())}
    ).sort_values("Confidence", ascending=True)  # ascending so highest bar is on top

    # Highlight the predicted class with a distinct colour
    colour_map = {cat: ("#1f77b4" if cat != label else "#ff7f0e") for cat in df["Category"]}

    fig = px.bar(
        df,
        x="Confidence",
        y="Category",
        orientation="h",
        color="Category",
        color_discrete_map=colour_map,
        text=df["Confidence"].map(lambda v: f"{v:.1%}"),
        title="Confidence Scores — All Categories",
    )
    fig.update_layout(
        showlegend=False,
        xaxis_tickformat=".0%",
        xaxis_range=[0, 1],
        height=280,
        margin=dict(l=0, r=20, t=40, b=0),
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------ #
    # Raw scores table (collapsed by default to keep the UI clean)
    # ------------------------------------------------------------------ #
    with st.expander("Raw probability scores"):
        raw_df = pd.DataFrame(
            {"Category": list(scores.keys()), "Probability": [f"{v:.4f}" for v in scores.values()]}
        )
        st.table(raw_df.set_index("Category"))

# ------------------------------------------------------------------ #
# Footer
# ------------------------------------------------------------------ #
st.divider()
st.caption(
    "Model: DistilBERT fine-tuned on AG News · "
    "Test accuracy: ~91% · "
    "Built with HuggingFace Transformers + Streamlit"
)

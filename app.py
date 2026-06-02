import streamlit as st
import torch
import torch.nn as nn
import pandas as pd

from transformers import BertTokenizer, BertModel
from preprocessing import clean_text, tokenize_text

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="Multi-Label News Classification", layout="wide")

st.title("📰 Multi-Label News Article Classification Using BERT")
st.write("Fusion of Headline, Summary, and Body for Category Prediction")

# ---------------------------
# Load BERT (cached for speed)
# ---------------------------
@st.cache_resource
def load_models():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, bert_model = load_models()

# ---------------------------
# Labels (Modify if needed)
# ---------------------------
LABELS = ["Business", "Technology", "Politics", "Sports", "Health"]

# ---------------------------
# Demo Classifier Layer
# ---------------------------
class Classifier(nn.Module):
    def __init__(self, hidden_size=768, num_labels=len(LABELS)):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)

        # Stable initialization for demo
        torch.manual_seed(42)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)

classifier = Classifier()
classifier.eval()

# ---------------------------
# Encode Text Using BERT
# ---------------------------
def encode_text(text):
    cleaned = clean_text(text)

    encoding = tokenize_text(cleaned, tokenizer)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)

    return pooled

# ---------------------------
# Sample Dataset Loader
# ---------------------------
if st.button("📂 Load Sample News Article"):
    st.session_state.headline = "AI Transforms Financial Markets"
    st.session_state.summary = "Automation reshapes investment strategies"
    st.session_state.body = (
        "Artificial intelligence is rapidly transforming financial analysis, "
        "risk assessment, and automated trading systems worldwide."
    )

headline = st.text_input("Headline", st.session_state.get("headline", ""))
summary = st.text_input("Summary", st.session_state.get("summary", ""))
body = st.text_area("Body", st.session_state.get("body", ""), height=200)

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Predict Labels"):

    if not headline or not summary or not body:
        st.warning("Please enter all fields before prediction.")
    else:
        h_vec = encode_text(headline)
        s_vec = encode_text(summary)
        b_vec = encode_text(body)

        # Fusion Strategy (Your Project Concept)
        fused_vector = (h_vec + s_vec + b_vec) / 3

        with torch.no_grad():
            probs = classifier(fused_vector).squeeze().tolist()

        # Thresholding for Multi-label Decision
        THRESHOLD = 0.5
        predicted_labels = [
            LABELS[i] for i, p in enumerate(probs) if p >= THRESHOLD
        ]

        if not predicted_labels:
            predicted_labels = ["No strong category detected"]

        # ---------------------------
        # Show Predictions
        # ---------------------------
        st.subheader("✅ Predicted Categories")
        for label in predicted_labels:
            st.success(label)

        # ---------------------------
        # Confidence Table
        # ---------------------------
        results = pd.DataFrame({
            "Category": LABELS,
            "Confidence Score": probs
        }).sort_values(by="Confidence Score", ascending=False)

        st.subheader("📊 Confidence Scores")
        st.dataframe(results, use_container_width=True)

        # ---------------------------
        # Visualization
        # ---------------------------
        st.subheader("📈 Classification Visualization")
        st.bar_chart(results.set_index("Category"))

st.markdown("---")
st.caption("Educational Implementation of Multi-Label Classification using BERT Feature Fusion.")

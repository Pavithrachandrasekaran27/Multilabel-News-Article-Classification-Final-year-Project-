import pandas as pd
from transformers import BertTokenizer
from preprocessing import tokenize_text

# Load tokenizer once
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def load_reuters_dataset():
    """
    Loads a small sample dataset for testing/demo purposes.
    (Simulates Reuters dataset structure: Headline, Summary, Body)
    """

    data = {
        "headline": [
            "Stock market rises",
            "New technology released"
        ],
        "summary": [
            "Investors show confidence",
            "Company launches AI product"
        ],
        "body": [
            "Global markets rallied today with strong economic signals.",
            "The new AI system is expected to transform automation."
        ],
        "label": [
            ["business"],
            ["technology"]
        ]
    }

    df = pd.DataFrame(data)

    # Fuse Headline + Summary + Body (your project fusion idea)
    df["text"] = df["headline"] + " " + df["summary"] + " " + df["body"]

    return df


def prepare_features(texts, max_len=512):
    """
    Converts text into BERT input features.
    """
    input_ids = []
    attention_masks = []

    for text in texts:
        encoding = tokenize_text(text, tokenizer, max_len)

        input_ids.append(encoding["input_ids"])
        attention_masks.append(encoding["attention_mask"])

    return input_ids, attention_masks

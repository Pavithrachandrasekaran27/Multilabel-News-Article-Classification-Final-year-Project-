import torch
import torch.nn as nn
from transformers import BertModel

class BERTSegmentClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_labels=10):  # Adjust num_labels
        super(BERTSegmentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fusion_layer = nn.Linear(hidden_size * 3, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, headline_ids, headline_mask,
                summary_ids, summary_mask,
                body_ids, body_mask):

        head_emb = self.bert(input_ids=headline_ids, attention_mask=headline_mask)[1]
        sum_emb = self.bert(input_ids=summary_ids, attention_mask=summary_mask)[1]
        body_emb = self.bert(input_ids=body_ids, attention_mask=body_mask)[1]

        fused = torch.cat([head_emb, sum_emb, body_emb], dim=1)
        fused = self.fusion_layer(fused)
        logits = self.classifier(fused)
        return self.sigmoid(logits)

# Helper functions for testing
def get_segment_embeddings(text, max_len=128):
    from preprocessing import tokenize_text
    input_ids, attention_mask = tokenize_text(text, max_len)
    model = BertModel.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        emb = model(input_ids, attention_mask)[1]  # CLS token
    return emb

def fuse_embeddings(emb_list):
    return torch.cat(emb_list, dim=1)

def classify_output(fused_emb, num_labels=10):
    classifier = nn.Sequential(
        nn.Linear(fused_emb.shape[1], num_labels),
        nn.Sigmoid()
    )
    return classifier(fused_emb)

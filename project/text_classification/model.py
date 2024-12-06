from transformers import DistilBertModel, DistilBertConfig
import torch.nn as nn

class DistilBertClassifier(nn.Module):
    def __init__(self, num_labels=4):
        super(DistilBertClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # [CLS]
        return logits

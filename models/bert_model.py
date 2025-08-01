from transformers import BertForSequenceClassification, BertTokenizer
import torch
from config import settings # our variables
import os

class BertModel:
    def __init__(self, num_labels=2): 
        """
        num_labels:
            - count of outputs (spam or not)
            - 0 spam / 1 not spam
        BertForSquenceClassification:
            - version customized for classification
        from_pretrained:
            - We use pretrained models
        BertTokenizer:
            - Converts it into a form that BERT can understand.
        device:
            - 'cuda' if you have gpu else 'cpu'
        """
        self.model=None
        self.tokenizer=None
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels=num_labels
    def load_pretrained(self, model_path=None):
        if model_path and os.path.exists(model_path):
            self.model=BertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer=BertTokenizer.from_pretrained(model_path)
        else:
            self.model=BertForSequenceClassification.from_pretrained(
                settings.MODEL_NAME,
                num_labels=self.num_labels
            )
            self.tokenizer=BertTokenizer.from_pretrained(settings.MODEL_NAME)
        self.model.to(self.device)
        return self
    def predict(self, text, threshold=0.5):
        """
        padding: fills in short sentences
        truncation: cuts long sentences
        sigmoid: sets the score to the range [0, 1]
        threshold: If it is above the threshold, it is spam 
        int(): it converts bool to 0 or 1
        logits: it converts variables to tensor
        """
        inputs=self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=settings.MAX_LENGTH,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs=self.model(**inputs)
        # probs=torch.sigmoid(outputs.logits)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).cpu().numpy()
        # return (probs>threshold).int().cpu().numpy()
        print(probs)
        return predicted_class
    def get_model(self):
        return self.model
    def get_tokenizer(self):
        return self.tokenizer
    
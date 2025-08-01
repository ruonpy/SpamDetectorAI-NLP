from torch.utils.data import Dataset
from config import settings
import torch
class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        """
        We use this class for data management.
        """
        self.texts=texts
        self.labels=labels
        self.tokenizer=tokenizer
        self.max_length=settings.MAX_LENGTH
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        text=str(self.texts[index])
        label=int(self.labels[index])
        encoding=self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding["attention_mask"].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
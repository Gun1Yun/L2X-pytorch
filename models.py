import torch
import torch.nn as nn
import torch.nn.functional as F


class OriginalModel(nn.Module):
    def __init__(
        self, vocab_size: int, emb_size: int, filter_size: int, kernel_size: int, hidden_size: int
    ):
        """Original Model to explain
        original model for generate prediction

        Args:
            vocab_size: the number of vocabs
            emb_size: hidden dimensions of embedding layer
            filter_size: filter size of convolution layer
            kernel_size: kerenel size of convolution layer
            hidden_size: hidden size of linear layer
        """
        super(OriginalModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.conv1d = nn.Conv1d(emb_size, filter_size, kernel_size)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        self.linear1 = nn.Linear(filter_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout1(x)
        x = self.conv1d(x.transpose(1, 2))
        x = F.relu(x)
        x = self.max_pooling(x)
        x = self.linear1(x.squeeze(axis=-1))
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.linear2(x)

        return x

import os
import random
import pickle
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

SEED = 1018
torch.manual_seed(SEED)
random.seed(SEED)

## parameter ##
max_features = 5000
maxlen = 400
batch_size = 40
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 5
k = 10  # Number of selected words by L2X.
PART_SIZE = 125
###############


def load_data():
    if "id_to_word_pkl" not in os.listdir("data"):
        print("loading data..")

        train_dataset, test_dataset = datasets.IMDB(root=".data", split=("train", "test"))
        tokenizer = get_tokenizer("basic_english")
        train_iter = datasets.IMDB(split="train")

        def yeild_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(yeild_tokens(train_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])

        print(vocab(["here", "is", "an", "example"]))


## Original model : for explain ##
class OriginalModel(nn.Module):
    def __init__(self):
        """
        original model to explain
        """
        super(OriginalModel, self).__init__()
        self.embedding = nn.Embedding(max_features, embedding_dims)
        self.conv1d = nn.Conv1d(embedding_dims, filters, kernel_size)
        self.activation = nn.ReLU()
        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        self.linear1 = nn.Linear(filters, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, 2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout1(x)
        x = self.conv1d(x.transpose(1, 2))
        x = self.activation(x)
        x = self.max_pooling(x)
        x = self.linear1(x.squeeze(axis=-1))
        x = self.dropout2(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


def generate_original_preds(train=True):
    """ """
    if train:
        # train the original model and save
        model = OriginalModel()


## L2X ##
class Selector(nn.Module):
    """
    Build L2X selector for selecting word
    """

    def __init__(self):
        super(Selector, self).__init__()
        self.tau = 0.1
        self.k = 10  # select constraint
        self.uniform_dist = torch.distributions.uniform.Uniform(
            torch.tensor([0.0]), torch.tensor([1.0])
        )

        # for layer
        self.embeddings = nn.Embedding(max_features, embedding_dims)
        self.dropout = nn.Dropout(0.2)
        self.conv1d = nn.Conv1d(embedding_dims, 100, kernel_size, padding="same")

        # glob
        self.glob_max_pooling = nn.AdaptiveMaxPool1d(1)
        self.glob_linear = nn.Linear(100, 100)

        # local
        self.local_conv1d = nn.Conv1d(100, 100, 3, padding="same")
        self.local_info_conv1d = nn.Conv1d(100, 100, 3, padding="same")

        # final
        self.dropout2 = nn.Dropout(0.2)
        self.final_conv1d = nn.Conv1d(200, 100, 1, padding="same")  # output channel x2
        self.final_conv1d2 = nn.Conv1d(100, 1, 1, padding="same")

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)

        x_base = self.conv1d(x.transpose(1, 2))

        global_x = self.glob_max_pooling(x_base)
        global_x = self.glob_linear(global_x.squeeze(axis=-1))

        local_x = self.local_conv1d(x_base)
        local_x = self.local_info_conv1d(local_x)
        local_x = local_x.transpose(1, 2)

        # combine
        # expand global
        global_x = global_x.unsqueeze(axis=-2)
        global_x = global_x.tile((1, local_x.shape[1], 1))
        combined = torch.cat((global_x, local_x), dim=-1)

        # output
        output = self.dropout2(combined)
        output = self.final_conv1d(output.transpose(1, 2))
        output = self.final_conv1d2(output)

        # model.train()
        if self.training:
            # sampling with geumbel softmax
            samples = self._gumbel_softmax(output)
            return samples
        # model.eval()
        else:
            # make threshold for top k and make discrete
            threshold = torch.topk(output, self.k, sorted=True)[0][:, -1].unsqueeze(-1)
            discreate_output = torch.tensor(
                torch.greater_equal(output, threshold), dtype=torch.float32
            )
            return discreate_output

    # make this to tensor
    def _gumbel_softmax(self, logits):
        # logits : batch, 1, len(x)
        dim = logits.shape[-1]
        uniform_shape = torch.Size([logits.shape[0], self.k, dim])
        uniform_sampled = self.uniform_dist.sample(torch.Size(uniform_shape)).squeeze(dim=-1)
        gumbel = -torch.log(-torch.log(uniform_sampled))
        noisy_dist = (gumbel + logits) / self.tau

        samples = F.softmax(noisy_dist, dim=-1)
        samples, _ = samples.max(dim=1)

        return samples.unsqueeze(-1)


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

    def forward(self):
        pass


def main():
    test_data = torch.randint(1, 100, (2, 55))
    # test for OriginalModel()
    test_model = OriginalModel()
    result = test_model(test_data)
    # print(result)

    # test for Selector
    test_model = Selector()
    test_model.eval()
    result = test_model(test_data)


if __name__ == "__main__":
    main()

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


class Selector(nn.Module):
    """
    Build L2X selector for selecting word
    """

    def __init__(self, vocab_size: int, emb_size: int, kernel_size: int):
        super(Selector, self).__init__()
        self.tau = 0.1
        self.k = 10  # select constraint
        self.uniform_dist = torch.distributions.uniform.Uniform(
            torch.tensor([0.0]), torch.tensor([1.0])
        )

        # for layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.dropout = nn.Dropout(0.2)
        self.conv1d = nn.Conv1d(emb_size, 100, kernel_size, padding="same")

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
        uniform_sampled = (
            self.uniform_dist.sample(torch.Size(uniform_shape))
            .squeeze(dim=-1)
            .to(torch.device(logits.device))
        )
        gumbel = -torch.log(-torch.log(uniform_sampled))
        noisy_dist = (gumbel + logits) / self.tau

        samples = F.softmax(noisy_dist, dim=-1)
        samples, _ = samples.max(dim=1)

        return samples.unsqueeze(-1)


class Predictor(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int, k: int):
        super(Predictor, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.linear1 = nn.Linear(emb_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)
        self.k = k

    def forward(self, x, logits):
        x = self.embeddings(x)
        x = torch.multiply(x, logits)
        x = (torch.sum(x, dim=1) / self.k).squeeze(1)
        x = self.linear1(x)
        x = F.relu(x)

        return self.linear2(x)


class L2XModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        kernel_size: int,
        hidden_size: int,
        k: int,
    ):
        super(L2XModel, self).__init__()
        self.selector = Selector(vocab_size, emb_size, kernel_size)
        self.predictor = Predictor(vocab_size, emb_size, hidden_size, k)

    def forward(self, x):
        x_s = self.selector(x)
        q_x = self.predictor(x, x_s)

        return q_x

import os
import time
import random
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.functional import truncate
from torchtext.vocab import build_vocab_from_iterator

from tqdm.auto import tqdm

from models import OriginalModel, L2XModel
from utils import create_dataset_from_score

## Argument parser ##
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices=["original", "L2X"], default="original")
parser.add_argument("--train", action="store_true")
parser.set_defaults(train=False)

## Set seed ##
SEED = 1018
random.seed(SEED)
torch.manual_seed(SEED)

## Hyperparameter ##
max_len = 400
vocab_size = 5000
emb_size = 50
batch_size = 40
filter_size = 250
kernel_size = 3
hidden_size = 250
n_epochs = 5
learning_rate = 1e-2
k = 10

## MODEL PATH ##
OG_MODEL_PATH = "./ckpt/original.pt"
L2X_MODEL_PATH = "./ckpt/l2x.pt"
OG_DATA_PATH = "./data/"
VOCAB_PATH = "./data/"


class ImdbDataset(Dataset):
    def __init__(self, x, y):
        super(ImdbDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x_tensor = torch.tensor(self.x[index])
        y_tensor = torch.tensor(self.y[index])
        y_tensor = F.softmax(y_tensor, dim=-1)

        return x_tensor, y_tensor

    def __len__(self):
        return len(self.x)


def load_data(vocab_size, tokenizer):
    """
    Args:
        vocab_size: max size of vocabulary
    """
    train_dataset, test_dataset = datasets.IMDB(root=".data", split=("train", "test"))
    train_iter = datasets.IMDB(split="train")

    def yeild_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(
        yeild_tokens(train_iter), specials=["<unk>"], max_tokens=vocab_size
    )
    vocab.set_default_index(vocab["<unk>"])

    vocab_path = os.path.join(VOCAB_PATH, "vocab.pkl")
    if os.path.isfile(vocab_path):
        with open(vocab_path, "rb") as f_vocab:
            vocab = pickle.load(f_vocab)
    else:
        with open(vocab_path, "wb") as f_vocab:
            pickle.dump(vocab, f_vocab)

    return train_dataset, test_dataset, vocab


def text_pipeline(x):
    return vocab(tokenizer(x))


def label_pipeline(x):
    return 0 if x == "neg" else 1


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(
            truncate(text_pipeline(_text), max_seq_len=max_len), dtype=torch.int64
        )
        processed_text = nn.ConstantPad1d((0, max_len - processed_text.size(0)), 0)(processed_text)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

    text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
    return label_list, text_list, offsets


def generate_original_prediction(train):
    original_model = OriginalModel(
        vocab_size, max_len, emb_size, filter_size, kernel_size, hidden_size
    )
    original_model.to(device)

    if train:
        train_loader = DataLoader(
            train_datasets, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
        )
        val_loader = DataLoader(
            val_datasets, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(original_model.parameters(), lr=learning_rate)
        # train
        original_model.train()
        for epoch in range(1, 1 + n_epochs):
            print(f"Epoch [{epoch}/{n_epochs}] : ", end="")
            total_acc, total_count, total_loss = 0, 0, 0
            steps = 0
            for _, (label, text, offsets) in tqdm(enumerate(train_loader)):
                optimizer.zero_grad()
                label, text, offsets = label.to(device), text.to(device), offsets.to(device)
                predicted_label = original_model(text)
                loss = criterion(predicted_label, label)
                loss.backward()
                optimizer.step()
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
                total_loss += loss.item()
                steps += 1
            print(f"Acc : {total_acc/total_count} Loss : {total_loss/steps}")

        torch.save(original_model.state_dict(), OG_MODEL_PATH)
        print(f"Train OG model is finished. saved OGmodel at {OG_MODEL_PATH}.")

    original_model.load_state_dict(torch.load(OG_MODEL_PATH), strict=False)
    original_model.eval()

    with torch.no_grad():
        train_preds, val_preds = [], []
        x_train, y_train, x_val, y_val = [], [], [], []

        train_loader = DataLoader(
            train_datasets, batch_size=1000, shuffle=False, collate_fn=collate_batch
        )
        val_loader = DataLoader(
            val_datasets, batch_size=1000, shuffle=False, collate_fn=collate_batch
        )

        # predicts train
        for _, (label, text, offsets) in tqdm(enumerate(train_loader)):
            label, text, offsets = label.to(device), text.to(device), offsets.to(device)
            pred = original_model(text)
            train_preds.append(pred)
            x_train.append(text)
            y_train.append(label)

        # predicts validation
        for _, (label, text, offsets) in tqdm(enumerate(val_loader)):
            label, text, offsets = label.to(device), text.to(device), offsets.to(device)
            pred = original_model(text)
            val_preds.append(pred)
            x_val.append(text)
            y_val.append(label)

        train_preds = torch.cat(train_preds)
        val_preds = torch.cat(val_preds)

        x_train = torch.cat(x_train)
        y_train = torch.cat(y_train)
        x_val = torch.cat(x_val)
        y_val = torch.cat(y_val)

        train_preds = train_preds.detach().cpu().numpy().tolist()
        val_preds = val_preds.detach().cpu().numpy().tolist()

        x_train = x_train.detach().cpu().numpy().tolist()
        y_train = y_train.detach().cpu().numpy().tolist()
        x_val = x_val.detach().cpu().numpy().tolist()
        y_val = y_val.detach().cpu().numpy().tolist()

        train_path = os.path.join(OG_DATA_PATH, "train_preds.pkl")
        val_path = os.path.join(OG_DATA_PATH, "val_preds.pkl")

        x_train_path = os.path.join(OG_DATA_PATH, "x_train.pkl")
        y_train_path = os.path.join(OG_DATA_PATH, "y_train.pkl")
        x_val_path = os.path.join(OG_DATA_PATH, "x_val.pkl")
        y_val_path = os.path.join(OG_DATA_PATH, "y_val.pkl")

        with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
            pickle.dump(train_preds, f_train)
            pickle.dump(val_preds, f_val)

        with open(x_train_path, "wb") as f_x_train, open(y_train_path, "wb") as f_y_train, open(
            x_val_path, "wb"
        ) as f_x_val, open(y_val_path, "wb") as f_y_val:
            pickle.dump(x_train, f_x_train)
            pickle.dump(y_train, f_y_train)
            pickle.dump(x_val, f_x_val)
            pickle.dump(y_val, f_y_val)

        print(f"Generate OG prediction is finished. Saved predictions at {OG_DATA_PATH}")


def L2X(train=True):
    # for validation
    train_loader = DataLoader(
        train_datasets, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_datasets, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    train_path = os.path.join(OG_DATA_PATH, "train_preds.pkl")
    val_path = os.path.join(OG_DATA_PATH, "val_preds.pkl")

    x_train_path = os.path.join(OG_DATA_PATH, "x_train.pkl")
    y_train_path = os.path.join(OG_DATA_PATH, "y_train.pkl")
    x_val_path = os.path.join(OG_DATA_PATH, "x_val.pkl")
    y_val_path = os.path.join(OG_DATA_PATH, "y_val.pkl")

    with open(train_path, "rb") as f_train, open(val_path, "rb") as f_val:
        train_preds = pickle.load(f_train)
        val_preds = pickle.load(f_val)

    with open(x_train_path, "rb") as f_x_train, open(y_train_path, "rb") as f_y_train, open(
        x_val_path, "rb"
    ) as f_x_val, open(y_val_path, "rb") as f_y_val:
        x_train = pickle.load(f_x_train)
        y_train = pickle.load(f_y_train)
        x_val = pickle.load(f_x_val)
        y_val = pickle.load(f_y_val)

    l2x_model = L2XModel(vocab_size, emb_size, kernel_size, hidden_size, k)
    l2x_model.to(device)

    if train:

        l2x_train_dataset = ImdbDataset(x_train, train_preds)
        l2x_val_dataset = ImdbDataset(x_val, val_preds)
        l2x_train_loader = DataLoader(l2x_train_dataset, batch_size=batch_size, shuffle=True)
        l2x_val_loader = DataLoader(l2x_val_dataset, batch_size=batch_size, shuffle=True)

        l2x_model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(l2x_model.parameters(), lr=learning_rate)

        for epoch in range(1, 1 + n_epochs):
            print(f"Epoch [{epoch}/{n_epochs}] : ", end="")
            total_acc, total_count, total_loss = 0, 0, 0
            for _, (text, label) in tqdm(enumerate(l2x_train_loader)):
                optimizer.zero_grad()
                text, label = text.to(device), label.to(device)
                pred = l2x_model(text)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()
                total_acc += (pred.argmax(1) == label.argmax(1)).sum().item()
                total_count += label.size(0)
                total_loss += loss.item()
            print(f"Acc : {total_acc/total_count}, Loss : {total_loss/len(l2x_train_loader)}")

        torch.save(l2x_model.state_dict(), L2X_MODEL_PATH)
        print(f"Train L2X model is finished. saved L2X Model at {L2X_MODEL_PATH}.")

    l2x_model.load_state_dict(torch.load(L2X_MODEL_PATH), strict=False)
    l2x_model.eval()

    val_loader = DataLoader(val_datasets, batch_size=1000, shuffle=False, collate_fn=collate_batch)

    with torch.no_grad():
        x_val, scores = [], []
        for _, (label, text, offsets) in tqdm(enumerate(val_loader)):
            label, text = label.to(device), text.to(device)
            pred = l2x_model.selector(text)
            score = l2x_model.selector.logits
            x_val.append(text)
            scores.append(score)

        x_val = torch.cat(x_val)
        scores = torch.cat(scores)
        scores = scores.reshape(scores.shape[0], -1)

    x_val = x_val.detach().cpu().numpy().tolist()
    scores = scores.detach().cpu().numpy().tolist()

    return x_val, scores


if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device("cuda:3")
    tokenizer = get_tokenizer("basic_english")
    train_datasets, val_datasets, vocab = load_data(vocab_size, tokenizer)

    if args.task == "original":
        generate_original_prediction(args.train)

    elif args.task == "L2X":
        x_val, scores = L2X(args.train)
        create_dataset_from_score(x_val, scores, vocab, k, max_len)

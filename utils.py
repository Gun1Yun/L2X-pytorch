import pickle
import numpy as np

from tqdm import tqdm


def get_selected_words(x_single, score, k, max_len):
    selected = np.argsort(score)[-k:]
    selected_k_hot = np.zeros(max_len)
    selected_k_hot[selected] = 1.0

    x_seleted = (x_single * selected_k_hot).astype(int)
    return x_seleted


def create_dataset_from_score(x, scores, vocab, k, max_len):
    """
    Args:
        x: original text dataset
        scores: score of each feature
        k: number of selection
        max_len: max length of sentence
    """
    new_data = []
    new_texts = []

    for i, x_single in tqdm(enumerate(x)):
        x_selected = get_selected_words(x_single, scores[i], k, max_len)
        selected_token = x_selected[x_selected != 0].tolist()
        text = vocab.lookup_tokens(selected_token)

        new_data.append(x_selected)
        new_texts.append(text)

    with open("./data/x_val_l2x.pkl", "wb") as fw:
        pickle.dump(new_data, fw)

    with open("./data/x_val_l2x_texts.pkl", "wb") as fw:
        pickle.dump(new_texts, fw)

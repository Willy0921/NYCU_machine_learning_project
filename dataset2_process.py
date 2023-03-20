import numpy as np
import pandas as pd
import openpyxl
import nltk
from nltk import tokenize, pos_tag
from nltk.corpus import wordnet
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-m", "--mode", help="Data mode", type=str, default=None)
args = parser.parse_args()


def filter_useful_words(X_raw):
    allowed_word_types = ["NN", "VB", "JJ", "RB"]
    words_X = []
    for sentence in X_raw:
        words = [word for word, tag in pos_tag(tokenize.word_tokenize(str(sentence))) if tag in allowed_word_types and word not in ['[', ']']]
        if words:
            words_X.append(words)
    print("Turning into words done!")
    return words_X


def calculate_synsets(words_X):

    synsets_X = [[wordnet.synsets(word)[0] for word in words if wordnet.synsets(word)] for words in words_X]
    print("Turning into synsets done!")
    return synsets_X


def calculate_similarity(synsets_X, feature_synsets):

    similarity_X = []
    for synsets in synsets_X:
        similarities = []
        for feature_synset in feature_synsets:
            scores = [synset.wup_similarity(feature_synset) if not np.isnan(synset.wup_similarity(feature_synset)) else 0 for synset in synsets]
            if scores:
                similarities.append(np.nanmean(scores))
        if similarities:
            similarity_X.append(similarities)
    print("Calculating similarity done!")
    return similarity_X


def preprocess_dataset(X_raw, feature_words):
    print("Start preprocessing, please wait...")
    feature_synsets = [wordnet.synsets(word)[0] for word in feature_words if wordnet.synsets(word)[0]]

    words_X = filter_useful_words(X_raw)

    synsets_X = calculate_synsets(words_X)

    similarity_X = calculate_similarity(synsets_X, feature_synsets)
    # print(similarity_X)
    return similarity_X


def load_dataset():
    print("Load raw dataset...")
    X_data = pd.read_excel(f"./dataset2/X_{args.mode}.xlsx").to_numpy()
    return X_data


def save_processed_dataset(X_data):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "dataset2"
    title = ["f1", "f2", "f3", "f4"]
    for i in range(len(title)):
        ws.cell(row=1, column=i + 1).value = title[i]
    for instance in X_data:
        ws.append(instance)

    wb.save(filename=f"./dataset2/X_{args.mode}_processed.xlsx")
    print("Saving processed dataset done!")


if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    X_raw = load_dataset()

    feature_words = np.array(["perfective", "good", "bad", "atrocious"])

    X_data = preprocess_dataset(X_raw, feature_words)

    save_processed_dataset(X_data)

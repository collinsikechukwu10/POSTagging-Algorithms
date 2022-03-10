import json
from collections import defaultdict
from typing import List, Type, Dict, AnyStr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os
from dataset import SentenceDataset, TreeBankDataset, START_TAG, END_TAG
from probability import create_transition_probability_matrix, create_emission_probability_matrix
from tagger import POSTagger
from utils import get_word, get_pos_tag, tokenize_text

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "outputs")


class JSONSerializable:
    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class POSTagConfusionMatrix(JSONSerializable):
    def __init__(self, tags):
        self._keys = tags
        self._table = defaultdict(dict)
        for tag in tags:
            self._table[tag] = dict((t, 0) for t in tags)

    def increment(self, key1, key2):
        self._table[key1][key2] += 1

    def get(self, key1, key2):
        return self._table[key1][key2]

    def get_keys(self) -> set:
        keys = self._keys
        if START_TAG in keys:
            keys.remove(START_TAG)
        if END_TAG in keys:
            keys.remove(END_TAG)
        return keys

    def to_dataframe(self):
        return pd.DataFrame(self._table)


class Performance(JSONSerializable):
    def __init__(self, dataset: SentenceDataset, cm: POSTagConfusionMatrix):
        self._p = dict()
        self._cm = cm
        keys = cm.get_keys()
        # remove start and end tag as they would cause zero division error
        count = 0
        for key in cm.get_keys():
            key_count = cm.get(key, key)
            self._p[key] = key_count / dataset.get_pos_tag_freq_dist()[key]
            count += key_count
        self._p["total_performance"] = count / dataset.get_pos_tag_freq_dist().N()

    def get_cm(self):
        return self._cm

    def get_performance(self):
        return self._p


def plot_performances(dict_of_performances):
    print("Generating plots...")
    folder_name = dt.datetime.now()
    folder_path = os.path.join(OUTPUT_PATH, str(folder_name))
    os.mkdir(folder_path)
    for key, performance in dict_of_performances.items():
        plot_confusion_matrix(performance.get_cm(), os.path.join(folder_path, f"confusion_matrix_{key}.png"))

    # create combined bar chart showing performance
    df = pd.DataFrame({k: v.get_performance() for k, v in dict_of_performances.items()})
    fig0 = plt.figure(figsize=(20, 20))
    ax0 = fig0.add_subplot(111)
    df.plot(kind="bar", ax=ax0)
    # plt.show()
    plt.savefig(os.path.join(folder_path, f"performance.png"), transparent=True)


def evaluate(sample_data: SentenceDataset, taggers: List[POSTagger]) -> Dict[AnyStr, Performance]:
    # should return a confusion matrix
    cms = dict()
    for tagger in taggers:
        print(f"Using {tagger.__class__.__name__} for tagging")
        cm = POSTagConfusionMatrix(sample_data.unique_tags())
        for sentence in sample_data.sentences():
            words = [get_word(t) for t in sentence]
            actual_tags = [get_pos_tag(t) for t in sentence]
            predicted_tags = tagger.get_tags(words, as_mapping=False)
            for (actual, predicted) in zip(actual_tags, predicted_tags):
                # print(f"Actual: {actual}, Predicted:{predicted}")
                cm.increment(actual, predicted)
        cms[tagger.name()] = Performance(sample_data, cm)
    # plot accuracy
    plot_performances(cms)
    return cms


def evaluate_treebank_dataset(treebank_dataset: TreeBankDataset, tagger_classes: List[Type[POSTagger]]) -> \
        Dict[AnyStr, Dict[AnyStr, Performance]]:
    train_dataset = treebank_dataset.train_data()
    test_dataset = treebank_dataset.test_data()
    performance = dict()
    # get transition probability and emission probability using training data
    transition_probability = create_transition_probability_matrix(train_dataset)
    emission_probability = create_emission_probability_matrix(train_dataset)
    taggers = [tagger_class(emission_probability, transition_probability) for tagger_class in tagger_classes]
    performance[treebank_dataset.name()] = evaluate(test_dataset, taggers)
    return performance


def test_treebank_dataset(treebank_dataset: TreeBankDataset, tagger_classes: List[Type[POSTagger]], string: AnyStr) -> \
        Dict[AnyStr, List[AnyStr]]:
    result = {}
    train_dataset = treebank_dataset.train_data()
    # get transition probability and emission probability using the training data
    transition_probability = create_transition_probability_matrix(train_dataset)
    emission_probability = create_emission_probability_matrix(train_dataset)
    for tagger_class in tagger_classes:
        tagger = tagger_class(emission_probability, transition_probability)
        result[tagger_class.name()] = tagger.get_tags(tokenize_text(string), as_mapping=True)
    return result


# -------------------------
# PLOTTING HELPER FUNCTIONS
# -------------------------
def plot_confusion_matrix(matrix: POSTagConfusionMatrix, path):
    # convert confusion matrix to a list of list so matplotlib can understand
    fig1 = plt.figure(figsize=(20, 20))
    ax = fig1.add_subplot(111)
    sns.heatmap(matrix.to_dataframe(), annot=True, vmin=-1, ax=ax, fmt=".5g")
    plt.savefig(path, transparent=True)

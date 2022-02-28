from collections import defaultdict
from typing import List, Type, Dict, AnyStr

from dataset import SentenceDataset, TreeBankDataset
from probability import create_transition_probability_matrix, create_emission_probability_matrix
from tagger import POSTagger
from utils import get_word, get_pos_tag, tokenize_text


class POSTagConfusionMatrix:
    def __init__(self, tags):
        self._keys = tags
        self._table = defaultdict(dict)
        for tag in tags:
            self._table[tag] = dict((t, 0) for t in tags)

    def increment(self, key1, key2):
        if len(set(self._keys).intersection([key1, key2])) == 2:
            self._table[key1][key2] += 1

    def print_matrix(self):
        format_row = "{:>10}" * (len(self._keys) + 1)
        print(format_row.format("", *self._keys))
        for key, val in self._table.items():
            print(format_row.format(key, val.values()))


def evaluate(sample_data: SentenceDataset, taggers: List[POSTagger]) -> Dict[AnyStr, POSTagConfusionMatrix]:
    # should return a confusion matrix
    cms = dict()
    for tagger in taggers:
        cm = POSTagConfusionMatrix(sample_data.unique_tags())
        for sentence in sample_data.sentences():
            words = [get_word(t) for t in sentence]
            actual_tags = [get_pos_tag(t) for t in sentence]
            predicted_tags = tagger.get_tags(words, as_mapping=True)
            for (ac, pred) in zip(actual_tags, predicted_tags):
                if ac != pred:
                    cm.increment(ac, pred)
        cms[tagger.name()] = cm
    return cms


def evaluate_treebank_dataset(treebank_dataset: TreeBankDataset, tagger_classes: List[Type[POSTagger]]) -> Dict[
    AnyStr, Dict[AnyStr, POSTagConfusionMatrix]]:
    train_dataset = treebank_dataset.train_data()
    test_dataset = treebank_dataset.test_data()
    confusion_matrices = dict()
    # get transition probability using training data
    transition_probability = create_transition_probability_matrix(train_dataset)
    # get emission probability
    emission_probability = create_emission_probability_matrix(train_dataset)
    taggers = [tagger_class(emission_probability, transition_probability) for tagger_class in tagger_classes]
    confusion_matrices[treebank_dataset.name() + "(train)"] = evaluate(train_dataset.sentences(), taggers)
    confusion_matrices[treebank_dataset.name() + "(test)"] = evaluate(test_dataset.sentences(), taggers)
    return confusion_matrices


def test_treebank_dataset(treebank_dataset: TreeBankDataset, tagger_classes: List[Type[POSTagger]], string: AnyStr) -> Dict[
    AnyStr, List[AnyStr]]:
    result = {}
    train_dataset = treebank_dataset.train_data()
    # get transition probability and emission probability using the training data
    transition_probability = create_transition_probability_matrix(train_dataset)
    emission_probability = create_emission_probability_matrix(train_dataset)
    for tagger_class in tagger_classes:
        tagger = tagger_class(emission_probability, transition_probability)
        result[tagger_class.name()] = tagger.get_tags(tokenize_text(string))
    return result

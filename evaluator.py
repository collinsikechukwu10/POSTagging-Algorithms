from collections import defaultdict
from typing import List

from tagger import POSTagger
from gettingstarted import SentenceDataset, get_word, get_pos_tag


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


class POSEvaluator:

    # prepare evaluation results for actual values
    @staticmethod
    def evaluate(sample_data: SentenceDataset, taggers: List[POSTagger]) -> List[POSTagConfusionMatrix]:
        # should return a confusion matrix
        cms = list()
        for tagger in taggers:
            cm = POSTagConfusionMatrix(sample_data.unique_tags())
            for sentence in sample_data.sentences():
                words = [get_word(t) for t in sentence]
                actual_tags = [get_pos_tag(t) for t in sentence]
                predicted_tags = tagger.get_tags(words, as_mapping=True)
                for (ac, pred) in zip(actual_tags, predicted_tags):
                    if ac != pred:
                        cm.increment(ac, pred)
            cms.append(cm)
        return cms

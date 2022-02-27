from typing import List, Iterable, AnyStr, Tuple, Set
from nltk.util import ngrams
import os
from conllu.models import Token

from utils import conllu_corpus, get_words_and_tags

START_TAG = "<s>"
END_TAG = "</s>"


class SentenceDataset:
    def __init__(self, list_of_sentences: List[List[Token]]):
        self._list_of_sentences = list_of_sentences
        emissions = []
        transitions = []
        self._unique_tags = set()
        self._unique_vocabulary = set()
        for sentence in self._list_of_sentences:
            words, tags = get_words_and_tags(sentence)
            transitions += ngrams(tags, 2, pad_left=True, pad_right=True, left_pad_symbol=START_TAG,
                                  right_pad_symbol=END_TAG)
            emissions += list(zip(tags, words))
            self._unique_tags.update(tags)
            self._unique_vocabulary.update(words)
        self._unique_tags.update([START_TAG, END_TAG])
        self._transitions = transitions
        self._emissions = emissions

    def __len__(self) -> int:
        return len(self._list_of_sentences)

    def emmisions(self) -> List[Tuple[AnyStr, AnyStr]]:
        return self._emissions

    def transitions(self) -> List[Tuple[AnyStr, AnyStr]]:
        return self._transitions

    def sentences(self):
        return self._list_of_sentences

    def unique_tags(self) -> Set[AnyStr]:
        return self._unique_tags

    def unique_vocabulary(self) -> Set[AnyStr]:
        return self._unique_vocabulary


class TreeBankDataset:
    def __init__(self, tree_bank_directory):
        conllu_files = os.listdir(tree_bank_directory)
        train_corpus = list(filter(lambda i: "train.conllu" in i, conllu_files))[0]
        test_corpus = list(filter(lambda i: "test.conllu" in i, conllu_files))[0]
        train_sentences = conllu_corpus(train_corpus)
        test_sentences = conllu_corpus(test_corpus)
        self._train_dataset = SentenceDataset(train_sentences)
        self._test_dataset = SentenceDataset(test_sentences)

        print("Generating dataset for ")
        print(f"{len(self._train_dataset)} training sentences, {len(self._test_dataset)} test sentences")

    def train_data(self) -> SentenceDataset:
        return self._train_dataset

    def test_data(self) -> SentenceDataset:
        return self._test_dataset


def prepare_datasets(main_directory) -> Iterable[TreeBankDataset]:
    return [TreeBankDataset(os.path.join(main_directory, treebank)) for treebank in os.listdir(main_directory)]

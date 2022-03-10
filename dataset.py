from typing import List, Iterable, AnyStr, Tuple, Set

from nltk import FreqDist
from nltk.util import ngrams
import os
from conllu.models import Token

from utils import conllu_corpus, get_words_and_tags, get_pos_tag

DATASET_LIST = {
    "en": "UD_English-EWT",
    "du": "UD_Dutch-Alpino",
    "esp": "UD_Spanish-GSD",
}


def get_dataset_from_code(code: str) -> str:
    return os.path.join(DEFAULT_TREEBANKS_PATH, DATASET_LIST.get(code, ""))


DEFAULT_TREEBANKS_PATH = os.path.join(os.path.dirname(__file__), "treebanks")
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
        # add start and end tag
        self._unique_tags.update([START_TAG, END_TAG])
        self._transitions = transitions
        self._emissions = emissions

        # get freq dist for tags, used later for testing
        self._pos_tag_freq_dist = FreqDist(
            get_pos_tag(token).lower() for sentence in self._list_of_sentences for token in sentence)

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

    def get_pos_tag_freq_dist(self) -> FreqDist:
        return self._pos_tag_freq_dist


class TreeBankDataset:
    def __init__(self, tree_bank_directory):
        self._name = os.path.basename(tree_bank_directory)
        conllu_files = os.listdir(tree_bank_directory)
        train_corpus = list(filter(lambda i: "train.conllu" in i, conllu_files))[0]
        test_corpus = list(filter(lambda i: "test.conllu" in i, conllu_files))[0]
        train_sentences = conllu_corpus(os.path.join(tree_bank_directory, train_corpus))
        test_sentences = conllu_corpus(os.path.join(tree_bank_directory, test_corpus))
        self._train_dataset = SentenceDataset(train_sentences)
        self._test_dataset = SentenceDataset(test_sentences)

        print(f"Generating dataset for {self._name}" +
              f" Sentences: [Training: {len(self._train_dataset)}], [Test: {len(self._test_dataset)}]")

    def train_data(self) -> SentenceDataset:
        return self._train_dataset

    def test_data(self) -> SentenceDataset:
        return self._test_dataset

    def name(self):
        return self._name


def is_conllu_language_dataset_folder(directory):
    # only has the conllu specific files for a language
    if os.path.isdir(directory):
        sub_files = list(map(lambda i: os.path.join(directory, i), os.listdir(directory)))
        return any(map(lambda i: ".conllu" in i, sub_files))
    return False


def get_datasets_folders(directory):
    folders = []
    if os.path.isdir(directory):
        if is_conllu_language_dataset_folder(directory):
            folders += [directory]
        else:
            for sub_file in list(map(lambda i: os.path.join(directory, i), os.listdir(directory))):
                folders += get_datasets_folders(sub_file)
    return folders


def resolve_datasets(dataset_code: str) -> Iterable[TreeBankDataset]:
    datasets = []
    if dataset_code == "all" or datasets == "":
        # run all data sets that exists based on the default path
        datasets += get_datasets_folders(DEFAULT_TREEBANKS_PATH)
    else:
        datasets += get_datasets_folders(get_dataset_from_code(dataset_code))
    for dataset in datasets:
        yield TreeBankDataset(dataset)

import json
from typing import List, Iterable, AnyStr, Tuple, Set, Dict

from nltk import FreqDist
from nltk.util import ngrams
import os
from conllu.models import Token

from utils import conllu_corpus, get_words_and_tags, get_pos_tag

DATASET_LIST = {
    "en": "UD_English-EWT",
    "du": "UD_Dutch-Alpino",
    "esp": "UD_Spanish-GSD",
    "fr": "UD_French-GSD"
}


def get_dataset_from_code(code: str) -> str:
    return os.path.join(DEFAULT_TREEBANKS_PATH, DATASET_LIST.get(code, ""))


BASE_PATH = os.path.dirname(__file__)
DEFAULT_TREEBANKS_PATH = os.path.join(BASE_PATH, "treebanks")
START_TAG = "<s>"
END_TAG = "</s>"


class SentenceDataset:
    def __init__(self, list_of_sentences: List[List[Token]], dataset_name):
        self.dataset_name = dataset_name
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

        print(f"Using {len(self._unique_tags)} unique tags and {len(self._unique_vocabulary)} unique tokens")

    def replace_tag(self, old_pos_tag, new_pos_tag):
        # just replace it in the list of sentences
        for sentence in self._list_of_sentences:
            for token in sentence:
                if get_pos_tag(token) == old_pos_tag:
                    token.update({'upos': new_pos_tag.upper()})
        # not really safe but im doing it anyways
        self.__init__(self._list_of_sentences, self.dataset_name)

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
        self._train_sentences = conllu_corpus(os.path.join(tree_bank_directory, train_corpus))
        self._test_sentences = conllu_corpus(os.path.join(tree_bank_directory, test_corpus))
        self._train_dataset = SentenceDataset(self._train_sentences, self._name)
        self._test_dataset = SentenceDataset(self._test_sentences, self._name)

        print(f"Importing {self._name} dataset: " +
              f"[Training: {len(self._train_dataset)}], [Test: {len(self._test_dataset)}]")

    def train_data(self) -> SentenceDataset:
        return self._train_dataset

    def test_data(self) -> SentenceDataset:
        return self._test_dataset

    def name(self):
        return self._name

    def resize(self, train_size, test_size):
        if train_size <= len(self._train_dataset) and test_size <= len(self._test_dataset):
            self._train_dataset = SentenceDataset(self._train_sentences[:train_size], self._name)
            self._test_dataset = SentenceDataset(self._test_sentences[:test_size], self._name)
            print(f"Resizing {self._name} dataset: " +
                  f"[Training: {len(self._train_dataset)}], [Test: {len(self._test_dataset)}]")
        else:
            raise Exception("Cannot resize dataset..")


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
    treebanks = [TreeBankDataset(dataset) for dataset in datasets]
    if len(treebanks) > 1:
        # this means we are using more than one dataset,
        # hence we need to make sure that the number of samples used in the training and testing
        # are the same throughout
        DatasetAdjuster.resize_datasets_by_min(treebanks)
    return treebanks


class DatasetAdjuster:
    @staticmethod
    def get_min_dataset_size(list_of_datasets: List[TreeBankDataset]):
        train_size = []
        test_size = []
        for x in list_of_datasets:
            train_size.append(len(x.train_data().sentences()))
            test_size.append(len(x.test_data().sentences()))
        return min(train_size), min(test_size)

    @staticmethod
    def merge_tags(dataset: TreeBankDataset, tags_to_delete: list = (), tag_to_merge_into: AnyStr = None):
        if tag_to_merge_into is not None:
            print(f"Applying replacements to {dataset.name()} dataset...")
            for tag_to_delete in tags_to_delete:
                # change tag for both training and test data
                print(f"\treplacing {tag_to_delete} with {tag_to_merge_into} training data...")
                dataset.train_data().replace_tag(tag_to_delete, tag_to_merge_into)
                print(f"\treplacing {tag_to_delete} with {tag_to_merge_into} test data...")
                dataset.test_data().replace_tag(tag_to_delete, tag_to_merge_into)
            print("Replacement complete...")

    @staticmethod
    def resize_datasets_by_min(list_of_datasets: List[TreeBankDataset]):
        min_train_size, min_test_size = DatasetAdjuster.get_min_dataset_size(list_of_datasets)
        for dataset in list_of_datasets:
            dataset.resize(min_train_size, min_test_size)

    @staticmethod
    def get_replacements():
        print("Importing replacements...")
        with open(os.path.join(BASE_PATH, "replacements.json"), "r") as f:
            replacements: Dict[AnyStr:List[AnyStr]] = json.load(f)
            if len(replacements) > 0:
                print("Replacements exist...")
                return replacements
        return {}

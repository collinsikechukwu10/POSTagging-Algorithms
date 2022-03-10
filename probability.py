from typing import List, Tuple, Dict, AnyStr, Set
from nltk import FreqDist, WittenBellProbDist

from dataset import START_TAG, END_TAG, SentenceDataset


# noinspection PyTypeChecker
class ProbabilityDistributionTable:
    def __init__(self, unique_prior_tokens, unique_target_tokens, ngrams_list: List[Tuple[str, str]]):
        """
        Constructor for creating a joint probability distribution table.
        :param unique_prior_tokens: prior tokens
        :param unique_target_tokens: target tokens
        :param ngrams_list: list of ngrams to use to populate probability table
        """
        # This is 2 dimensional
        table = {}
        self._prior_unique_tags = unique_prior_tokens
        self._target_unique_tags = unique_target_tokens

        for prior in self._prior_unique_tags:
            table[prior] = FreqDist()

        # populate probability table from ngrams
        for prior, target in ngrams_list:
            table[prior][target] += 1

        # add special cases
        # P(<s>| </s> ) should be 1, in the table there would be 0 occurences so we just add one to it,
        # NB, this is only for transition probability
        if START_TAG in unique_target_tokens and END_TAG in unique_target_tokens:
            table[END_TAG][START_TAG] += 1
        # apply smoothening and convert to a 2D Matrix
        for key in table:
            table[key] = WittenBellProbDist(table[key], bins=1e5)
            # dict((target, smoothed.prob(target)) for target in self._target_unique_tags)
        self._matrix: Dict[AnyStr, WittenBellProbDist] = table  # : Dict[AnyStr, Dict[AnyStr, float]] = table

    def infer(self, prior, target) -> float:
        """
        Infers the probability of a target given a prior P(target | prior)
        :param prior: prior value
        :param target: target value
        :return:
        """
        return self._matrix[prior].prob(target)  # [target]

    def get_unique_prior_tokens(self) -> Set[AnyStr]:
        return self._prior_unique_tags

    def get_unique_target_tokens(self) -> Set[AnyStr]:
        return self._target_unique_tags

    def print_table(self):
        """
        Prints the joint probability distribution table
        """
        format_row = "{:>10}" * (len(self._target_unique_tags) + 1)
        print(format_row.format("", *self._target_unique_tags))
        for key, val in self._matrix.items():
            print(format_row.format(key, (val.prob(t) for t in self._target_unique_tags)))


def create_emission_probability_matrix(train_dataset: SentenceDataset):
    """
    Create emission probability distribution table
    :param train_dataset: training dataset
    :return:
    """
    return ProbabilityDistributionTable(
        unique_prior_tokens=train_dataset.unique_tags(),
        unique_target_tokens=train_dataset.unique_vocabulary(),
        ngrams_list=train_dataset.emmisions())


def create_transition_probability_matrix(train_dataset: SentenceDataset):
    """
    Create transition probability distribution table

    :param train_dataset: training dataset
    :return:
    """
    return ProbabilityDistributionTable(
        unique_prior_tokens=train_dataset.unique_tags(),
        unique_target_tokens=train_dataset.unique_tags(),
        ngrams_list=train_dataset.transitions())

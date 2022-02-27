from typing import List, Tuple, Dict, AnyStr
from nltk import FreqDist, WittenBellProbDist

from gettingstarted import START_TAG, END_TAG


# noinspection PyTypeChecker
class BiGramProbabilityMatrix:
    def __init__(self, unique_prior_tokens, unique_target_tokens, ngrams_list: List[Tuple[str, str]]):
        # This is 2 dimensional
        table = dict()
        self._prior_unique_tags = unique_prior_tokens
        self._target_unique_tags = unique_target_tokens

        # # get unique tags
        # for (prior, target) in ngrams_list:
        #     self._prior_unique_tags.add(prior)
        #     self._target_unique_tags.add(target)

        # # add start and end tags
        # self._prior_unique_tags.add([START_TAG, END_TAG])
        # self._target_unique_tags.add([START_TAG, END_TAG])
        for prior in self._prior_unique_tags:
            table[prior] = FreqDist()

        # populate probability table from ngrams
        for prior, target in ngrams_list:
            table[prior][target] += 1

        # Work on the base cases
        # WITH BIGRAMS, the following 2 cases are automatically applied
        # if START_TAG in self._prior_unique_tags and START_TAG in self._target_unique_tags:
        #     self.update(START_TAG, START_TAG, 0)
        # if END_TAG in self._prior_unique_tags and END_TAG in self._target_unique_tags:
        #     self.update(END_TAG, END_TAG, 0)

        if END_TAG in self._prior_unique_tags and START_TAG in self._target_unique_tags:
            # its alredy 0
            # TODO, this isnt working
            table[END_TAG][START_TAG] += len(table[END_TAG])

        # apply smoothening and convert to a 2D Matrix
        for key in table:
            smoothed = WittenBellProbDist(table[key], bins=1e5)
            table[key] = dict((target, smoothed.prob(target)) for target in self._target_unique_tags)
        self._matrix: Dict[AnyStr, Dict[AnyStr, float]] = table

    def infer(self, prior, target) -> float:
        return self._matrix[prior][target]

    def get_unique_prior_tokens(self):
        return self._prior_unique_tags

    def get_unique_target_tokens(self):
        return self._target_unique_tags

    def print_table(self):
        format_row = "{:>10}" * (len(self._target_unique_tags) + 1)
        print(format_row.format("", *self._target_unique_tags))
        for key, val in self._matrix.items():
            print(format_row.format(key, val.values()))

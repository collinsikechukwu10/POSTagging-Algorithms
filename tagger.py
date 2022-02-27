from abc import abstractmethod
from collections import defaultdict
from math import log
from typing import AnyStr, List

from gettingstarted import START_TAG, END_TAG
from probability import BiGramProbabilityMatrix


class POSTagger:
    def __init__(self, emission_matrix: BiGramProbabilityMatrix, transition_matrix: BiGramProbabilityMatrix):
        self._emission_matrix = emission_matrix
        self._transition_matrix = transition_matrix

    @abstractmethod
    def get_tags(self, list_of_tokens, as_mapping=False) -> List[AnyStr]:
        raise NotImplemented()


class EagerTagger(POSTagger):
    def infer_tag(self, previous_tag, current_token) -> str:
        # get all instances of t_i
        tagset = list(self._transition_matrix.get_unique_prior_tokens())
        tagset_probabilities = []
        for tag in tagset:
            transition_prob = self._transition_matrix.infer(prior=previous_tag, target=tag)
            emission_prob = self._emission_matrix.infer(prior=tag, target=current_token)
            tagset_probabilities.append(transition_prob * emission_prob)
        return tagset[tagset_probabilities.index(max(tagset_probabilities))]

    def get_tags(self, list_of_tokens, as_mapping=False):
        # gets tag from a sentence
        t_i = START_TAG
        predicted_tags = []
        for token in list_of_tokens:
            t_i = self.infer_tag(t_i, token)
            predicted_tags.append(t_i)
        if as_mapping:
            return dict((k, v) for k, v in zip(list_of_tokens, predicted_tags))
        return predicted_tags


class VibertiPOSTagger(POSTagger):
    def get_tags(self, list_of_tokens, as_mapping=False):
        v_table = defaultdict(list)
        b_table = defaultdict(list)
        predicted_tags = list()
        # perform forward pass
        tagset = self._transition_matrix.get_unique_prior_tokens()
        self._forward_track(tagset, list_of_tokens, v_table, b_table)
        # backtrack the b table to find your optimal set
        # find the tag for the highest end tag probability
        i = len(list_of_tokens)
        while i >= 0:
            max_tag_prob_pair = sorted([(k, v[i]) for k, v in b_table.items()], key=lambda v: v[1])[-1]
            predicted_tags.append(max_tag_prob_pair[0])
            i -= 1
        if as_mapping:
            return dict((k, v) for k, v in zip(list_of_tokens, reversed(predicted_tags)[1:-2]))
        return predicted_tags

    def _forward_track(self, tag_set, list_of_tokens, v_table, b_table):
        for token_idx in range(0, len(list_of_tokens) + 1):
            for tag in tag_set:
                emission_prob = self._emission_matrix.infer(prior=tag, target=list_of_tokens[token_idx])
                if token_idx == 0:
                    transition_prob = self._transition_matrix.infer(prior=START_TAG, target=tag)
                    max_v_tag_prob_pair = (tag, log(transition_prob) + log(emission_prob))
                elif token_idx == len(list_of_tokens):
                    transition_prob = self._transition_matrix.infer(prior=tag, target=END_TAG)
                    previous_v = v_table[tag][token_idx - 1]  # already in log
                    max_v_tag_prob_pair = (tag, log(transition_prob) + previous_v)
                else:
                    trellis = []
                    for key in v_table.keys():
                        transition_prob = self._transition_matrix.infer(prior=key, target=tag)
                        previous_v = v_table[key][token_idx - 1]  # already in log
                        trellis.append((key, log(transition_prob) * log(emission_prob) * previous_v))
                    max_v_tag_prob_pair = sorted(trellis, key=lambda i: i[1])[-1]
                v_table[tag].append(max_v_tag_prob_pair[1])
                b_table[tag].append(max_v_tag_prob_pair[0])


class MostProbablePOSTagger(POSTagger):
    def get_tags(self, list_of_tokens, as_mapping=False) -> List[AnyStr]:
        pass

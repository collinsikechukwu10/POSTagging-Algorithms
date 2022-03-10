from abc import abstractmethod
from collections import defaultdict
from math import log, exp
from sys import float_info
from typing import AnyStr, List, Dict, Union, Type

from dataset import START_TAG, END_TAG
from probability import ProbabilityDistributionTable

# This effectively acts as probability 0 in the form of log probability.
min_log_prob = -float_info.max


class POSTagger:
    def __init__(self, emission_matrix: ProbabilityDistributionTable, transition_matrix: ProbabilityDistributionTable):
        """
        Constructor fo POS tagger class.
        :param emission_matrix: emission probability table
        :param transition_matrix: transition probability table
        """
        self._emission_matrix = emission_matrix
        self._transition_matrix = transition_matrix

    @abstractmethod
    def get_tags(self, list_of_tokens: List[AnyStr], as_mapping: bool = False) -> Union[List[AnyStr], Dict[AnyStr,
                                                                                                           AnyStr]]:
        """
        Get tags for a list of tokens
        :param list_of_tokens:  list of tokens
        :param as_mapping: result should be returned as a list of pos tags or a mapping of tokens to pos tags
        """
        raise NotImplemented()

    @classmethod
    def name(cls):
        """
        Gets class name
        :return:
        """
        return cls.__name__

    @staticmethod
    def as_mapping(list_of_tokens, list_of_tags):
        """
        Returns list of tokens and list of tags as a list of paired tokens to tags
        :param list_of_tokens: list of tokens
        :param list_of_tags: list of pos tags
        :return:
        """
        return dict((k, v) for k, v in zip(list_of_tokens, list_of_tags))


class EagerTagger(POSTagger):
    def _infer_tag(self, previous_tag, current_token) -> str:
        """
        Infers the current tag (t_i) using the previous tag (t_i-1) and current token (w_i).
        :param previous_tag: previous tag
        :param current_token: current token
        :return: proposed tag
        """
        # get all instances of t_i
        tagset_probabilities = []
        for tag in self._transition_matrix.get_unique_prior_tokens():
            transition_prob = self._transition_matrix.infer(prior=previous_tag, target=tag)
            emission_prob = self._emission_matrix.infer(prior=tag, target=current_token)
            tagset_probabilities.append((tag, transition_prob * emission_prob))
        max_tag_prob_pair = sorted([(k, v) for (k, v) in tagset_probabilities], key=lambda v: v[1])[-1]
        return max_tag_prob_pair[0]

    def get_tags(self, list_of_tokens: List[AnyStr], as_mapping: bool = False) -> Union[List[AnyStr], Dict[AnyStr,
                                                                                                           AnyStr]]:
        """
        Get tags for a list of tokens
        :param list_of_tokens:  list of tokens
        :param as_mapping: result should be returned as a list of pos tags or a mapping of tokens to pos tags
        """
        # gets tag from a sentence
        t_i = START_TAG
        predicted_tags = []
        for token in list_of_tokens:
            t_i = self._infer_tag(t_i, token)
            predicted_tags.append(t_i)
        if as_mapping:
            return self.as_mapping(list_of_tokens, predicted_tags)
        return predicted_tags


class VibertiPOSTagger(POSTagger):
    def get_tags(self, list_of_tokens, as_mapping=False):
        """
        Get tags for a list of tokens
        :param list_of_tokens:  list of tokens
        :param as_mapping: result should be returned as a list of pos tags or a mapping of tokens to pos tags
        """
        predicted_tags = list()
        # perform forward pass
        tagset = self._transition_matrix.get_unique_prior_tokens()
        v_table, b_table = self._forward_track(tagset, list_of_tokens)
        # backtrack the b table to find your optimal set
        # find the tag for the highest end tag probability
        i = len(list_of_tokens)
        while i >= 0:
            max_tag_prob_pair = sorted([(k, v[i]) for k, v in b_table.items()], key=lambda v: v[1])[-1]
            predicted_tags.append(max_tag_prob_pair[0])
            i -= 1
        if as_mapping:
            return self.as_mapping(list_of_tokens, reversed(predicted_tags)[1:-2])
        return predicted_tags

    def _forward_track(self, tag_set, list_of_tokens, ):
        v_table = defaultdict(list)
        b_table = defaultdict(list)
        for token_idx in range(0, len(list_of_tokens) + 1):
            for tag in tag_set:
                if token_idx == 0:
                    emission_prob = self._emission_matrix.infer(prior=tag, target=list_of_tokens[token_idx])
                    transition_prob = self._transition_matrix.infer(prior=START_TAG, target=tag)
                    max_v_tag_prob_pair = (tag, log(transition_prob) + log(emission_prob))
                elif token_idx == len(list_of_tokens):
                    transition_prob = self._transition_matrix.infer(prior=tag, target=END_TAG)
                    max_v_tag_prob_pair = (tag, log(transition_prob) + v_table[tag][token_idx - 1])
                else:
                    trellis = []
                    emission_prob = self._emission_matrix.infer(prior=tag, target=list_of_tokens[token_idx])
                    for key in v_table.keys():
                        transition_prob = self._transition_matrix.infer(prior=key, target=tag)
                        previous_v = v_table[key][token_idx - 1]  # already in log
                        trellis.append((key, log(transition_prob) + log(emission_prob) + previous_v))
                    max_v_tag_prob_pair = sorted(trellis, key=lambda i: i[1])[-1]
                v_table[tag].append(max_v_tag_prob_pair[1])
                b_table[tag].append(max_v_tag_prob_pair[0])
        return v_table, b_table


class MostProbablePOSTagger(POSTagger):
    def get_tags(self, list_of_tokens, as_mapping=False):
        """
        Get tags for a list of tokens
        :param list_of_tokens:  list of tokens
        :param as_mapping: result should be returned as a list of pos tags or a mapping of tokens to pos tags
        """
        predicted_tags = list()
        # perform forward pass
        tagset = self._transition_matrix.get_unique_prior_tokens()
        alpha_table = self._forward_track(tagset, list_of_tokens)
        beta_table = self._backward_track(tagset, list_of_tokens)
        # confirmed that alpha[q_f] == beta[q_0]
        # alpha and beta table should be the same size
        gamma_table = self._combine_table(alpha_table, beta_table)
        # get tags for the seentence
        for i in range(1, len(list_of_tokens) + 1):
            max_tag_prob_pair = sorted(gamma_table[i].items(), key=lambda v: v[1])[-1]
            predicted_tags.append(max_tag_prob_pair[0])
        if as_mapping:
            return self.as_mapping(list_of_tokens, predicted_tags)
        return predicted_tags

    def _forward_track(self, tag_set, list_of_tokens):
        """
        Forward probability algorithm for generating alpha table
        :param tag_set: list of tags
        :param list_of_tokens: list of tokens
        :return: alpha table
        """
        alpha_table = defaultdict(dict)
        alpha_table[0] = {START_TAG: 1}
        for token_idx, token in enumerate(list_of_tokens):
            for current_tag in tag_set:
                if token_idx == 0:
                    emission_prob = self._emission_matrix.infer(prior=current_tag, target=token)
                    transition_prob = self._transition_matrix.infer(prior=START_TAG, target=current_tag)
                    log_alpha = log(emission_prob) + log(transition_prob)
                else:
                    temp_alphas = []
                    emission_prob = self._emission_matrix.infer(prior=current_tag, target=token)
                    for previous_tag in tag_set:
                        transition_prob = self._transition_matrix.infer(prior=previous_tag, target=current_tag)
                        temp_alphas.append(
                            log(emission_prob) + log(transition_prob) + alpha_table[token_idx][previous_tag])
                    log_alpha = self.logsumexp(temp_alphas)
                alpha_table[token_idx + 1][current_tag] = log_alpha
        # do for closing tag
        alpha_table[len(list_of_tokens) + 1] = {
            END_TAG: self.logsumexp([log(self._transition_matrix.infer(prior=t, target=END_TAG))
                                     + alpha_table[len(list_of_tokens)][t] for t in tag_set])
        }

        return alpha_table

    def _backward_track(self, tag_set, list_of_tokens):
        """
        Backward probability algorithm for generating beta table
        :param tag_set: list of tags
        :param list_of_tokens: list of tokens
        :return: beta table
        """
        beta_table = defaultdict(dict)
        beta_table[len(list_of_tokens) + 1] = {END_TAG: 1}
        for token_idx, token in reversed(list(enumerate(list_of_tokens))):
            for current_tag in tag_set:
                if token_idx + 1 == len(list_of_tokens):
                    log_beta = log(self._transition_matrix.infer(prior=current_tag, target=END_TAG))
                else:
                    temp_betas = []
                    for previous_tag in tag_set:
                        emission_prob = self._emission_matrix.infer(prior=previous_tag,
                                                                    target=list_of_tokens[
                                                                        token_idx + 1])  # [token_idx + 1])
                        transition_prob = self._transition_matrix.infer(prior=current_tag, target=previous_tag)
                        temp_betas.append(
                            log(emission_prob) + log(transition_prob) + beta_table[token_idx + 2][
                                previous_tag])  # [token_idx + 1][previous_tag])
                    log_beta = self.logsumexp(temp_betas)
                beta_table[token_idx + 1][current_tag] = log_beta
        # do for opening tag
        beta_table[0] = {
            START_TAG: self.logsumexp([log(self._transition_matrix.infer(prior=START_TAG, target=t))
                                       + log(self._emission_matrix.infer(t, list_of_tokens[0]))
                                       + beta_table[1][t] for t in tag_set])
        }
        return beta_table

    def _combine_table(self, alpha_table, beta_table):
        """
        Joins alpha and beta table
        :param alpha_table: alpha table from forward probability algorithm
        :param beta_table: beta table from backward probability algorithm
        :return: gamma table where gamma[i][j] = alpha[i][j] * beta[i][j]
        """
        gamma_table = dict()
        for step_key, alpha_step_dict in alpha_table.items():
            # multiply alpha and beta table elementwise
            gamma_step_dict = {}
            beta_step_dict = beta_table[step_key]
            for tag_key in alpha_step_dict.keys():
                # since both alpha and beta tables are in ln form. we just add them and take the exponent
                gamma_step_dict[tag_key] = exp(alpha_step_dict[tag_key] + beta_step_dict[tag_key])
            gamma_table[step_key] = gamma_step_dict
        return gamma_table

    @staticmethod
    def logsumexp(vals):
        """
        Adding a list of probabilities represented as log probabilities.
        :param vals: log values
        :return: sum of log values
        """

        if len(vals) == 0:
            return min_log_prob
        m = max(vals)
        if m == min_log_prob:
            return min_log_prob
        else:
            return m + log(sum([exp(val - m) for val in vals]))


def resolve_taggers(tagger_type) -> List[Type[POSTagger]]:
    taggers = []
    if tagger_type == "eager":
        taggers.append(EagerTagger)
    elif tagger_type == "viterbi":
        taggers.append(VibertiPOSTagger)
    elif tagger_type == "local_decoding":
        taggers.append(MostProbablePOSTagger)
    elif tagger_type == "all":
        taggers.extend([EagerTagger, VibertiPOSTagger, MostProbablePOSTagger])
    else:
        raise Exception(f"Cannot resolve the tagger type [{tagger_type}]")
    return taggers

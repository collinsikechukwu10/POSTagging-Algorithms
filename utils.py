from typing import List, AnyStr

from sys import float_info
from math import log, exp
from conllu import parse_incr
from conllu.models import Token
from nltk.tokenize import WhitespaceTokenizer

# This effectively acts as probability 0 in the form of log probability.
min_log_prob = -float_info.max


def get_words_and_tags(tokens: List[Token]):
    words = []
    tags = []
    for t in tokens:
        words.append(get_word(t))
        tags.append(get_pos_tag(t))
    return words, tags


def get_word(token: Token):
    return token['form'].lower()


def get_pos_tag(token: Token):
    return token['upos'].lower()


def prune_sentence(sent):
    return [token for token in sent if type(token['id']) is int]


def conllu_corpus(path):
    data_file = open(path, 'r', encoding='utf-8')
    sents = list(parse_incr(data_file))
    return [prune_sentence(sent) for sent in sents]


# Adding a list of probabilities represented as log probabilities.
def logsumexp(vals):
    if len(vals) == 0:
        return min_log_prob
    m = max(vals)
    if m == min_log_prob:
        return min_log_prob
    else:
        return m + log(sum([exp(val - m) for val in vals]))


def tokenize_text(text: AnyStr) -> List[AnyStr]:
    return WhitespaceTokenizer().tokenize(text)

# prob1 = 0.0001
# prob2 = 0.000001
#
# # Now represented as log probabilities.
# logprob1 = log(prob1)
# logprob2 = log(prob2)
#
# # The log of the sum of the two probabilities.
# logsummed = logsumexp([logprob1, logprob2])
#
# # Converting log probability back to probability by exponentiation.
# summed = exp(logsummed)
# print("prob1 + prob2 =", summed)
#
# # Obviously, multiplication of probabilities is simply by adding
# # log probabilities together.
# logmultiplied = logprob1 + logprob2
# multiplied = exp(logmultiplied)
# print("prob1 * prob2 =", multiplied)

from typing import List, AnyStr

from conllu import parse_incr
from conllu.models import Token
from nltk.tokenize import WhitespaceTokenizer
import seaborn as sns
import matplotlib.pyplot as plt

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


def tokenize_text(text: AnyStr) -> List[AnyStr]:
    return WhitespaceTokenizer().tokenize(text)




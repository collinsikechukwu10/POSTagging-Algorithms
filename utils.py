from typing import List, AnyStr

from conllu import parse_incr
from conllu.models import Token
from nltk.tokenize import WhitespaceTokenizer


def get_words_and_tags(tokens: List[Token]):
    """
    Gets list of words(tokens) and tags seperately
    :param tokens: list of conllu token objects
    :return:
    """
    words = []
    tags = []
    for t in tokens:
        words.append(get_word(t))
        tags.append(get_pos_tag(t))
    return words, tags


def get_word(token: Token):
    """
    Gets word from conllu token object
    :param token:
    :return:
    """
    return token['form'].lower()


def get_pos_tag(token: Token):
    """
    Gets parts of speech tag from conllu token object
    :param token:
    :return:
    """
    return token['upos'].lower()


def prune_sentence(sent):
    return [token for token in sent if type(token['id']) is int]


def conllu_corpus(path):
    """
    Imports conllu corpus from its .conllu file
    :param path: path to .conllu file
    :return: conllu dataset
    """
    data_file = open(path, 'r', encoding='utf-8')
    sents = list(parse_incr(data_file))
    return [prune_sentence(sent) for sent in sents]


def tokenize_text(text: AnyStr) -> List[AnyStr]:
    """
    Splits text by whitespace.
    :param text: text to split
    :return:
    """
    return WhitespaceTokenizer().tokenize(text)

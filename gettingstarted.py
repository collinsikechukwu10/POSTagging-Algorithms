from io import open
import os
from typing import List
from nltk.util import ngrams
from conllu import parse_incr
from conllu.models import Token

from utils import prune_sentence

treebank = {}


treebank['en'] = 'UD_English-EWT/en_ewt'
treebank['es'] = 'UD_Spanish-GSD/es_gsd'
treebank['nl'] = 'UD_Dutch-Alpino/nl_alpino'

for key in treebank.keys():
    treebank[key] = os.path.join(TREE_BANKS_DIR, treebank[key])

# def train_corpus(lang):
#     return treebank[lang] + '-ud-train.conllu'
#
#
# def test_corpus(lang):
#     return treebank[lang] + '-ud-test.conllu'






import gensim

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


filename = 'GoogleNews-vector-negative300.bin'


model = KeyedVectors.load_word2vec_format(filename, binary=True)

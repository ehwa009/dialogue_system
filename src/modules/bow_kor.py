#!/usr/bin/env python
#-*- encoding: utf8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# import modules.util_kor as util
import util_kor as util
import numpy as np

class BoW_encoder():

    def __init__(self):
        self.vocab = self.get_vocab()
        self.vocab_size = len(self.vocab)
    
    def get_vocab(self):
        content = util.read_content()
        vocab = sorted(set(content.split(' ')))

        return [ item for item in vocab if item ]

    def encode(self, utterance):
        bow = np.zeros([self.vocab_size], dtype=np.int32)
        for word in utterance.split(' '):
            if word in self.vocab:
                idx = self.vocab.index(word)
                bow[idx] += 1
        return bow


    
        
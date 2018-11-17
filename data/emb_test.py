#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from gensim.models import word2vec
from gensim.models.wrappers import FastText

import numpy as np
import os

class UtteranceEmbed():

    def __init__(self, lang='eng', dim=200):
       
        self.dim = dim
        # self.file_path = 'ko.bin'
        self.file_path = 'wiki.ko.bin'
        
        self.model = FastText.load_fasttext_format(self.file_path)
        # self.model = word2vec.Word2Vec.load(self.file_path)
        
    
    def encode(self, utterance):
        utterance = unicode(utterance)


        embs = [self.model[word] for word in utterance.split(' ') if word and word in self.model]
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim], np.float32)

if __name__ == "__main__":
    u = UtteranceEmbed()
    utterance = '안녕'
    result = u.encode(utterance)
    print(result)
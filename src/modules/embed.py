#!/usr/bin/env python
#-*- encoding: utf8 -*-

from gensim.models import word2vec
import numpy as np
import os
import rospkg, rospy

class UtteranceEmbed():

    def __init__(self, lang='eng', dim=300):
        self.dim = dim
        
        if lang == 'eng':
            self.file_path = os.path.join(rospkg.RosPack().get_path('dialogue_system'), 'data', 'text8.model')
        elif lang == 'kor':
            self.file_path = os.path.join(rospkg.RosPack().get_path('dialogue_system'), 'data', 'korean_word2vec')
        
        self.model = word2vec.Word2Vec.load(self.file_path)
    
    def encode(self, utterance):
        embs = [self.model[word] for word in utterance.split(' ') if word and word in self.model]
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim], np.float32)
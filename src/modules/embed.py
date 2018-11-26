#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from gensim.models import word2vec
from gensim.models.wrappers import FastText

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import os
import rospkg, rospy

class UtteranceEmbed():

    def __init__(self, lang='eng', dim=300):
       
        if lang == 'eng':
            self.dim = dim
            self.file_path = os.path.join(rospkg.RosPack().get_path('dialogue_system'), 'data', 'text8.model')
            self.model = word2vec.Word2Vec.load(self.file_path)
            
        elif lang == 'kor':
            self.dim = dim
            # self.file_path = os.path.join(rospkg.RosPack().get_path('dialogue_system'), 'data', 'cc.ko.300.bin')
            self.file_path = os.path.join(rospkg.RosPack().get_path('dialogue_system'), 'data', 'wiki.ko.bin')
            # self.file_path = os.path.join(rospkg.RosPack().get_path('dialogue_system'), 'data', 'ko.bin')
            
            self.model = FastText.load_fasttext_format(self.file_path)
            # self.model = word2vec.Word2Vec.load(self.file_path)
        
    
    def encode(self, utterance):
        # for korean 
        utterance = unicode(utterance)
        embs = [self.model[word] for word in utterance.split(' ') if word and word in self.model]
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim], np.float32)
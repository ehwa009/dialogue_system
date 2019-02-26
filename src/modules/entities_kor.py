#!/usr/bin/env python
#-*- encoding: utf8 -*-

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from enum import Enum

class EntityTracker():

    def __init__(self):

        self.entities = {
            '<name>' : None,
            '<address_number>' : None,
            '<address_name>' : None,
            '<time>' : None,
            '<pm_am>' : None,
            '<location>': None,
        }

        self.num_features = len(self.entities)
        self.name = ['황의준', '윤혜정', '황희준', '김점옥', '황성호', '김철수']
        self.address = ['조례동']
        self.address_number = ['123']
        self.time = ['4시', '3시']
        self.pm_am = ['오전', '오후']
        self.location = ['화장실']

        self.EntType = Enum('Entity Type', '<name> <address_name> <address_number> <time> <pm_am> <location> <non_ent>')

    def set_entities(self, ent):
        self.entities = ent

    def ent_type(self, ent):
        if ent.startswith(tuple(self.name)):
            entity_word = [word for word in self.name if word in ent][0]
            return self.EntType['<name>'].name, entity_word
        
        elif ent.startswith(tuple(self.address)):
            entity_word = [word for word in self.address if word in ent][0]
            return self.EntType['<address_name>'].name, entity_word

        elif ent.startswith(tuple(self.address_number)):
            entity_word = [word for word in self.address_number if word in ent][0]
            return self.EntType['<address_number>'].name, entity_word

        elif ent.startswith(tuple(self.time)):
            entity_word = [word for word in self.time if word in ent][0]
            return self.EntType['<time>'].name, entity_word
        
        elif ent.startswith(tuple(self.pm_am)):
            entity_word = [word for word in self.pm_am if word in ent][0]
            return self.EntType['<pm_am>'].name, entity_word

        elif ent.startswith(tuple(self.location)):
            entity_word = [word for word in self.location if word in ent][0]
            return self.EntType['<location>'].name, entity_word

        else:
            return ent, None

    def extract_entities(self, utterance, update=True, is_test=False):
        tokenized = []
        for word in utterance.split(' '):
            entity, entity_word = self.ent_type(word)
            if word != entity and update:
                self.entities[entity] = entity_word
            tokenized.append(entity)
        tokenized_str = ' '.join(tokenized)
        
        if is_test is True:
            return tokenized_str, self.entities
        else:
            return tokenized_str

    def context_features(self):
        keys = list(set(self.entities.keys()))
        keys = sorted(keys)
        self.ctxt_features = np.array( [bool(self.entities[key]) for key in keys], 
                                    dtype=np.float32 )
        return self.ctxt_features

    def do_clear_entities(self):
        self.entities = {
            '<name>' : None,
            '<address_number>' : None,
            '<address_name>' : None,
            '<time>' : None,
            '<pm_am>' : None,
            '<location>': None,
        }

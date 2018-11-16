#!/usr/bin/env python
#-*- encoding: utf8 -*-

import modules.util_kor as util

class Data():

    def __init__(self, entity_tracker, action_tracker):

        self.action_templates = action_tracker.get_action_templates()
        self.et = entity_tracker
        self.trainset = self.prepare_data()

    def prepare_data(self):
        # get dialogs from raw text
        dialogs, dialog_indices = util.read_dialogs(with_indices=True)
        # get utteracnes
        utterances = util.get_utterances(dialogs)
        # get responses
        responses = util.get_response(dialogs)
        responses = [self.get_template_id(response) for response in responses]

        # make actual trainset
        trainset = []
        for u,r in zip(utterances, responses):
            trainset.append((u,r))
        
        return trainset, dialog_indices

    def get_template_id(self, response):
        return self.action_templates.index(self.et.extract_entities(response))



            

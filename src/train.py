#!/usr/bin/env python
#-*- encoding: utf8 -*-
from __future__ import division

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from modules.embed import UtteranceEmbed

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# load rnn models
from modules.rnn_model.gru import GRU
from modules.rnn_model.lstm import LSTM
from modules.rnn_model.reversed_input_lstm import ReversingLSTM
from modules.rnn_model.reversed_input_gru import ReversingGRU
from modules.rnn_model.stacked_gru import StackedGRU
from modules.rnn_model.stacked_lstm import StackedLSTM
from modules.rnn_model.bidirectional_lstm import BidirectionalLSTM
from modules.rnn_model.bidirectional_gru import BidirectionalGRU

import numpy as np
import sys, os
import rospy, rospkg
import logging
import time
import random
import json



class Train():

    def __init__(self, args):

        self.response_accuracy = []
        self.dialog_accuracy = []
        try:
            ###################### selective import #############################
            if args[0] == 'am':
                self.is_action_mask = True
            else:
                self.is_action_mask = False
            if args[1] == 'emb':
                self.is_emb = True
            else:
                self.is_emb = False
            
            self.network_type = args[2]
            self.lang_type = args[3]

            if self.lang_type == 'eng':
                from modules.entities import EntityTracker
                from modules.data_utils import Data
                from modules.actions import ActionTracker
                from modules.bow import BoW_encoder
                
            elif self.lang_type == 'kor':
                from modules.entities_kor import EntityTracker
                from modules.data_utils_kor import Data
                from modules.actions_kor import ActionTracker
                from modules.bow_kor import BoW_encoder
                
            ###################################################################
        except:
            rospy.logwarn("please try again. i.e. ... train.py <am> <emb> <bidirectional_lstm> <eng>")

        if self.is_emb:
            if self.lang_type == 'eng':
                self.emb = UtteranceEmbed(lang=self.lang_type)
            elif self.lang_type == 'kor':
                self.emb = UtteranceEmbed(lang=self.lang_type)
        
        et = EntityTracker()
        self.bow_enc = BoW_encoder()

        at = ActionTracker(et)
        
        at.do_display_template()
        self.dataset, dialog_indices = Data(et, at).trainset
        # self.dialog_indices_tr = dialog_indices[:200]
        self.dialog_indices_tr = random.sample(dialog_indices, 200)
        # self.dialog_indices_dev = dialog_indices[200:250]
        self.dialog_indices_dev = random.sample(dialog_indices, 50)

        self.action_templates = at.get_action_templates()
        action_size = at.action_size
        nb_hidden = 128
        
        # set feature input
        if self.is_action_mask and self.is_emb:
            obs_size = self.emb.dim + self.bow_enc.vocab_size + et.num_features + at.action_size
        elif self.is_action_mask and not(self.is_emb):
            obs_size = self.bow_enc.vocab_size + et.num_features + at.action_size
        elif not(self.is_action_mask) and self.is_emb:
            obs_size = self.emb.dim + self.bow_enc.vocab_size + et.num_features
        elif not(self.is_action_mask) and not(self.is_emb):
            obs_size = self.bow_enc.vocab_size + et.num_features

        # set network_type type
        if self.network_type == 'gru':
            self.net = GRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_action_mask)
        elif self.network_type == 'reversed_lstm':
            self.net = ReversingLSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_action_mask)
        elif self.network_type == 'reversed_gru':
            self.net = ReversingGRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_action_mask)
        elif self.network_type == 'stacked_gru':
            self.net = StackedGRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_action_mask)
        elif self.network_type == 'stacked_lstm':
            self.net = StackedLSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_action_mask)
        elif self.network_type == 'lstm':
            self.net = LSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_action_mask)
        elif self.network_type == 'bidirectional_lstm':
            self.net = BidirectionalLSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_action_mask)
        elif self.network_type == 'bidirectional_gru':
            self.net = BidirectionalGRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_action_mask)

        file_path = os.path.join(rospkg.RosPack().get_path('dialogue_system'),'log', self.network_type)
        # init logging
        self.logger = self.get_logger(file_path)
        msg = "'\033[94m[%s trainer]\033[0m initialized - %s (action_mask: %s, embedding: %s, lang: %s, obs_size: %s, bow: %s, context_feature: %s, action_size: %s)"%(rospy.get_name(), self.net.__class__.__name__, self.is_action_mask, self.is_emb, self.lang_type, obs_size, self.bow_enc.vocab_size, et.num_features, action_size)
        rospy.loginfo(msg)

    def train(self, cont=False):
        # logging and print
        msg = "training started."
        rospy.loginfo(msg)
        # call previous trained model
        if cont:
            self.net.restore()

        epochs = 10
        # start measuring time
        for j in range(epochs):
            num_tr_examples = len(self.dialog_indices_tr)
            loss = 0.

            for i,dialog_idx in enumerate(self.dialog_indices_tr):
                start, end = dialog_idx['start'], dialog_idx['end']
                # train dialog
                loss += self.dialog_train(self.dataset[start:end])
                sys.stdout.write('\r{}.[{}/{}]'.format(j+1, i+1, num_tr_examples))
                             
            # logging and print
            msg = '\n\n {}.tr loss {}'.format(j+1, loss/num_tr_examples)
            rospy.loginfo(msg)
                
            turn_accuracy, dialog_accuracy = self.evaluate()
            msg = '\n{}.dev turn_accuracy {}, dialog_accuracy {}'.format(j+1, turn_accuracy, dialog_accuracy)
            rospy.loginfo(msg)

            if dialog_accuracy > 0.999:
                self.net.save()
                break
        # save checkpoint
        self.net.save()
    
    def dialog_train(self, dialog):
        ###################################################################
        if self.lang_type == 'eng':
            from modules.entities import EntityTracker
            from modules.data_utils import Data
            from modules.actions import ActionTracker
            from modules.bow import BoW_encoder
            
        elif self.lang_type == 'kor':
            from modules.entities_kor import EntityTracker
            from modules.data_utils_kor import Data
            from modules.actions_kor import ActionTracker
            from modules.bow_kor import BoW_encoder
        ###################################################################

        et = EntityTracker()
        at = ActionTracker(et)
        # reset state in network_type
        self.net.reset_state()

        loss = 0.
        for (u,r) in dialog:
            u_ent = et.extract_entities(u)
            u_ent_features = et.context_features()
            u_bow = self.bow_enc.encode(u)
            if self.is_emb:
                u_emb = self.emb.encode(u)
            if self.is_action_mask:
                action_mask = at.action_mask()

            # concatenated features
            if self.is_action_mask and self.is_emb:
                features = np.concatenate((u_ent_features, u_emb, u_bow, action_mask), axis=0)
            elif self.is_action_mask and not(self.is_emb):
                features = np.concatenate((u_ent_features, u_bow, action_mask), axis=0)
            elif not(self.is_action_mask) and self.is_emb:
                features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
            elif not(self.is_action_mask) and not(self.is_emb):
                features = np.concatenate((u_ent_features, u_bow), axis=0)
            
            # forward propagation with cumulative loss
            if self.is_action_mask:
                loss += self.net.train_step(features, r, action_mask)
            else:
                loss += self.net.train_step(features, r)
    
        return loss/len(dialog)

    def evaluate(self, eval=False):
        ###################################################################
        if self.lang_type == 'eng':
            from modules.entities import EntityTracker
            from modules.data_utils import Data
            from modules.actions import ActionTracker
            from modules.bow import BoW_encoder
            
        elif self.lang_type == 'kor':
            from modules.entities_kor import EntityTracker
            from modules.data_utils_kor import Data
            from modules.actions_kor import ActionTracker
            from modules.bow_kor import BoW_encoder
        ###################################################################

        et = EntityTracker()
        at = ActionTracker(et)
        # only for evaluation purpose
        if eval:
            self.net.restore()
        # reset entities extractor
        
        turn_accuracy = 0.
        dialog_accuracy = 0.
        for dialog_idx in self.dialog_indices_dev:
            start, end = dialog_idx['start'], dialog_idx['end']
            dialog = self.dataset[start:end]
            num_dev_examples = len(self.dialog_indices_dev)

            et = EntityTracker()
            at = ActionTracker(et)
            # reset network_type before evaluate.
            self.net.reset_state()

            correct_examples = 0
            for (u,r) in dialog:                
                u_ent = et.extract_entities(u)
                u_ent_features = et.context_features()
                u_bow = self.bow_enc.encode(u)
                if self.is_emb:
                    u_emb = self.emb.encode(u)
                if self.is_action_mask:
                    action_mask = at.action_mask()

                # concatenated features
                if self.is_action_mask and self.is_emb:
                    features = np.concatenate((u_ent_features, u_emb, u_bow, action_mask), axis=0)
                elif self.is_action_mask and not(self.is_emb):
                    features = np.concatenate((u_ent_features, u_bow, action_mask), axis=0)
                elif not(self.is_action_mask) and self.is_emb:
                    features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                elif not(self.is_action_mask) and not(self.is_emb):
                    features = np.concatenate((u_ent_features, u_bow), axis=0)

                if self.is_action_mask:
                    probs, prediction = self.net.forward(features, action_mask)
                else:
                    probs, prediction = self.net.forward(features)

                correct_examples += int(prediction == r)

            turn_accuracy += correct_examples/len(dialog)

            accuracy = correct_examples/len(dialog)
            if (accuracy == 1.0):
                dialog_accuracy += 1
        
        turn_accuracy = turn_accuracy/num_dev_examples
        dialog_accuracy = dialog_accuracy/num_dev_examples
        
        return turn_accuracy, dialog_accuracy

    
    def get_logger(self, filename):
        """Return a logger instance that writes in filename

        Args:
            filename: (string) path to log.txt

        Returns:
            logger: (instance of logger)

        """
        logger = logging.getLogger('logger')
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
                '%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)

        return logger


if __name__ == '__main__':
    rospy.init_node('dialogue_trainer', anonymous=False)
    
    t = Train(sys.argv[1:])
    # t.train()
    t.train(cont=True)
    # print(t.evaluate(eval=True))
    rospy.signal_shutdown('training finish')
    

        
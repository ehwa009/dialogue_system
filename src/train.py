#!/usr/bin/env python
#-*- encoding: utf8 -*-
from __future__ import division

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from modules.bow import BoW_encoder
from modules.embed import UtteranceEmbed

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from modules.rnn_model.lstm_model import LSTM
from modules.rnn_model.gru_model import GRU
from modules.rnn_model.inverted_gru_model import InvertedGRU
from modules.rnn_model.inverted_lstm_model import InvertedLSTM
from modules.rnn_model.multi_gru_model import MultiGRU
from modules.rnn_model.multi_lstm_model import MultiLSTM
from modules.rnn_model.hcn_lstm_model import HCN_LSTM
from modules.rnn_model.bidirectional_lstm import Bidirectional_LSTM

import numpy as np
import sys, os
import rospy, rospkg
import logging
import time


class Train():

    def __init__(self, args):

        ###################### selective import #############################
        self.feature = int(args[0])
        self.network = args[1]
        self.lang = args[2]

        if self.lang == 'eng':
            from modules.entities import EntityTracker
            from modules.data_utils import Data
            from modules.actions import ActionTracker
        elif self.lang == 'kor':
            from modules.entities_kor import EntityTracker
            from modules.data_utils_kor import Data
            from modules.actions_kor import ActionTracker
        ###################################################################

        if self.lang == 'eng':
            self.emb = UtteranceEmbed(lang=self.lang)
        elif self.lang == 'kor':
            self.emb = UtteranceEmbed(lang=self.lang, dim=200)
        et = EntityTracker()
        self.bow_enc = BoW_encoder()
        
        at = ActionTracker(et)
        
        at.do_display_template()

        self.dataset, dialog_indices = Data(et, at).trainset
        self.dialog_indices_tr = dialog_indices[:200]
        self.dialog_indices_dev = dialog_indices[200:250]

        self.action_templates = at.get_action_templates()
        action_size = at.action_size
        nb_hidden = 128
        
        # set feature input
        if self.feature == 1:
            obs_size = self.emb.dim
        elif self.feature == 2:
            obs_size = self.emb.dim + self.bow_enc.vocab_size
        elif self.feature == 3:
            obs_size = self.emb.dim + et.num_features
        elif self.feature == 4:
            obs_size = self.emb.dim + self.bow_enc.vocab_size + et.num_features

        # set network type
        if self.network == 'lstm':
            self.net = LSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size)
        elif self.network == 'gru':
            self.net = GRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang)
        elif self.network == 'inverted_gru':
            self.net = InvertedGRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size)
        elif self.network == 'inverted_lstm':
            self.net = InvertedLSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size)
        elif self.network == 'multi_gru':
            self.net = MultiGRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size)
        elif self.network == 'multi_lstm':
            self.net = MultiLSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size)
        elif self.network == 'hcn_lstm':
            self.net = HCN_LSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size)

        file_path = os.path.join(rospkg.RosPack().get_path('dialogue_system'),'log', self.network)
        # init logging
        self.logger = self.get_logger(file_path)
        msg = "'\033[94m[%s trainer]\033[0m initialized - %s (input condition: %s, lang: %s)"% (rospy.get_name(), self.net.__class__.__name__, self.feature, self.lang)
        rospy.loginfo(msg)

    def train(self):
        # logging and print
        msg = "training started."
        rospy.loginfo(msg)

        epochs = 20
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
            
            accuracy = self.evaluate()
            
            msg = '{}.dev accuracy {}\n'.format(j+1, accuracy)
            rospy.loginfo(msg)

            if accuracy > 0.99:
                self.net.save()
                break
        # save checkpoint
        # self.net.save()
                
    def dialog_train(self, dialog):
        ###################################################################
        if self.lang == 'eng':
            from modules.entities import EntityTracker
            from modules.data_utils import Data
            from modules.actions import ActionTracker
        elif self.lang == 'kor':
            from modules.entities_kor import EntityTracker
            from modules.data_utils_kor import Data
            from modules.actions_kor import ActionTracker
        ###################################################################

        et = EntityTracker()
        at = ActionTracker(et)
        # reset state in network
        self.net.reset_state()

        loss = 0.
        for (u,r) in dialog:
            u_ent = et.extract_entities(u)
            u_ent_features = et.context_features()
            u_bow = self.bow_enc.encode(u)
            u_emb = self.emb.encode(u)
            action_mask = at.action_mask()

            # print(unicode(u), r)
            # print('fuck: %s' % u_ent_features)            
                
            # concatenated features
            if self.feature == 1:
                features = u_emb
            elif self.feature == 2: 
                features = np.concatenate((u_emb, u_bow), axis=0)
            elif self.feature == 3:
                features = np.concatenate((u_ent_features, u_emb), axis=0)
            elif self.feature == 4:
                features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
            
            # forward propagation with cumulative loss
            if self.net.__class__.__name__ == "HCN_LSTM":
                loss += self.net.train_step(features, r, action_mask)
            else:
                loss += self.net.train_step(features, r)
    
        return loss/len(dialog)

    def evaluate(self, eval=False):
        ###################################################################
        if self.lang == 'eng':
            from modules.entities import EntityTracker
            from modules.data_utils import Data
            from modules.actions import ActionTracker
        elif self.lang == 'kor':
            from modules.entities_kor import EntityTracker
            from modules.data_utils_kor import Data
            from modules.actions_kor import ActionTracker
        ###################################################################

        et = EntityTracker()
        at = ActionTracker(et)
        # only for evaluation purpose
        if eval:
            self.net.restore()
        # reset entities extractor
        
        dialog_accuracy = 0.
        for dialog_idx in self.dialog_indices_dev:
            start, end = dialog_idx['start'], dialog_idx['end']
            dialog = self.dataset[start:end]
            num_dev_examples = len(self.dialog_indices_dev)

            et = EntityTracker()
            at = ActionTracker(et)
            # reset network before evaluate.
            self.net.reset_state()

            correct_examples = 0
            for (u,r) in dialog:
                u_ent = et.extract_entities(u)
                u_ent_features = et.context_features()
                u_bow = self.bow_enc.encode(u)
                u_emb = self.emb.encode(u)
                action_mask = at.action_mask()
                
                # concatenated features
                if self.feature == 1:
                    features = u_emb
                elif self.feature == 2: 
                    features = np.concatenate((u_emb, u_bow), axis=0)
                elif self.feature == 3:
                    features = np.concatenate((u_ent_features, u_emb), axis=0)
                elif self.feature == 4:
                    features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                
                if self.net.__class__.__name__ == "HCN_LSTM":
                    probs, prediction = self.net.forward(features, action_mask)
                else:
                    probs, prediction = self.net.forward(features)

                correct_examples += int(prediction == r)

            dialog_accuracy += correct_examples/len(dialog)
        
        if eval:
            rospy.logwarn("TEST accuracy: %f" %(dialog_accuracy/num_dev_examples))
        
        return dialog_accuracy/num_dev_examples

    
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
    t.train()
    rospy.signal_shutdown('training finish')
    

        
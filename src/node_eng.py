#!/usr/bin/env python
#-*- encoding: utf8 -*-

from modules.entities import EntityTracker
from modules.bow import BoW_encoder
from modules.embed import UtteranceEmbed
from modules.actions import ActionTracker
from modules.data_utils import Data

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from mind_msgs.msg import LearningFigure

import modules.util as util
import numpy as np
import sys
import rospy
import json
import random

from modules.rnn_model.lstm_model import LSTM
from modules.rnn_model.gru_model import GRU
from modules.rnn_model.inverted_gru_model import InvertedGRU
from modules.rnn_model.inverted_lstm_model import InvertedLSTM
from modules.rnn_model.multi_gru_model import MultiGRU
from modules.rnn_model.multi_lstm_model import MultiLSTM

from mind_msgs.msg import Reply, RaisingEvents, DBQuery
from mind_msgs.srv import ReloadWithResult, ReadData, WriteData

PRESCRIPTION = ['''I'm sorry {first_name}, your doctor has not yet written your prescription \n
                and so it is not ready for collection at the moment.\n
                However, I have sent a message to your doctor.\n
                Once the prescription has been written, someone will call you and let you know.''']
APPOINTMENT = ['''No problem {first_name}, I can see that you have an appointment with Dr Jones today and have checked you in''']
BATHROOM = ['''Certainly, the bathroom is located down the hall, second door on the right''']
WAITING = ['''{first_name}, you are next to see Dr. jones, he will be around 5 more minutes.''']

class Dialogue():

    def __init__(self):
        self.et = EntityTracker()
        self.at = ActionTracker(self.et)
        self.bow_enc = BoW_encoder()
        self.emb = UtteranceEmbed()

        obs_size = self.emb.dim + self.bow_enc.vocab_size + self.et.num_features
        self.action_template = self.at.get_action_templates()

        self.at.do_display_template()

        # must clear entities space
        self.et.do_clear_entities()

        action_size = self.at.action_size
        nb_hidden = 128

        # selected network and langauge
        network = rospy.get_param('~network_model', 'lstm')
        lang = rospy.get_param('~lang', 'eng')

        if network == 'lstm':
            self.net = LSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size)
        elif network == 'gru':
            self.net = GRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=lang)
        elif network == 'inverted_gru':
            self.net = InvertedGRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size)
        elif network == 'inverted_lstm':
            self.net = InvertedLSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size)
        elif network == 'multi_gru':
            self.net = MultiGRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size)
        elif network == 'multi_lstm':
            self.net = MultiLSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size)
        
        # restore trained file
        self.net.restore()

        self.pub_reply = rospy.Publisher('reply', Reply, queue_size=10)
        # self.pub_qeury = rospy.Publisher('query', DBQuery, queue_size=10)

        rospy.Subscriber('raising_events', RaisingEvents, self.handle_raise_events)
        rospy.loginfo('\033[94m[%s]\033[0m initialized.'%rospy.get_name())

    def handle_raise_events(self, msg):
        utterance = msg.recognized_word
        
        if utterance == 'clear':
            self.net.reset_state()
            self.et.do_clear_entities()

            response = 'context has been cleared.'

        else:
            if 'silency_detected' in msg.events:
                utterance = '<SILENCE>'

            rospy.loginfo("actual input: %s" %utterance)
            
            u_ent, u_entities = self.et.extract_entities(utterance, is_test=True)
            u_ent_features = self.et.context_features()
            u_emb = self.emb.encode(utterance)
            u_bow = self.bow_enc.encode(utterance)

            features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
            probs, prediction = self.net.forward(features)
            
            response = self.action_template[prediction]

            # check validity
            prediction = self.pre_action_process(prediction, u_entities)
            
            if self.post_process(prediction, u_entities):
                if prediction == 1:
                    response = 'api_call appointment {} {} {} {} {} {} {}'.format(
                        u_entities['<first_name>'], u_entities['<last_name>'],
                        u_entities['<address_number>'], u_entities['<address_name>'],
                        u_entities['<address_type>'], u_entities['<time>'], u_entities['<pm_am>']
                    )
                elif prediction == 2:
                    response = 'api_call location {}'.format(u_entities['<location>'])
                elif prediction == 3:
                    response = 'api_call prescription {} {} {} {} {}'.format(
                        u_entities['<first_name>'], u_entities['<last_name>'],
                        u_entities['<address_number>'], u_entities['<address_name>'],
                        u_entities['<address_type>']
                    )
                elif prediction == 4:
                    response = 'api_call waiting_time {} {} {} {} {} {} {}'.format(
                        u_entities['<first_name>'], u_entities['<last_name>'],
                        u_entities['<address_number>'], u_entities['<address_name>'],
                        u_entities['<address_type>'], u_entities['<time>'], u_entities['<pm_am>']
                    )
                # TODO: implement real api call 

            else:
                require_name = [7,10,12]
                prediction = self.action_post_process(prediction, u_entities)

                if prediction in require_name:
                    response = self.action_template[prediction]
                    response = response.split(' ')
                    response = [word.replace('<first_name>', u_entities['<first_name>']) for word in response]
                    response = ' '.join(response)
                else:
                    response = self.action_template[prediction]

        # print status of entities and actual response
        rospy.loginfo(json.dumps(self.et.entities, indent=2))
        try:
            rospy.logwarn("System: [conf: %f, predict: %d] / %s\n" %(max(probs), prediction, response))
        except UnboundLocalError:
            rospy.logwarn("System: [conf: ] / %s\n" %(response))

        reply_msg = Reply()
        reply_msg.header.stamp = rospy.Time.now()
        reply_msg.reply = response

        self.pub_reply.publish(reply_msg)

    def post_process(self, prediction, u_ent_features):
        api_call_list = [1,2,3,4]
        if prediction in api_call_list:
            return True
        attr_list = [0,9,10,11,12]
        if all(u_ent_featur == 1 for u_ent_featur in u_ent_features) and prediction in attr_list:
            return True
        else:
            return False

    def action_post_process(self, prediction, u_entities):

        attr_mapping_dict = {
            11: '<first_name>',
            11: '<last_name>',
            12: '<address_number>',
            12: '<address_name>',
            12: '<address_type>',
            10: '<time>',
            10: '<pm_am>' ,
        }
        
        # find exist and non-exist entity
        exist_ent_index = [key for key, value in u_entities.items() if value != None]
        non_exist_ent_index = [key for key, value in u_entities.items() if value == None]
        
        # if predicted key is already in exist entity index then find non exist entity index
        # and leads the user to input non exist entity.
        
        if prediction in attr_mapping_dict:
            pred_key = attr_mapping_dict[prediction]
            if pred_key in exist_ent_index:
                for key, value in attr_mapping_dict.items():
                    if value == non_exist_ent_index[0]:
                        return key
            else:
                return prediction
        else:
            return prediction

    def pre_action_process(self, prediction, u_entities):
        
        api_call_list = [1,2,3,4]

        attr_mapping_dict = {
            '<first_name>': 11,
            '<last_name>': 11,
            '<address_number>': 12,
            '<address_name>': 12,
            '<address_type>': 12,
            '<time>': 10,
            '<pm_am>': 10 ,
        }

        # find exist and non-exist entity
        non_exist_ent_index = [key for key, value in u_entities.items() if value == None]

        if prediction in api_call_list:
            if '<first_name>' in non_exist_ent_index:
                prediction = attr_mapping_dict['<first_name>']

        return prediction

if __name__ == '__main__':
    rospy.init_node('dialogue_system', anonymous=False)
    d = Dialogue()
    rospy.spin()


                
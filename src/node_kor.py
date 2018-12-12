#!/usr/bin/env python
#-*- encoding: utf8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from modules.bow import BoW_encoder
from modules.embed import UtteranceEmbed

from modules.entities_kor import EntityTracker
from modules.data_utils_kor import Data
from modules.actions_kor import ActionTracker

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from mind_msgs.msg import LearningFigure

import modules.util as util
import numpy as np
import sys
import rospy
import json
import random

from modules.rnn_model.hcn_lstm_model import HCN_LSTM
from modules.rnn_model.gru_model import GRU
from modules.rnn_model.inverted_gru_model import InvertedGRU
from modules.rnn_model.inverted_lstm_model import InvertedLSTM
from modules.rnn_model.multi_gru_model import MultiGRU
from modules.rnn_model.multi_lstm_model import MultiLSTM

from mind_msgs.msg import Reply, RaisingEvents
from mind_msgs.srv import ReloadWithResult, ReadData, WriteData, DBQuery

class Dialogue():

    def __init__(self):
        # selected network and langauge
        network = rospy.get_param('~network_model', 'lstm')
        lang = rospy.get_param('~lang', 'kor')
        self.is_action_mask = rospy.get_param('~action_mask', "true")

        self.et = EntityTracker()
        self.at = ActionTracker(self.et)
        self.bow_enc = BoW_encoder()
        self.emb = UtteranceEmbed(lang=lang)

        obs_size = self.emb.dim + self.bow_enc.vocab_size + self.et.num_features
        self.action_template = self.at.get_action_templates()

        self.at.do_display_template()

        # must clear entities space
        self.et.do_clear_entities()

        action_size = self.at.action_size
        nb_hidden = 128

        if network == 'hcn_lstm':
            self.net = HCN_LSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=lang, is_action_mask=self.is_action_mask)
        elif network == 'gru':
            self.net = GRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=lang, is_action_mask=self.is_action_mask)
        elif network == 'inverted_gru':
            self.net = InvertedGRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=lang, is_action_mask=self.is_action_mask)
        elif network == 'inverted_lstm':
            self.net = InvertedLSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=lang, is_action_mask=self.is_action_mask)
        elif network == 'multi_gru':
            self.net = MultiGRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=lang, is_action_mask=self.is_action_mask)
        elif network == 'multi_lstm':
            self.net = MultiLSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=lang, is_action_mask=self.is_action_mask)
        
        # restore trained file
        self.net.restore()

        self.pub_reply = rospy.Publisher('reply', Reply, queue_size=10)
        # self.pub_qeury = rospy.Publisher('query', DBQuery, queue_size=10)

        rospy.Subscriber('raising_events', RaisingEvents, self.handle_raise_events)
        
        try:
            rospy.wait_for_service('reception_db/query_data')
            self.get_response_db = rospy.ServiceProxy('reception_db/query_data', DBQuery)
        except rospy.exceptions.ROSInterruptException as e:
            rospy.logerr(e)
            quit()
        
        rospy.loginfo('\033[94m[%s]\033[0m initialized.'%rospy.get_name())

    def handle_raise_events(self, msg):
        utterance = msg.recognized_word
        
        if utterance == 'clear':
            self.net.reset_state()
            self.et.do_clear_entities()

            response = '초기화 완료.'

        else:
            if 'silency_detected' in msg.events:
                utterance = '<SILENCE>'

            rospy.loginfo("actual input: %s" %utterance)
            
            u_ent, u_entities = self.et.extract_entities(utterance, is_test=True)
            u_ent_features = self.et.context_features()
            u_emb = self.emb.encode(utterance)
            u_bow = self.bow_enc.encode(utterance)
            
            if self.is_action_mask:
                action_mask = self.at.action_mask()

            # print(u_ent_features)
            # print(u_emb)
            # print(u_bow)

            features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
            
            if self.is_action_mask:
                probs, prediction = self.net.forward(features, action_mask)
            else:
                probs, prediction = self.net.forward(features)
    
            print('TEST: %d' % prediction)
            
            response = self.action_template[prediction]

            # check validity
            prediction = self.pre_action_process(prediction, u_entities)
            
            if self.post_process(prediction, u_entities):
                if prediction == 1:
                    response = 'api_call 예약 {} {} {}'.format(
                        u_entities['<name>'], u_entities['<address>'],
                        u_entities['<time>'])
                elif prediction == 2:
                    response = 'api_call 위치 {}'.format(u_entities['<location>'])
                elif prediction == 3:
                    response = 'api_call 처방전 {} {}'.format(
                        u_entities['<name>'], u_entities['<address>'])
                elif prediction == 0:
                    response = 'api_call 대기시간 {} {} {}'.format(
                        u_entities['<name>'], u_entities['<address>'],
                        u_entities['<time>'])
                
                # TODO: implement real api call 
                response = self.get_response_db(response)
                response = response.response

            else:
                require_name = [4,7,12]
                prediction = self.action_post_process(prediction, u_entities)

                if prediction in require_name:
                    response = self.action_template[prediction]
                    response = response.split(' ')
                    response = [word.replace('<name>', u_entities['<name>']) for word in response]
                    response = ' '.join(response)
                else:
                    response = self.action_template[prediction]

        # print status of entities and actual response
        rospy.loginfo(json.dumps(self.et.entities, indent=2))
        try:
            rospy.logwarn("System: [conf: %f, predict: %d] / %s\n" %(max(probs), prediction, unicode(response)))
        except UnboundLocalError:
            rospy.logwarn("System: [conf: ] / %s\n" %(response))

        reply_msg = Reply()
        reply_msg.header.stamp = rospy.Time.now()
        reply_msg.reply = response

        self.pub_reply.publish(reply_msg)

    def post_process(self, prediction, u_ent_features):
        api_call_list = [0,1,2,3]
        if prediction in api_call_list:
            return True
        attr_list = [5,6,7,8,10]
        if all(u_ent_featur == 1 for u_ent_featur in u_ent_features) and prediction in attr_list:
            return True
        else:
            return False

    def action_post_process(self, prediction, u_entities):

        attr_mapping_dict = {
            8: '<name>',
            4: '<address>',
            7: '<time>',
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
        api_call_list = [0,1,2,3]
        attr_mapping_dict = {
            '<name>': 8,
            '<address>': 4,
            '<time>': 7,
        }
        # find exist and non-exist entity
        non_exist_ent_index = [key for key, value in u_entities.items() if value == None]

        if prediction in api_call_list:
            if '<name>' in non_exist_ent_index:
                prediction = attr_mapping_dict['<name>']

        return prediction

if __name__ == '__main__':
    rospy.init_node('dialogue_system', anonymous=False)
    d = Dialogue()
    rospy.spin()
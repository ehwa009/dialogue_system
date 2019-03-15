#!/usr/bin/env python
#-*- encoding: utf8 -*-

from modules.entities import EntityTracker
from modules.bow import BoW_encoder
from modules.embed import UtteranceEmbed
from modules.actions import ActionTracker
from modules.data_utils import Data

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import modules.util as util
import numpy as np
import sys, os, re
import rospy,rospkg
import json
import random

# load rnn models
from modules.rnn_model.gru import GRU
from modules.rnn_model.lstm import LSTM
from modules.rnn_model.reversed_input_lstm import ReversingLSTM
from modules.rnn_model.reversed_input_gru import ReversingGRU
from modules.rnn_model.stacked_gru import StackedGRU
from modules.rnn_model.stacked_lstm import StackedLSTM
from modules.rnn_model.bidirectional_lstm import BidirectionalLSTM
from modules.rnn_model.bidirectional_gru import BidirectionalGRU

from mind_msgs.msg import Reply, RaisingEvents
from mind_msgs.srv import DBQuery
from std_msgs.msg import Int16, Bool, Empty

PRESCRIPTION = ['''I'm sorry {first_name}, your doctor has not yet written your prescription \n
                and so it is not ready for collection at the moment.\n
                However, I have sent a message to your doctor.\n
                Once the prescription has been written, someone will call you and let you know.''']
APPOINTMENT = ['''No problem {first_name}, I can see that you have an appointment with Dr Jones today and have checked you in''']
BATHROOM = ['''Certainly, the bathroom is located down the hall, second door on the right''']
WAITING = ['''{first_name}, you are next to see Doctor jones, he will be around 5 more minutes.''']
REPROMPT = ["I missed that, can you say that again?", "sorry I can't understand, please say that again."]

BOUNDARY_CONFIDENCE = 0.4

class Dialogue():

    def __init__(self):
        # stor whole dialogues
        self.story = []
        self.sp_confidecne = []
        self.file_path = os.path.join(rospkg.RosPack().get_path('dialogue_system'),'log', 'dialogue.txt')
        
        # count turn taking
        self.usr_count = 0
        self.sys_count = 0

        # paramaters
        self.network_type = rospy.get_param('~network_model', 'stacked_lstm')
        self.lang_type = rospy.get_param('~lang', 'eng')
        self.is_emb = rospy.get_param('~embedding', 'false')
        self.is_am = rospy.get_param('~action_mask', "true")
        self.user_num = rospy.get_param('~user_number', '0')

        # call rest of modules
        self.et = EntityTracker()
        self.at = ActionTracker(self.et)
        self.bow_enc = BoW_encoder()
        self.emb = UtteranceEmbed(lang=self.lang_type)

        # select observation size for RNN
        if self.is_am and self.is_emb:
            obs_size = self.emb.dim + self.bow_enc.vocab_size + self.et.num_features + self.at.action_size
        elif self.is_am and not(self.is_emb):
            obs_size = self.bow_enc.vocab_size + self.et.num_features + self.at.action_size
        elif not(self.is_am) and self.is_emb:
            obs_size = self.emb.dim + self.bow_enc.vocab_size + self.et.num_features
        elif not(self.is_am) and not(self.is_emb):
            obs_size = self.bow_enc.vocab_size + self.et.num_features
        
        self.action_template = self.at.get_action_templates()
        self.at.do_display_template()
        # must clear entities space
        self.et.do_clear_entities()
        action_size = self.at.action_size
        nb_hidden = 128

        if self.network_type == 'gru':
            self.net = GRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_am)
        elif self.network_type == 'reversed_lstm':
            self.net = ReversingLSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_am)
        elif self.network_type == 'reversed_gru':
            self.net = ReversingGRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_am)
        elif self.network_type == 'stacked_gru':
            self.net = StackedGRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_am)
        elif self.network_type == 'stacked_lstm':
            self.net = StackedLSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_am)
        elif self.network_type == 'lstm':
            self.net = LSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_am)
        elif self.network_type == 'bidirectional_lstm':
            self.net = BidirectionalLSTM(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_am)
        elif self.network_type == 'bidirectional_gru':
            self.net = BidirectionalGRU(obs_size=obs_size, nb_hidden=nb_hidden, action_size=action_size, lang=self.lang_type, is_action_mask=self.is_am)
        
        # restore trained model
        self.net.restore()

        # rostopics
        self.pub_reply = rospy.Publisher('reply', Reply, queue_size=10)
        self.pub_complete = rospy.Publisher('complete_execute_scenario', Empty, queue_size=10)
        rospy.Subscriber('raising_events', RaisingEvents, self.handle_raise_events)

        try:
            rospy.wait_for_service('reception_db/query_data')
            self.get_response_db = rospy.ServiceProxy('reception_db/query_data', DBQuery)
            rospy.logwarn("waiting for reception DB module...")
        except rospy.exceptions.ROSInterruptException as e:
            rospy.logerr(e)
            quit()
        rospy.logwarn("network: {}, lang: {}, action_mask: {}, embedding: {}, user_number: {}".format(self.network_type, self.lang_type, self.is_am, self.is_emb, self.user_num))
        self.story.append('user number: %s'%self.user_num)
        rospy.loginfo('\033[94m[%s]\033[0m initialized.'%rospy.get_name())

        # if utterance == 'clear':
        #     self.net.reset_state()
        #     self.et.do_clear_entities()
        #     response = 'context has been cleared.'

    def get_response(self, utterance):
        rospy.loginfo("actual input: %s" %utterance) # check actual user input
        
        # clean utterance
        utterance = re.sub(r'[^ a-z A-Z 0-9]', " ", utterance)
        # utterance preprocessing
        u_ent, u_entities = self.et.extract_entities(utterance, is_test=True)
        u_ent_features = self.et.context_features()
        u_bow = self.bow_enc.encode(utterance)
        
        if self.is_emb:
            u_emb = self.emb.encode(utterance)
        if self.is_am:
            action_mask = self.at.action_mask()
        
        # concatenated features
        if self.is_am and self.is_emb:
            features = np.concatenate((u_ent_features, u_emb, u_bow, action_mask), axis=0)
        elif self.is_am and not(self.is_emb):
            features = np.concatenate((u_ent_features, u_bow, action_mask), axis=0)
        elif not(self.is_am) and self.is_emb:
            features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
        elif not(self.is_am) and not(self.is_emb):
            features = np.concatenate((u_ent_features, u_bow), axis=0)
        
        try:
            # predict template number
            if self.is_am:
                probs, prediction = self.net.forward(features, action_mask)
            else:
                probs, prediction = self.net.forward(features)

            # check response confidence
            if max(probs) > BOUNDARY_CONFIDENCE:
                response = self.action_template[prediction]
                prediction = self.pre_action_process(prediction, u_entities)       
                
                # handle api call
                if self.post_process(prediction, u_entities):
                    if prediction == 1:
                        response = 'api_call appointment {} {} {} {} {} {} {}'.format(
                            u_entities['<first_name>'], u_entities['<last_name>'],  
                            u_entities['<address_number>'], u_entities['<address_name>'],
                            u_entities['<address_type>'], u_entities['<time>'], u_entities['<pm_am>'])
                    elif prediction == 2:
                        response = 'api_call location {}'.format(u_entities['<location>'])
                    elif prediction == 3:
                        response = 'api_call prescription {} {} {} {} {}'.format(
                            u_entities['<first_name>'], u_entities['<last_name>'],
                            u_entities['<address_number>'], u_entities['<address_name>'],
                            u_entities['<address_type>'])
                    elif prediction == 4:
                        response = 'api_call waiting_time {} {} {} {} {} {} {}'.format(
                            u_entities['<first_name>'], u_entities['<last_name>'],
                            u_entities['<address_number>'], u_entities['<address_name>'],
                            u_entities['<address_type>'], u_entities['<time>'], u_entities['<pm_am>'])
                    response = self.get_response_db(response) # query knowledge base; here we use dynamo db
                    response = response.response 
                elif prediction in [6,9,11]:
                    response = self.action_template[prediction]
                    response = response.split(' ')
                    response = [word.replace('<first_name>', u_entities['<first_name>']) for word in response]
                    response = ' '.join(response)
                else:
                    response = self.action_template[prediction]

            else:
                response = random.choice(REPROMPT) # if prediction confidence less than 40%, reprompt    
        except:
            response = random.choice(REPROMPT)

        return prediction, probs, response

    def handle_raise_events(self, msg):
        utterance = msg.recognized_word
        try:
            # get confidence
            data = json.loads(msg.data[0])
            confidence = data['confidence']
        except:
            confidence = None

        if confidence > BOUNDARY_CONFIDENCE or confidence == None:       
            if 'silency_detected' in msg.events:
                utterance = '<SILENCE>'
            else:
                try:
                    self.story.append("U%i: %s (sp_conf:%f)"%(self.usr_count+1, utterance, confidence))
                    self.sp_confidecne.append(confidence)
                except:
                    self.story.append("U%i: %s"%(self.usr_count+1, utterance))
                self.usr_count += 1
                utterance = utterance.lower()
            # generate system response
            prediction, probs, response = self.get_response(utterance)
        else:
            prediction = -1
            probs = -1
            response = random.choice(REPROMPT)
        
        # add system turn
        self.story.append("A%i: %s"%(self.sys_count+1, response))
        self.sys_count += 1
        
        # finish interaction
        if (prediction == 6):
            self.pub_complete.publish()
            # logging user and system turn
            self.story.append("user: %i, system: %i"%(self.usr_count, self.sys_count))
            self.story.append("mean_sp_conf: %f"%(reduce(lambda x, y: x + y, self.sp_confidecne) / len(self.sp_confidecne)))
            self.story.append('===================================================================')
            self.write_file(self.file_path, self.story)    
        
        # display system response
        rospy.loginfo(json.dumps(self.et.entities, indent=2)) # recognized entity values
        try:
            rospy.logwarn("System: [conf: %f, predict: %d] / %s\n" %(max(probs), prediction, response))
        except:
            rospy.logwarn("System: [] / %s\n" %(response))
       
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
        
        api_call_list = [1,3,4]

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

    ''' 
    writing story log file
    '''
    def write_file(self, path, story_list):
        with open(path, 'a') as f:
            for item in story_list:
                f.write("%s\n"%item)
        rospy.logwarn('save dialogue histories.')

if __name__ == '__main__':
    rospy.init_node('dialogue_system', anonymous=False)
    d = Dialogue()
    rospy.spin()


                
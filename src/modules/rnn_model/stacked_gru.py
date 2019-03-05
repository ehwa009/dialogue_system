#!/usr/bin/env python
#-*- encoding: utf8 -*-

import tensorflow as tf
import numpy as np
import rospkg, rospy
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from tensorflow.contrib.layers import xavier_initializer as xav

class StackedGRU():

    def __init__(self, obs_size, nb_hidden=128, action_size=16, lang='eng', is_action_mask=False):

        self.obs_size = obs_size
        self.nb_hidden = nb_hidden
        self.action_size = action_size
        self.lang = lang
        self.is_action_mask = is_action_mask

        if self.lang == 'eng':
            self.file_path = os.path.join(rospkg.RosPack().get_path('dialogue_system'),'ckpt')
        elif self.lang == 'kor':
            self.file_path = os.path.join(rospkg.RosPack().get_path('dialogue_system'),'ckpt_kor')

        def __graph__():
            tf.reset_default_graph()

            # entry points
            features_ = tf.placeholder(tf.float32, [1, obs_size], name='input_features')
            init_state_f_, init_state_s_ = ( tf.placeholder(tf.float32, [1, nb_hidden]) for _ in range(2) )
            action_ = tf.placeholder(tf.int32, name='ground_truth_action')
            if self.is_action_mask:
                action_mask_ = tf.placeholder(tf.float32, [action_size], name='action_mask')
            
            # input projection - 인풋 dimention을 맞춰주기 위한 trick ##############
            Wi = tf.get_variable('Wi', [obs_size, nb_hidden], initializer=xav())
            bi = tf.get_variable('bi', [nb_hidden], initializer=tf.constant_initializer(0.))
            
            projected_features = tf.matmul(features_, Wi) + bi
            ########################################################################

            gru_f = tf.contrib.rnn.GRUCell(num_units=nb_hidden)
            stacked_gru = tf.contrib.rnn.MultiRNNCell([gru_f]*2)

            output, state = stacked_gru(projected_features, state=(init_state_f_, init_state_s_))

            # ouput projection - 아웃풋 dimention을 맞춰주기 위한 trick ###########
            state_reshaped = tf.concat(axis=1, values=(state[0], state[1]))

            Wo = tf.get_variable('Wo', [2*nb_hidden, action_size], initializer=xav())
            bo = tf.get_variable('bo', [action_size], initializer=tf.constant_initializer(0.))
            
            logits = tf.matmul(state_reshaped, Wo) + bo
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action_)
            train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
            ########################################################################

            if self.is_action_mask:
                probs = tf.multiply(tf.squeeze(tf.nn.softmax(logits)), action_mask_)
            else:
                probs = tf.squeeze(tf.squeeze(tf.nn.softmax(logits)))
            
            prediction = tf.arg_max(probs, dimension=0)


            # each output values
            self.loss = loss
            self.prediction = prediction
            self.probs = probs
            self.logits = logits
            self.state = state
            self.train_op = train_op

            # attach placeholder
            self.features_ = features_
            self.init_state_f_ = init_state_f_
            self.init_state_s_ = init_state_s_
            self.action_ = action_
            if self.is_action_mask:
                self.action_mask_ = action_mask_

        # build graph
        __graph__()

        # start a session; attach to self
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.sess = sess
        # set init state to zeros
        self.init_state_f = np.zeros([1,self.nb_hidden], dtype=np.float32)
        self.init_state_s = np.zeros([1,self.nb_hidden], dtype=np.float32)

    # forward propagation
    def forward(self, features, action_mask=None):
        if action_mask is None:
            probs, prediction, state_f, state_s = self.sess.run( [self.probs, self.prediction, self.state[0], self.state[1]], 
                    feed_dict = { 
                        self.features_ : features.reshape([1,self.obs_size]), 
                        self.init_state_f_ : self.init_state_f,
                        self.init_state_s_ : self.init_state_s,
                        })
        else:
            probs, prediction, state_f, state_s = self.sess.run( [self.probs, self.prediction, self.state[0], self.state[1]], 
                    feed_dict = { 
                        self.features_ : features.reshape([1,self.obs_size]), 
                        self.init_state_f_ : self.init_state_f,
                        self.init_state_s_ : self.init_state_s,
                        self.action_mask_ : action_mask
                        })
        # maintain state
        self.init_state_f = state_f
        self.init_state_s = state_s
        # return argmax
        return probs, prediction

    # training
    def train_step(self, features, action, action_mask=None):
        if action_mask is None:
            _, loss_value, state_f, state_s = self.sess.run( [self.train_op, self.loss, self.state[0], self.state[1]],
                    feed_dict = {
                        self.features_ : features.reshape([1, self.obs_size]),
                        self.action_ : [action],
                        self.init_state_f_ : self.init_state_f,
                        self.init_state_s_ : self.init_state_s,
                        })
        else:
            _, loss_value, state_f, state_s = self.sess.run( [self.train_op, self.loss, self.state[0], self.state[1]],
                    feed_dict = {
                        self.features_ : features.reshape([1, self.obs_size]),
                        self.action_ : [action],
                        self.init_state_f_ : self.init_state_f,
                        self.init_state_s_ : self.init_state_s,
                        self.action_mask_ : action_mask
                        })
        # maintain state
        self.init_state_f = state_f
        self.init_state_s = state_s
        return loss_value

    def reset_state(self):
        # set init state to zeros
        self.init_state_f = np.zeros([1,self.nb_hidden], dtype=np.float32)
        self.init_state_s = np.zeros([1,self.nb_hidden], dtype=np.float32)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, '%s/hcn.ckpt'%self.file_path, global_step=0)
        rospy.logwarn('saved to ckpt/hcn.ckpt\n')

    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.file_path)
        if ckpt and ckpt.model_checkpoint_path:
            rospy.logwarn('restoring checkpoint from' + ckpt.model_checkpoint_path + '\n')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            rospy.logwarn('no checkpoint found...\n')



            
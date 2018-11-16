#!/usr/bin/env python
#-*- encoding: utf8 -*-

import tensorflow as tf
import numpy as np
import rospkg, rospy
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from tensorflow.contrib.layers import xavier_initializer as xav

class InvertedLSTM():

    def __init__(self, obs_size, nb_hidden=128, action_size=16):

        self.obs_size = obs_size
        self.nb_hidden = nb_hidden
        self.action_size = action_size

        self.file_path = os.path.join(rospkg.RosPack().get_path('dialogue_system'),'ckpt')

        def __graph__():
            tf.reset_default_graph()

            # entry points
            features_ = tf.placeholder(tf.float32, [1, obs_size], name='input_features')
            features_ = tf.reverse(features_, [1], name='inverted_input')

            init_state_c_, init_state_h_ = ( tf.placeholder(tf.float32, [1, nb_hidden]) for _ in range(2) )
            action_ = tf.placeholder(tf.int32, name='ground_truth_action')
            
            # input projection - 인풋 dimention을 맞춰주기 위한 trick ##############
            Wi = tf.get_variable('Wi', [obs_size, nb_hidden], initializer=xav())
            bi = tf.get_variable('bi', [nb_hidden], initializer=tf.constant_initializer(0.))
            
            projected_features = tf.matmul(features_, Wi) + bi
            ########################################################################

            lstm_f = tf.contrib.rnn.LSTMCell(num_units=nb_hidden, state_is_tuple=True)
            lstm_op, state = lstm_f(inputs=projected_features, state=(init_state_c_, init_state_h_))

            # ouput projection - 아웃풋 dimention을 맞춰주기 위한 trick ###########
            state_reshaped = tf.concat(axis=1, values=(state.c, state.h))

            Wo = tf.get_variable('Wo', [2*nb_hidden, action_size], initializer=xav())
            bo = tf.get_variable('bo', [action_size], initializer=tf.constant_initializer(0.))
            
            logits = tf.matmul(state_reshaped, Wo) + bo
            ########################################################################

            probs = tf.squeeze(tf.nn.softmax(logits))
            
            prediction = tf.arg_max(probs, dimension=0)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action_)

            # default was 0.1
            train_op = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

            # each output values
            self.loss = loss
            self.prediction = prediction
            self.probs = probs
            self.logits = logits
            self.state = state
            self.train_op = train_op

            # attach placeholder
            self.features_ = features_
            self.init_state_c_ = init_state_c_
            self.init_state_h_ = init_state_h_
            self.action_ = action_

        # build graph
        __graph__()

        # start a session; attach to self
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.sess = sess
        # set init state to zeros
        self.init_state_c = np.zeros([1,self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1,self.nb_hidden], dtype=np.float32)

    # forward propagation
    def forward(self, features):
        # forward
        probs, prediction, state_c, state_h = self.sess.run( [self.probs, self.prediction, self.state.c, self.state.h], 
                feed_dict = { 
                    self.features_ : features.reshape([1,self.obs_size]), 
                    self.init_state_c_ : self.init_state_c,
                    self.init_state_h_ : self.init_state_h
                    })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h
        # return argmax
        return probs, prediction

    # training
    def train_step(self, features, action):
        _, loss_value, state_c, state_h = self.sess.run( [self.train_op, self.loss, self.state.c, self.state.h],
                feed_dict = {
                    self.features_ : features.reshape([1, self.obs_size]),
                    self.action_ : [action],
                    self.init_state_c_ : self.init_state_c,
                    self.init_state_h_ : self.init_state_h
                    })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h
        return loss_value

    def reset_state(self):
        # set init state to zeros
        self.init_state_c = np.zeros([1,self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1,self.nb_hidden], dtype=np.float32)

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



            

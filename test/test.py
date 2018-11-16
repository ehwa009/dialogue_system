#!/usr/bin/env python
#-*- encoding: utf8 -*-

import rospy

from mind_msgs.msg import Reply, RaisingEvents, DBQuery
from mind_msgs.srv import ReloadWithResult, ReadData, WriteData

class Node():

    def __init__(self):
        self.utter_pub = rospy.pub_utterance = rospy.Publisher('raising_events', RaisingEvents, queue_size=10)
        # rospy.Subscriber('reply', Reply, self.handle_resp)        
        
        while True:
            utterance = raw_input(":: ")

            if utterance == "exit":
                rospy.signal_shutdown('exit')
                break
            elif utterance == "":
                utterance = "<SILENCE>"
            
            self.handle_input(utterance)
            
    def handle_input(self, utterance):
        utter_pub = RaisingEvents()
        utter_pub.header.stamp = rospy.Time.now()
        utter_pub.recognized_word = utterance

        self.utter_pub.publish(utter_pub)

    def handle_resp(self, msg):
        resp = msg.reply
        print("\n" + resp)

if __name__ == '__main__':
    rospy.init_node('test_node', anonymous=False)
    n = Node()
    rospy.spin()
    

    


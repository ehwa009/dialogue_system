#!/usr/bin/env python
#-*- encoding: utf8 -*-

import modules.util_kor as util
import numpy as np

from modules.entities_kor import EntityTracker as et

'''
############## Action Template ###############
0. api_call 대기시간 <name> <address> <time>
1. api_call 예약 <name> <address> <time>
2. api_call 위치 <location>
3. api_call 처방전 <name> <address>
4. 감사합니다 <name> 님. 주소를 알려주시겠어요
5. 네 알겠습니다. 오늘 보려는 의사선생님 성함을 알려주세요.
6. 안녕하세요, 저는 리셉션 로봇 나오에요. 어떻게 도와드릴까요?
7. 알겠습니다 <name> 님, 오늘 약속시간은 어떻게 되시나요?
8. 알겠습니다. 이름을 알려주세요
9. 오늘 만나려는 의사선생님 성함은 어떻게 되시나요?
10. 제가 도와드릴게 있을까요?
11. 좋은 하루 보내세요.
12. 좋은 하루 보내시길 바라겠습니다. <name> 님
##############################################

{'<address>': None, 
'<location>': None, 
'<time>': None, 
'<name>': None}

'''

class ActionTracker():

    def __init__(self, ent_tracker):
        self.et = ent_tracker
        self.action_templates = self.get_action_templates()
        self.action_size = len(self.action_templates)

        self.am = np.zeros([self.action_size], dtype=np.float32)
        # action mask lookup, built on intuition
        self.am_dict = {
            '0000': [5,6,8,9,10,11],
            '0001': [4,7,12],
            '0010': [8, 9, 10, 2],
            '0011': [8, 9, 10],
            '0100': [8, 9, 10, 7, 2],
            '0101': [8, 9, 10, 7],
            '0110': [8, 9, 10, 2],
            '0111': [8, 9, 10],
            '1000': [8, 9, 7, 2],
            '1001': [8, 9, 7],
            '1010': [8, 9, 2],
            '1011': [8, 9],
            '1100': [8, 9, 7, 2],
            '1101': [8, 9, 7],
            '1110': [8, 9, 2],
            '1111': [1, 11, 12, 9, 6, 4, 5, 3]
        }
        
    
    def get_action_templates(self):
        responses = sorted(set([self.et.extract_entities(response, update=False) 
                        for response in util.get_response()]))

        return responses

    def do_display_template(self):
        print('\n############## Action Template ###############')
        i=0
        while (i < len(self.action_templates)):
            print('%i. %s' %(i, self.action_templates[i]))
            i += 1
        print('##############################################\n')

    def action_mask(self):
        # get context features as string of ints (0/1)
        ctxt_f = ''.join([ str(flag) for flag in self.et.context_features().astype(np.int32) ])

        def construct_mask(ctxt_f):
            indices = self.am_dict[ctxt_f]
            for index in indices:
                self.am[index-1] = 1.
            return self.am
    
        return construct_mask(ctxt_f)

if __name__ == "__main__":
    action = ActionTracker(et)
    action.do_display_template()
        
        
        
            
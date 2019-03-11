#!/usr/bin/env python
#-*- encoding: utf8 -*-

import modules.util as util
import numpy as np

from modules.entities import EntityTracker as et

'''
############## Action Template ###############                                                                   
0. and which doctor are you seeing?                                                                              
1. api_call appointment <first_name> <last_name> <address_number> <address_name> <address_type> <time> <pm_am>   
2. api_call location <location>                                                                                  
3. api_call prescription <first_name> <last_name> <address_number> <address_name> <address_type>                 
4. api_call waiting_time <first_name> <last_name> <address_number> <address_name> <address_type> <time> <pm_am>  
5. hi there, my name is Nao, the receptionist robot. how may I help you?                                         
6. i hope you have a nice day                                                                                    
7. i hope you have a nice day <first_name>                                                                       
8. is there anything else i can help you with?                                                                   
9. no problem, which doctor are you seeing?                                                                      
10. okay <first_name> , what time is your appointment                                                            
11. okay, can you please tell me your name?                                                                      
12. thanks <first_name> , and what is your address?                                                              
##############################################                                                                   

['<address_name>', '<address_number>', '<address_type>', '<first_name>', '<last_name>', '<location>', '<pm_am>', '<time>'] 

'''

class ActionTracker():

    def __init__(self, ent_tracker):
        self.et = ent_tracker
        self.action_templates = self.get_action_templates()
        self.action_size = len(self.action_templates)

        self.am = np.zeros([self.action_size], dtype=np.float32)
        # action mask lookup, built on intuition
        self.am_dict = {
            '00000000' : [0,5,6,8,9,11],
            '11100000' : [0,5,6,8,9,11],
            '00011000' : [0,5,6,7,8,9,10,11,12],
            '00000011' : [0,5,6,8,9,11],
            '11111000' : [0,3,5,6,7,8,9,10,11,12],
            '00000100' : [0,2,5,6,8,9,11],
            '11111111' : [0,1,2,3,4,5,6,7,8,9,10,11,12], 
            '11111011' : [0,1,3,4,5,6,7,8,9,10,11,12],
            '11111100' : [0,2,3,5,6,7,8,9,10,11,12],           
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
                self.am[index] = 1.
            return self.am
    
        return construct_mask(ctxt_f)

if __name__ == "__main__":
    action = ActionTracker(et)
    action.do_display_template()
        
        
        
            
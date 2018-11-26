#!/usr/bin/env python
#-*- encoding: utf8 -*-

import os
import rospkg

def read_dialogs(with_indices=False):
    training_data_path = os.path.join(rospkg.RosPack().get_path('dialogue_system'), 'data', 'reception.txt')
    
    with open(training_data_path) as f:
        dialogs = [row.split('\t') for row in f.read().split('\n')]
        prev_idx = -1
        n = 1
        dialog_indices = []
        updated_dialogs = []
        for i, dialog in enumerate(dialogs):
            if len(dialog) == 1:
                dialog_indices.append({
                    'start' : prev_idx + 1,
                    'end' : i - n + 1
                })
                prev_idx = i - n
                n += 1
            else:
                updated_dialogs.append(dialog)        

        if with_indices:
            return updated_dialogs, dialog_indices[:-1]

        return updated_dialogs

def get_utterances(dialogs=[]):
    dialogs = dialogs if len(dialogs) else read_dialogs()
    return [row[0] for row in dialogs]

def get_response(dialogs=[]):
    dialogs = dialogs if len(dialogs) else read_dialogs()
    return [row[1] for row in dialogs]

def read_content():
    return ' '.join(get_utterances())


#!/bin/bash   

# eng
# rosrun dialogue_system train.py 4 hcn_lstm eng action_mask
rosrun dialogue_system train.py 4 hcn_lstm eng
# rosrun dialogue_system train.py 4 gru eng action_mask
rosrun dialogue_system train.py 4 gru eng
# rosrun dialogue_system train.py 4 inverted_lstm eng action_mask
rosrun dialogue_system train.py 4 inverted_lstm eng
# rosrun dialogue_system train.py 4 inverted_gru eng action_mask
rosrun dialogue_system train.py 4 inverted_gru eng
# rosrun dialogue_system train.py 4 multi_lstm eng action_mask
rosrun dialogue_system train.py 4 multi_lstm eng
# rosrun dialogue_system train.py 4 multi_gru eng action_mask
rosrun dialogue_system train.py 4 multi_gru eng
# rosrun dialogue_system train.py 4 bidirectional_lstm eng action_mask
rosrun dialogue_system train.py 4 bidirectional_lstm eng
# rosrun dialogue_system train.py 4 bidirectional_gru eng action_mask
rosrun dialogue_system train.py 4 bidirectional_gru eng

# kor
# rosrun dialogue_system train.py 4 hcn_lstm kor action_mask
# rosrun dialogue_system train.py 4 hcn_lstm kor
# rosrun dialogue_system train.py 4 gru kor action_mask
# rosrun dialogue_system train.py 4 gru kor
# rosrun dialogue_system train.py 4 inverted_lstm kor action_mask
# rosrun dialogue_system train.py 4 inverted_lstm kor
# rosrun dialogue_system train.py 4 inverted_gru kor action_mask
# rosrun dialogue_system train.py 4 inverted_gru kor
# rosrun dialogue_system train.py 4 multi_lstm kor action_mask
# rosrun dialogue_system train.py 4 multi_lstm kor
# rosrun dialogue_system train.py 4 multi_gru kor action_mask
# rosrun dialogue_system train.py 4 multi_gru kor
# rosrun dialogue_system train.py 4 bidirection_lstm kor action_mask
# rosrun dialogue_system train.py 4 bidirectional_lstm kor
# rosrun dialogue_system train.py 4 bidirection_gru kor action_mask
# rosrun dialogue_system train.py 4 bidirectional_gru kor
#!/bin/bash   

rosrun dialogue_system train.py 1 lstm
rosrun dialogue_system train.py 2 lstm
rosrun dialogue_system train.py 3 lstm
rosrun dialogue_system train.py 4 lstm

rosrun dialogue_system train.py 1 gru
rosrun dialogue_system train.py 2 gru
rosrun dialogue_system train.py 3 gru
rosrun dialogue_system train.py 4 gru

rosrun dialogue_system train.py 1 inverted_lstm
rosrun dialogue_system train.py 2 inverted_lstm
rosrun dialogue_system train.py 3 inverted_lstm
rosrun dialogue_system train.py 4 inverted_lstm

rosrun dialogue_system train.py 1 inverted_gru
rosrun dialogue_system train.py 2 inverted_gru
rosrun dialogue_system train.py 3 inverted_gru
rosrun dialogue_system train.py 4 inverted_gru

rosrun dialogue_system train.py 1 multi_gru
rosrun dialogue_system train.py 2 multi_gru
rosrun dialogue_system train.py 3 multi_gru
rosrun dialogue_system train.py 4 multi_gru

rosrun dialogue_system train.py 1 multi_lstm
rosrun dialogue_system train.py 2 multi_lstm
rosrun dialogue_system train.py 3 multi_lstm
rosrun dialogue_system train.py 4 multi_lstm


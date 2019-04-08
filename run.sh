#!/bin/bash   

# eng
counter=1
while [ $counter -le 5 ]
do
    # rosrun dialogue_system train.py am emb lstm eng
    # rosrun dialogue_system train.py am no lstm eng
    # rosrun dialogue_system train.py no emb lstm eng
    # rosrun dialogue_system train.py no no lstm eng

    # rosrun dialogue_system train.py am emb gru eng
    # rosrun dialogue_system train.py am no gru eng
    # rosrun dialogue_system train.py no emb gru eng
    # rosrun dialogue_system train.py no no gru eng

    # rosrun dialogue_system train.py am emb reversed_lstm eng
    # rosrun dialogue_system train.py am no reversed_lstm eng
    # rosrun dialogue_system train.py no emb reversed_lstm eng
    # rosrun dialogue_system train.py no no reversed_lstm eng

    # rosrun dialogue_system train.py am emb reversed_gru eng
    # rosrun dialogue_system train.py am no reversed_gru eng
    # rosrun dialogue_system train.py no emb reversed_gru eng
    # rosrun dialogue_system train.py no no reversed_gru eng

    # rosrun dialogue_system train.py am emb stacked_lstm eng
    # rosrun dialogue_system train.py am no stacked_lstm eng
    # rosrun dialogue_system train.py no emb stacked_lstm eng
    # rosrun dialogue_system train.py no no stacked_lstm eng

    # rosrun dialogue_system train.py am emb stacked_gru eng
    # rosrun dialogue_system train.py am no stacked_gru eng
    # rosrun dialogue_system train.py no emb stacked_gru eng
    # rosrun dialogue_system train.py no no stacked_gru eng

    # rosrun dialogue_system train.py am emb bidirectional_lstm eng
    # rosrun dialogue_system train.py am no bidirectional_lstm eng
    # rosrun dialogue_system train.py no emb bidirectional_lstm eng
    # rosrun dialogue_system train.py no no bidirectional_lstm eng

    # rosrun dialogue_system train.py am emb bidirectional_gru eng
    # rosrun dialogue_system train.py am no bidirectional_gru eng
    # rosrun dialogue_system train.py no emb bidirectional_gru eng
    # rosrun dialogue_system train.py no no bidirectional_gru eng


    # kor ################################################################################
    rosrun dialogue_system train.py am emb lstm kor
    rosrun dialogue_system train.py am no lstm kor
    rosrun dialogue_system train.py no emb lstm kor
    rosrun dialogue_system train.py no no lstm kor
  
    rosrun dialogue_system train.py am emb gru kor
    rosrun dialogue_system train.py am no gru kor
    rosrun dialogue_system train.py no emb gru kor
    rosrun dialogue_system train.py no no gru kor

    rosrun dialogue_system train.py am emb reversed_lstm kor
    rosrun dialogue_system train.py am no reversed_lstm kor
    rosrun dialogue_system train.py no emb reversed_lstm kor
    rosrun dialogue_system train.py no no reversed_lstm kor

    rosrun dialogue_system train.py am emb reversed_gru kor
    rosrun dialogue_system train.py am no reversed_gru kor
    rosrun dialogue_system train.py no emb reversed_gru kor
    rosrun dialogue_system train.py no no reversed_gru kor

    rosrun dialogue_system train.py am emb stacked_lstm kor
    rosrun dialogue_system train.py am no stacked_lstm kor
    rosrun dialogue_system train.py no emb stacked_lstm kor
    rosrun dialogue_system train.py no no stacked_lstm kor

    rosrun dialogue_system train.py am emb stacked_gru kor
    rosrun dialogue_system train.py am no stacked_gru kor
    rosrun dialogue_system train.py no emb stacked_gru kor
    rosrun dialogue_system train.py no no stacked_gru kor

    rosrun dialogue_system train.py am emb bidirectional_lstm kor
    rosrun dialogue_system train.py am no bidirectional_lstm kor
    rosrun dialogue_system train.py no emb bidirectional_lstm kor
    rosrun dialogue_system train.py no no bidirectional_lstm kor

    rosrun dialogue_system train.py am emb bidirectional_gru kor
    rosrun dialogue_system train.py am no bidirectional_gru kor
    rosrun dialogue_system train.py no emb bidirectional_gru kor
    rosrun dialogue_system train.py no no bidirectional_gru kor

    ((counter++))
done
echo training finished

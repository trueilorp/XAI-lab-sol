#!/bin/bash

# Map sizes and training steps

MAP_SIZE="5x5" #small debug map, once your code is ready run your experiments at least on the 6x6 map

STEPS=100000 # this is way too low for a real training, once your code is ready use at least 300k steps

python train_script.py --map $MAP_SIZE --steps $STEPS
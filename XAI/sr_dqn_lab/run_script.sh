#!/bin/bash

# Map sizes and training steps
MAP_SIZE="5x5" #small debug map, once your code is ready run your experiments at least on the 6x6 map
STEPS=100000 # increase this number to let the training converge

python train_script.py --map $MAP_SIZE --steps $STEPS
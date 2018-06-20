#!/bin/bash
time python3 src/train.py	--training_data		data/train.data		\
							--training_caption	data/train.caption	\
							--model_path		models/my_model	

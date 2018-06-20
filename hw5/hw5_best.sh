#!/bin/bash
python3 src/hw5_train.py --training_data	data/train.csv	\
						 --testing_data		$1   \
						 --users_data		$4	\
						 --movies_data		$3 \
						 --model_path		models/best.h5 \
						 --history			history/123normalized_his.pkl	\
						 --output			$2

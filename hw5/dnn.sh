#!/bin/bash
python3 src/dnn.py		 --training_data	data/train.csv	\
						 --testing_data		data/test.csv   \
						 --users_data		data/users.csv	\
						 --movies_data		data/movies.csv \
						 --model_path		models/dnn_model.h5 \
						 --history			history/123normalized_his.pkl	\
						 --output			prediction.csv

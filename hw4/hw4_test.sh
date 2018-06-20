#!/bin/bash
python3 src/hw4_keras_test.py --model			models/my_model \
							  --word2vec_model	models/unlabeled_word2vec_model \
							  --test_data		$1 \
							  --output			$2


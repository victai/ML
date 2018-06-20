#!/bin/bash
echo '-----training w2v-----'
time python3 src/word2vec_train.py	--w2v_model	models/model.w2v	\
							  	--data		data/merge_with_stopwords.txt \
								--sg		1	\
								--size		200
echo '-------testing-------'
time python3 src/test.py		--data			data/testing_data.csv	\
								--output		prediction.csv	\
								--no_stopwords	0	\
								--stopword_file	data/stop_words.txt	\
								--size			200		\
								--w2v_model		models/model.w2v
#no_stopwords = 1 表示去掉stopwords

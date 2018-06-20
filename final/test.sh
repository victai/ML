#!/bin/bash
echo '-------testing-------'
wget --output-document=model.w2v https://www.dropbox.com/s/wosoas2fcamcs7f/best.w2v?dl=1
time python3 src/test.py		--data			$1	\
								--output		$2	\
								--no_stopwords	0	\
								--stopword_file	data/stop_words.txt	\
								--size			200		\
								--w2v_model		model.w2v
#no_stopwords = 1 表示去掉stopwords

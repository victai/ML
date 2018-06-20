#!/bin/bash
while getopts ":a:" opt; do
	case $opt in
		a)
			echo "-a was triggered, Training Word2Vec....." >&2
			python3 src/preprocess.py \
				--labeled_path		data/training_label.txt \
			 	--unlabeled_path	data/training_nolabel.txt \
				--test_path			data/testing_data.txt	\
				--word2vec_model	models/unlabeled_word2vec_model
			read -n 1 -s -r -p "Press any key to continue"
			;;
		\?)
			echo "Invalid option: -$OPTARG" >&2
			exit 1
			;;
		:)
			echo "-a was triggered, Training Word2Vec....." >&2
			python3 src/preprocess.py \
				--labeled_path		data/training_label.txt \
			 	--unlabeled_path	data/training_nolabel.txt \
				--test_path			data/testing_data.txt	\
				--word2vec_model	models/unlabeled_word2vec_model
			read -n 1 -s -r -p "Press any key to continue"
			;;
	esac
done
#cat << block
#echo "============= Start Training =============="
python3 src/hw4_keras.py 	--labeled_path		$1 \
			 			 	--unlabeled_path	$2 \
						  	--model_save_path	models/my_model	\
							--word2vec_model	models/unlabeled_word2vec_model
#block
#read -n 1 -s -r -p "Press any key to continue"
#echo ""
#echo "============= Start Testing =============="
#time python3 src/hw4_keras_test.py --model			models/new_model \
#								   --word2vec_model models/unlabeled_word2vec_model \
#								   --test_data		data/testing_data.txt \
#								   --output			prediction.csv


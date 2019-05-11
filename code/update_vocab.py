import os
import numpy as np
import torch
import json
import re
import sys
from data import DatasetHandler
from vocab import *


def create_word2vec_vocab():
	
	dataset_methods = [getattr(DatasetHandler, method) for method in dir(DatasetHandler) 
					   if method.startswith("load") and callable(getattr(DatasetHandler, method))]
	print("Found the following dataset loading function: " + str(dataset_methods))
	dataset_list = list()
	for m in dataset_methods:
		dataset_list += list(m())
	print("Loaded " + str(len(dataset_list)) + " datasets")

	datasets_word_list = list()
	for d in dataset_list:
		datasets_word_list += d.get_word_list()

	if os.path.isfile("small_glove_words.txt"):
		old_glove = [l.strip() for l in open("small_glove_words.txt")]
		print("Found " + str(len(old_glove)) + " words in old GloVe embeddings")
	else:
		old_glove = []

	word_list = list(set(datasets_word_list + old_glove + ['<s>', '</s>', '<p>', 'UNK']))
	# Allow both with "-" and without "-" words to cover all possible preprocessing steps
	print("Created word list with " + str(len(word_list)) + " words. Checking for \"-\" confusion...")
	for word in word_list:
		if "-" in word:
			for w in word.split("-"):
				if len(w) >= 1 and w not in word_list:
					word_list.append(w)
	print("Number of unique words in all datasets: " + str(len(word_list)))

	voc = build_vocab(word_list)
	np_word_list = []
	with open('small_glove_words.txt', 'w') as f:
		# json.dump(voc, f)
		for key, val in voc.items():
			f.write(key + "\n")
			np_word_list.append(val)
	np_word_array = np.stack(np_word_list, axis=0)
	np.save('small_glove_embed.npy', np_word_array)


if __name__ == '__main__':
	create_word2vec_vocab()
	save_word2vec_as_GloVe()
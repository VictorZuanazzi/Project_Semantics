import os
import numpy as np
import torch
import json
import re
import sys
from random import shuffle


WORD2VEC_DICT = None
WORD2ID_DICT = None
WORDVEC_TENSOR = None

def load_word2vec_from_file(word_file="small_glove_words.txt", numpy_file="small_glove_embed.npy"):
	global WORD2VEC_DICT, WORD2ID_DICT, WORDVEC_TENSOR
	
	if WORD2VEC_DICT is None or WORD2ID_DICT is None or WORDVEC_TENSOR is None:
		
		word2vec = dict()
		word2id = dict()
		word_vecs = np.load(numpy_file)
		with open(word_file, "r") as f:
			for i, l in enumerate(f):
				word2vec[l.replace("\n","")] = word_vecs[i,:]
		index = 0
		for key, _ in word2vec.items():
			word2id[key] = index
			index += 1

		print("Loaded vocabulary of size " + str(word_vecs.shape[0]))
		WORD2VEC_DICT, WORD2ID_DICT, WORDVEC_TENSOR = word2vec, word2id, word_vecs

	return WORD2VEC_DICT, WORD2ID_DICT, WORDVEC_TENSOR


def save_word2vec_as_GloVe(output_file="small_glove_torchnlp.txt"):
	word2vec, word2id, word_vecs = load_word2vec_from_file()
	s = ""
	for key, val in word2vec.items():
		s += key + " " + " ".join([("%g" % (x)) for x in val]) + "\n"
	with open(output_file, "w") as f:
		f.write(s)


def build_vocab(word_list, glove_path='../glove.840B.300d.txt'):
	word2vec = {}
	num_ignored_words = 0
	num_missed_words = 0
	num_found_words = 0
	word_list = set(word_list)
	overall_num_words = len(word_list)
	with open(glove_path, "r") as f:
		lines = f.readlines()
		number_lines = len(lines)
		for i, line in enumerate(lines):
			# if debug_level() == 0:
			print("Processed %4.2f%% of the glove (found %4.2f%% of words yet)" % (100.0 * i / number_lines, 100.0 * num_found_words / overall_num_words), end="\r")
			if num_found_words == overall_num_words:
				break
			word, vec = line.split(' ', 1)
			if word in word_list:
				glove_vec = [float(x) for x in vec.split()]
				word2vec[word] = np.array(glove_vec)
				num_found_words += 1
			else:
				num_ignored_words += 1

	example_missed_words = list()
	for word in word_list:
		if word not in word2vec:
			num_missed_words += 1
			if num_missed_words < 30:
				example_missed_words.append(word)

	print("Created vocabulary with %i words. %i words were ignored from Glove, %i words were not found in embeddings." % (len(word2vec.keys()), num_ignored_words, num_missed_words))
	if num_missed_words > 0:
		print("Example missed words: " + " +++ ".join(example_missed_words))

	return word2vec


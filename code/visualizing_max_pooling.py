import torch 
import torch.nn as nn
import argparse
import random
import numpy as np
import math
import datetime
import os
import sys
import json
import pickle
import copy
import matplotlib.pyplot as plt
from glob import glob
from shutil import copyfile

from tensorboardX import SummaryWriter

from model import NLIModel
from data import load_SNLI_datasets, debug_level, set_debug_level, DatasetTemplate, SentData
from mutils import load_model, load_model_from_args, load_args, args_to_params


def retrieve_word_importance(model, dataset, batch_size=64, output_path="", overwrite=False):
	if model.model_type != NLIModel.BILSTM_MAX:
		print("[!] ERROR: Only support BiLSTM MAX models for word importance visualization.")
		return

	_, _, _, _, word2id, _ = load_SNLI_datasets(debug_dataset = False)
	number_words = len(list(word2id.keys()))
	word_occurences = np.zeros(shape=(number_words,), dtype=np.int32)
	word_max_index = np.zeros(shape=(number_words,model.model_params["embed_sent_dim"]), dtype=np.int32)

	np_file_path = os.path.join(output_path, "np_combined.npz")

	if not overwrite and os.path.isfile(np_file_path):
		loaded_np_file = np.load(np_file_path)
		word_occurences = loaded_np_file['occurences']
		word_max_index = loaded_np_file['indices']
	else:
		number_batches = int(math.ceil(dataset.get_num_examples()/batch_size))
		for i in range(number_batches):
			print("Processed %4.2f%% of dataset" % (i*100.0/number_batches), end="\r") 
			embeds, lengths, _ = dataset.get_batch(batch_size=batch_size, loop_dataset=False, toTorch=True)
			embeds = [embeds] if not isinstance(embeds, list) else embeds 
			lengths = [lengths] if not isinstance(lengths, list) else lengths
			for e, l in zip(embeds, lengths):
				_, pool_indices = model.encode_sentence(e, l, debug=True)
				pool_indices = pool_indices.data.cpu().numpy()
				word_id_embeds = e.data.cpu().numpy()
				for batch_index in range(word_id_embeds.shape[0]):
					word_occurences[word_id_embeds[batch_index]] += 1
					pooled_words = word_id_embeds[batch_index, pool_indices[batch_index]]
					word_max_index[pooled_words, range(word_max_index.shape[1])] += 1
		np.savez_compressed(np_file_path, indices=word_max_index, occurences=word_occurences)

	word_frequency = word_max_index / (1e-10 + word_occurences[:,None])
	word_importance = np.mean(word_frequency, axis=1)

	print_top_bottom(word_importance, word_occurences)
	# visualize_word_distribution(word_frequency, "<s>", file_path=os.path.join(output_path, "word_dist_s.pdf"))
	# visualize_word_distribution(word_frequency, ",", file_path=os.path.join(output_path, "word_dist_comma.pdf"))
	# visualize_word_distribution(word_frequency, "bikinis", file_path=os.path.join(output_path, "word_dist_bikinis.pdf"))
	# visualize_word_distribution(word_frequency, "as", file_path=os.path.join(output_path, "word_dist_as.pdf"))
	# visualize_word_distribution(word_frequency, "like", file_path=os.path.join(output_path, "word_dist_like.pdf"))
	export_sample_word_per_feature(word_frequency, word_occurences, N=25)
	return word_frequency, word_occurences


def print_top_bottom(word_importance, word_occurences, N=25):	
	_, _, _, _, word2id, _ = load_SNLI_datasets(debug_dataset = False)
	id2word = {v:k for k,v in word2id.items()}
	
	word_importance_sort_ascending = np.array([x for x in np.argsort(word_importance) if word_occurences[x] > 100])
	word_importance_sort_descending = word_importance_sort_ascending[::-1]
	print("="*50+"\nMost important words\n" + "="*50)
	for i in range(N):
		word_id = word_importance_sort_descending[i]
		print("Top %i: %s (%4.2f%% in %i sentences)" % (i, str(id2word[word_id]), 100.0 * word_importance[word_id], word_occurences[word_id]))
	print("="*50+"\nLeast important words\n" + "="*50)
	for i in range(N):
		word_id = word_importance_sort_ascending[i]
		print("Bottom %i: %s (%4.2f%% in %i sentences)" % (i, str(id2word[word_id]), 100.0 * word_importance[word_id], word_occurences[word_id]))


def visualize_single_sentence(model, sentence, file_path=None):
	_, _, _, _, word2id, _ = load_SNLI_datasets(debug_dataset = False)
	id2word = {v:k for k,v in word2id.items()}

	new_d = SentData(sentence=sentence, label=-1)
	new_d.translate_to_dict(word2id)
	dataset = DatasetTemplate("all")
	dataset.set_data_list([new_d])

	embeds, lengths, _ = dataset.get_batch(batch_size=1, loop_dataset=False, toTorch=True)
	_, pool_indices = model.encode_sentence(embeds, lengths, debug=True)
	pool_indices = pool_indices.data.cpu().numpy()[0]
	word_id_embeds = embeds.data.cpu().numpy()[0]

	word_index = range(word_id_embeds.shape[0])
	word_importance = [np.sum(i == pool_indices) * 100.0 / pool_indices.shape[0] for i in range(word_id_embeds.shape[0])]
	plt.xticks(word_index, [id2word[w] for w in word_id_embeds], rotation=45)
	plt.bar(word_index, word_importance)
	plt.ylabel("Proportion of pool indices")
	plt.title("Word importance for sample sentence")
	if file_path is None:
		plt.show()
	else:
		plt.savefig(file_path)
		plt.close()


def visualize_word_distribution(word_frequency, word, file_path=None):
	_, _, _, _, word2id, _ = load_SNLI_datasets(debug_dataset = False)
	word_id = word2id[word]

	word_dist = word_frequency[word_id]
	feature_index = range(word_frequency.shape[1])
	plt.bar(feature_index, word_dist * 100.0)
	plt.ylabel("Frequency of having the highest value")
	plt.xlabel("Features")
	plt.title("Feature pool distribution for the word \"%s\"" % (word))
	
	if file_path is None:
		plt.show()
	else:
		plt.savefig(file_path)
		plt.close()


def export_sample_word_per_feature(word_frequency, word_occurences, output_path="", N=10, min_occ=100):
	_, _, _, _, word2id, _ = load_SNLI_datasets(debug_dataset = False)
	id2word = {v:k for k,v in word2id.items()}

	word_frequency = word_frequency * (word_occurences[:,None] > min_occ) # Masking rare words
	s = "="*50+"\nWords to feature mapping\n" + "="*50+"\n"
	for i in range(word_frequency.shape[1]):
		print("Processing feature %i..." % (i+1), end="\r")
		most_popular_words = np.argsort(word_frequency[:,i])[::-1]
		s += "Feature %4i: [%s]\n" % (i, ", ".join(["\"%s\" (%4.2f%%)" % (id2word[most_popular_words[j]], word_frequency[most_popular_words[j], i]) for j in range(N)]))  

	with open(os.path.join(output_path, "feature_word_mapping.txt"), "w") as f:
		f.write(s)


def visualize_embeddings(word_frequency, word_occurences, tensorboard_writer, max_examples=4000):
	_, _, _, _, word2id, _ = load_SNLI_datasets(debug_dataset = False)
	id2word = {v:k for k,v in word2id.items()}

	most_frequent_words = np.argsort(word_occurences)[::-1]
	most_frequent_words = most_frequent_words[:max_examples]
	print("Using all words with a frequency of more than %i." % (word_occurences[most_frequent_words[-1]]))

	embeddings = word_frequency[most_frequent_words]
	labels = [id2word[w] for w in most_frequent_words]
	tensorboard_writer.add_embedding(embeddings, metadata=labels, tag="MaxPoolDistribution", global_step=None)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints are saved", type=str, required=True)
	parser.add_argument("--overwrite", help="Whether to re-create the distributions if they are already in the folder or not", action="store_true")
	args = parser.parse_args()
	model = load_model_from_args(load_args(args.checkpoint_path), args.checkpoint_path)

	train_dataset, val_dataset, test_dataset, _, _, _ = load_SNLI_datasets(debug_dataset = True)
	dataset = copy.deepcopy(test_dataset)
	dataset.set_data_list(train_dataset.data_list + val_dataset.data_list + test_dataset.data_list)

	if os.path.isfile(args.checkpoint_path):
		output_path = args.checkpoint_path.rsplit("/",1)[0]
	else:
		output_path = args.checkpoint_path
	word_freq, word_occ = retrieve_word_importance(model, dataset, output_path=output_path, overwrite=args.overwrite)
	# visualize_single_sentence(model, "Two women are sleeping in their bikinis on a beach .")
	# writer = SummaryWriter(args.checkpoint_path)
	# visualize_embeddings(word_freq, word_occ, writer)

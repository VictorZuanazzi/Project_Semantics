import os
import numpy as np
import torch
import json
import re
import sys
from random import shuffle
from vocab import load_word2vec_from_file
from model import get_device
import pandas as pd

# 0 => Full debug
# 1 => Reduced output
# 2 => No output at all (on cluster)
DEBUG_LEVEL = 0

def set_debug_level(level):
	global DEBUG_LEVEL
	DEBUG_LEVEL = level

def debug_level():
	global DEBUG_LEVEL
	return DEBUG_LEVEL


###############################
## Dataset class definitions ##
###############################

class DatasetHandler:

	SNLI_DATASETS = None
	SNLI_EXTRA_DATASETS = None
	MNLI_DATASETS = None
	SST_DATASETS = None
	VUA_DATASETS = None
	VUA_SEQ_DATASETS = None
	WIC_DATASETS = None
	POS_MNLI_DATASETS = None


	@staticmethod
	def _load_all_type_datasets(dataset_fun, debug_dataset=False, data_types=None, data_path=None, name=None):
		_, word2id_dict, _ = load_word2vec_from_file()
		dataset_list = list()
		if data_types is None:
			data_types = ['train' if not debug_dataset else 'dev', 'dev', 'test']
		elif debug_dataset:
			data_types[0] = data_types[1]
		for data_type in data_types:
			if data_path is None:
				dataset = dataset_fun(data_type, shuffle_data=('train' in data_type))
			else:
				dataset = dataset_fun(data_type, data_path=data_path, shuffle_data=('train' in data_type), name=name)
			dataset.set_vocabulary(word2id_dict)
			dataset.print_statistics()
			dataset_list.append(dataset)
		return dataset_list

	@staticmethod
	def load_SNLI_datasets(debug_dataset=False):
		if DatasetHandler.SNLI_DATASETS is None:
			DatasetHandler.SNLI_DATASETS = DatasetHandler._load_all_type_datasets(SNLIDataset, debug_dataset=debug_dataset)
		return DatasetHandler.SNLI_DATASETS[0], DatasetHandler.SNLI_DATASETS[1], DatasetHandler.SNLI_DATASETS[2]

	@staticmethod
	def load_SNLI_splitted_datasets():
		if DatasetHandler.SNLI_EXTRA_DATASETS is None:
			DatasetHandler.SNLI_EXTRA_DATASETS = DatasetHandler._load_all_type_datasets(SNLIDataset, data_types=['test_hard', 'test_easy']) 
		return DatasetHandler.SNLI_EXTRA_DATASETS[0], DatasetHandler.SNLI_EXTRA_DATASETS[1]
	
	@staticmethod
	def load_MultiNLI_datasets(debug_dataset=False):
		if DatasetHandler.MNLI_DATASETS is None:
			DatasetHandler.MNLI_DATASETS = DatasetHandler._load_all_type_datasets(SNLIDataset, data_path="../data/multinli_1.0", data_types=['train', 'dev.matched', 'dev.mismatched'], debug_dataset=debug_dataset, name="MultiNLI")
		return DatasetHandler.MNLI_DATASETS[0], DatasetHandler.MNLI_DATASETS[1], DatasetHandler.MNLI_DATASETS[2]

	@staticmethod
	def load_SST_datasets(debug_dataset=False):
		if DatasetHandler.SST_DATASETS is None:
			DatasetHandler.SST_DATASETS = DatasetHandler._load_all_type_datasets(SSTDataset, debug_dataset=debug_dataset)
		return DatasetHandler.SST_DATASETS[0], DatasetHandler.SST_DATASETS[1], DatasetHandler.SST_DATASETS[2]

	@staticmethod
	def load_VUA_datasets(debug_dataset=False):
		if DatasetHandler.VUA_DATASETS is None:
			DatasetHandler.VUA_DATASETS = DatasetHandler._load_all_type_datasets(VUADataset, debug_dataset=debug_dataset)
		return DatasetHandler.VUA_DATASETS[0], DatasetHandler.VUA_DATASETS[1], DatasetHandler.VUA_DATASETS[2]

	@staticmethod
	def load_VUAseq_datasets(debug_dataset=False):
		if DatasetHandler.VUA_SEQ_DATASETS is None:
			DatasetHandler.VUA_SEQ_DATASETS = DatasetHandler._load_all_type_datasets(VUASeqDataset, debug_dataset=debug_dataset)
		return DatasetHandler.VUA_SEQ_DATASETS[0], DatasetHandler.VUA_SEQ_DATASETS[1], DatasetHandler.VUA_SEQ_DATASETS[2]

	@staticmethod
	def load_WiC_datasets(debug_dataset=False):
		if DatasetHandler.WIC_DATASETS is None:
			DatasetHandler.WIC_DATASETS = DatasetHandler._load_all_type_datasets(WiCDataset, debug_dataset=debug_dataset)
		return DatasetHandler.WIC_DATASETS[0], DatasetHandler.WIC_DATASETS[1], DatasetHandler.WIC_DATASETS[2]

	@staticmethod
	def load_POS_MNLI_datasets(debug_dataset=False):
		if DatasetHandler.POS_MNLI_DATASETS is None:
			DatasetHandler.POS_MNLI_DATASETS = DatasetHandler._load_all_type_datasets(POSDataset, data_path="../data/POS/", data_types=['train_mnli', 'dev_mnli', 'test_mnli'], name="POS_MNLI", debug_dataset=debug_dataset)
		return DatasetHandler.POS_MNLI_DATASETS[0], DatasetHandler.POS_MNLI_DATASETS[1], DatasetHandler.POS_MNLI_DATASETS[2]


class DatasetTemplate:

	def __init__(self, data_type="train", shuffle_data=True, name=""):
		self.data_type = data_type
		self.shuffle_data = shuffle_data
		self.set_data_list(list())
		self.label_dict = dict()
		self.num_invalids = 0
		self.dataset_name = name

	def set_data_list(self, new_data):
		self.data_list = new_data
		self.example_index = 0
		self.perm_indices = list(range(len(self.data_list)))
		if self.shuffle_data:
			shuffle(self.perm_indices)

	def _get_next_example(self):
		exmp = self.data_list[self.perm_indices[self.example_index]]
		self.example_index += 1
		if self.example_index >= len(self.perm_indices):
			if self.shuffle_data:
				shuffle(self.perm_indices)
			self.example_index = 0
		return exmp

	@staticmethod
	def sents_to_Tensors(batch_stacked_sents, batch_labels=None, toTorch=False):
		lengths = []
		embeds = []
		for batch_sents in batch_stacked_sents:
			lengths_sents = np.array([x.shape[0] for x in batch_sents])
			max_len = np.max(lengths_sents)
			sent_embeds = np.zeros((len(batch_sents), max_len), dtype=np.int32)
			for s_index, sent in enumerate(batch_sents):
				sent_embeds[s_index, :sent.shape[0]] = sent
			if toTorch:
				sent_embeds = torch.LongTensor(sent_embeds).to(get_device())
				lengths_sents = torch.LongTensor(lengths_sents).to(get_device())
			lengths.append(lengths_sents)
			embeds.append(sent_embeds)
		if batch_labels is not None and toTorch:
			if isinstance(batch_labels[0], (list, np.ndarray)):
				padded_labels = np.zeros((len(batch_labels), max_len), dtype=np.int32) - 1
				for label_index, lab in enumerate(batch_labels):
					padded_labels[label_index, :lab.shape[0]] = np.array(lab)
				batch_labels = padded_labels
			batch_labels = torch.LongTensor(np.array(batch_labels)).to(get_device())
		return embeds, lengths, batch_labels
	
	@staticmethod
	def object_to_Tensors(some_object, toTorch=False):
		"""wraps the given object in a torch tensor or numpy array.
		inputs:
			some_object (list(int), tuple(int), int), objec to be wraped.
			toTorch (bool), if True wraps some_object with a torch tensor, if False
				wraps it with an numpy array
		output:
			some_boject (torch.LongTensor(some_object) or np.array(some_object)), 
				if cuda is available, torch tensor is returned in cuda.
		"""
		if toTorch:
			some_object = torch.LongTensor(some_object).to(get_device())
		else:
			some_object = np.array(some_object)
			
		return some_object

	def get_num_examples(self):
		return len(self.data_list)

	def get_word_list(self):
		all_words = dict()
		for i, data in enumerate(self.data_list):
			if debug_level() == 0:
				print("Processed %4.2f%% of the dataset %s" % (100.0 * i / len(self.data_list), self.dataset_name), end="\r")
			if isinstance(data, NLIData):
				data_words = data.premise_words + data.hypothesis_words
			elif isinstance(data, WiCData):
				data_words = data.s1_words + data.s2_words
			else:
				data_words = data.sent_words
			for w in data_words:
				if w not in all_words:
					all_words[w] = ''
		all_words = list(all_words.keys())
		print("Found " + str(len(all_words)) + " unique words")
		return all_words

	def set_vocabulary(self, word2vec):
		missing_words = 0
		overall_words = 0
		for data in self.data_list:
			data.translate_to_dict(word2vec)
			mw, ow = data.number_words_not_in_dict(word2vec)
			missing_words += mw 
			overall_words += ow 
		print("Amount of missing words: %4.2f%%" % (100.0 * missing_words / overall_words))

	def get_batch(self, batch_size, loop_dataset=True, toTorch=False):
		# Default: assume that dataset entries contain object of SentData
		if not loop_dataset:
			batch_size = min(batch_size, len(self.perm_indices) - self.example_index)
		batch_sents = []
		batch_labels = []
		for _ in range(batch_size):
			data = self._get_next_example()
			batch_sents.append(data.sent_vocab)
			batch_labels.append(data.label)
		embeds, lengths, labels = DatasetTemplate.sents_to_Tensors([batch_sents], batch_labels=batch_labels, toTorch=toTorch)
		return (embeds[0], lengths[0], labels)

	def get_num_classes(self):
		c = 0
		for key, val in self.label_dict.items():
			if val >= 0:
				c += 1
		return c

	def add_label_explanation(self, label_dict):
		# The keys should be the labels, the explanation strings
		if isinstance(list(label_dict.keys())[0], str) and not isinstance(list(label_dict.values())[0], str):
			label_dict = {v: k for k, v in label_dict.items()}
		self.label_dict = label_dict

	def label_to_string(self, label):
		if label in self.label_dict:
			return self.label_dict[label]
		else:
			return str(label)

	def print_statistics(self):
		print("="*50)
		print("Dataset statistics " + ((self.dataset_name + " ") if self.dataset_name is not None else "") + self.data_type)
		print("-"*50)
		print("Number of examples: " + str(len(self.data_list)))
		if len(self.data_list) > 0 and isinstance(self.data_list[0].label, (list, np.ndarray)):
			print("Number of token-level labels: " + str(sum([d.label.shape[0] for d in self.data_list])))
			if len(self.data_list) < 30000:
				print("Labelwise amount:")
				label_list = [l for d in self.data_list for l in (d.label if len(d.label.shape) == 1 else d.label[:,0])]
				for key, val in self.label_dict.items():
					print("\t- " + val + ": " + str(label_list.count(key)))
		else:
			print("Labelwise amount:")
			for key, val in self.label_dict.items():
				print("\t- " + val + ": " + str(sum([d.label == key for d in self.data_list])))
		print("Number of invalid examples: " + str(self.num_invalids))
		print("="*50)


class SNLIDataset(DatasetTemplate):

	# Data type either train, dev or test
	def __init__(self, data_type, data_path="../data/snli_1.0", add_suffix=True, shuffle_data=True, name="SNLI"):
		super(SNLIDataset, self).__init__(data_type, shuffle_data, name=name)
		if data_path is not None:
			self.load_data(data_path, data_type)
		else:
			self.data_list = list()
		super().set_data_list(self.data_list)
		super().add_label_explanation(NLIData.LABEL_LIST)

	def load_data(self, data_path, data_type):
		self.data_list = list()
		self.num_invalids = 0
		s1 = [line.rstrip() for line in open(data_path + "/s1." + data_type, 'r')]
		s2 = [line.rstrip() for line in open(data_path + "/s2." + data_type, 'r')]
		labels = [NLIData.LABEL_LIST[line.rstrip('\n')] for line in open(data_path + "/labels." + data_type, 'r')]
		
		i = 0
		for prem, hyp, lab in zip(s1, s2, labels):
			if debug_level() == 0:
				print("Read %4.2f%% of the dataset %s" % (100.0 * i / len(s1), self.dataset_name), end="\r")
			i += 1
			if lab == -1:
				self.num_invalids += 1
				continue
			d = NLIData(premise = prem, hypothesis = hyp, label = lab)
			self.data_list.append(d)

	def get_batch(self, batch_size, loop_dataset=True, toTorch=False):
		# Output sentences with dimensions (bsize, max_len)
		if not loop_dataset:
			batch_size = min(batch_size, len(self.perm_indices) - self.example_index)
		batch_s1 = []
		batch_s2 = []
		batch_labels = []
		for _ in range(batch_size):
			data = self._get_next_example()
			batch_s1.append(data.premise_vocab)
			batch_s2.append(data.hypothesis_vocab)
			batch_labels.append(data.label)
			
		return DatasetTemplate.sents_to_Tensors([batch_s1, batch_s2], batch_labels=batch_labels, toTorch=toTorch)


class SSTDataset(DatasetTemplate):

	LABEL_LIST = {
		0 : "Negative",
		1 : "Positive"
	}

	# Data type either train, dev or test
	def __init__(self, data_type, data_path="../data/SST", add_suffix=True, shuffle_data=True):
		super(SSTDataset, self).__init__(data_type, shuffle_data, name="SST")
		if data_path is not None:
			self.load_data(data_path, data_type)
		else:
			self.data_list = list()
		super().set_data_list(self.data_list)
		super().add_label_explanation(SSTDataset.LABEL_LIST)

	def load_data(self, data_path, data_type):
		self.data_list = list()
		self.num_invalids = 0
		filepath = os.path.join(data_path, data_type + ".txt")
		with open(filepath, mode="r", encoding="utf-8") as f:
			for line in f:
				sent = line.strip().replace("\\","")
				tokens = re.sub(r"\([0-9] |\)", "", sent).split()
				label = int(sent[1])
				if label == 2:
					self.num_invalids += 1
					continue
				label = 0 if label < 2 else 1
				d = SentData(sentence=" ".join(tokens), label=label)
				self.data_list.append(d)


class POSDataset(DatasetTemplate):

	def __init__(self, data_type, data_path="../data/POS/", shuffle_data=True, name="POS"):
		super(POSDataset, self).__init__(data_type, shuffle_data, name=name)
		if data_path is not None:
			self.load_data(data_path, data_type)
		else:
			self.data_list = list()
		super().set_data_list(self.data_list)
		super().add_label_explanation(POSData.LABEL_LIST)


	def load_data(self, data_path, data_type):
		with open(os.path.join(data_path, "sents." + data_type), mode="r", encoding="utf-8") as f:
			sentences = f.readlines()
		with open(os.path.join(data_path, "labels." + data_type), mode="r", encoding="utf-8") as f:
			labels = f.readlines()
		self.data_list = list()
		for i, sent, lab in zip(range(len(sentences)), sentences, labels):
			if debug_level() == 0:
				print("Read %4.2f%% of the dataset %s" % (100.0 * i / len(sentences), self.dataset_name), end="\r")
			new_d = POSData(sentence=sent.replace("\n",""), pos_tags=[l for l in lab.replace("\n","").split(" ") if len(l)>0])
			self.data_list.append(new_d)


	def export_to_file(self, output_dir, suffix):
		pure_sentences = "\n".join([" ".join(d.sent_words) for d in self.data_list])
		pos_tags = "\n".join([" ".join([POSData.label_to_string(d_ind) for d_ind in d.label_words]) for d in self.data_list])
		with open(os.path.join(output_dir, "sents." + suffix), "w", encoding="utf-8") as f:
			f.write(pure_sentences)
		with open(os.path.join(output_dir, "labels." + suffix), "w", encoding="utf-8") as f:
			f.write(pos_tags)
		print("Testing reading the exported dataset...")
		self.load_data(data_path=output_dir, data_type=suffix)
		print("Successfully passed test")


class VUADataset(DatasetTemplate):

	# Data type either train, dev or test
	def __init__(self, data_type, data_path="../data/VUA/", shuffle_data=True):
		"""Initializes the VUA dataset.
		inputs:
		data_type: (srt), chose between the datastes 'train', 'dev', 'test'.
		data_path: (str), path to the directory of the VUA dataset.
		shuffle_data: (bool), True for shuffling the data, False not to.
		"""
		super(VUADataset, self).__init__(data_type, shuffle_data, name="VUA")
			
		if data_path is not None: 
			#load the data from file
			self.load_data(data_path, data_type)
		else:
			#empty data_list if no path is specified
			self.data_list == list()
		
		super().set_data_list(self.data_list)
		super().add_label_explanation(VUAData.LABEL_LIST)

	def load_data(self, data_path, data_type):
		"""loads the data as intances of the class VUAData.
		input:
			data_path: (str), path to the directory of the VUA dataset.
			data_type: (srt), chose between the datastes 'train', 'dev', 'test'.
		output:
			data_list: (list(VUAData)) a list of datapoints in as instances of
			the class VUAData."""
			
		self.data_list = list()
		self.num_invalids = 0
		
		#maps data_type to file name
		file = {"train": "VUA_formatted_train_augmented.csv",
						"dev": "VUA_formatted_val.csv", 
						"test": "VUA_formatted_test.csv"}
		word_index_string = {"train": "word_idx",
							 "dev": "verb_idx",
							 "test": "verb_idx"}
		
		#reads the wanted data
		df_data = pd.read_csv(data_path + file.get(data_type, "train"),
							  encoding = 'latin-1')
		
		for i in df_data.index.tolist():
			if debug_level() == 0:
				print("Read %4.2f%% of the dataset %s" % (100.0 * i / len(df_data), self.dataset_name), end="\r")
			
			#reads the relevant parts of the dataset
			sentence = df_data.at[i, "sentence"]
			verb_position = df_data.at[i, word_index_string[data_type]]
			label = df_data.at[i, "label"]
			
			#initializes the data as an instance of the class VUAData
			d = VUAData(sentence, verb_position, label)
			
			#appends everything in a beautiful list.
			self.data_list.append(d)

	def get_batch(self, batch_size, loop_dataset=True, toTorch=False):
		"""get a batch of examples from VUAData
		input:
			batch_size: (int), the number of datapoints in a batch,
			loop_dataset: (bool), when False it ensures the batch size over all
				batches. When True it is possible that the last batch of the 
				epoch has fewer examples.
			toTorch: (bool), if True the data is wraped in a torch tensor, if 
				False numpy arrays are used instead.
		output:
			outputs of DatasetTemplate.sents_to_Tensors:
				embeds: (np.array or torch.LongTensor), embeddings for the words
					in the sentences.
				lengths: (np.array or torch.LongTensor), the length of each 
					sentence of the batch.
				batch_labels:(np.array or torch.LongTensor), the labels of each
					sentence.
			batch_verb_p: (np.array or torch.LongTensor), indicate the position
				of the verb of interest.
		"""
		# Output sentences with dimensions (bsize, max_len)
		if not loop_dataset:
			batch_size = min(batch_size, len(self.perm_indices) - self.example_index)
		
		batch_sentence = []
		batch_verb_p = []
		batch_labels = []
		for _ in range(batch_size):
			
			data = self._get_next_example()
			
			batch_sentence.append(data.sent_vocab)
			batch_verb_p.append(data.verb_position)
			batch_labels.append(data.label)
		
		#converts batch_verb_p to torch or numpy
		batch_verb_p = DatasetTemplate.object_to_Tensors(batch_verb_p, toTorch=toTorch)
		
		#get the embeds, lengtghs and labels
		embeds, lengths, batch_labels = DatasetTemplate.sents_to_Tensors([batch_sentence], 
												batch_labels=batch_labels, 
												toTorch=toTorch)
		
		return embeds[0], lengths[0], batch_labels, batch_verb_p
	

class VUASeqDataset(DatasetTemplate):

	# Data type either train, dev or test
	def __init__(self, data_type, data_path="../data/VUAsequence/", shuffle_data=True):
		"""Initializes the VUA sequence dataset.
		inputs:
		data_type: (srt), chose between the datastes 'train', 'dev', 'test'.
		data_path: (str), path to the directory of the VUA sequence dataset.
		shuffle_data: (bool), True for shuffling the data, False not to.
		"""
		super(VUASeqDataset, self).__init__(data_type, shuffle_data, name="VUA Sequence")
			
		if data_path is not None: 
			#load the data from file
			self.load_data(data_path, data_type)
		else:
			#empty data_list if no path is specified
			self.data_list == list()
		
		super().set_data_list(self.data_list)
		super().add_label_explanation(VUASeqData.LABEL_LIST)

	def load_data(self, data_path, data_type):
		"""loads the data as intances of the class VUASeqData.
		input:
			data_path: (str), path to the directory of the VUA dataset.
			data_type: (srt), chose between the datastes 'train', 'dev', 'test'.
		output:
			data_list: (list(VUASeqData)) a list of datapoints in as instances of
			the class VUASeqData."""
			
		self.data_list = list()
		self.num_invalids = 0
		
		#maps data_type to file name
		file = {"train": "VUA_seq_formatted_train.csv",
				"dev": "VUA_seq_formatted_val.csv", 
				"test": "VUA_seq_formatted_test.csv"}
		
		#reads the wanted data
		df_data = pd.read_csv(data_path + file.get(data_type, "train"),
							  encoding = 'latin-1')
		
		for i in df_data.index.tolist():
			if debug_level() == 0:
				print("Read %4.2f%% of the dataset %s" % (100.0 * i / len(df_data), self.dataset_name), end="\r")
			
			#reads the relevant parts of the dataset
			sentence = df_data.at[i, "sentence"]
			pos = eval(df_data.at[i, "pos_seq"])
			label = eval(df_data.at[i, "label_seq"])
			
			#initializes the data as an instance of the class VUASeqData
			d = VUASeqData(sentence, pos, label)
			
			#appends everything in a beautiful list.
			self.data_list.append(d)

	def get_batch(self, batch_size, loop_dataset=True, toTorch=False):
		"""get a batch of examples from VUASeqData
		input:
			batch_size: (int), the number of datapoints in a batch,
			loop_dataset: (bool), when False it ensures the batch size over all
				batches. When True it is possible that the last batch of the 
				epoch has fewer examples.
			toTorch: (bool), if True the data is wraped in a torch tensor, if 
				False numpy arrays are used instead.
		output:
			outputs of DatasetTemplate.sents_to_Tensors:
				embeds: (np.array or torch.LongTensor), embeddings for the words
					in the sentences.
				lengths: (np.array or torch.LongTensor), the length of each 
					sentence of the batch.
				batch_labels:(np.array or torch.LongTensor), the labels of each
					sentence.
		"""
		# Output sentences with dimensions (bsize, max_len)
		if not loop_dataset:
			batch_size = min(batch_size, len(self.perm_indices) - self.example_index)
		
		batch_sentence = []
		batch_labels = []
		for _ in range(batch_size):
			
			data = self._get_next_example()
			sent_vocab, label = data.get_data()
			batch_sentence.append(sent_vocab)
			batch_labels.append(label)
		
		#get the embeds, lengtghs and labels
		embeds, lengths, batch_labels = DatasetTemplate.sents_to_Tensors([batch_sentence], 
												batch_labels=batch_labels, 
												toTorch=toTorch)
		
		return embeds[0], lengths[0], batch_labels


class WiCDataset(DatasetTemplate):

	# Data type either train, dev or test
	def __init__(self, data_type, data_path="../data/WiC_dataset/", shuffle_data=True):
		"""Initializes the Word in Context dataset.
		inputs:
		data_type: (srt), chose between the datastes 'train', 'dev', 'test'.
		data_path: (str), path to the directory of the WiC dataset.
		shuffle_data: (bool), True for shuffling the data, False not to.
		"""
		super(WiCDataset, self).__init__(data_type, shuffle_data, name="WiC")
			
		if data_path is not None: 
			#load the data from file
			self.load_data(data_path, data_type)
		else:
			#empty data_list if no path is specified
			self.data_list == list()
		
		super().set_data_list(self.data_list)
		super().add_label_explanation(WiCData.LABEL_LIST)

	def load_data(self, data_path, data_type):
		"""loads the data as intances of the class WiCData.
		input:
			data_path: (str), path to the directory of the WiC dataset.
			data_type: (srt), chose between the datastes 'train', 'dev', 'test'.
		output:
			data_list: (list(WiCData)) a list of datapoints in as instances of
			the class WiCData."""
			
		self.data_list = list()
		self.num_invalids = 0
		
		#maps data_type to file name
		data_name = {"train": "train/train.data.txt",
					 "dev": "dev/dev.data.txt", 
					 "test": "test/test.data.txt"}
		
		label_name ={"train": "train/train.gold.txt",
					 "dev": "dev/dev.gold.txt", 
					 "test": None}
   
		#reads the wanted data
		data_file = open(data_path + data_name[data_type], 
						 "r", 
						 encoding="utf8")
		#parse the file
		data_lines = data_file.read()
		data_lines = data_lines.split("\n")
		data_line = [l.split("\t") for l in data_lines]
		data_file.close()
		
		if not data_type == "test":
			#test does not have labels
			label_file = open(data_path + label_name[data_type], 
							  "r", 
							  encoding="utf8")
			label_lines = label_file.read()
			label_line = label_lines.split("\n")
			#converts
			label = [WiCData.LABEL_LIST[l] for l in label_line]
			label_file.close()
		else:
			#unknown labels are given when test set is loaded.
			label = [WiCData.LABEL_LIST[""]]*len(data_line)
		
		for i, data in enumerate(data_line):
			
			if debug_level() == 0:
				print("Read %4.2f%% of the dataset %s" % (100.0 * i / len(data_line), self.dataset_name), end="\r")
			 
			if len(data) != 5:
				#skip sequences that are not in the correct format
				continue
			
			#gets the sentences 
			s1 = data[3]
			s2 = data[4]
			
			#get the position fo the word in each sentence
			p1, p2 = tuple(data[2].split("-"))
			p1 = int(p1)
			p2 = int(p2)
			
			#get the word of interest
			word = data[0]
			pos = data[1]
			
			#initializes the data as an instance of the class WiCData
			d = WiCData(word, pos, s1, s2, p1, p2, label[i])
			
			#appends everything in a beautiful list.
			self.data_list.append(d)

	def get_batch(self, batch_size, loop_dataset=True, toTorch=False):
		"""get a batch of examples from WiCData
		input:
			batch_size: (int), the number of datapoints in a batch,
			loop_dataset: (bool), when False it ensures the batch size over all
				batches. When True it is possible that the last batch of the 
				epoch has fewer examples.
			toTorch: (bool), if True the data is wraped in a torch tensor, if 
				False numpy arrays are used instead.
		output:
			outputs of DatasetTemplate.sents_to_Tensors:
				embeds: (np.array or torch.LongTensor), embeddings for the words
					in the sentences with dimensions (batch_size, max_len).
				lengths: (np.array or torch.LongTensor), the length of each 
					sentence of the batch.
				batch_labels:(np.array or torch.LongTensor), the labels of each
					sentence.
			batch_p1, batch_p2: (np.array or torch.LongTensor), indicate the 
				position of the word of interest.
		"""
		if not loop_dataset:
			batch_size = min(batch_size, len(self.perm_indices) - self.example_index)
		
		batch_s1 = []
		batch_s2 = []
		batch_p1 = []
		batch_p2 = []
		batch_labels = []
		
		for _ in range(batch_size):
			
			data = self._get_next_example()
			
			batch_s1.append(data.s1_vocab)
			batch_s2.append(data.s2_vocab)
			batch_p1.append(data.p1)
			batch_p2.append(data.p2)
			batch_labels.append(data.label)
			
		#converts batch_pX to torch or numpy
		batch_p1 = DatasetTemplate.object_to_Tensors(batch_p1, toTorch=toTorch)
		batch_p2 = DatasetTemplate.object_to_Tensors(batch_p2, toTorch=toTorch)
		
		#get the embeds, lengtghs and labels
		embeds, lengths, batch_labels = DatasetTemplate.sents_to_Tensors([batch_s1, batch_s2],
												batch_labels=batch_labels, 
												toTorch=toTorch)
		
		return embeds, lengths, batch_labels, batch_p1, batch_p2  


############################################
## DATA OBJECT CLASSES FOR CLASSIFICATION ##
############################################

class SentData:

	def __init__(self, sentence, label=None):
		self.sent_words = SentData._preprocess_sentence(sentence)
		self.sent_vocab = None
		self.label = label

	def translate_to_dict(self, word_dict):
		self.sent_vocab = SentData._sentence_to_dict(word_dict, self.sent_words)

	def number_words_not_in_dict(self, word_dict):
		missing_words = 0
		for w in self.sent_words:
			if w not in word_dict:
				missing_words += 1
		return missing_words, len(self.sent_words)

	@staticmethod
	def _preprocess_sentence(sent):
		sent_words = list(sent.lower().strip().split(" "))
		if "." in sent_words[-1] and len(sent_words[-1]) > 1:
			sent_words[-1] = sent_words[-1].replace(".","")
			sent_words.append(".")
		sent_words = [w for w in sent_words if len(w) > 0]
		for i in range(len(sent_words)):
			if len(sent_words[i]) > 1 and "." in sent_words[i]:
				sent_words[i] = sent_words[i].replace(".","")
		sent_words = [w for w in sent_words if len(w) > 0]
		return sent_words

	@staticmethod
	def _sentence_to_dict(word_dict, sent):
		vocab_words = list()
		vocab_words += [word_dict['<s>']]
		vocab_words += SentData._word_seq_to_dict(sent, word_dict)
		vocab_words += [word_dict['</s>']]
		vocab_words = np.array(vocab_words, dtype=np.int32)
		return vocab_words

	@staticmethod
	def _word_seq_to_dict(word_seq, word_dict):
		vocab_words = list()
		for w in word_seq:
			if len(w) <= 0:
				continue
			if w in word_dict:
				vocab_words.append(word_dict[w])
			elif "-" in w:
				vocab_words += SentData._word_seq_to_dict(w.split("-"), word_dict)
			elif "/" in w:
				vocab_words += SentData._word_seq_to_dict(w.split("/"), word_dict)
			else:
				subword = re.sub('\W+','', w)
				if subword in word_dict:
					vocab_words.append(word_dict[subword])
		return vocab_words


class NLIData:

	LABEL_LIST = {
		'-': -1,
		"neutral": 0, 
		"entailment": 1,
		"contradiction": 2
	}

	def __init__(self, premise, hypothesis, label):
		self.premise_words = SentData._preprocess_sentence(premise)
		self.hypothesis_words = SentData._preprocess_sentence(hypothesis)
		self.premise_vocab = None
		self.hypothesis_vocab = None
		self.label = label

	def translate_to_dict(self, word_dict):
		self.premise_vocab = SentData._sentence_to_dict(word_dict, self.premise_words)
		self.hypothesis_vocab = SentData._sentence_to_dict(word_dict, self.hypothesis_words)

	def number_words_not_in_dict(self, word_dict):
		missing_words = 0
		for w in (self.premise_words + self.hypothesis_words):
			if w not in word_dict:
				missing_words += 1
		return missing_words, (len(self.premise_words) + len(self.hypothesis_words))
		
	def get_data(self):
		return self.premise_vocab, self.hypothesis_vocab, self.label

	def get_premise(self):
		return " ".join(self.premise_words)

	def get_hypothesis(self):
		return " ".join(self.hypothesis_words)

	@staticmethod
	def label_to_string(label):
		for key, val in NLIData.LABEL_LIST.items():
			if val == label:
				return key


class VUAData(SentData):
	
	LABEL_LIST = {
		"metaphor": 1, 
		"literal": 0
	}

	def __init__(self, sentence, verb_position, label):
		super(VUAData, self).__init__(sentence, label)
		self.verb_position = verb_position
		
	def get_data(self):
		return self.sent_vocab, self.label

	def get_sentence(self):
		return " ".join(self.sent_words)

	@staticmethod
	def label_to_string(label):
		for key, val in VUAData.LABEL_LIST.items():
			if val == label:
				return key


class WiCData:
	
	LABEL_LIST = {
		"": -1, 
		"F": 0,
		"T": 1
	}

	def __init__(self, word, pos, s1, s2, p1, p2, label):
		
		self.s1_words = SentData._preprocess_sentence(s1)
		self.s2_words = SentData._preprocess_sentence(s2)
		self.s1_vocab = None 
		self.s2_vocab = None
		self.p1 = p1
		self.p2 = p2
		self.label = label
		self.word = word
		self.pos = pos

	def translate_to_dict(self, word_dict):
		self.s1_vocab = SentData._sentence_to_dict(word_dict, 
												   self.s1_words)
		self.s2_vocab = SentData._sentence_to_dict(word_dict, 
												   self.s2_words)

	def number_words_not_in_dict(self, word_dict):
		missing_words = 0
		for w in (self.s1_words + self.s2_words):
			if w not in word_dict:
				missing_words += 1
		return missing_words, (len(self.s1_words) + len(self.s2_words))
		
	def get_data(self):
		return self.s1_vocab, self.s2_vocab, self.label

	def get_s1(self):
		return " ".join(self.s1_words)
	
	def get_s2(self):
		return " ".join(self.s2_words)

	@staticmethod
	def label_to_string(label):
		for key, val in WiCData.LABEL_LIST.items():
			if val == label:
				return key


#######################################
## DATA OBJECT FOR SEQUENTAIL LABELS ##
#######################################

class SeqData:

	def __init__(self, sentence, label, default_label=None):
		self.sent_words, self.label_words = SeqData._preprocess_sentence(sentence, label, default_label=default_label)
		assert len(self.label_words) == len(self.sent_words), "Number of labels have to fit to number of words in the sentence. \n" + \
															  "Original sentence: \"%s\"\n" % (str(sentence)) + \
															  "Splitted sentence: \"%s\"\n" % (str(self.sent_words)) + \
															  "Label list: \"%s\"\n" % (str(self.label_words)) + \
															  "Length sentence: %i, Length labels: %i" % (len(self.sent_words), len(self.label_words))
		self.sent_vocab = None
		self.label = None

	def translate_to_dict(self, word_dict):
		self.sent_vocab, self.label = SeqData._sentence_to_dict(word_dict, self.sent_words, self.label_words)

	def number_words_not_in_dict(self, word_dict):
		missing_words = 0
		for w in (self.sent_words):
			if w not in word_dict:
				missing_words += 1
		return missing_words, len(self.sent_words)

	@staticmethod
	def _preprocess_sentence(sent, labels, default_label=None):
		sent_words = list(sent.lower().strip().split(" "))
		sent_words = [w for w in sent_words if len(w) > 0]
		if "." in sent_words[-1] and len(sent_words[-1]) > 1:
			sent_words[-1] = sent_words[-1].replace(".","")
			sent_words.append(".")
			sent_words = [w for w in sent_words if len(w) > 0]
			if len(sent_words) == len(labels) + 1 and default_label is not None:
				labels.append(default_label)
		
		for i in range(len(sent_words)):
			if len(sent_words[i]) > 1 and "." in sent_words[i]:
				sent_words[i] = sent_words[i].replace(".","")
		return sent_words, labels

	@staticmethod
	def _sentence_to_dict(word_dict, sent, labels):
		vocab_words = list()
		vocab_words += [word_dict['<s>']]
		vocab_words += SeqData._word_seq_to_dict(sent, word_dict)
		vocab_words += [word_dict['</s>']]
		vocab_words = np.array(vocab_words, dtype=np.int32)

		if isinstance(labels[0], list):
			labels = np.array([[-1]*len(labels[0])] + labels + [[-1]*len(labels[0])], dtype=np.int32)
		else:
			labels = np.array([-1] + labels + [-1], dtype=np.int32)

		if labels.shape[0] != vocab_words.shape[0]:
			print("Labels and vocab words do not fit for sentence " + str(sent))
			print("Label shape %i, vocab words shape %i" % (labels.shape[0], vocab_words.shape[0]))
			print("Before: labels %i, sentence %i " % (len(labels), len(sent)))

		return vocab_words, labels

	@staticmethod
	def _word_seq_to_dict(word_seq, word_dict):
		"""
		In difference to SentData, this processing strictly keeps the length of the sequence equal.
		Thus, we don't split words like "blue-shiny" but replace it by the unknown token.
		"""
		vocab_words = list()
		for w_index, w in enumerate(word_seq):
			if len(w) <= 0:
				print("[!] SKIPPING WORD")
				continue
			if w in word_dict:
				vocab_words.append(word_dict[w])
			else:
				subword = re.sub('\W+','', w)
				if subword in word_dict:
					vocab_words.append(word_dict[subword])
				else:
					vocab_words.append(word_dict['UNK'])
		return vocab_words

class POSData(SeqData):

	LABEL_LIST = {
		"ADJ": 0,
		"ADP": 1,
		"ADV": 2,
		"CCONJ": 3,
		"CONJ": 3,
		"DET": 4,
		"INTJ": 5,
		"NOUN": 6,
		"NUM": 7,
		"PART": 8,
		"PRT": 8,
		"PRON": 9,
		"PROPN": 10,
		"PUNCT": 11,
		".": 11,
		"SYM": 12,
		"VERB": 13,
		"X": -1
	}

	def __init__(self, sentence, pos_tags):
		super(POSData, self).__init__(sentence, [POSData.LABEL_LIST[p] for p in pos_tags])

	@staticmethod
	def label_to_string(label):
		for key, val in POSData.LABEL_LIST.items():
			if val == label:
				return key

	@staticmethod
	def num_classes():
		return max(list(POSData.LABEL_LIST.values())) + 1


class VUASeqData(SeqData):
	
	#it is called LABEL_LIST, but it is a dictionary.
	LABEL_LIST = {
		"metaphor": 1, 
		"literal": 0
	}

	def __init__(self, sentence, pos, label):
		super(VUASeqData, self).__init__(sentence, [[l, POSData.LABEL_LIST[p]] for l, p in zip(label, pos)], default_label=[-1, -1])
		
	def get_data(self, chose_label = "metaphor"):
		"""access to data and label.
		chose_label == 'metaphor' gives the sequence metaphor labels,
		chose_label == 'pos' gives the POS labels."""
		if chose_label == "metaphor":
			return self.sent_vocab, self.label[:,0]
		else: 
			return self.sent_vocab, self.label[:,1]

	def get_sentence(self):
		return " ".join(self.sent_words)

	@staticmethod
	def label_to_string(label):
		for key, val in VUASeqData.LABEL_LIST.items():
			if val == label:
				return key


if __name__ == "__main__":
	train_data, _, _ = DatasetHandler.load_VUAseq_datasets()
	train_data.print_statistics()
	batch = train_data.get_batch(4, toTorch=True)
	for e in batch:
		print(e)


import torch 
import torch.nn as nn
import argparse
import numpy as np
import math
from random import shuffle
import os
import sys
import matplotlib.pyplot as plt

from model import SimpleClassifier, NLIClassifier, ESIM_Head
from data import DatasetTemplate, DatasetHandler, debug_level
from vocab import get_id2word_dict


class TaskTemplate:

	def __init__(self, model, model_params, name, load_data=True):
		self.name = name 
		self.model = model
		self.model_params = model_params
		self.classifier_params = model_params[name + "_head"] if (name + "_head") in model_params else model_params
		self.classifier = None
		self.loss_module = None
		self.train_dataset = None 
		self.val_dataset = None 
		self.test_dataset = None
		if load_data:
			self._load_datasets()


	def _load_datasets(self):
		raise NotImplementedError


	def train_step(self, batch_size, loop_dataset=True):
		# Function to perform single step given the batch size; returns the loss and the accuracy of the batch
		raise NotImplementedError


	def _eval_batch(self, batch):
		raise NotImplementedError


	def eval(self, dataset=None, batch_size=64):
		# Default: if no dataset is specified, we use validation dataset
		if dataset is None:
			assert self.val_dataset is not None, "[!] ERROR: Validation dataset not loaded. Please load the dataset beforehand for evaluation."
			dataset = self.val_dataset

		self.model.eval()
		self.classifier.eval()
		# Prepare metrics
		number_batches = int(math.ceil(dataset.get_num_examples() * 1.0 / batch_size))
		label_list = []
		preds_list = []

		# Evaluation loop
		for batch_ind in range(number_batches):
			if debug_level() == 0:
				print("Evaluation process: %4.2f%%" % (100.0 * batch_ind / number_batches), end="\r")
			# Evaluate single batch
			batch = dataset.get_batch(batch_size, loop_dataset=False, toTorch=True)
			pred_labels, batch_labels = self._eval_batch(batch)
			preds_list += torch.squeeze(pred_labels).tolist()
			label_list += torch.squeeze(batch_labels).tolist()
		
		# Metric output
		preds_list = np.array(preds_list)
		label_list = np.array(label_list)
		accuracy = np.sum(preds_list == label_list) * 1.0 / preds_list.shape[0]
		detailed_acc = {"accuracy": accuracy, 
						"predictions": preds_list, 
						"labels": label_list,
						"class_scores": dict()}

		print("-"*75)
		print("Evaluation accuracy: %4.2f%%" % (accuracy * 100.0))
		print("Accuracy per class: ")
		for c in list(set(label_list)):
			TP = np.sum(np.logical_and(preds_list == c, label_list == c))
			FP = np.sum(np.logical_and(preds_list == c, label_list != c))
			FN = np.sum(np.logical_and(preds_list != c, label_list == c))
			recall = TP * 1.0 / max(1e-5, TP + FN) 
			precision = TP * 1.0 / max(1e-5, TP + FP)
			F1_score = 2.0 * TP / max(1e-5, 2 * TP + FP + FN)
			print("\t- Class %s: Recall=%4.2f%%, Precision=%4.2f%%, F1 score=%4.2f%%" % (dataset.label_to_string(c), recall, precision, F1_score))
			detailed_acc["class_scores"][dataset.label_to_string(c)] = {"recall": recall, "precision": precision, "f1": F1_score}
		print("-"*75)

		self.classifier.train()
		
		return accuracy, detailed_acc


	def dict_to_save(self):
		state_dict = {}
		if self.classifier is not None:
			state_dict[self.name + "_classifier"] = self.classifier.state_dict()
		return state_dict


	def load_from_dict(self, checkpoint_dict):
		if self.name + "_classifier" in checkpoint_dict:
			self.classifier.load_state_dict(checkpoint_dict[self.name + "_classifier"])
		else:
			print("[%] WARNING: State dict to load was passed without a entry for the classifier. Task: " + self.name)


	def get_parameters(self):
		return self.classifier.parameters()


	def add_to_summary(self, writer, iteration):
		pass


	@staticmethod
	def _create_CrossEntropyLoss():
		loss_module = nn.CrossEntropyLoss()
		if torch.cuda.is_available():
			loss_module = loss_module.cuda()
		return loss_module


####################################
## MULTITASK SAMPLER FOR TRAINING ##
####################################

class MultiTaskSampler:

	def __init__(self, tasks, multitask_params, batch_size):
		self.tasks = tasks 
		self.multitask_params = multitask_params
		self.batch_size = batch_size

		self.epoch_size = multitask_params["epoch_size"]
		self.separate_batches = multitask_params["batchwise"]
		# TODO: Implement anti curriculum learning
		self.anti_curriculum_learning = multitask_params["anti_curriculum_learning"]
		
		self.batch_index = 0
		self.batch_list = []
		print("Task frequency")
		for task_index, t in enumerate(tasks):
			tnum = int(self.epoch_size * multitask_params["freq"][t.name] * (self.batch_size if not self.separate_batches else 1))
			self.batch_list += [task_index] * tnum
			print("\t - %s: %i" % (t.name, tnum))
		shuffle(self.batch_list)

		# For keeping track of 
		self.loss_counters = np.zeros((len(self.tasks) + 1, 3), dtype=np.float32)


	def _get_next_batch_index(self):
		new_batch = self.batch_list[self.batch_index]
		self.batch_index += 1
		if self.batch_index >= len(self.batch_list):
			self.batch_index = 0
			shuffle(self.batch_list)
		return new_batch


	def sample_batch_loss(self, index_iter):

		if self.separate_batches:
			task_index = self._get_next_batch_index()
			loss, acc = self.tasks[task_index].train_step(self.batch_size)
			self._add_loss_to_record(task_index, loss, 1, acc=acc)
		else:
			loss = 0
			batch_indices = [self._get_next_batch_index() for _ in range(self.batch_size)]
			for task_index, t in enumerate(self.tasks):
				task_batch_size = sum([b == task_index for b in batch_indices])
				if task_batch_size <= 0:
					continue

				task_weight = (task_batch_size * 1.0 / self.batch_size)
				task_loss, task_acc = t.train_step(int(task_batch_size))
				task_loss = task_loss * task_weight
				loss += task_loss
				self._add_loss_to_record(task_index, task_loss, task_weight, acc=task_acc*task_weight)

		self._add_loss_to_record(-1, loss, 1)

		return loss


	def evaluate_all(self):
		print("Evaluation...")
		accuracy_dict = dict()
		for t in self.tasks:
			acc, detailed_acc = t.eval()
			print("Task " + t.name + ": %4.2f%%" % (acc*100.0))
			accuracy_dict[t.name] = detailed_acc
		return accuracy_dict


	def _add_loss_to_record(self, task_index, loss, weight, acc=None):
		self.loss_counters[task_index + 1, 0] += loss.item()
		self.loss_counters[task_index + 1, 1] += weight 
		if acc is not None:
			self.loss_counters[task_index + 1, 2] += acc.item()


	def get_average_metrics(self):
		loss_avg = self.loss_counters[:,0] / np.maximum(self.loss_counters[:,1], 1e-5)
		acc_avg = self.loss_counters[:,2] / np.maximum(self.loss_counters[:,1], 1e-5)
		return loss_avg, acc_avg


	def reset_loss_counter(self):
		self.loss_counters[:,:] = 0






#########################
## TASK SPECIFIC TASKS ##
#########################

class SNLITask(TaskTemplate):

	NAME = "Stanford_NLI"

	CLASSIFIER_INFERSENT = 0
	CLASSIFIER_ESIM = 1

	def __init__(self, model, model_params, load_data=True):
		super(SNLITask, self).__init__(model=model, model_params=model_params, load_data=load_data, name=SNLITask.NAME)
		self.classifier = self._create_classifier()
		self.loss_module = TaskTemplate._create_CrossEntropyLoss()
		if torch.cuda.is_available():
			self.classifier = self.classifier.cuda()
			self.loss_module = self.loss_module.cuda()


	def _create_classifier(self):
		if "model" not in self.classifier_params or self.classifier_params["model"] == SNLITask.CLASSIFIER_INFERSENT:
			return NLIClassifier(self.classifier_params)
		elif self.classifier_params["model"] == SNLITask.CLASSIFIER_ESIM:
			return ESIM_Head(self.classifier_params)
		else:
			print("[!] ERROR: Unknown classifier for SNLI Task: " + str(self.classifier_params["model"]) + \
				  "Supported options are: [" + ",".join([str(o) for o in [SNLITask.CLASSIFIER_INFERSENT, SNLITask.CLASSIFIER_ESIM]]) + "]")
			sys.exit(1)


	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = DatasetHandler.load_SNLI_datasets(debug_dataset=False)


	def train_step(self, batch_size, loop_dataset=True):
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
		embeds, lengths, batch_labels = self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True)
		out = self._forward_model(embeds, lengths, applySoftmax=False)

		loss = self.loss_module(out, batch_labels)
		_, pred_labels = torch.max(out, dim=-1)
		acc = torch.sum(pred_labels == batch_labels).float() / pred_labels.shape[-1]

		return loss, acc


	def _eval_batch(self, batch):
		embeds, lengths, batch_labels = batch
		preds = self._forward_model(embeds, lengths, applySoftmax=True)
		
		_, pred_labels = torch.max(preds, dim=-1)
		
		return pred_labels, batch_labels


	def _forward_model(self, embeds, lengths, applySoftmax=False):
		embed_s1 = self.model.encode_sentence(embeds[0], lengths[0], word_level=self.classifier.is_word_level())
		embed_s2 = self.model.encode_sentence(embeds[1], lengths[1], word_level=self.classifier.is_word_level())

		if not self.classifier.is_word_level():
			out = self.classifier(embed_s1, embed_s2, applySoftmax=applySoftmax)
		else:
			out = self.classifier(embed_s1, lengths[0], embed_s2, lengths[1], applySoftmax=applySoftmax)
		return out


	def add_to_summary(self, writer, iteration, num_examples=4):
		if isinstance(self.classifier, ESIM_Head):
			random_samples = np.random.randint(0, len(self.val_dataset.data_list), size=num_examples)
			random_data = [self.val_dataset.data_list[i] for i in random_samples]
			batch_prem = [data.premise_vocab for data in random_data]
			batch_hyp = [data.hypothesis_vocab for data in random_data]
			batch_labels = [data.label for data in random_data]
			embeds, lengths, _ = DatasetTemplate.sents_to_Tensors([batch_prem, batch_hyp], batch_labels=None, toTorch=True)
			with torch.no_grad():
				_ = self._forward_model(embeds, lengths, applySoftmax=False)
			prem_attention_map = self.classifier.last_prem_attention_map
			hyp_attention_map = self.classifier.last_hyp_attention_map
			id2word = get_id2word_dict()
			fig = plt.figure()
			figure_list = list()
			ncols = int(math.ceil(math.sqrt(num_examples)))
			for i in range(len(random_data)):
				for attention_map, main_w in zip([prem_attention_map, np.transpose(hyp_attention_map, (0,2,1))], ["premise", "hypothesis"]):
					fig = plt.figure()
					ax = fig.add_subplot(111)
					cax = ax.matshow(attention_map[i,:batch_prem[i].shape[0],:batch_hyp[i].shape[0]], cmap=plt.cm.gray)
					ax.set_yticklabels([id2word[x] for x in batch_prem[i]])
					ax.set_xticklabels([id2word[x] for x in batch_hyp[i]])
					plt.yticks(range(batch_prem[i].shape[0]))
					plt.xticks(range(batch_hyp[i].shape[0]), rotation=90)
					# print("Attention map %i shape: " % (i) + str(attention_map[i,:batch_prem[i].shape[0],:batch_hyp[i].shape[0]].shape))
					# print("Premise %i: %s" % (i, " ".join([id2word[x] for x in batch_prem[i]])))
					# print("Hypothesis %i: %s" % (i, " ".join([id2word[x] for x in batch_hyp[i]])))
					writer.add_figure(tag="train/sample_attention_maps_%i_%s"%(i, main_w), figure=fig, global_step=iteration)

			

class SSTTask(TaskTemplate):

	NAME = "Stanford_Sentiment_Treebank"

	def __init__(self, model, model_params, load_data=True):
		super(SSTTask, self).__init__(model=model, model_params=model_params, load_data=load_data, name=SSTTask.NAME)
		self.classifier = SimpleClassifier(self.classifier_params, 2)
		self.loss_module = TaskTemplate._create_CrossEntropyLoss()
		if torch.cuda.is_available():
			self.classifier = self.classifier.cuda()
			self.loss_module = self.loss_module.cuda()


	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = DatasetHandler.load_SST_datasets()


	def train_step(self, batch_size, loop_dataset=True):
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
		embeds, lengths, batch_labels = self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True)
		
		sent_embeds = self.model.encode_sentence(embeds, lengths)
		out = self.classifier(sent_embeds, applySoftmax=False)

		loss = self.loss_module(out, batch_labels)

		_, pred_labels = torch.max(out, dim=-1)
		acc = torch.sum(pred_labels == batch_labels).float() / pred_labels.shape[-1]

		return loss, acc


	def _eval_batch(self, batch):
		embeds, lengths, batch_labels = batch
		
		sent_embeds = self.model.encode_sentence(embeds, lengths)
		preds = self.classifier(sent_embeds, applySoftmax=True)
		
		_, pred_labels = torch.max(preds, dim=-1)
		
		return pred_labels, batch_labels

# class POSTask(TaskTemplate):

# 	NAME = "POS_Tagging"

# 	def __init__(self, model, classifier_params, load_data=True):
# 		super(SNLITask, self).__init__(model=model, load_data=load_data, name=SNLITask.NAME)
# 		self.classifier = SimpleClassifier(classifier_params, 10)
# 		self.loss_module = TaskTemplate._create_CrossEntropyLoss()
# 		if torch.cuda.is_available():
# 			self.classifier = self.classifier.cuda()
# 			self.loss_module = self.loss_module.cuda()


# 	def _load_datasets(self):
# 		self.train_dataset, self.val_dataset, self.test_dataset = DatasetHandler.load_SNLI_datasets()


# 	def train_step(self, batch_size, loop_dataset=True):
# 		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
# 		embeds, lengths, batch_labels = self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True)
		
# 		embed_s1 = self.model.encode_sentence(embeds[0], lengths[0])
# 		embed_s2 = self.model.encode_sentence(embeds[1], lengths[1])

# 		out = self.classifier(embed_s1, embed_s2, applySoftmax=False)

# 		loss = self.loss_module(out, batch_labels)

# 		_, pred_labels = torch.max(out, dim=-1)
# 		acc = torch.sum(pred_labels == batch_labels) / pred_labels.shape[-1]

# 		return loss, acc


# 	def _eval_batch(self, batch):
# 		embeds, lengths, batch_labels = batch
		
# 		embed_s1 = self.model.encode_sentence(embeds[0], lengths[0])
# 		embed_s2 = self.model.encode_sentence(embeds[1], lengths[1])

# 		preds = self.classifier(embed_s1, embed_s2, applySoftmax=True)
		
# 		_, pred_labels = torch.max(preds, dim=-1)
		
# 		return pred_labels, batch_labels
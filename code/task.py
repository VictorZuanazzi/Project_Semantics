import torch 
import torch.nn as nn
import argparse
import numpy as np
import math
from random import shuffle
import os
import sys
# Disable matplotlib screen support
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from model import SimpleClassifier, NLIClassifier, ESIM_Head, get_device
from data import DatasetTemplate, DatasetHandler, debug_level, VUAData, POSData
from vocab import get_id2word_dict



def create_task(model, task, model_params, debug=False):
	if task == SNLITask.NAME:
		return SNLITask(model, model_params, load_data=True, debug=debug)
	if task == MNLITask.NAME:
		return MNLITask(model, model_params, load_data=True, debug=debug)
	elif task == SSTTask.NAME:
		return SSTTask(model, model_params, load_data=True, debug=debug)
	elif task == VUATask.NAME:
		return VUATask(model, model_params, load_data=True, debug=debug)
	elif task == VUASeqTask.NAME:
		return VUASeqTask(model, model_params, load_data=True, debug=debug)
	elif task == POSTask.NAME:
		return POSTask(model, model_params, load_data=True, debug=debug)
	else:
		print("[!] ERROR: Unknown task " + str(task))
		print("If the task exists but could not be found, add it in the function 'create_task' in the file 'task.py'.")
		sys.exit(1)


class TaskTemplate:

	def __init__(self, model, model_params, name, load_data=True, debug=False):
		self.name = name 
		self.model = model
		self.model_params = model_params
		self.classifier_params = model_params[name + "_head"] if (name + "_head") in model_params else model_params
		self.debug = debug
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


	def eval(self, dataset=None, batch_size=64, add_predictions=False):
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

		# to_remove = [i for i, l in enumerate(label_list) if l < 0]
		# for r_index in sorted(to_remove)[::-1]:
		# 	del preds_list[r_index]
		# 	del label_list[r_index]
		
		# Metric output
		preds_list = np.array(preds_list)
		label_list = np.array(label_list)
		preds_list = preds_list[label_list >= 0]
		label_list = label_list[label_list >= 0]
		accuracy = np.sum(preds_list == label_list) * 1.0 / preds_list.shape[0]
		detailed_acc = {"accuracy": accuracy, 
						"class_scores": dict()}

		if add_predictions:
			detailed_acc["predictions"] = preds_list
			detailed_acc["labels"] = label_list

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
			print("\t- Class %s: Recall=%4.2f%%, Precision=%4.2f%%, F1 score=%4.2f%%" % (dataset.label_to_string(c), recall*100.0, precision*100.0, F1_score*100.0))
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


	def print_classifier(self):
		print("="*75 + "\n" + self.name + " classifier:\n"+"-"*75)
		print(self.classifier)
		print("="*75)


	def eval_metric(self, eval_dict):
		return eval_dict["accuracy"]


	@staticmethod
	def _create_CrossEntropyLoss(weight=None, ignore_index=-1):
		loss_module = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index).to(get_device())
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
		self.highest_eval_accs = {t.name: -1 for t in self.tasks}


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
		reached_new_opt = {}
		for t in self.tasks:
			acc, detailed_acc = t.eval()
			print("Task " + t.name + ": %4.2f%%" % (acc*100.0))
			reached_new_opt[t.name] = (acc > self.highest_eval_accs[t.name])
			if acc > self.highest_eval_accs[t.name]:
				self.highest_eval_accs[t.name] = acc
				print("Highest accuracy so far for task " + t.name)
			accuracy_dict[t.name] = detailed_acc
		return accuracy_dict, reached_new_opt


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

class NLITask(TaskTemplate):

	CLASSIFIER_INFERSENT = 0
	CLASSIFIER_ESIM = 1

	def __init__(self, model, model_params, load_data=True, debug=False, name=None):
		super(NLITask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name=name)
		self.classifier = self._create_classifier()
		self.loss_module = TaskTemplate._create_CrossEntropyLoss()
		self.print_classifier()


	def _create_classifier(self):
		if "model" not in self.classifier_params or self.classifier_params["model"] == NLITask.CLASSIFIER_INFERSENT:
			return NLIClassifier(self.classifier_params)
		elif self.classifier_params["model"] == NLITask.CLASSIFIER_ESIM:
			return ESIM_Head(self.classifier_params)
		else:
			print("[!] ERROR: Unknown classifier for SNLI Task: " + str(self.classifier_params["model"]) + \
				  "Supported options are: [" + ",".join([str(o) for o in [NLITask.CLASSIFIER_INFERSENT, NLITask.CLASSIFIER_ESIM]]) + "]")
			sys.exit(1)


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
		embed_s1 = self.model.encode_sentence(embeds[0], lengths[0], word_level=self.classifier.is_word_level(), layer=-1)
		embed_s2 = self.model.encode_sentence(embeds[1], lengths[1], word_level=self.classifier.is_word_level(), layer=-1)

		if not self.classifier.is_word_level():
			out = self.classifier(embed_s1, embed_s2, applySoftmax=applySoftmax)
		else:
			out = self.classifier(embed_s1[1], lengths[0], embed_s2[1], lengths[1], applySoftmax=applySoftmax)
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
					sent_attention_map = attention_map[i,:batch_prem[i].shape[0],:batch_hyp[i].shape[0]]
					bias_prem = self.classifier.bias_prem is not None and main_w == "premise"
					bias_hyp = self.classifier.bias_hyp is not None and main_w == "hypothesis"
					if bias_prem:
						sent_attention_map = np.concatenate([sent_attention_map, 1 - np.sum(sent_attention_map, axis=1, keepdims=True)], axis=1)
					if bias_hyp:
						sent_attention_map = np.concatenate([sent_attention_map, 1 - np.sum(sent_attention_map, axis=0, keepdims=True)], axis=0)
					# print(sent_attention_map)
					cax = ax.matshow(sent_attention_map, cmap=plt.cm.gray)
					ax.set_yticklabels([id2word[x] for x in batch_prem[i]] + ["bias"])
					ax.set_xticklabels([id2word[x] for x in batch_hyp[i]] + ["bias"])
					plt.yticks(range(batch_prem[i].shape[0]+(1 if bias_hyp else 0)))
					plt.xticks(range(batch_hyp[i].shape[0]+(1 if bias_prem else 0)), rotation=90)
					# print("Attention map %i shape: " % (i) + str(attention_map[i,:batch_prem[i].shape[0],:batch_hyp[i].shape[0]].shape))
					# print("Premise %i: %s" % (i, " ".join([id2word[x] for x in batch_prem[i]])))
					# print("Hypothesis %i: %s" % (i, " ".join([id2word[x] for x in batch_hyp[i]])))
					writer.add_figure(tag="train_" + self.name + "/sample_attention_maps_%i_%s"%(i, main_w), figure=fig, global_step=iteration)
			plt.close()


class SNLITask(NLITask):

	NAME = "Stanford_NLI"

	def __init__(self, model, model_params, load_data=True, debug=False):
		super(SNLITask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name=SNLITask.NAME)


	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = DatasetHandler.load_SNLI_datasets(debug_dataset=self.debug)


class MNLITask(NLITask):

	NAME = "MultiNLI"

	def __init__(self, model, model_params, load_data=True, debug=False):
		super(MNLITask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name=MNLITask.NAME)

	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = DatasetHandler.load_MultiNLI_datasets(debug_dataset=self.debug)



class SSTTask(TaskTemplate):

	NAME = "Stanford_Sentiment_Treebank"

	def __init__(self, model, model_params, load_data=True, debug=False):
		super(SSTTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name=SSTTask.NAME)
		self.classifier = SimpleClassifier(self.classifier_params, 2)
		self.loss_module = TaskTemplate._create_CrossEntropyLoss()
		self.print_classifier()


	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = DatasetHandler.load_SST_datasets(debug_dataset=self.debug)


	def train_step(self, batch_size, loop_dataset=True):
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
		embeds, lengths, batch_labels = self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True)
		
		sent_embeds = self.model.encode_sentence(embeds, lengths, word_level=False, layer=-1)
		out = self.classifier(sent_embeds, applySoftmax=False)

		loss = self.loss_module(out, batch_labels)

		_, pred_labels = torch.max(out, dim=-1)
		acc = torch.sum(pred_labels == batch_labels).float() / pred_labels.shape[-1]

		return loss, acc


	def _eval_batch(self, batch):
		embeds, lengths, batch_labels = batch
		
		sent_embeds = self.model.encode_sentence(embeds, lengths, word_level=False, layer=-1)
		preds = self.classifier(sent_embeds, applySoftmax=True)
		
		_, pred_labels = torch.max(preds, dim=-1)
		
		return pred_labels, batch_labels


class VUATask(TaskTemplate):

	NAME = "VUA_Metaphor_Detection"

	def __init__(self, model, model_params, load_data=True, debug=False):
		super(VUATask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name=VUATask.NAME)
		self.classifier_params["embed_sent_dim"] *= 2
		self.classifier = SimpleClassifier(self.classifier_params, 2)
		self.loss_module = TaskTemplate._create_CrossEntropyLoss()
		self.print_classifier()


	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = DatasetHandler.load_VUA_datasets(debug_dataset=self.debug)


	def train_step(self, batch_size, loop_dataset=True):
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
		embeds, lengths, batch_labels, word_pos = self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True)
		
		out = self._forward_model(embeds, lengths, word_pos, applySoftmax=False)
		loss = self.loss_module(out, batch_labels)

		_, pred_labels = torch.max(out, dim=-1)
		acc = torch.sum(pred_labels == batch_labels).float() / pred_labels.shape[-1]

		return loss, acc


	def _eval_batch(self, batch):
		embeds, lengths, batch_labels, word_pos = batch
		
		preds = self._forward_model(embeds, lengths, word_pos, applySoftmax=True)
		_, pred_labels = torch.max(preds, dim=-1)
		
		return pred_labels, batch_labels


	def _forward_model(self, embeds, lengths, word_pos, applySoftmax=False):
		batch_size = embeds.size(0)
		time_dim = embeds.size(1)
		sent_embeds, word_embeds = self.model.encode_sentence(embeds, lengths, word_level=True)
		word_embeds = word_embeds.view(-1, word_embeds.size(2))
		indexes = (lengths - 1) + torch.arange(batch_size, device=word_embeds.device, dtype=lengths.dtype) * time_dim
		sel_word_embeds = word_embeds[indexes,:]
		metaphor_embeds = torch.cat((sent_embeds, sel_word_embeds), dim=-1)
		out = self.classifier(metaphor_embeds, applySoftmax=False)
		return out

	def eval_metric(self, eval_dict):
		return eval_dict["class_scores"][self.train_dataset.label_to_string(VUAData.LABEL_LIST["metaphor"])]["f1"]


class VUASeqTask(TaskTemplate):

	NAME = "VUA_Sequential_Metaphor_Detection"
	LAYER = 1

	def __init__(self, model, model_params, load_data=True, debug=False):
		super(VUASeqTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name=VUASeqTask.NAME)
		self.classifier_params["embed_sent_dim"] = self.model.get_layer_size(VUASeqTask.LAYER)
		self.classifier = SimpleClassifier(self.classifier_params, 2)
		self.loss_module = TaskTemplate._create_CrossEntropyLoss()
		self.print_classifier()


	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = DatasetHandler.load_VUAseq_datasets(debug_dataset=self.debug)


	def train_step(self, batch_size, loop_dataset=True):
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
		embeds, lengths, batch_labels = self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True)
		batch_labels = batch_labels.view(-1)

		out = self._forward_model(embeds, lengths, applySoftmax=False)
		loss = self.loss_module(out, batch_labels)

		_, pred_labels = torch.max(out, dim=-1)
		acc = torch.sum(pred_labels == batch_labels).float() / torch.sum(batch_labels >= 0).float()

		return loss, acc


	def _eval_batch(self, batch):
		embeds, lengths, batch_labels = batch
		
		preds = self._forward_model(embeds, lengths, applySoftmax=True)
		_, pred_labels = torch.max(preds, dim=-1)
		
		return pred_labels, batch_labels.view(-1)


	def _forward_model(self, embeds, lengths, applySoftmax=False):
		_, word_embeds = self.model.encode_sentence(embeds, lengths, word_level=True, layer=VUASeqTask.LAYER)
		word_embeds = word_embeds.view(-1, word_embeds.shape[2])
		out = self.classifier(word_embeds, applySoftmax=False)
		return out

	def eval_metric(self, eval_dict):
		return eval_dict["class_scores"][self.train_dataset.label_to_string(VUAData.LABEL_LIST["metaphor"])]["f1"]



class POSTask(TaskTemplate):

	NAME = "POS_Tagging"
	LAYER = 0

	def __init__(self, model, model_params, load_data=True, debug=False):
		super(POSTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name=POSTask.NAME)
		self.classifier_params["embed_sent_dim"] = self.model.get_layer_size(POSTask.LAYER)
		self.classifier = SimpleClassifier(self.classifier_params, POSData.num_classes())
		self.loss_module = TaskTemplate._create_CrossEntropyLoss()


	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = DatasetHandler.load_POS_MNLI_datasets(debug_dataset=self.debug)


	def train_step(self, batch_size, loop_dataset=True):
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
		embeds, lengths, batch_labels = self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True)
		batch_labels = batch_labels.view(-1)
		
		out = self._forward_model(embeds, lengths, applySoftmax=False)
		loss = self.loss_module(out, batch_labels)

		_, pred_labels = torch.max(out, dim=-1)
		acc = torch.sum(pred_labels == batch_labels).float() / torch.sum(batch_labels >= 0).float()

		return loss, acc


	def _eval_batch(self, batch):
		embeds, lengths, batch_labels = batch
		
		preds = self._forward_model(embeds, lengths, applySoftmax=True)
		_, pred_labels = torch.max(preds, dim=-1)
		
		return pred_labels, batch_labels.view(-1)


	def _forward_model(self, embeds, lengths, applySoftmax=False):
		_, word_embeds = self.model.encode_sentence(embeds, lengths, word_level=True, layer=POSTask.LAYER)
		word_embeds = word_embeds.view(-1, word_embeds.shape[2])
		out = self.classifier(word_embeds, applySoftmax=False)
		return out
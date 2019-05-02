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
from glob import glob

from tensorboardX import SummaryWriter
# from visualdl import LogWriter
# from torchviz import make_dot

from eval import SNLIEval
from model import NLIModel
from data import load_SNLI_datasets, debug_level, set_debug_level
from mutils import load_model, load_args, args_to_params, get_dict_val, PARAM_CONFIG_FILE

class SNLITrain:

	OPTIMIZER_SGD = 0
	OPTIMIZER_ADAM = 1


	def __init__(self, model_type, model_params, optimizer_params, batch_size, checkpoint_path, debug=False):
		self.train_dataset, _, _, _, self.word2id, wordvec_tensor = load_SNLI_datasets(debug_dataset = debug)
		self.id2word = {v: k for k, v in self.word2id.items()}
		self.model = NLIModel(model_type, model_params, wordvec_tensor, self.id2word)  # fixme
		self.evaluater = SNLIEval(self.model)
		self.batch_size = batch_size
		self._create_optimizer(optimizer_params)
		self._prepare_checkpoint(checkpoint_path)
		

	def _create_optimizer(self, optimizer_params):
		if optimizer_params["optimizer"] == SNLITrain.OPTIMIZER_SGD:
			self.optimizer = torch.optim.SGD(self.model.parameters(), 
											 lr=optimizer_params["lr"], 
											 weight_decay=optimizer_params["weight_decay"],
											 momentum=optimizer_params["momentum"])
		elif optimizer_params["optimizer"] == SNLITrain.OPTIMIZER_ADAM:
			self.optimizer = torch.optim.Adam(self.model.parameters(), 
											  lr=optimizer_params["lr"])
		else:
			print("[!] ERROR: Unknown optimizer: " + str(optimizer_params["optimizer"]))
			sys.exit(1)
		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=optimizer_params["lr_decay_step"])
		self.lr_scheduler.step() # The first step is not recognized...
		self.max_red_steps = optimizer_params["lr_max_red_steps"]


	def _prepare_checkpoint(self, checkpoint_path):
		if checkpoint_path is None:
			current_date = datetime.datetime.now()
			checkpoint_path = "checkpoints/%02d_%02d_%02d__%02d_%02d_%02d/" % (current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute, current_date.second)
		if not os.path.exists(checkpoint_path):
			os.makedirs(checkpoint_path)
		self.checkpoint_path = checkpoint_path


	def train_model(self, epochs=50, loss_freq=50, enable_tensorboard=False, intermediate_evals=False):

		loss_module = nn.CrossEntropyLoss()
		if torch.cuda.is_available():
			loss_module = loss_module.cuda()

		checkpoint_dict = self.load_recent_model()
		start_epoch = get_dict_val(checkpoint_dict, "epoch", 0)
		eval_accuracies = get_dict_val(checkpoint_dict, "eval_accuracies", list())
		loss_avg_list = get_dict_val(checkpoint_dict, "loss_avg_list", list())
		lr_red_step = get_dict_val(checkpoint_dict, "lr_red_step", list())
		start_step = get_dict_val(checkpoint_dict, "step_index", 0)
		if start_step != 0:
			self.train_dataset.perm_indices = get_dict_val(checkpoint_dict, "dataset_perm", [])
			self.train_dataset.example_index = get_dict_val(checkpoint_dict, "dataset_exm_index", 0)
		intermediate_accs = dict()
		
		if enable_tensorboard:
			writer = SummaryWriter(self.checkpoint_path)
		else:
			writer = None
		
		try:
			print("="*50 + "\nStarting training...\n"+"="*50)
			for index_epoch in range(start_epoch, epochs):

				if writer is not None:
					writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], index_epoch+1)
				if intermediate_evals:
					intermediate_accs[index_epoch] = list()
				self.model.train()
				num_steps = int(math.ceil(self.train_dataset.get_num_examples() * 1.0 / self.batch_size))
				if start_step == 0:
					loss_avg_list.append(0)
				for step_index in range(start_step, num_steps):

					embeds, lengths, batch_labels = self.train_dataset.get_batch(self.batch_size, loop_dataset=False, toTorch=True, bidirectional=self.model.is_bidirectional())
					preds = self.model(words_s1 = embeds[0], lengths_s1 = lengths[0], words_s2 = embeds[1], lengths_s2 = lengths[1], applySoftmax=False)
					loss = loss_module(preds, batch_labels)

					self.model.zero_grad()
					loss.backward()
					self.optimizer.step()

					# if index_epoch == 0 and step_index == 0 and writer is not None:
					# 	param_dict = dict()
					# 	for p in self.model.parameters():
					# 		if p.requires_grad:
					# 		 	print(p.name, p.data)
					# 		 	param_dict[p.name] = p
					# 	make_dot(loss, param_dict)
					# 	# dummy_input = (embeds[0], lengths[0], embeds[1], lengths[1], True, False)
					# 	# torch.onnx.export(self.model, dummy_input, "test_graph.onnx")
					# 	# writer.add_graph(self.model.cpu(), (embeds[0].cpu(), lengths[0].cpu().int(), embeds[1].cpu(), lengths[1].cpu().int()), operator_export_type="ONNX")

					loss_avg_list[-1] += loss.item()
					if (step_index + 1) % loss_freq == 0:
						loss_avg_list[-1] = loss_avg_list[-1]  / loss_freq
						print("Training epoch %i|%i, step %i|%i. Loss: %6.5f" % (index_epoch+1, epochs, step_index+1, num_steps, loss_avg_list[-1]))
						if writer is not None:
							writer.add_scalar("train/loss", loss_avg_list[-1], num_steps * index_epoch + step_index + 1)
						loss_avg_list.append(0)

					if intermediate_evals and (step_index + 1) % 2000 == 0:
						intermediate_acc = self.evaluater.eval()
						intermediate_accs[index_epoch].append(intermediate_acc)
						self.model.train()
						if writer is not None:
							writer.add_scalar("eval/acc_per_step", intermediate_acc, num_steps * index_epoch + step_index + 1)

				del loss_avg_list[-1]
				start_step = 0

				acc = self.evaluater.eval(index_epoch)
				eval_accuracies.append(acc)

				if writer is not None:
					writer.add_scalar("eval/acc", acc, index_epoch+1)
					if intermediate_evals:
						writer.add_scalar("eval/acc_per_step", acc, num_steps * (index_epoch + 1) )

				if len(eval_accuracies) > 2:
					if (not intermediate_evals and eval_accuracies[-1] < (eval_accuracies[-2] + eval_accuracies[-3]) / 2) or \
					   (intermediate_evals and sum(intermediate_accs[index_epoch]) < sum(intermediate_accs[index_epoch-2])):
						print("Reducing learning rate")
						self.lr_scheduler.step()
						lr_red_step.append(index_epoch + 1)

				checkpoint_dict = {
					"epoch": index_epoch + 1,
					"eval_accuracies": eval_accuracies,
					"loss_avg_list": loss_avg_list,
					"lr_red_step": lr_red_step
				}
				self.save_model(index_epoch + 1, checkpoint_dict)

				if len(lr_red_step) > self.max_red_steps:
					print("Reached maximum number of learning rate reduction steps")
					break

			with open(os.path.join(self.checkpoint_path, "results.txt"), "w") as f:
				f.write("".join(["Epoch %i: %4.2f%%\n" % (i+1,eval_accuracies[i]*100.0) for i in range(len(eval_accuracies))]))
				f.write("Best accuracy achieved: %4.2f%%" % (max(eval_accuracies) * 100.0))

		except KeyboardInterrupt:
			print("User keyboard interrupt detected. Saving model at step %i..." % (step_index))
			checkpoint_dict = {
				"epoch": index_epoch,
				"step_index": step_index,
				"dataset_perm": self.train_dataset.perm_indices,
				"dataset_exm_index": self.train_dataset.example_index,
				"eval_accuracies": eval_accuracies,
				"loss_avg_list": loss_avg_list,
				"lr_red_step": lr_red_step
			}
			self.save_model(index_epoch + 1, checkpoint_dict, step=step_index+1)

		if writer is not None:
			writer.close()

	def save_model(self, epoch, add_param_dict, step=None, save_embeddings=False):
		checkpoint_file = os.path.join(self.checkpoint_path, 'checkpoint_' + str(epoch).zfill(3) + ("_step_%i" % (step) if step is not None else "") + ".tar")
		model_dict = self.model.state_dict()
		if not save_embeddings:
			model_dict = {k:v for k,v in model_dict.items() if not k.startswith("embeddings")}
		checkpoint_dict = {
				'model_state_dict': model_dict,
				'optimizer_state_dict': self.optimizer.state_dict(),
				'scheduler_state_dict': self.lr_scheduler.state_dict()
		}
		for key, val in add_param_dict.items():
			checkpoint_dict[key] = val
		torch.save(checkpoint_dict, checkpoint_file)

	def load_recent_model(self):
		return load_model(self.checkpoint_path, model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--cluster", help="Enable option if code is executed on cluster. Reduces output size", action="store_true")
	parser.add_argument("-d", "--debug", help="Whether debug output should be activated or not", action="store_true")
	parser.add_argument("--epochs", help="Maximum number of epochs to train. Default: dynamic with learning rate threshold", type=int, default=50)
	parser.add_argument("--eval_freq", help="Frequency of evaluation on validation set (in number of steps/iterations). Default: once per epoch", type=int, default=-1)
	parser.add_argument("--batch_size", help="Batch size used during training", type=int, default=64)
	parser.add_argument("--learning_rate", help="Learning rate of the optimizer", type=float, default=0.1)
	parser.add_argument("--lr_decay", help="Decay of learning rate of the optimizer. Always applied if eval accuracy droped compared to mean of last two epochs", type=float, default=0.2)
	parser.add_argument("--lr_max_red_steps", help="Maximum number of times learning rate should be decreased before terminating", type=int, default=4)
	parser.add_argument("--weight_decay", help="Weight decay of the SGD optimizer", type=float, default=1e-2)
	parser.add_argument("--optimizer", help="Which optimizer to use. 0: SGD, 1: Adam", type=int, default=0)
	parser.add_argument("--momentum", help="Apply momentum to SGD optimizer", type=float, default=0.0)
	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints should be saved", type=str, default=None)
	parser.add_argument("--load_config", help="Tries to find parameter file in checkpoint path, and loads all given parameters from there", action="store_true")
	parser.add_argument("--fc_dim", help="Number of hidden units in fully connected layers (classifier)", type=int, default=512)
	parser.add_argument("--fc_dropout", help="Dropout probability in FC classifier", type=float, default=0.0)
	parser.add_argument("--fc_nonlinear", help="Whether to add a non-linearity (tanh) between classifier layers or not", action="store_true")
	parser.add_argument("--embed_dim", help="Embedding dimensionality of sentence", type=int, default=2048)
	parser.add_argument("--model", help="Which encoder model to use. 0: BOW, 1: LSTM, 2: Bi-LSTM, 3: Bi-LSTM with max pooling, 4: Bi-LSTM skip connections", type=int, default=0)
	parser.add_argument("--tensorboard", help="Activates tensorboard support while training", action="store_true")
	parser.add_argument("--restart", help="Does not load old checkpoints, and deletes those if checkpoint path is specified (including tensorboard file etc.)", action="store_true")
	parser.add_argument("--intermediate_evals", help="Whether validations should also be performed within a epoch. NO CHECKPOINTS WILL BE SAVED FOR THOSE, and the values are only saved in the tensorboard!", action="store_true")
	parser.add_argument("--seed", help="Seed to make experiments reproducable", type=int, default=42)

	args = parser.parse_args()
	print(args)
	if args.cluster:
		set_debug_level(2)
		loss_freq = 500
	else:
		set_debug_level(0)
		loss_freq = 50

	if args.load_config:
		if args.checkpoint_path is None:
			print("[!] ERROR: Please specify the checkpoint path to load the config from.")
			sys.exit(1)
		args = load_args(args.checkpoint_path)

	# Setup training
	model_type, model_params, optimizer_params = args_to_params(args)
	trainModule = SNLITrain(model_type=args.model, 
							model_params=model_params,
							optimizer_params=optimizer_params, 
							batch_size=args.batch_size,
							checkpoint_path=args.checkpoint_path, 
							debug=args.debug
							)

	if args.restart and args.checkpoint_path is not None and os.path.isdir(args.checkpoint_path):
		print("Cleaning up directiory " + str(args.checkpoint_path) + "...")
		for ext in [".tar", ".out.tfevents.*", ".txt"]:
			for file_in_dir in sorted(glob(os.path.join(args.checkpoint_path, "*" + ext))):
				print("Removing file " + file_in_dir)
				os.remove(file_in_dir)

	args_filename = os.path.join(trainModule.checkpoint_path, PARAM_CONFIG_FILE)
	with open(args_filename, "wb") as f:
		pickle.dump(args, f)

	trainModule.train_model(args.epochs, loss_freq=loss_freq, enable_tensorboard=args.tensorboard, intermediate_evals=args.intermediate_evals)

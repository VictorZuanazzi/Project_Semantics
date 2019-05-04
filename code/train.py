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
import time
from glob import glob

from tensorboardX import SummaryWriter
# from visualdl import LogWriter
# from torchviz import make_dot

from task import create_task, TaskTemplate, MultiTaskSampler, SNLITask, SSTTask, VUATask
from model import MultiTaskEncoder
from data import debug_level, set_debug_level
from vocab import load_word2vec_from_file
from mutils import load_model, load_args, args_to_params, get_dict_val, PARAM_CONFIG_FILE, write_dict_to_tensorboard

class MultiTaskTrain:

	OPTIMIZER_SGD = 0
	OPTIMIZER_ADAM = 1


	def __init__(self, tasks, model_type, model_params, optimizer_params, multitask_params, batch_size, checkpoint_path, debug=False):
		_, self.word2id, wordvec_tensor = load_word2vec_from_file()
		self.batch_size = batch_size
		self.model = MultiTaskEncoder(model_type, model_params, wordvec_tensor)
		self.tasks = [create_task(self.model, t, model_params, debug=debug) for t in tasks]
		assert len(self.tasks) > 0, "Please specify at least one task to train on."
		self.multitask_sampler = MultiTaskSampler(self.tasks, multitask_params, batch_size)
		self._create_optimizer(optimizer_params)
		self._prepare_checkpoint(checkpoint_path) 
		

	def _get_all_parameters(self):
		parameters_to_optimize = list(self.model.parameters())
		for t in self.tasks:
			parameters_to_optimize += list(t.get_parameters())
		return parameters_to_optimize


	def _create_optimizer(self, optimizer_params):
		parameters_to_optimize = self._get_all_parameters()
		if optimizer_params["optimizer"] == MultiTaskTrain.OPTIMIZER_SGD:
			self.optimizer = torch.optim.SGD(parameters_to_optimize, 
											 lr=optimizer_params["lr"], 
											 weight_decay=optimizer_params["weight_decay"],
											 momentum=optimizer_params["momentum"])
		elif optimizer_params["optimizer"] == MultiTaskTrain.OPTIMIZER_ADAM:
			self.optimizer = torch.optim.Adam(parameters_to_optimize, 
											  lr=optimizer_params["lr"],
											  weight_decay=optimizer_params["weight_decay"])
		else:
			print("[!] ERROR: Unknown optimizer: " + str(optimizer_params["optimizer"]))
			sys.exit(1)
		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, optimizer_params["lr_decay_step"], gamma=optimizer_params["lr_decay_factor"])
		self.max_red_steps = optimizer_params["lr_max_red_steps"]


	def _prepare_checkpoint(self, checkpoint_path):
		if checkpoint_path is None:
			current_date = datetime.datetime.now()
			checkpoint_path = "checkpoints/%02d_%02d_%02d__%02d_%02d_%02d/" % (current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute, current_date.second)
		if not os.path.exists(checkpoint_path):
			os.makedirs(checkpoint_path)
		self.checkpoint_path = checkpoint_path


	def train_model(self, max_iterations=1e6, loss_freq=50, eval_freq=2000, save_freq=1e5, enable_tensorboard=False, max_gradient_norm=10.0):

		# Setup training parameters
		parameters_to_optimize = self._get_all_parameters()
		checkpoint_dict = self.load_recent_model()
		start_iter = get_dict_val(checkpoint_dict, "iteration", 0)
		evaluation_dict = get_dict_val(checkpoint_dict, "evaluation_dict", dict())
		
		if enable_tensorboard:
			writer = SummaryWriter(self.checkpoint_path)
		else:
			writer = None

		# Function for saving model. Add here in the dictionary necessary parameters that should be saved
		def save_train_model(iteration):
			checkpoint_dict = {
				"evaluation_dict": evaluation_dict,
				"iteration": iteration
			}
			self.save_model(iteration, checkpoint_dict)

		def export_weight_parameters(iteration):
			# Export weight distributions
			for name, param in self.model.named_parameters():
				writer.add_histogram(name, param.data.view(-1), global_step=iteration)
			for t in self.tasks:
				for name, param in t.classifier.named_parameters():
					writer.add_histogram(t.name+"/"+name, param.data.view(-1), global_step=iteration)
		
		time_per_step = np.zeros((2,), dtype=np.float32)

		if start_iter == 0 and writer is not None:
			export_weight_parameters(0)
		# Try-catch if user terminates
		try:
			print("="*50 + "\nStarting training...\n"+"="*50)
			self.model.train()
			
			for index_iter in range(start_iter, int(max_iterations)):
				
				# Training step
				start_time = time.time()
				self.lr_scheduler.step()
				loss = self.multitask_sampler.sample_batch_loss(index_iter)
				self.model.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(parameters_to_optimize, max_gradient_norm)
				self.optimizer.step()
				end_time = time.time()
				time_per_step[0] += end_time - start_time
				time_per_step[1] += 1

				# Debug loss printing
				if (index_iter + 1) % loss_freq == 0:
					loss_avg, acc_avg = self.multitask_sampler.get_average_metrics()
					print("Training iteration %i|%i. Loss: %6.5f" % (index_iter+1, max_iterations, loss_avg[0]))
					if writer is not None:
						writer.add_scalar("train/loss/all_combined", loss_avg[0], index_iter + 1)
						for task_ind, t in enumerate(self.tasks):
							writer.add_scalar("train/loss/"+t.name, loss_avg[task_ind + 1], index_iter + 1)
							writer.add_scalar("train/accuracy/"+t.name, acc_avg[task_ind + 1], index_iter + 1)
						writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], index_iter+1)
						writer.add_scalar("train/training_time", time_per_step[0] / max(1e-5, time_per_step[1]), index_iter+1)
					time_per_step[:] = 0
					self.multitask_sampler.reset_loss_counter()

				# Evaluation
				if (index_iter + 1) % eval_freq == 0:
					self.model.eval()
					evaluation_dict[index_iter+1] = self.multitask_sampler.evaluate_all()
					self.model.train()

					if writer is not None:
						write_dict_to_tensorboard(writer, evaluation_dict[index_iter+1], base_name="eval", iteration=index_iter+1)
						export_weight_parameters(index_iter+1)
						for t in self.tasks:
							t.add_to_summary(writer, index_iter+1)

				# Saving
				if (index_iter + 1) % save_freq == 0:
					save_train_model(index_iter + 1)

		except KeyboardInterrupt:
			print("User keyboard interrupt detected. Saving model at step %i..." % (index_iter))
			save_train_model(index_iter + 1)

		with open(os.path.join(self.checkpoint_path, "results.txt"), "w") as f:
			for eval_iter, eval_dict in evaluation_dict.items():
				f.write("Iteration %i: " % (eval_iter))
				for task_name, task_eval_dict in eval_dict.items():
					f.write("%s=%4.2f%%, " % (task_name, task_eval_dict["accuracy"]))
				f.write("\n")

		if writer is not None:
			writer.close()


	def save_model(self, iteration, add_param_dict, save_embeddings=False):
		checkpoint_file = os.path.join(self.checkpoint_path, 'checkpoint_' + str(iteration).zfill(7) + ".tar")
		model_dict = self.model.state_dict()
		if not save_embeddings:
			model_dict = {k:v for k,v in model_dict.items() if not k.startswith("embeddings")}
		checkpoint_dict = {
			'model_state_dict': model_dict,
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.lr_scheduler.state_dict()
		}
		checkpoint_dict.update(add_param_dict)
		for t in self.tasks:
			checkpoint_dict.update(t.dict_to_save())
		torch.save(checkpoint_dict, checkpoint_file)


	def load_recent_model(self):
		checkpoint_dict = load_model(self.checkpoint_path, model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
		if len(checkpoint_dict.keys()) > 0: # If checkpoint is not empty, load heads as well
			for t in self.tasks:
				t.load_from_dict(checkpoint_dict)
		return checkpoint_dict




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Training extras
	parser.add_argument("--max_iterations", help="Maximum number of epochs to train. Default: dynamic with learning rate threshold", type=int, default=1e7)
	parser.add_argument("--batch_size", help="Batch size used during training", type=int, default=64)
	parser.add_argument("--tensorboard", help="Activates tensorboard support while training", action="store_true")
	parser.add_argument("--eval_freq", help="In which frequency the model should be evaluated (in number of iterations). Default: 2000", type=int, default=2000)
	parser.add_argument("--save_freq", help="In which frequency the model should be saved (in number of iterations). Default: 10,000", type=int, default=1e4)
	# Loading experiment
	parser.add_argument("--restart", help="Does not load old checkpoints, and deletes those if checkpoint path is specified (including tensorboard file etc.)", action="store_true")
	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints should be saved", type=str, default=None)
	parser.add_argument("--load_config", help="Tries to find parameter file in checkpoint path, and loads all given parameters from there", action="store_true")
	# Model parameters
	parser.add_argument("--embed_dim", help="Embedding dimensionality of sentence", type=int, default=2048)
	parser.add_argument("--model", help="Which encoder model to use. 0: BOW, 1: LSTM, 2: Bi-LSTM, 3: Bi-LSTM with max pooling", type=int, default=0)
	# Classifier parameters (TODO: MAKE IT TASK DEPENDENT)
	parser.add_argument("--fc_dim", help="Number of hidden units in fully connected layers (classifier)", type=int, default=512)
	parser.add_argument("--fc_dropout", help="Dropout probability in FC classifier", type=float, default=0.0)
	parser.add_argument("--fc_nonlinear", help="Whether to add a non-linearity (tanh) between classifier layers or not", action="store_true")
	# Output control
	parser.add_argument("--seed", help="Seed to make experiments reproducable", type=int, default=42)
	parser.add_argument("--cluster", help="Enable option if code is executed on cluster. Reduces output size", action="store_true")
	parser.add_argument("-d", "--debug", help="Whether debug output should be activated or not", action="store_true")
	# Tasks
	parser.add_argument("--task_SNLI", help="Frequency with which the task SNLI should be used. Default: 0 (not used at all)", type=float, default=0)
	parser.add_argument("--task_SNLI_head", help="Specification of SNLI task head. Use string encoding, for example: \"--task_SNLI_head model=0,dp=0.5,dim=300\". Default: use default values defined by parameters \"fc_dim\" etc.", type=str, default="")
	parser.add_argument("--task_POS", help="Frequency with which the task POS tagging should be used. Default: 0 (not used at all)", type=float, default=0)
	parser.add_argument("--task_SST", help="Frequency with which the task Stanford Sentiment Treebank should be used. Default: 0 (not used at all)", type=float, default=0)
	parser.add_argument("--task_SST_head", help="Specification of SST task head. Use string encoding, for example: \"--task_SST_head model=0,dp=0.5,dim=300\". Default: use default values defined by parameters \"fc_dim\" etc.", type=str, default="")
	parser.add_argument("--task_VUA", help="Frequency with which the task VUMetaphor should be used. Default: 0 (not used at all)", type=float, default=0)
	parser.add_argument("--task_VUA_head", help="Specification of VUA task head. Use string encoding, for example: \"--task_VUA_head model=0,dp=0.5,dim=300\". Default: use default values defined by parameters \"fc_dim\" etc.", type=str, default="")
	parser.add_argument("--task_VUAseq", help="Frequency with which the task VU Sequential Metaphor should be used. Default: 0 (not used at all)", type=float, default=0)
	parser.add_argument("--task_VUAseq_head", help="Specification of VUA Sequential task head. Use string encoding, for example: \"--task_VUA_head model=0,dp=0.5,dim=300\". Default: use default values defined by parameters \"fc_dim\" etc.", type=str, default="")
	# Multitask training
	parser.add_argument("--multi_epoch_size", help="Size of epoch for which the batch indices are shuffled", type=int, default=1e3)
	parser.add_argument("--multi_batchwise", help="Whether the multi-task learning should be done per batch, or the elements within a batch come from multiple tasks", action="store_true")
	parser.add_argument("--anti_curriculum_learning", help="If enabled, hard tasks will be trained for a couple of iterations before \"easy\" tasks are added.", action="store_true")
	# Optimizer parameters
	parser.add_argument("--learning_rate", help="Learning rate of the optimizer", type=float, default=0.1)
	parser.add_argument("--lr_decay", help="Decay of learning rate of the optimizer. Always applied if eval accuracy droped compared to mean of last two epochs", type=float, default=0.2)
	parser.add_argument("--lr_decay_step", help="Number of steps after which learning rate should be decreased", type=float, default=1e6)
	parser.add_argument("--lr_max_red_steps", help="Maximum number of times learning rate should be decreased before terminating", type=int, default=4)
	parser.add_argument("--weight_decay", help="Weight decay of the optimizer", type=float, default=0.0)
	parser.add_argument("--optimizer", help="Which optimizer to use. 0: SGD, 1: Adam", type=int, default=0)
	parser.add_argument("--momentum", help="Apply momentum to SGD optimizer", type=float, default=0.0)

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
	tasks, model_type, model_params, optimizer_params, multitask_params = args_to_params(args)
	trainModule = MultiTaskTrain(tasks=tasks,
								 model_type=args.model, 
								 model_params=model_params,
								 optimizer_params=optimizer_params, 
								 multitask_params=multitask_params,
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

	trainModule.train_model(args.max_iterations, loss_freq=loss_freq, enable_tensorboard=args.tensorboard, eval_freq=args.eval_freq, save_freq=args.save_freq)

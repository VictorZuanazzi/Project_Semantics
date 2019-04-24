import torch 
import torch.nn
import argparse
import math
import os
import sys
import logging
from glob import glob

from model import NLIModel
from data import load_SNLI_datasets, debug_level, NLIData, SNLIDataset
from mutils import load_model, load_model_from_args, load_args, args_to_params

# Sent initials for SentEval
PATH_TO_SENTEVAL = "../../SentEval"
PATH_TO_DATA = PATH_TO_SENTEVAL + '/data'
sys.path.insert(0, PATH_TO_SENTEVAL)

import importlib
spam_spec = importlib.util.find_spec("senteval")
FOUND_SENTEVAL = spam_spec is not None
if FOUND_SENTEVAL:
	import senteval
else:
	print("[!] WARNING: Could not find senteval!")

def create_model(checkpoint_path, model_type, model_params):
	_, _, _, word2vec, word2id, wordvec_tensor = load_SNLI_datasets(debug_dataset = True)
	model = NLIModel(model_type, model_params, wordvec_tensor)
	_ = load_model(checkpoint_path, model=model)
	for param in model.parameters():
		param.requires_grad = False
	model.eval()
	return model

# SentEval prepare and batcher
UNKNOWN_WORDS = dict()
def prepare(params, samples):
	global UNKNOWN_WORDS
	_, _, _, _, word2id, _ = load_SNLI_datasets(debug_dataset = True)
	params.word2id = word2id
	# for s in samples:
	# 	print(s)
	words = ' '.join([' '.join([w if isinstance(w, str) else w.decode('UTF-8') for w in s]).lower() for s in samples]).split(" ")

	for w in words:
		if w not in word2id:
			UNKNOWN_WORDS[w] = ''
	print("Number of unknown words: " + str(len(UNKNOWN_WORDS.keys())))
	# sys.exit(1)
	with open("senteval_unknown_words.txt", "w") as f:
		f.write("\n".join(list(UNKNOWN_WORDS.keys())))
	return

def batcher(params, batch):
	global MODEL
	data_batch = list()
	for sent in batch:
		str_sent = " ".join([w if isinstance(w, str) else w.decode('UTF-8') for w in sent])
		new_d = NLIData(premise=str_sent, hypothesis='.', label=-1)
		new_d.translate_to_dict(params.word2id)
		data_batch.append(new_d)
	sents, lengths, _ = SNLIDataset.sents_to_Tensors([[d.premise_vocab for d in data_batch]], toTorch=True)
	
	sent_embeddings = MODEL.encode_sentence(sents[0], lengths[0])
	return sent_embeddings.cpu()

def perform_SentEval(model, fast_eval=False):
	global MODEL, FOUND_SENTEVAL
	MODEL = model
	# Set params for SentEval
	if fast_eval:
		params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': torch.cuda.is_available(), 'kfold': 5}
		params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
										 'tenacity': 3, 'epoch_size': 2}
		transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICKEntailment', 'SICKRelatedness', 'STS14']
	else:
		params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': torch.cuda.is_available(), 'kfold': 5}
		params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
		transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICKEntailment', 'SICKRelatedness', 'STS14', 'ImageCaptionRetrieval']
		
	# Set up logger
	logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

	if FOUND_SENTEVAL:
		se = senteval.engine.SE(params_senteval, batcher, prepare)
		# transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
		# 					'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
		# 					'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
		# 					'Length', 'WordContent', 'Depth', 'TopConstituents',
		# 					'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
		# 					'OddManOut', 'CoordinationInversion']
		results = se.eval(transfer_tasks)
		print(results)
	else:
		results = None
	return results


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints are saved", type=str, required=True)
	parser.add_argument("--fast", help="Whether to use the fast evaluation setting or the full version", action="store_true")
	args = parser.parse_args()
	model = load_model_from_args(load_args(args.checkpoint_path), args.checkpoint_path)
	for param in model.parameters():
		param.requires_grad = False

	perform_SentEval(model, args.fast)

	with open("senteval_unknown_words.txt", "w") as f:
		f.write("\n".join(list(UNKNOWN_WORDS.keys())))


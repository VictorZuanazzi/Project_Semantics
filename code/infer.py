import torch 
import torch.nn
import argparse
import math
import os
import sys
from glob import glob

from model import NLIModel
from data import load_SNLI_datasets, debug_level, NLIData, SNLIDataset
from mutils import load_model, load_model_from_args, load_args, args_to_params



def create_dataset_from_file(input_file, load_file=True):
	data_list = list()
	if load_file:
		with open(input_file, "r") as f:
			lines = f.readlines()
	else:
		lines = input_file.split("\n")

	for line in lines:
		if not "#SEP#" in line:
			print("[!] WARNING: Every line in the input file must contain the string \"#SEP#\" in between the premise and the hypothesis. Skipping line \"%s\"" % (line))
			continue
		line_sep = line.replace("\n","").split("#SEP#")
		premise = line_sep[0]
		hypothesis = line_sep[1]
		data_point = NLIData(premise, hypothesis, -1)
		data_list.append(data_point)
		print("Premise: " + data_point.get_premise() + "\nHypothesis: " + data_point.get_hypothesis())

	infer_dataset = SNLIDataset(data_type="infer", data_path=None, shuffle_data=False)
	infer_dataset.set_data_list(data_list)
	_, _, _, _, word2id, _ = load_SNLI_datasets(debug_dataset = False)
	infer_dataset.set_vocabulary(word2id)
	return infer_dataset

def run_inference(model, input_file, output_file=None, batch_size=64, load_file=True):
	infer_dataset = create_dataset_from_file(input_file, load_file=load_file)

	num_batches = int(math.ceil(infer_dataset.get_num_examples()*1.0/batch_size))
	predictions = list()
	for batch_index in range(num_batches):
		
		if debug_level() == 0:
			print("Inference process: %4.2f%%" % (100.0 * batch_index / num_batches), end="\r")
		
		embeds, lengths, _ = infer_dataset.get_batch(batch_size, loop_dataset=False, toTorch=True)
		preds = model(words_s1 = embeds[0], lengths_s1 = lengths[0], words_s2 = embeds[1], lengths_s2 = lengths[1], applySoftmax=True)
		_, pred_labels = torch.max(preds, dim=-1)
		out = torch.squeeze(pred_labels).tolist()
		predictions += out if isinstance(out, list) else [out]
		print(preds)

	out_s = ""
	for i in range(len(infer_dataset.data_list)):
		out_s += "="*100 + "\n"
		out_s += " Example %i\n" % (i+1) 
		out_s += "-"*100 + "\n"
		out_s += " Premise: " + infer_dataset.data_list[i].get_premise() + "\n"
		out_s += " Hypothesis: " + infer_dataset.data_list[i].get_hypothesis() + "\n"
		out_s += " Prediction: " + NLIData.label_to_string(predictions[i]) + "\n"
		out_s += "="*100 + "\n\n"

	if output_file is not None:
		with open(output_file, "w") as f:
			f.write(out_s)

	print(out_s)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints are saved", type=str, required=True)
	parser.add_argument("--input_file", help="Input file which contains the sentences. Format of each line: \"premise\" #SEP# \"hypothesis\"", type=str, required=True)
	parser.add_argument("--output_file", help="File to which the predictions should be written out. Default: infer_out.txt", type=str, default="infer_out.txt")
	args = parser.parse_args()
	model = load_model_from_args(load_args(args.checkpoint_path), args.checkpoint_path, load_best_model=True)

	run_inference(model, args.input_file, args.output_file, load_file=True)

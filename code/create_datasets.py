import nltk
from glob import glob

from data import DatasetHandler, POSData, POSDataset, SentData


def create_POS_dataset(text_files, output_dir="../data/POS/", suffix="train"):
	if not isinstance(text_files, list):
		text_files = list(text_files)
	lines = list()
	for txt_file in text_files:
		with open(txt_file, "r") as f:
			lines += f.readlines()
			lines = list(set(lines))
	data_batch = list()
	for line_index, line in enumerate(lines):
		print("Processed %4.2f%% of the dataset..." % (100.0 * line_index / len(lines)), end="\r")
		tokenized_data = SentData._preprocess_sentence(line)
		try:
			pos_tags = [w[1] for w in nltk.pos_tag(tokenized_data, tagset="universal")]
		except IndexError:
			print("Problems with sentence: " + str(tokenized_data)+ " (originally: " + line + ")")
		new_d = POSData(sentence=" ".join(tokenized_data), pos_tags=pos_tags)
		data_batch.append(new_d)
	print("Dataset successfully parsed")
	dataset = POSDataset("all", data_path=None)
	dataset.set_data_list(data_batch)
	print("Exporting to file...")
	dataset.export_to_file(output_dir=output_dir, suffix=suffix)
	print("Finished")


if __name__ == '__main__':
	for suffix in ["dev", "test", "train"]:
		create_POS_dataset(text_files=["../data/snli_1.0/s1."+suffix, "../data/snli_1.0/s2."+suffix], suffix=suffix+"_snli")
	for suffix, out_suffix in zip(["dev.matched", "dev.mismatched", "train"], ["dev", "test", "train"]):
		create_POS_dataset(text_files=["../data/multinli_1.0/s1."+suffix, "../data/multinli_1.0/s2."+suffix], suffix=out_suffix+"_mnli")
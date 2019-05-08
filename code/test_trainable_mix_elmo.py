from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
from time import time

options_file = 'elmo/medium/elmo_2x2048_256_2048cnn_1xhighway_options.json'
weight_file = 'elmo/medium/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file,
						num_output_representations=1)

# use batch_to_ids to convert sentences to character ids
sentences = [['Victor', 'is', 'a', 'good', 'fella', '.'],
             ['Frank', 'is', 'better', 'tho', '.'],
             ['I', 'beg', 'to', 'disagree', '.'],
             ['All', 'is', 'well', 'which', 'ends', 'well', '.']]

start = time()

character_ids = batch_to_ids(sentences)

elmo.eval()
embeddings = elmo(character_ids)

elmo_embeds = embeddings['elmo_representations'][0]

print("Elapsed:", round(time() - start, 6), "s")

print(elmo_embeds.shape)
print(elmo_embeds[0, 1, 0:10])

from allennlp.commands.elmo import ElmoEmbedder
import torch
from time import time

options_file = 'elmo/medium/elmo_2x2048_256_2048cnn_1xhighway_options.json'
weight_file = 'elmo/medium/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'

elmo = ElmoEmbedder(options_file, weight_file)

sentences = [['Victor', 'is', 'a', 'good', 'fella', '.'],
             ['Frank', 'is', 'better', 'tho', '.'],
             ['I', 'beg', 'to', 'disagree', '.'],
             ['All', 'is', 'well', 'which', 'ends', 'well', '.']]

start = time()

# 'vectors' represents activations (batch_size, 3, num_timesteps, 1024)
vectors = elmo.batch_to_embeddings(sentences)[0]

# concatenate the two interesting layers
elmo_embeds = torch.cat((vectors[:, 1], vectors[:, 2]), 2)

print(elmo_embeds.shape)

print("Elapsed:", round(time() - start, 6), "s")

print(elmo_embeds.shape)
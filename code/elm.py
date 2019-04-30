from allennlp.modules.elmo import Elmo, batch_to_ids

# options_file = open('elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json', 'rb')
# weight_file = open('elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5', 'rb')
options_file = 'elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json'
weight_file = 'elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

# Compute two different representations for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file, 2, dropout=0)

# options_file.close()
# weight_file.close()

# use batch_to_ids to convert sentences to character ids
sentences = [['Frank', '.'], ['A', 'good', 'name', '.']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)

# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector

print("First layer:", embeddings['elmo_representations'][0].size())
print("Second layer:", embeddings['elmo_representations'][1].size())

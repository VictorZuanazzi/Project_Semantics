import torch
import torch.nn as nn
import numpy as np 
import sys
import math


class MultiTaskEncoder(nn.Module):

	AVERAGE_WORD_VECS = 0
	LSTM = 1
	BILSTM = 2
	BILSTM_MAX = 3

	def __init__(self, model_type, model_params, wordvec_tensor):
		super(MultiTaskEncoder, self).__init__()

		self.embeddings = nn.Embedding(wordvec_tensor.shape[0], wordvec_tensor.shape[1])
		with torch.no_grad():
			self.embeddings.weight.data.copy_(torch.from_numpy(wordvec_tensor))
			self.embeddings.weight.requires_grad = False

		self.model_type = model_type
		self.model_params = model_params
		self._choose_encoder(model_type, model_params)

		if torch.cuda.is_available():
			self.embeddings = self.embeddings.cuda()
			self.encoder = self.encoder.cuda()


	def _choose_encoder(self, model_type, model_params):
		if model_type == MultiTaskEncoder.AVERAGE_WORD_VECS:
			self.encoder = EncoderBOW()
		elif model_type == MultiTaskEncoder.LSTM:
			self.encoder = EncoderLSTM(model_params)
		elif model_type == MultiTaskEncoder.BILSTM:
			self.encoder = EncoderBILSTM(model_params)
		elif model_type == MultiTaskEncoder.BILSTM_MAX:
			self.encoder = EncoderBILSTMPool(model_params)
		else:
			print("Unknown encoder: " + str(model_type))
			sys.exit(1)


	def forward(self, words, lengths, dummy_input=False, debug=False):
		return self.encode_sentence(words, lengths, dummy_input=dummy_input, debug=debug)


	def encode_sentence(self, words, lengths, dummy_input=False, debug=False):
		word_embeds = self.embeddings(words)
		sent_embeds = self.encoder(word_embeds, lengths, dummy_input=dummy_input, debug=debug)
		return sent_embeds



class NLIClassifier(nn.Module):

	def __init__(self, model_params):
		super(NLIClassifier, self).__init__()
		embed_sent_dim = model_params["embed_sent_dim"]
		fc_dropout = model_params["fc_dropout"] 
		fc_dim = model_params["fc_dim"]
		n_classes = model_params["n_classes"]

		input_dim = 4 * embed_sent_dim
		if model_params["fc_nonlinear"]:
			self.classifier = nn.Sequential(
				nn.Dropout(p=fc_dropout),
				nn.Linear(input_dim, fc_dim),
				nn.Tanh(),
				nn.Linear(fc_dim, n_classes)
			)
		else:
			self.classifier = nn.Sequential(
				nn.Dropout(p=fc_dropout),
				nn.Linear(input_dim, fc_dim),
				nn.Linear(fc_dim, n_classes)
			)
		self.softmax_layer = nn.Softmax(dim=-1)


	def forward(self, embed_s1, embed_s2, applySoftmax=False):
		input_features = torch.cat((embed_s1, embed_s2, 
									torch.abs(embed_s1 - embed_s2), 
									embed_s1 * embed_s2), dim=1)

		out = self.classifier(input_features)
		if applySoftmax:
			out = self.softmax_layer(out)
		
		return out

####################
## ENCODER MODELS ##
####################

class EncoderModule(nn.Module):

	def __init__(self):
		super(EncoderModule, self).__init__()

	def forward(self, embed_words, lengths, dummy_input=False, debug=False):
		raise NotImplementedError


class EncoderBOW(EncoderModule):

	def __init__(self):
		super(EncoderBOW, self).__init__()

	def forward(self, embed_words, lengths, dummy_input=False, debug=False):
		# Embeds are of shape [batch, time, embed_dim]
		# Lengths is of shape [batch]
		word_positions = torch.arange(start=0, end=embed_words.shape[1], dtype=lengths.dtype, device=embed_words.device)
		mask = (word_positions.reshape(shape=[1, -1, 1]) < lengths.reshape([-1, 1, 1])).float()
		out = torch.sum(mask * embed_words, dim=1) / lengths.reshape([-1, 1]).float()
		return out


class EncoderLSTM(EncoderModule):

	def __init__(self, model_params):
		super(EncoderLSTM, self).__init__()
		self.lstm_chain = PyTorchLSTMChain(input_size=model_params["embed_word_dim"], 
									hidden_size=model_params["embed_sent_dim"])

	def forward(self, embed_words, lengths, dummy_input=False, debug=False):
		final_states, _ = self.lstm_chain(embed_words, lengths, dummy_input=dummy_input)
		return final_states


class EncoderBILSTM(EncoderModule):

	def __init__(self, model_params):
		super(EncoderBILSTM, self).__init__()
		self.lstm_chain = PyTorchLSTMChain(input_size=model_params["embed_word_dim"], 
										   hidden_size=int(model_params["embed_sent_dim"]/2),
										   bidirectional=True)

	def forward(self, embed_words, lengths, dummy_input=False, debug=False):
		# embed words is of shape [batch_size, time, word_dim]
		final_states, _ = self.lstm_chain(embed_words, lengths, dummy_input=dummy_input)
		return final_states


class EncoderBILSTMPool(EncoderModule):

	def __init__(self, model_params, skip_connections=False):
		super(EncoderBILSTMPool, self).__init__()
		self.lstm_chain = PyTorchLSTMChain(input_size=model_params["embed_word_dim"], 
										   hidden_size=int(model_params["embed_sent_dim"]/2),
										   bidirectional=True)

	def forward(self, embed_words, lengths, dummy_input=False, debug=False):
		# embed words is of shape [batch_size * 2, time, word_dim]
		_, outputs = self.lstm_chain(embed_words, lengths, dummy_input=dummy_input)
		# Max time pooling
		pooled_features, pool_indices = EncoderBILSTMPool.pool_over_time(outputs, lengths)
		if debug:
			return pooled_features, pool_indices
		else:
			return pooled_features

	@staticmethod
	def pool_over_time(outputs, lengths):
		time_dim = outputs.shape[1]
		word_positions = torch.arange(start=0, end=time_dim, dtype=lengths.dtype, device=outputs.device)
		mask = (word_positions.reshape(shape=[1, -1, 1]) < lengths.reshape([-1, 1, 1])).float()
		outputs = outputs * mask + (torch.min(outputs) - 1) * (1 - mask)
		final_states, max_indices = torch.max(outputs, dim=1)
		return final_states, max_indices



###################################
## LOW LEVEL LSTM IMPLEMENTATION ##
###################################

class PyTorchLSTMChain(nn.Module):

	def __init__(self, input_size, hidden_size, bidirectional=False):
		super(PyTorchLSTMChain, self).__init__()
		self.lstm_cell = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional)
		self.hidden_size = hidden_size

	def forward(self, word_embeds, lengths, dummy_input=False):
		batch_size = word_embeds.shape[0]
		time_dim = word_embeds.shape[1]
		embed_dim = word_embeds.shape[2]

		# For graph creation: dummy function 
		if dummy_input:
			outputs, (final, _) = self.lstm_cell(word_embeds)
			return final, outputs


		sorted_lengths, perm_index = lengths.sort(0, descending=True)
		word_embeds = word_embeds[perm_index]

		packed_word_embeds = torch.nn.utils.rnn.pack_padded_sequence(word_embeds, sorted_lengths, batch_first=True)
		packed_outputs, (final_hidden_states, _) = self.lstm_cell(packed_word_embeds)
		outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

		# Redo sort
		_, unsort_indices = perm_index.sort(0, descending=False)
		outputs = outputs[unsort_indices]
		final_hidden_states = final_hidden_states[:,unsort_indices]

		final_states = final_hidden_states.transpose(0, 1).transpose(1, 2)
		final_states = final_states.reshape([batch_size, self.hidden_size * final_states.shape[-1]])

		return final_states, outputs # final



##################
## PSEUDO TESTS ##
##################

class ModuleTests():

	def __init__(self):
		pass

	def testEncoderBOW(self):
		enc = EncoderBOW()
		input_embeds = torch.ones((4, 16, 4))
		lengths = torch.Tensor(np.array([16, 8, 4, 2]))
		out = enc(input_embeds, lengths)
		print("Result: " + str(out))


	def testBiLSTMReshaping(self):
		output = torch.FloatTensor(np.array([[1, 3], [-2, -2], [2, -4], [-6, -1], [0, 0], [0, 0], [1, 2], [3, 4]], dtype=np.float32))
		output = output.reshape(shape=[-1, 2 * output.shape[1]])
		print("Result: " + str(output))

	def testTimePooling(self):
		output = torch.FloatTensor(np.array([[[1, 3], [-2, -2], [5, 1]], [[2, -4], [-6, -1], [0, 0]]], dtype=np.float32))
		lengths = torch.LongTensor(np.array([2, 3]))
		res = EncoderBILSTMPool.pool_over_time(output, lengths)
		print("Result: " + str(res))

	def testSorting(self):
		vals = torch.FloatTensor([1, 7, 5, 8, 2, 2, 10, 0])
		sorted_vals, perm_index = vals.sort(0, descending=True)
		_, unsort_indices = perm_index.sort(0, descending=False)
		print("Indices: " + str(perm_index))
		print("Unsort indices: " + str(unsort_indices))
		print("Vals: " + str(vals))
		print("Vals sorted: " + str(vals[perm_index]))
		print("Vals unsorted: " + str(vals[perm_index][unsort_indices]))


if __name__ == '__main__':
	print(torch.__version__)
	tester = ModuleTests()
	tester.testEncoderBOW()
	tester.testTimePooling()
	tester.testBiLSTMReshaping()
	tester.testSorting()







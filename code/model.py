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


	def forward(self, words, lengths, debug=False):
		return self.encode_sentence(words, lengths, debug=debug)


	def encode_sentence(self, words, lengths, word_level=False, debug=False):
		word_embeds = self.embeddings(words)
		sent_embeds = self.encoder(word_embeds, lengths, word_level=word_level)
		return sent_embeds


####################################
## CLASSIFIER/TASK-SPECIFIC HEADS ##
####################################

class ClassifierHead(nn.Module):
	"""
		Base class for classifiers and heads.
		Inputs:
			* model_params: dict of all possible parameters that might be necessary to specify model
			* word_level: Whether the classifier operates on word level of a sentence to get single label or not.
						  A classifier that operates on sequence level is included in "False" (as it could also be applied on sentences)
	"""
	def __init__(self, model_params, word_level=False):
		super(ClassifierHead, self).__init__()
		self.model_params = model_params
		self.word_level = word_level

	def forward(self, input):
		raise NotImplementedError

	def is_word_level(self):
		return self.word_level


class SimpleClassifier(ClassifierHead):
	"""
		General purpose classifier with shallow MLP. 
	"""
	def __init__(self, model_params, num_classes):
		super(SimpleClassifier, self).__init__(model_params, word_level=False)
		embed_sent_dim = model_params["embed_sent_dim"]
		fc_dropout = model_params["fc_dropout"] 
		fc_dim = model_params["fc_dim"]

		input_dim = embed_sent_dim
		self.classifier = nn.Sequential(
			nn.Dropout(p=fc_dropout),
			nn.Linear(input_dim, fc_dim),
			nn.ReLU(),
			nn.Linear(fc_dim, num_classes)
		)
		self.softmax_layer = nn.Softmax(dim=-1)


	def forward(self, input_features, applySoftmax=False):
		out = self.classifier(input_features)
		if applySoftmax:
			out = self.softmax_layer(out)
		return out


class NLIClassifier(ClassifierHead):
	"""
		Classifier as proposed in InferSent paper. Requires sentence embedding of hypothesis and premise
	"""
	def __init__(self, model_params):
		super(NLIClassifier, self).__init__(model_params, word_level=False)
		embed_sent_dim = model_params["embed_sent_dim"]
		fc_dropout = model_params["fc_dropout"] 
		fc_dim = model_params["fc_dim"]
		n_classes = model_params["nli_classes"]

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


class ESIM_Head(ClassifierHead):
	"""
		Head based on the "Enhanced Sequential Inference Model" (ESIM). Takes cross-attention on word level over premise and hypothesis,
		and has additional Bi-LSTM decoder. 
		Paper: https://www.aclweb.org/anthology/P17-1152
	"""
	def __init__(self, model_params):
		super(ESIM_Head, self).__init__(model_params, word_level=True)
		embed_sent_dim = model_params["embed_sent_dim"]
		fc_dropout = model_params["fc_dropout"] 
		fc_dim = model_params["fc_dim"]
		n_classes = model_params["nli_classes"]
		use_bias = get_param_val(model_params, "use_bias", False)
		self.use_scaling = get_param_val(model_params, "use_scaling", False)
		print(model_params)
		if use_bias:
			self.bias_prem = nn.Parameter(torch.zeros(1), requires_grad=True)
			self.bias_hyp = nn.Parameter(torch.zeros(1), requires_grad=True)
		else:
			self.bias_prem, self.bias_hyp = None, None


		hidden_size = int(embed_sent_dim/2)
		self.projection_layer = nn.Sequential(
			nn.Linear(4 * embed_sent_dim, hidden_size),
			nn.ReLU(),
			nn.Dropout(p=fc_dropout)
		)
		self.BiLSTM_decoder = PyTorchLSTMChain(input_size=hidden_size, 
											   hidden_size=hidden_size,
											   bidirectional=True)
		self.classifier = nn.Sequential(
			nn.Dropout(p=fc_dropout),
			nn.Linear(4 * embed_sent_dim, hidden_size),
			nn.Tanh(),
			nn.Dropout(p=fc_dropout),
			nn.Linear(hidden_size, n_classes)
		)
		self.softmax_layer = nn.Softmax(dim=-1)
		self.last_attention_map = None


	def forward(self, word_embed_premise, length_premise, word_embed_hypothesis, length_hypothesis, applySoftmax=False):
		# Cross attention
		prem_opponent, hyp_opponent = self._cross_attention(word_embed_premise, length_premise, word_embed_hypothesis, length_hypothesis)
		# Decode both sentences
		prem_sentence = self._decode_sentence(word_embed_premise, length_premise, prem_opponent)
		hyp_sentence = self._decode_sentence(word_embed_hypothesis, length_hypothesis, hyp_opponent)
		# Concat both sentence embeddings as input features to classifier
		input_features = torch.cat((prem_sentence, hyp_sentence), dim=1)
		# Final classifier
		out = self.classifier(input_features)
		if applySoftmax:
			out = self.softmax_layer(out)
		return out


	def _decode_sentence(self, word_embeds, lengths, opponents):
		# Combine features
		decode_features = torch.cat((word_embeds, opponents, 
									 torch.abs(word_embeds - opponents), 
									 word_embeds * opponents), dim=2)
		# Projection layer to reduce dimension
		proj_features = self.projection_layer(decode_features)
		# Bi-LSTM decoder
		_, dec_features = self.BiLSTM_decoder(proj_features, lengths)
		# Pooling over last hidden states
		max_features, _ = EncoderBILSTMPool.pool_over_time(dec_features, lengths, pooling='MAX')
		mean_features, _ = EncoderBILSTMPool.pool_over_time(dec_features, lengths, pooling='MEAN')
		# Final sentence embedding
		sent_embed = torch.cat((max_features, mean_features), dim=1)
		return sent_embed


	def _cross_attention(self, word_embed_premise, length_premise, word_embed_hypothesis, length_hypothesis):
		# Function bmm: If batch1 is a (b,n,m) tensor, batch2 is a (b,m,p) tensor, out will be a (b,n,p) tensor.
		similarity_matrix = torch.bmm(word_embed_premise,
									  word_embed_hypothesis.transpose(2, 1).contiguous()) # Shape: [batch, prem len, hyp len]
		if self.use_scaling:
			similarity_matrix = similarity_matrix / math.sqrt(word_embed_premise.shape[-1])
		
		prem_to_hyp_attn = self._masked_softmax(similarity_matrix, length_hypothesis, bias=self.bias_prem)
		hyp_to_prem_attn = self._masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), # Input shape: [batch, hyp len, prem len]
											   length_premise, bias=self.bias_hyp)
		self.last_prem_attention_map = prem_to_hyp_attn.cpu().data.numpy()
		self.last_hyp_attention_map = hyp_to_prem_attn.cpu().data.numpy()

		prem_opponent = torch.bmm(prem_to_hyp_attn, word_embed_hypothesis)
		hyp_opponent = torch.bmm(hyp_to_prem_attn, word_embed_premise)
		return prem_opponent, hyp_opponent


	def _masked_softmax(self, _input, lengths, bias=None):
		word_positions = torch.arange(start=0 if bias is None else -1, end=_input.shape[2], dtype=lengths.dtype, device=_input.device)
		mask = (word_positions.reshape(shape=[1, 1, -1]) < lengths.reshape([-1, 1, 1])).float()
		if bias is not None:
			expanded_bias = bias.view(1,1,1).expand(_input.size(0), _input.size(1), 1)
			_input = torch.cat((expanded_bias, _input), dim=-1)
		softmax_act = self.softmax_layer(_input) * mask
		softmax_act = softmax_act / torch.sum(softmax_act, dim=-1, keepdim=True)
		if bias is not None:
			softmax_act = softmax_act[:,:,1:]
		return softmax_act

	@staticmethod
	def _init_esim_weights(module):
		"""
		Initialise the weights of the ESIM model.
		Copied from https://github.com/coetaur0/ESIM/blob/master/esim/model.py
		TODO: Incorporate it here. Is initialization so crucial here?
		"""
		if isinstance(module, nn.Linear):
			nn.init.xavier_uniform_(module.weight.data)
			nn.init.constant_(module.bias.data, 0.0)

		elif isinstance(module, nn.LSTM):
			nn.init.xavier_uniform_(module.weight_ih_l0.data)
			nn.init.orthogonal_(module.weight_hh_l0.data)
			nn.init.constant_(module.bias_ih_l0.data, 0.0)
			nn.init.constant_(module.bias_hh_l0.data, 0.0)
			hidden_size = module.bias_hh_l0.data.shape[0] // 4
			module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

			if (module.bidirectional):
				nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
				nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
				nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
				nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
				module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0



####################
## ENCODER MODELS ##
####################

class EncoderModule(nn.Module):

	def __init__(self):
		super(EncoderModule, self).__init__()

	def forward(self, embed_words, lengths, word_level=False, debug=False):
		raise NotImplementedError


class EncoderBOW(EncoderModule):

	def __init__(self):
		super(EncoderBOW, self).__init__()

	def forward(self, embed_words, lengths, word_level=False, debug=False):
		# Embeds are of shape [batch, time, embed_dim]
		# Lengths is of shape [batch]
		if not word_level:
			word_positions = torch.arange(start=0, end=embed_words.shape[1], dtype=lengths.dtype, device=embed_words.device)
			mask = (word_positions.reshape(shape=[1, -1, 1]) < lengths.reshape([-1, 1, 1])).float()
			out = torch.sum(mask * embed_words, dim=1) / lengths.reshape([-1, 1]).float()
			return out
		else:
			return embed_words


class EncoderLSTM(EncoderModule):

	def __init__(self, model_params):
		super(EncoderLSTM, self).__init__()
		self.lstm_chain = PyTorchLSTMChain(input_size=model_params["embed_word_dim"], 
									hidden_size=model_params["embed_sent_dim"])

	def forward(self, embed_words, lengths, word_level=False, debug=False):
		final_states, word_outputs = self.lstm_chain(embed_words, lengths)
		if not word_level:
			return final_states
		else:
			return word_outputs


class EncoderBILSTM(EncoderModule):

	def __init__(self, model_params):
		super(EncoderBILSTM, self).__init__()
		self.lstm_chain = PyTorchLSTMChain(input_size=model_params["embed_word_dim"], 
										   hidden_size=int(model_params["embed_sent_dim"]/2),
										   bidirectional=True)

	def forward(self, embed_words, lengths, word_level=False, debug=False):
		# embed words is of shape [batch_size, time, word_dim]
		final_states, word_outputs = self.lstm_chain(embed_words, lengths)
		if not word_level:
			return final_states
		else:
			return word_outputs


class EncoderBILSTMPool(EncoderModule):

	def __init__(self, model_params, skip_connections=False):
		super(EncoderBILSTMPool, self).__init__()
		self.lstm_chain = PyTorchLSTMChain(input_size=model_params["embed_word_dim"], 
										   hidden_size=int(model_params["embed_sent_dim"]/2),
										   bidirectional=True)

	def forward(self, embed_words, lengths, word_level=False, debug=False):
		# embed words is of shape [batch_size * 2, time, word_dim]
		_, word_outputs = self.lstm_chain(embed_words, lengths)
		if not word_level:
			# Max time pooling
			pooled_features, pool_indices = EncoderBILSTMPool.pool_over_time(word_outputs, lengths, pooling='MAX')
			if debug:
				return pooled_features, pool_indices
			else:
				return pooled_features
		else:
			return word_outputs

	@staticmethod
	def pool_over_time(outputs, lengths, pooling='MAX'):
		time_dim = outputs.shape[1]
		word_positions = torch.arange(start=0, end=time_dim, dtype=lengths.dtype, device=outputs.device)
		mask = (word_positions.reshape(shape=[1, -1, 1]) < lengths.reshape([-1, 1, 1])).float()
		if pooling == 'MAX':
			outputs = outputs * mask + (torch.min(outputs) - 1) * (1 - mask)
			final_states, pool_indices = torch.max(outputs, dim=1)
		elif pooling == 'MIN':
			outputs = outputs * mask + (torch.max(outputs) + 1) * (1 - mask)
			final_states, pool_indices = torch.min(outputs, dim=1)
		elif pooling == 'MEAN':
			final_states = torch.sum(outputs * mask, dim=1) / lengths.reshape([-1, 1]).float()
			pool_indices = None
		else:
			print("[!] ERROR: Unknown pooling option \"" + str(pooling) + "\"")
			sys.exit(1)
		return final_states, pool_indices
	



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


#####################
## Helper function ##
#####################

def get_param_val(param_dict, key, default_val):
	if key in param_dict:
		return param_dict[key]
	else:
		return default_val

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







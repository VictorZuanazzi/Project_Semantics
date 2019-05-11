import torch
import torch.nn as nn
import numpy as np 
import sys
import math

def get_device():
	return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
			self.embeddings.weight.requires_grad = get_param_val(model_params, "finetune_embeds", False)
		self.embed_dropout = RNNDropout(get_param_val(model_params, "embed_dropout", 0.0))

		self.model_type = model_type
		self.model_params = model_params
		self._choose_encoder(model_type, model_params)

		self.embeddings = self.embeddings.to(get_device())
		self.encoder = self.encoder.to(get_device())

		self.apply(ESIM_Head._init_esim_weights)
		print("Encoder: \n" + str(self.encoder))


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


	def encode_sentence(self, words, lengths, word_level=False, debug=False, layer=-1):
		word_embeds = self.embeddings(words)
		word_embeds = self.embed_dropout(word_embeds)
		sent_embeds = self.encoder(word_embeds, lengths, word_level=word_level, max_layer=layer)
		if not word_level:
			sent_embeds = sent_embeds[-1]
		else:
			sent_embeds = [p[-1] for p in sent_embeds]
		return sent_embeds


	def get_layer_size(self, layer):
		return self.encoder.lstm_stack.layer_size(layer)


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
		fc_nonlinear = get_param_val(model_params, "fc_nonlinear", False)
		fc_num_layers = get_param_val(model_params, "fc_num_layers", 1)

		input_dim = embed_sent_dim
		layer_list = list()
		for n in range(fc_num_layers):
			layer_list.append(nn.Dropout(p=fc_dropout))
			layer_list.append(nn.Linear(input_dim, fc_dim))
			if fc_nonlinear:
				layer_list.append(nn.ReLU())
			input_dim = fc_dim

		layer_list.append(nn.Dropout(p=fc_dropout))
		layer_list.append(nn.Linear(input_dim, num_classes))

		self.classifier = nn.Sequential(*layer_list)
		self.softmax_layer = nn.Softmax(dim=-1)
		self.to(get_device())


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
		self.to(get_device())


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
		fc_dropout = get_param_val(model_params, "fc_dropout", 0.0) 
		n_classes = get_param_val(model_params, "nli_classes", 3)
		use_bias = get_param_val(model_params, "use_bias", False)
		self.use_scaling = get_param_val(model_params, "use_scaling", False)
		attn_proj_dim = get_param_val(model_params, "attn_proj", 0)

		if use_bias:
			self.bias_prem = nn.Parameter(torch.zeros(1), requires_grad=True)
			self.bias_hyp = nn.Parameter(torch.zeros(1), requires_grad=True)
		else:
			self.bias_prem, self.bias_hyp = None, None

		if attn_proj_dim == 0:
			self.attn_projection = None # nn.Identity()
		else:
			self.attn_projection = nn.Linear(embed_sent_dim, attn_proj_dim)

		hidden_size = int(embed_sent_dim/2)
		self.projection_layer = nn.Sequential(
			nn.Linear(4 * embed_sent_dim, hidden_size),
			nn.ReLU(),
			RNNDropout(p=fc_dropout)
		)
		self.BiLSTM_decoder = PyTorchLSTMChain(input_size=hidden_size, 
											   hidden_size=hidden_size,
											   per_direction=True,
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

		self.apply(ESIM_Head._init_esim_weights)
		self.to(get_device())


	def forward(self, word_embed_premise, length_premise, word_embed_hypothesis, length_hypothesis, applySoftmax=False):
		# Cross attention
		prem_to_hyp_attn, hyp_to_prem_attn = self._cross_attention(word_embed_premise, length_premise, word_embed_hypothesis, length_hypothesis)
		prem_opponent = torch.bmm(prem_to_hyp_attn, word_embed_hypothesis)
		hyp_opponent = torch.bmm(hyp_to_prem_attn, word_embed_premise)
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
									 (word_embeds - opponents), # torch.abs
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
		if self.attn_projection is not None:
			word_embed_premise = self.attn_projection(word_embed_premise)
			word_embed_hypothesis = self.attn_projection(word_embed_hypothesis)
		similarity_matrix = torch.bmm(word_embed_premise,
									  word_embed_hypothesis.transpose(2, 1).contiguous()) # Shape: [batch, prem len, hyp len]
		if self.use_scaling:
			similarity_matrix = similarity_matrix / math.sqrt(word_embed_premise.shape[-1])
		
		prem_to_hyp_attn = self._masked_softmax(similarity_matrix, length_hypothesis, bias=self.bias_prem)
		hyp_to_prem_attn = self._masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), # Input shape: [batch, hyp len, prem len]
											   length_premise, bias=self.bias_hyp)
		self.last_prem_attention_map = prem_to_hyp_attn.cpu().data.numpy()
		self.last_hyp_attention_map = hyp_to_prem_attn.cpu().data.numpy()

		return prem_to_hyp_attn, hyp_to_prem_attn


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

	def forward(self, embed_words, lengths, word_level=False, max_layer=-1, debug=False):
		raise NotImplementedError


class EncoderBOW(EncoderModule):

	def __init__(self):
		super(EncoderBOW, self).__init__()

	def forward(self, embed_words, lengths, word_level=False, max_layer=-1, debug=False):
		# Embeds are of shape [batch, time, embed_dim]
		# Lengths is of shape [batch]
		word_positions = torch.arange(start=0, end=embed_words.shape[1], dtype=lengths.dtype, device=embed_words.device)
		mask = (word_positions.reshape(shape=[1, -1, 1]) < lengths.reshape([-1, 1, 1])).float()
		out = torch.sum(mask * embed_words, dim=1) / lengths.reshape([-1, 1]).float()
		if not word_level:
			return [out]
		else:
			return [out], [embed_words]


class EncoderLSTM(EncoderModule):

	def __init__(self, model_params):
		super(EncoderLSTM, self).__init__()
		self.lstm_stack = create_StackedLSTM_from_params(model_params, bidirectional=False)

	def forward(self, embed_words, lengths, word_level=False, max_layer=-1, debug=False):
		final_states, word_outputs = self.lstm_stack(embed_words, lengths, max_layer=max_layer)
		if not word_level:
			return final_states
		else:
			return final_states, word_outputs


class EncoderBILSTM(EncoderModule):

	def __init__(self, model_params):
		super(EncoderBILSTM, self).__init__()
		self.lstm_stack = create_StackedLSTM_from_params(model_params, bidirectional=True)

	def forward(self, embed_words, lengths, word_level=False, max_layer=-1, debug=False):
		# embed words is of shape [batch_size, time, word_dim]
		final_states, word_outputs = self.lstm_stack(embed_words, lengths, max_layer=max_layer)
		if not word_level:
			return final_states
		else:
			return final_states, word_outputs


class EncoderBILSTMPool(EncoderModule):

	def __init__(self, model_params, skip_connections=False):
		super(EncoderBILSTMPool, self).__init__()
		self.lstm_stack = create_StackedLSTM_from_params(model_params, bidirectional=True)

	def forward(self, embed_words, lengths, word_level=False, max_layer=-1, debug=False):
		# embed words is of shape [batch_size * 2, time, word_dim]
		_, word_outputs = self.lstm_stack(embed_words, lengths, max_layer=max_layer)
		# Max time pooling
		pooled_features = list()
		for n_layer in range(len(word_outputs)):
			pooled_layer_features, pool_indices = EncoderBILSTMPool.pool_over_time(word_outputs[n_layer], lengths, pooling='MAX')
			pooled_features.append(pooled_layer_features)
		if not word_level:
			if debug:
				return pooled_features, pool_indices # Are the pooling indices for the last layer
			else:
				return pooled_features
		else:
			return pooled_features, word_outputs

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

def create_StackedLSTM_from_params(model_params, bidirectional=False):
	input_size = get_param_val(model_params, "embed_word_dim", 0)
	hidden_dims = get_param_val(model_params, "hidden_dims", list())
	hidden_dims.append(get_param_val(model_params, "embed_sent_dim", 2048))
	projection_dims = get_param_val(model_params, "proj_dims", list())
	if len(projection_dims) == 0:
		projection_dims = None
	else:
		assert len(projection_dims) == (len(hidden_dims) - 1), \
			   "[!] WARNING: The number of projection layers/dimensions (%i) do not fit to the number of hidden dimensions (%i) that are specified." % (len(projection_dims), len(hidden_dims))
	print("Projection dims: " + str(projection_dims))
	projection_dropout = get_param_val(model_params, "proj_dropout", 0.0)
	input_dropout = get_param_val(model_params, "input_dropout", 0.0)
	skip_connections = not get_param_val(model_params, "no_skip_connections", False)

	return StackedLSTMChain(
			input_size=input_size,
			hidden_dims=hidden_dims,
			proj_dims=projection_dims,
			proj_dropout=projection_dropout,
			input_dropout=input_dropout,
			bidirectional=bidirectional,
			skip_connections=skip_connections
		)


class StackedLSTMChain(nn.Module):
	"""
		Stacked (bi)-LSTM chains to increase model complexity. Hyperparameters:
		- Number of layers. Specified by the lengths of the list passed to the input parameter "hidden_dims"
		- Size of the hidden dimensions for each layer. Specified by the numbers in the list of "hidden_dims"
		- Stacking output of all previous layers as input to next layer (as done in DenseNet). Can be deactivated with parameter "skip_connections" 
		- Projection layers between LSTM chains to reduce number of features. If not specified, outputs of LSTM layers are (eventuall concatenated 
			with previous) passed to next layer without any processing. Only adviced if outputs are stacked.
		- NOT IMPLEMENTED YET: layer on top of output of LSTM states. Used for do linear classification on it, guaranteeing that information between
			tasks are shared.
	"""

	def __init__(self, input_size, hidden_dims, proj_dims=None, proj_dropout=0.0, input_dropout=0.0, bidirectional=False, skip_connections=True):
		super(StackedLSTMChain, self).__init__()
		if not isinstance(hidden_dims, list):
			hidden_dims = list(hidden_dims)
		if proj_dims is not None and not isinstance(proj_dims, list):
			proj_dims = [proj_dims] * (len(hidden_dims) - 1)
		self.num_layers = len(hidden_dims)
		self.input_size = input_size
		self.hidden_dims = hidden_dims
		self.proj_dims = proj_dims
		self.proj_dropout = proj_dropout
		self.input_dropout = input_dropout
		self.bidirectional = bidirectional
		self.skip_connections = skip_connections
		self._build_network()

	def _build_network(self):
		self.lstm_chains = list()
		self.proj_layers = list()
		self.input_dropout = RNNDropout(self.input_dropout)
		for n in range(len(self.hidden_dims)):
			if n == 0:
				inp_dim = self.input_size
			else:
				inp_dim = self.hidden_dims[n-1] if not self.skip_connections else (self.input_size + sum(self.hidden_dims[:n]))
				if self.proj_dims is not None:
					self.proj_layers.append(self._get_projection_layer(inp_dim, self.proj_dims[n-1], self.proj_dropout))
					inp_dim = self.proj_dims[n-1]
			n_chain = PyTorchLSTMChain(inp_dim, self.hidden_dims[n], per_direction=False, bidirectional=self.bidirectional)
			self.lstm_chains.append(n_chain)
		self.proj_layers = nn.ModuleList(self.proj_layers)
		self.lstm_chains = nn.ModuleList(self.lstm_chains)

	def _get_projection_layer(self, input_dim, output_dim, output_dropout):
		return nn.Sequential(
				nn.Linear(input_dim, output_dim),
				nn.ReLU(),
				RNNDropout(output_dropout)
			)

	def layer_size(self, layer):
		return self.hidden_dims[layer]


	def forward(self, word_embeds, lengths, max_layer=-1):
		final_states = list()
		outputs = list()
		input_stack = [word_embeds]

		num_layers = max_layer+1 if (max_layer < len(self.lstm_chains) and max_layer >= 0) else len(self.lstm_chains)
		for layer_index in range(num_layers):
			if self.skip_connections:
				stacked_inputs = torch.cat(input_stack, dim=-1)
			else:
				stacked_inputs = input_stack[-1]
			if layer_index > 0 and len(self.proj_layers) > 0:
				stacked_inputs = self.proj_layers[layer_index-1](stacked_inputs)
			layer_states, layer_outputs = self.lstm_chains[layer_index](stacked_inputs, lengths)
			final_states.append(layer_states)
			outputs.append(layer_outputs)
			input_stack.append(self.input_dropout(layer_outputs))
		return final_states, outputs


class PyTorchLSTMChain(nn.Module):

	def __init__(self, input_size, hidden_size, per_direction=False, bidirectional=False):
		super(PyTorchLSTMChain, self).__init__()
		if not per_direction and bidirectional:
			hidden_size = int(hidden_size/2)
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

# Class widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/modules/input_variational_dropout.py
# Here copied from https://github.com/coetaur0/ESIM/blob/master/esim/layers.py
class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.

    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """

    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.

        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).

        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0],
                                             sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training,
                                             inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch


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







import torch
import torch.nn as nn
import numpy as np 
import sys
import math


class NLIModel(nn.Module):

    AVERAGE_WORD_VECS = 0
    LSTM = 1
    BILSTM = 2
    BILSTM_MAX = 3
    BILSTM_ATTN = 4

    def __init__(self, model_type, model_params, wordvec_tensor):
        super(NLIModel, self).__init__()

        self.embeddings = nn.Embedding(wordvec_tensor.shape[0], wordvec_tensor.shape[1])
        with torch.no_grad():
            self.embeddings.weight.data.copy_(torch.from_numpy(wordvec_tensor))
            self.embeddings.weight.requires_grad = False

        self.model_type = model_type
        self.model_params = model_params
        self._choose_encoder(model_type, model_params)
        self.classifier = NLIClassifier(model_params, model_type)

        if torch.cuda.is_available():
            self.embeddings = self.embeddings.cuda()
            self.encoder = self.encoder.cuda()
            self.classifier = self.classifier.cuda()

    def _choose_encoder(self, model_type, model_params):
        if model_type == NLIModel.AVERAGE_WORD_VECS:
            self.encoder = EncoderBOW()
        elif model_type == NLIModel.LSTM:
            self.encoder = EncoderLSTM(model_params)
        elif model_type == NLIModel.BILSTM:
            self.encoder = EncoderBILSTM(model_params)
        elif model_type == NLIModel.BILSTM_MAX:
            self.encoder = EncoderBILSTMPool(model_params)
        elif model_type == NLIModel.BILSTM_ATTN:
            self.encoder = EncoderBILSTMattentionPool(model_params)
        else:
            print("Unknown encoder: " + str(model_type))
            sys.exit(1)


    def forward(self, words_s1, lengths_s1=None, words_s2=None, lengths_s2=None, dummy_input=False, applySoftmax=False):
        # If only one element is given, we assume that the first one must be a tuple of all inputs
        # Required for e.g. graph creation in tensorboard
        if lengths_s1 is None:
            words_s1, lengths_s1, words_s2, lengths_s2 = words_s1[0][0], words_s1[1][0], words_s1[0][1], words_s1[1][1]

        # Input must be [batch, time]
        embed_s1 = self.encode_sentence(words_s1, lengths_s1, dummy_input=dummy_input)
        embed_s2 = self.encode_sentence(words_s2, lengths_s2, dummy_input=dummy_input)

        out = self.classifier(embed_s1, embed_s2, applySoftmax=applySoftmax)
        return out

    def encode_sentence(self, words, lengths, dummy_input=False, debug=False):
        word_embeds = self.embeddings(words)
        sent_embeds = self.encoder(word_embeds, lengths, dummy_input=dummy_input, debug=debug)
        return sent_embeds

    # TODO: REMOVE FUNCTION
    def is_bidirectional(self):
        if self.model_type in [NLIModel.BILSTM, NLIModel.BILSTM_MAX]:
            return False
        return False



class NLIClassifier(nn.Module):

    def __init__(self, model_params, model_type):
        super(NLIClassifier, self).__init__()
        embed_sent_dim = model_params["embed_sent_dim"]
        fc_dropout = model_params["fc_dropout"] 
        fc_dim = model_params["fc_dim"]
        n_classes = model_params["n_classes"]
        self.non_linear = model_params["fc_nonlinear"]

        if model_type == 4:
            input_dim = 4 * embed_sent_dim *5
        else: 
            input_dim = 4 * embed_sent_dim
        if model_params["fc_nonlinear"]:
            self.seq1 = nn.Sequential(
                nn.Dropout(p=fc_dropout),
                nn.Linear(input_dim, fc_dim),
                nn.ReLU()
            )
            
            self.seq2 = nn.Sequential(
                nn.Dropout(p=fc_dropout),
                nn.Linear(fc_dim+ input_dim, fc_dim),
                nn.ReLU(),
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

        
        if self.non_linear:
            out = self.seq1(input_features)
            out = torch.cat((out,input_features), dim =1)
            out = self.seq2(out)
        else:
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
    
    
class EncoderBILSTMattentionPool(EncoderModule):

    def __init__(self, model_params, skip_connections=False):
        super(EncoderBILSTMattentionPool, self).__init__()
        self.lstm_chain = PyTorchLSTMChain(input_size=model_params["embed_word_dim"], 
                                           hidden_size=int(model_params["embed_sent_dim"]/2), num_layers = 3,
                                           bidirectional=True)
        self.attention_1 = nn.Sequential(nn.Linear(model_params["embed_sent_dim"],600, bias = True), nn.ReLU(), nn.Linear(600,model_params["embed_sent_dim"], bias = True), nn.Softmax(dim =1))
        self.attention_2 = nn.Sequential(nn.Linear(model_params["embed_sent_dim"],600, bias = True), nn.ReLU(), nn.Linear(600,model_params["embed_sent_dim"], bias = True), nn.Softmax(dim =1))
        self.attention_3 = nn.Sequential(nn.Linear(model_params["embed_sent_dim"],600, bias = True), nn.ReLU(), nn.Linear(600,model_params["embed_sent_dim"], bias = True), nn.Softmax(dim =1))
        self.attention_4 = nn.Sequential(nn.Linear(model_params["embed_sent_dim"],600, bias = True), nn.ReLU(), nn.Linear(600,model_params["embed_sent_dim"], bias = True), nn.Softmax(dim =1))
        self.attention_5 = nn.Sequential(nn.Linear(model_params["embed_sent_dim"],600, bias = True), nn.ReLU(), nn.Linear(600,model_params["embed_sent_dim"], bias = True), nn.Softmax(dim =1))
       
        
        

    def forward(self, embed_words, lengths, dummy_input=False, debug=False):
        # embed words is of shape [batch_size * 2, time, word_dim]
        _, outputs = self.lstm_chain(embed_words, lengths, dummy_input=dummy_input)
        # attention pooling
        
#         print("output size = ",outputs.size())
#         print("output elements: ", outputs[1,1,123:128])
        A1 = self.attention_1(outputs)
        A2 = self.attention_2(outputs)
        A3 = self.attention_3(outputs)
        A4 = self.attention_4(outputs)
        A5 = self.attention_5(outputs)
#         print("raw A1 size: ", A1.size())
#         print("A1 elements: ", A1[1,1,123:128])
#         print("A2 elements: ", A1[1,1,123:128])
#         print("A3 elements: ", A1[1,1,123:128])
#         print("A4 elements: ", A1[1,1,123:128])

        M1 = torch.sum(torch.mul(A1, outputs),1)
        M2 = torch.sum(torch.mul(A2, outputs),1)
        M3 = torch.sum(torch.mul(A3, outputs),1)
        M4 = torch.sum(torch.mul(A4, outputs),1)
        M5 = torch.sum(torch.mul(A5, outputs),1)
#         print("size after alpha and output multiplication = ",M1.size())
        pooled_features = torch.cat((M1, M2, M3, M4, M5),1)
#         print("pooled = ", pooled_features.size()) 
        
        if debug:
            return pooled_features
        else:
            return pooled_features

# class EncoderBILSTM(nn.Module):

#     def __init__(self, model_params):
#         super(EncoderBILSTM, self).__init__()
#         self.lstm_chain = LSTMChain(input_size=model_params["embed_word_dim"], 
#                                     hidden_size=int(model_params["embed_sent_dim"]/2))

#     def forward(self, embed_words, lengths):
#         # embed words is of shape [batch_size * 2, time, word_dim]
#         final_states, _ = self.lstm_chain(embed_words, lengths)
#         # Reshape to [batch_size, sent_dim]
#         final_states = final_states.reshape(shape=[-1, 2 * final_states.shape[1]])
#         return final_states


# class EncoderBILSTMPool(nn.Module):

#     def __init__(self, model_params, skip_connections=False):
#         super(EncoderBILSTMPool, self).__init__()
#         self.lstm_chain = LSTMChain(input_size=model_params["embed_word_dim"], 
#                                     hidden_size=int(model_params["embed_sent_dim"]/2))

#     def forward(self, embed_words, lengths):
#         # embed words is of shape [batch_size * 2, time, word_dim]
#         _, outputs = self.lstm_chain(embed_words, lengths)
#         # Max time pooling
#         pooled_features = EncoderBILSTMPool.pool_over_time(outputs, lengths)
#         # Reshape to [batch_size, sent_dim]
#         pooled_features = pooled_features.reshape(shape=[-1, 2 * pooled_features.shape[1]])
#         return pooled_features

#     @staticmethod
#     def pool_over_time(outputs, lengths):
#         time_dim = outputs.shape[1]
#         word_positions = torch.arange(start=0, end=time_dim, dtype=lengths.dtype, device=outputs.device)
#         mask = (word_positions.reshape(shape=[1, -1, 1]) < lengths.reshape([-1, 1, 1])).float()
#         outputs = outputs * mask + (torch.min(outputs) - 1) * (1 - mask)
#         final_states, _ = torch.max(outputs, dim=1)
#         return final_states
        



###################################
## LOW LEVEL LSTM IMPLEMENTATION ##
###################################

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        """Creates the weights for this LSTM"""
        super(LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.tanh_act = nn.Tanh()
        self.sigmoid_act = nn.Sigmoid()
        self.combined_gate = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        # Pytorch default initialization for LSTM weights
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)  

    def forward(self, input_, hx):
        """
        input is (batch, input_size)
        hx is ((batch, hidden_size), (batch, hidden_size))
        """
        prev_h, prev_c = hx

        # project input and prev state
        cat_input = torch.cat([input_,prev_h], dim=1)

        # main LSTM computation    
        combined_output = self.combined_gate(cat_input)
        i = self.sigmoid_act(combined_output[:,0 * self.hidden_size:1 * self.hidden_size])
        f = self.sigmoid_act(combined_output[:,1 * self.hidden_size:2 * self.hidden_size])
        g = self.tanh_act(combined_output[:,2 * self.hidden_size:3 * self.hidden_size])
        o = self.sigmoid_act(combined_output[:,3 * self.hidden_size:4 * self.hidden_size])

        c = f * prev_c + i * g
        h = o * self.tanh_act(c)

        return h, c

    def __repr__(self):
        return "{}({:d}, {:d})".format(
        self.__class__.__name__, self.input_size, self.hidden_size)


class LSTMChain(nn.Module):

    def __init__(self, input_size, hidden_size, lstm_cell = None):
        super(LSTMChain, self).__init__()

        if lstm_cell is None:
            self.lstm_cell = LSTMCell(input_size, hidden_size)
        else:
            self.lstm_cell = lstm_cell


    def forward(self, word_embeds, lengths):
        batch_size = word_embeds.shape[0]
        time_dim = word_embeds.shape[1]
        embed_dim = word_embeds.shape[2]
        
        # Prepare default initial states
        hx = word_embeds.new_zeros(batch_size, self.lstm_cell.hidden_size)
        cx = word_embeds.new_zeros(batch_size, self.lstm_cell.hidden_size)
        
        # Iterate over time steps
        outputs = []   
        for i in range(time_dim):
            hx, cx = self.lstm_cell(word_embeds[:, i], (hx, cx))
            outputs.append(hx)

        # Stack output over time dimension => Output states per time step
        outputs = torch.stack(outputs, dim=0) 
        outputs = outputs.transpose(0, 1).contiguous()

        # Get final hidden state per sentence
        reshaped_outputs = outputs.view(-1, outputs.shape[-1] * 2) # Reshape for better indexing
        indexes = (lengths - 1) + torch.arange(batch_size, device=word_embeds.device, dtype=lengths.dtype) * time_dim # Index of final states
        final = reshaped_outputs[indexes,:] # Final states

        return final, outputs


class PyTorchLSTMChain(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1, bidirectional=False):
        super(PyTorchLSTMChain, self).__init__()
        self.lstm_cell = nn.LSTM(input_size, hidden_size, num_layers = num_layers , bidirectional=bidirectional)
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

        reshaped_outputs = outputs.view(-1, outputs.shape[-1]) # Reshape for better indexing
        # print((time_dim).dtype)
        indexes = (lengths - 1) + torch.arange(batch_size, device=word_embeds.device, dtype=lengths.dtype) * int(time_dim) # Index of final states
        final = reshaped_outputs[indexes.long(),:] # Final states

        final_states = final_hidden_states.transpose(0, 1).transpose(1, 2)
        # print("Final hidden states shape: " + str(final_hidden_states.shape))
        final_states = final_states.reshape([batch_size, self.hidden_size * final_states.shape[-1]])

        # print("Final: " + str(final[0:4,0]))
        # print("Initial 0: " + str(final_hidden_states[0,0:4,0]))
        # print("Initial 1: " + str(final_hidden_states[1,0:4,0]))
        # print("Pre-final: " + str(final_states[0:4,0]))

        return final_states, outputs # final

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
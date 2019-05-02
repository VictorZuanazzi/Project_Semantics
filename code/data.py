import os
import numpy as np
import torch
import json
import re
import sys
from random import shuffle
from vocab import load_word2vec_from_file
import pandas as pd

# 0 => Full debug
# 1 => Reduced output
# 2 => No output at all (on cluster)
DEBUG_LEVEL = 0

def set_debug_level(level):
    global DEBUG_LEVEL
    DEBUG_LEVEL = level

def debug_level():
    global DEBUG_LEVEL
    return DEBUG_LEVEL


###############################
## Dataset class definitions ##
###############################

class DatasetHandler:

    SNLI_DATASETS = None
    SNLI_EXTRA_DATASETS = None
    SST_DATASETS = None

    @staticmethod
    def _load_all_type_datasets(dataset_fun, debug_dataset=False, data_types=None):
        _, word2id_dict, _ = load_word2vec_from_file()
        dataset_list = list()
        if data_types is None:
            data_types = ['train' if not debug_dataset else 'dev', 'dev', 'test']
        for data_type in data_types:
            dataset = dataset_fun(data_type, shuffle_data=('train' in data_type))
            dataset.print_statistics()
            dataset.set_vocabulary(word2id_dict)
            dataset_list.append(dataset)
        return dataset_list

    @staticmethod
    def load_SNLI_datasets(debug_dataset=False):
        if DatasetHandler.SNLI_DATASETS is None:
            DatasetHandler.SNLI_DATASETS = DatasetHandler._load_all_type_datasets(SNLIDataset, debug_dataset=debug_dataset)
        return DatasetHandler.SNLI_DATASETS[0], DatasetHandler.SNLI_DATASETS[1], DatasetHandler.SNLI_DATASETS[2]

    @staticmethod
    def load_SNLI_splitted_datasets():
        if DatasetHandler.SNLI_EXTRA_DATASETS is None:
            DatasetHandler.SNLI_EXTRA_DATASETS = DatasetHandler._load_all_type_datasets(SNLIDataset, data_types=['test_hard', 'test_easy']) 
        return DatasetHandler.SNLI_EXTRA_DATASETS[0], DatasetHandler.SNLI_EXTRA_DATASETS[1]

    @staticmethod
    def load_SST_datasets(debug_dataset=False):
        if DatasetHandler.SST_DATASETS is None:
            DatasetHandler.SST_DATASETS = DatasetHandler._load_all_type_datasets(SSTDataset, debug_dataset=debug_dataset)
        return DatasetHandler.SST_DATASETS[0], DatasetHandler.SST_DATASETS[1], DatasetHandler.SST_DATASETS[2]


class DatasetTemplate:

    def __init__(self, data_type="train", shuffle_data=True):
        self.data_type = data_type
        self.shuffle_data = shuffle_data
        self.set_data_list(list())
        self.label_dict = dict()
        self.num_invalids = 0

    def set_data_list(self, new_data):
        self.data_list = new_data
        self.example_index = 0
        self.perm_indices = list(range(len(self.data_list)))
        if self.shuffle_data:
            shuffle(self.perm_indices)

    def _get_next_example(self):
        exmp = self.data_list[self.perm_indices[self.example_index]]
        self.example_index += 1
        if self.example_index >= len(self.perm_indices):
            if self.shuffle_data:
                shuffle(self.perm_indices)
            self.example_index = 0
        return exmp

    @staticmethod
    def sents_to_Tensors(batch_stacked_sents, batch_labels=None, toTorch=False):
        lengths = []
        embeds = []
        for batch_sents in batch_stacked_sents:
            lengths_sents = np.array([x.shape[0] for x in batch_sents])
            max_len = np.max(lengths_sents)
            sent_embeds = np.zeros((len(batch_sents), max_len), dtype=np.int32)
            for s_index, sent in enumerate(batch_sents):
                sent_embeds[s_index, :sent.shape[0]] = sent
            if toTorch:
                sent_embeds = torch.LongTensor(sent_embeds)
                lengths_sents = torch.LongTensor(lengths_sents)
                if torch.cuda.is_available():
                    sent_embeds = sent_embeds.cuda()
                    lengths_sents = lengths_sents.cuda()
            lengths.append(lengths_sents)
            embeds.append(sent_embeds)
        if batch_labels is not None and toTorch:
            batch_labels = torch.LongTensor(np.array(batch_labels))
            if torch.cuda.is_available():
                batch_labels = batch_labels.cuda()
        return embeds, lengths, batch_labels
    
    @staticmethod
    def object_to_Tensors(some_object, toTorch=False):
        """wraps the given object in a torch tensor or numpy array.
        inputs:
            some_object (list(int), tuple(int), int), objec to be wraped.
            toTorch (bool), if True wraps some_object with a torch tensor, if False
                wraps it with an numpy array
        output:
            some_boject (torch.LongTensor(some_object) or np.array(some_object)), 
                if cuda is available, torch tensor is returned in cuda.
        """
        if toTorch:
            some_object = torch.LongTensor(some_object)
            if torch.cuda.is_available():
                some_object = some_object.cuda()
        else:
            some_object = np.array(some_object)
            
        return some_object

    def get_num_examples(self):
        return len(self.data_list)

    def get_word_list(self):
        all_words = dict()
        for i, data in enumerate(self.data_list):
            if debug_level() == 0:
                print("Processed %4.2f%% of the dataset" % (100.0 * i / len(self.data_list)), end="\r")
            if isinstance(data, NLIData):
                data_words = data.premise_words + data.hypothesis_words
            else:
                data_words = data.sent_words
            for w in data_words:
                if w not in all_words:
                    all_words[w] = ''
        all_words = list(all_words.keys())
        print("Found " + str(len(all_words)) + " unique words")
        return all_words

    def set_vocabulary(self, word2vec):
        missing_words = 0
        overall_words = 0
        for data in self.data_list:
            data.translate_to_dict(word2vec)
            mw, ow = data.number_words_not_in_dict(word2vec)
            missing_words += mw 
            overall_words += ow 
        print("Amount of missing words: %4.2f%%" % (100.0 * missing_words / overall_words))

    def get_batch(self, batch_size, loop_dataset=True, toTorch=False):
        # Default: assume that dataset entries contain object of SentData
        if not loop_dataset:
            batch_size = min(batch_size, len(self.perm_indices) - self.example_index)
        batch_sents = []
        batch_labels = []
        for _ in range(batch_size):
            data = self._get_next_example()
            batch_sents.append(data.sent_vocab)
            batch_labels.append(data.label)
        embeds, lengths, labels = DatasetTemplate.sents_to_Tensors([batch_sents], batch_labels=batch_labels, toTorch=toTorch)
        return (embeds[0], lengths[0], labels)

    def get_num_classes(self):
        c = 0
        for key, val in self.label_dict.items():
            if val >= 0:
                c += 1
        return c

    def add_label_explanation(self, label_dict):
        # The keys should be the labels, the explanation strings
        if isinstance(list(label_dict.keys())[0], str) and not isinstance(list(label_dict.values())[0], str):
            label_dict = {v: k for k, v in label_dict.items()}
        self.label_dict = label_dict

    def label_to_string(self, label):
        if label in self.label_dict:
            return self.label_dict[label]
        else:
            return str(label)

    def print_statistics(self):
        print("="*50)
        print("Dataset statistics " + self.data_type)
        print("-"*50)
        print("Number of examples: " + str(len(self.data_list)))
        print("Labelwise amount:")
        for key, val in self.label_dict.items():
            print("\t- " + val + ": " + str(sum([d.label == key for d in self.data_list])))
        print("Number of invalid examples: " + str(self.num_invalids))
        print("="*50)


class SNLIDataset(DatasetTemplate):

    # Data type either train, dev or test
    def __init__(self, data_type, data_path="../data/snli_1.0", add_suffix=True, shuffle_data=True):
        super(SNLIDataset, self).__init__(data_type, shuffle_data)
        if data_path is not None:
            self.load_data(data_path, data_type)
        else:
            self.data_list == list()
        super().set_data_list(self.data_list)
        super().add_label_explanation(NLIData.LABEL_LIST)

    def load_data(self, data_path, data_type):
        self.data_list = list()
        self.num_invalids = 0
        s1 = [line.rstrip() for line in open(data_path + "/s1." + data_type, 'r')]
        s2 = [line.rstrip() for line in open(data_path + "/s2." + data_type, 'r')]
        labels = [NLIData.LABEL_LIST[line.rstrip('\n')] for line in open(data_path + "/labels." + data_type, 'r')]
        
        i = 0
        for prem, hyp, lab in zip(s1, s2, labels):
            if debug_level() == 0:
                print("Read %4.2f%% of the dataset" % (100.0 * i / len(s1)), end="\r")
            i += 1
            if lab == -1:
                self.num_invalids += 1
                continue
            d = NLIData(premise = prem, hypothesis = hyp, label = lab)
            self.data_list.append(d)

    def get_batch(self, batch_size, loop_dataset=True, toTorch=False, bidirectional=False):
        # Output sentences with dimensions (bsize, max_len)
        if not loop_dataset:
            batch_size = min(batch_size, len(self.perm_indices) - self.example_index)
        batch_s1 = []
        batch_s2 = []
        batch_labels = []
        for _ in range(batch_size):
            data = self._get_next_example()
            batch_s1.append(data.premise_vocab)
            batch_s2.append(data.hypothesis_vocab)
            batch_labels.append(data.label)
            if bidirectional:
                batch_s1.append(data.premise_vocab[::-1])
                batch_s2.append(data.hypothesis_vocab[::-1])
        return DatasetTemplate.sents_to_Tensors([batch_s1, batch_s2], batch_labels=batch_labels, toTorch=toTorch)


class SSTDataset(DatasetTemplate):

    LABEL_LIST = {
        0 : "Negative",
        1 : "Positive"
    }

    # Data type either train, dev or test
    def __init__(self, data_type, data_path="../data/SST", add_suffix=True, shuffle_data=True):
        super(SSTDataset, self).__init__(data_type, shuffle_data)
        if data_path is not None:
            self.load_data(data_path, data_type)
        else:
            self.data_list == list()
        super().set_data_list(self.data_list)
        super().add_label_explanation(SSTDataset.LABEL_LIST)

    def load_data(self, data_path, data_type):
        self.data_list = list()
        self.num_invalids = 0
        filepath = os.path.join(data_path, data_type + ".txt")
        with open(filepath, mode="r", encoding="utf-8") as f:
            for line in f:
                sent = line.strip().replace("\\","")
                tokens = re.sub(r"\([0-9] |\)", "", sent).split()
                label = int(sent[1])
                if label == 2:
                    self.num_invalids += 1
                    continue
                label = 0 if label < 2 else 1
                d = SentData(sentence=" ".join(tokens), label=label)
                self.data_list.append(d)

class VUADataset(DatasetTemplate):

    # Data type either train, dev or test
    def __init__(self, data_type, data_path="../data/VUA/", shuffle_data=True):
        """Initializes the VUA dataset.
        inputs:
        data_type: (srt), chose between the datastes 'train', 'dev', 'test'.
        data_path: (str), path to the directory of the VUA dataset.
        shuffle_data: (bool), True for shuffling the data, False not to.
        """
        super(VUADataset, self).__init__(data_type, shuffle_data)
            
        if data_path is not None: 
            #load the data from file
            self.load_data(data_path, data_type)
        else:
            #empty data_list if no path is specified
            self.data_list == list()
        
        super().set_data_list(self.data_list)
        super().add_label_explanation(VUAData.LABEL_LIST)

    def load_data(self, data_path, data_type):
        """loads the data as intances of the class VUAData.
        input:
            data_path: (str), path to the directory of the VUA dataset.
            data_type: (srt), chose between the datastes 'train', 'dev', 'test'.
        output:
            data_list: (list(VUAData)) a list of datapoints in as instances of
            the class VUAData."""
            
        self.data_list = list()
        self.num_invalids = 0
        
        #maps data_type to file name
        file = {"train": "VUA_formatted_train_augmented.csv",
                        "dev": "VUA_formatted_val.csv", 
                        "test": "VUA_formatted_test.csv"}
        
        #reads the wanted data
        df_data = pd.read_csv(data_path + file.get(data_type, "train"),
                              encoding = 'latin-1')
        
        for i in df_data.index.tolist():
            if debug_level() == 0:
                print("Read %4.2f%% of the dataset" % (100.0 * i / len(df_data)), end="\r")
            
            #reads the relevant parts of the dataset
            sentence = df_data.at[i, "sentence"]
            verb_position = df_data.at[i, "verb_idx"]
            label = df_data.at[i, "label"]
            
            #initializes the data as an instance of the class VUAData
            d = VUAData(sentence, verb_position, label)
            
            #appends everything in a beautiful list.
            self.data_list.append(d)

    def get_batch(self, batch_size, loop_dataset=True, toTorch=False, bidirectional=False):
        """get a batch of examples from VUAData
        input:
            batch_size: (int), the number of datapoints in a batch,
            loop_dataset: (bool), when False it ensures the batch size over all
                batches. When True it is possible that the last batch of the 
                epoch has fewer examples.
            toTorch: (bool), if True the data is wraped in a torch tensor, if 
                False numpy arrays are used instead.
            bidirectional: (bool) deprecated, not used here. The parameter is 
                kept in the signature to keep consistency with other classes.
        output:
            outputs of DatasetTemplate.sents_to_Tensors:
                embeds: (np.array or torch.LongTensor), embeddings for the words
                    in the sentences.
                lengths: (np.array or torch.LongTensor), the length of each 
                    sentence of the batch.
                batch_labels:(np.array or torch.LongTensor), the labels of each
                    sentence.
            batch_verb_p: (np.array or torch.LongTensor), indicate the position
                of the verb of interest.
        """
        # Output sentences with dimensions (bsize, max_len)
        if not loop_dataset:
            batch_size = min(batch_size, len(self.perm_indices) - self.example_index)
        
        batch_sentence = []
        batch_verb_p = []
        batch_labels = []
        for _ in range(batch_size):
            
            data = self._get_next_example()
            
            batch_sentence.append(data.sentence_vocab)
            batch_verb_p.append(data.verb_position)
            batch_labels.append(data.label)
        
        #converts batch_verb_p to torch or numpy
        batch_verb_p = DatasetTemplate.object_to_Tensors(batch_verb_p)
        
        #get the embeds, lengtghs and labels
        embeds, lengths, batch_labels = DatasetTemplate.sents_to_Tensors(batch_sentence, 
                                                batch_labels=batch_labels, 
                                                toTorch=toTorch)
        
        return embeds, lengths, batch_labels, batch_verb_p

class WiCDataset(DatasetTemplate):

    # Data type either train, dev or test
    def __init__(self, data_type, data_path="../data/WiC_dataset/", shuffle_data=True):
        """Initializes the Word in Context dataset.
        inputs:
        data_type: (srt), chose between the datastes 'train', 'dev', 'test'.
        data_path: (str), path to the directory of the WiC dataset.
        shuffle_data: (bool), True for shuffling the data, False not to.
        """
        super(WiCDataset, self).__init__(data_type, shuffle_data)
            
        if data_path is not None: 
            #load the data from file
            self.load_data(data_path, data_type)
        else:
            #empty data_list if no path is specified
            self.data_list == list()
        
        super().set_data_list(self.data_list)
        super().add_label_explanation(WiCData.LABEL_LIST)

    def load_data(self, data_path, data_type):
        """loads the data as intances of the class WiCData.
        input:
            data_path: (str), path to the directory of the WiC dataset.
            data_type: (srt), chose between the datastes 'train', 'dev', 'test'.
        output:
            data_list: (list(WiCData)) a list of datapoints in as instances of
            the class WiCData."""
            
        self.data_list = list()
        self.num_invalids = 0
        
        #maps data_type to file name
        data_name = {"train": "train/train.data.txt",
                     "dev": "dev/dev.data.txt", 
                     "test": "test/test.data.txt"}
        
        label_name ={"train": "train/train.gold.txt",
                     "dev": "dev/dev.gold.txt", 
                     "test": None}
   
        #reads the wanted data
        data_file = open(data_path + data_name[data_type], 
                         "r", 
                         encoding="utf8")
        #parse the file
        data_lines = data_file.read()
        data_lines = data_lines.split("\n")
        data_line = [l.split("\t") for l in data_lines]
        data_file.close()
        
        if not data_type == "test":
            #test does not have labels
            label_file = open(data_path + label_name[data_type], 
                              "r", 
                              encoding="utf8")
            label_lines = label_file.read()
            label_line = label_lines.split("\n")
            #converts
            label = [WiCData.LABEL_LIST[l] for l in label_line]
            label_file.close()
        else:
            #unknown labels are given when test set is loaded.
            label = [WiCData.LABEL_LIST[""]]*len(data_line)
        
        for i, data in enumerate(data_line):
            
            if debug_level() == 0:
                print("Read %4.2f%% of the dataset" % (100.0 * i / len(data_line)), end="\r")
             
            if len(data) != 5:
                #skip sequences that are not in the correct format
                continue
            
            #gets the sentences 
            s1 = data[3]
            s2 = data[4]
            
            #get the position fo the word in each sentence
            p1, p2 = tuple(data[2].split("-"))
            p1 = int(p1)
            p2 = int(p2)
            
            #get the word of interest
            word = data[0]
            pos = data[1]
            
            #initializes the data as an instance of the class WiCData
            d = WiCData(word, pos, s1, s2, p1, p2, label[i])
            
            #appends everything in a beautiful list.
            self.data_list.append(d)

    def get_batch(self, batch_size, loop_dataset=True, toTorch=False, bidirectional=False):
        """get a batch of examples from WiCData
        input:
            batch_size: (int), the number of datapoints in a batch,
            loop_dataset: (bool), when False it ensures the batch size over all
                batches. When True it is possible that the last batch of the 
                epoch has fewer examples.
            toTorch: (bool), if True the data is wraped in a torch tensor, if 
                False numpy arrays are used instead.
            bidirectional: (bool) deprecated, not used here. The parameter is 
                kept in the signature to keep consistency with other classes.
        output:
            outputs of DatasetTemplate.sents_to_Tensors:
                embeds: (np.array or torch.LongTensor), embeddings for the words
                    in the sentences with dimensions (batch_size, max_len).
                lengths: (np.array or torch.LongTensor), the length of each 
                    sentence of the batch.
                batch_labels:(np.array or torch.LongTensor), the labels of each
                    sentence.
            batch_p1, batch_p2: (np.array or torch.LongTensor), indicate the 
                position of the word of interest.
        """
        if not loop_dataset:
            batch_size = min(batch_size, len(self.perm_indices) - self.example_index)
        
        batch_s1 = []
        batch_s2 = []
        batch_p1 = []
        batch_p2 = []
        batch_labels = []
        
        for _ in range(batch_size):
            
            data = self._get_next_example()
            
            batch_s1.append(data.s1_vocab)
            batch_s2.append(data.s2_vocab)
            batch_p1.append(data.p1)
            batch_p2.append(data.p2)
            batch_labels.append(data.label)
            
        #converts batch_pX to torch or numpy
        batch_p1 = DatasetTemplate.object_to_Tensors(batch_p1)
        batch_p2 = DatasetTemplate.object_to_Tensors(batch_p2)
        
        #get the embeds, lengtghs and labels
        embeds, lengths, batch_labels = DatasetTemplate.sents_to_Tensors([batch_s1, batch_s2],
                                                batch_labels=batch_labels, 
                                                toTorch=toTorch)
        
        return embeds, lengths, batch_labels, batch_p1, batch_p2  

class NLIData:

    LABEL_LIST = {
        '-': -1,
        "neutral": 0, 
        "entailment": 1,
        "contradiction": 2
    }

    def __init__(self, premise, hypothesis, label):
        self.premise_words = SentData._preprocess_sentence(premise)
        self.hypothesis_words = SentData._preprocess_sentence(hypothesis)
        self.premise_vocab = None
        self.hypothesis_vocab = None
        self.label = label

    def translate_to_dict(self, word_dict):
        self.premise_vocab = SentData._sentence_to_dict(word_dict, self.premise_words)
        self.hypothesis_vocab = SentData._sentence_to_dict(word_dict, self.hypothesis_words)

    def number_words_not_in_dict(self, word_dict):
        missing_words = 0
        for w in (self.premise_words + self.hypothesis_words):
            if w not in word_dict:
                missing_words += 1
        return missing_words, (len(self.premise_words) + len(self.hypothesis_words))
        
    def get_data(self):
        return self.premise_vocab, self.hypothesis_vocab, self.label

    def get_premise(self):
        return " ".join(self.premise_words)

    def get_hypothesis(self):
        return " ".join(self.hypothesis_words)

    @staticmethod
    def label_to_string(label):
        for key, val in NLIData.LABEL_LIST.items():
            if val == label:
                return key



class SentData:

    def __init__(self, sentence, label=None):
        self.sent_words = SentData._preprocess_sentence(sentence)
        self.sent_vocab = None
        self.label = label

    def translate_to_dict(self, word_dict):
        self.sent_vocab = SentData._sentence_to_dict(word_dict, self.sent_words)

    def number_words_not_in_dict(self, word_dict):
        missing_words = 0
        for w in self.sent_words:
            if w not in word_dict:
                missing_words += 1
        return missing_words, len(self.sent_words)

    @staticmethod
    def _preprocess_sentence(sent):
        sent_words = list(sent.lower().strip().split(" "))
        if "." in sent_words[-1] and len(sent_words[-1]) > 1:
            sent_words[-1] = sent_words[-1].replace(".","")
            sent_words.append(".")
        sent_words = [w for w in sent_words if len(w) > 0]
        for i in range(len(sent_words)):
            if len(sent_words[i]) > 1 and "." in sent_words[i]:
                sent_words[i] = sent_words[i].replace(".","")
        return sent_words

    @staticmethod
    def _sentence_to_dict(word_dict, sent):
        vocab_words = list()
        vocab_words += [word_dict['<s>']]
        vocab_words += SentData._word_seq_to_dict(sent, word_dict)
        vocab_words += [word_dict['</s>']]
        vocab_words = np.array(vocab_words, dtype=np.int32)
        return vocab_words

    @staticmethod
    def _word_seq_to_dict(word_seq, word_dict):
        vocab_words = list()
        for w in word_seq:
            if len(w) <= 0:
                continue
            if w in word_dict:
                vocab_words.append(word_dict[w])
            elif "-" in w:
                vocab_words += SentData._word_seq_to_dict(w.split("-"), word_dict)
            elif "/" in w:
                vocab_words += SentData._word_seq_to_dict(w.split("/"), word_dict)
            else:
                subword = re.sub('\W+','', w)
                if subword in word_dict:
                    vocab_words.append(word_dict[subword])
        return vocab_words

class VUAData:
    
    LABEL_LIST = {
        "methaphore": 1, 
        "literal": 0
    }

    def __init__(self, sentence, verb_position, label):
        
        self.sentence_words = SentData._preprocess_sentence(sentence)
        self.sentence_vocab = None 
        self.verb_position = verb_position
        self.label = label

    def translate_to_dict(self, word_dict):
        self.sentence_vocab = SentData._sentence_to_dict(word_dict, self.sentence_words)

    def number_words_not_in_dict(self, word_dict):
        missing_words = 0
        for w in (self.sentence_words):
            if w not in word_dict:
                missing_words += 1
        return missing_words, len(self.sentence_words)
        
    def get_data(self):
        return self.sentence_vocab, self.label

    def get_sentence(self):
        return " ".join(self.sentence_words)

    @staticmethod
    def label_to_string(label):
        for key, val in VUAData.LABEL_LIST.items():
            if val == label:
                return key

class WiCData:
    
    LABEL_LIST = {
        "": -1, 
        "F": 0,
        "T": 1
    }

    def __init__(self, word, pos, s1, s2, p1, p2, label):
        
        self.s1_words = SentData._preprocess_sentence(s1)
        self.s2_words = SentData._preprocess_sentence(s2)
        self.s1_vocab = None 
        self.s2_vocab = None
        self.p1 = p1
        self.p2 = p2
        self.label = label
        self.word = word
        self.pos = pos

    def translate_to_dict(self, word_dict):
        self.s1_vocab = SentData._sentence_to_dict(word_dict, 
                                                   self.s1_words)
        self.s2_vocab = SentData._sentence_to_dict(word_dict, 
                                                   self.s2_words)

    def number_words_not_in_dict(self, word_dict):
        missing_words = 0
        for w in (self.s1_words + self.s2_words):
            if w not in word_dict:
                missing_words += 1
        return missing_words, (len(self.s1_words) + len(self.s2_words))
        
    def get_data(self):
        return self.s1_vocab, self.s2_vocab, self.label

    def get_s1(self):
        return " ".join(self.s1_words)
    
    def get_s2(self):
        return " ".join(self.s2_words)

    @staticmethod
    def label_to_string(label):
        for key, val in WiCData.LABEL_LIST.items():
            if val == label:
                return key
# class SeqData:

#     def __init__(self, sentence, label):
#         self.sent_words = SeqData._preprocess_sentence(sentence)
#         self.label = label
#         assert len(self.label) == len(self.sent_words), "Number of labels have to fit to number of words in the sentence"
#         self.sent_vocab = None

#     def translate_to_dict(self, word_dict):
#         self.sent_vocab = SentData._sentence_to_dict(word_dict, self.sent_words)

#     @staticmethod
#     def _preprocess_sentence(sent, labels):
#         sent_words = list(sent.lower().strip().split(" "))
#         if "." in sent_words[-1] and len(sent_words[-1]) > 1:
#             sent_words[-1] = sent_words[-1].replace(".","")
#             sent_words.append(".")
#         sent_words = [w for w in sent_words if len(w) > 0]
#         for i in range(len(sent_words)):
#             if len(sent_words[i]) > 1 and "." in sent_words[i]:
#                 sent_words[i] = sent_words[i].replace(".","")
#         return sent_words

#     @staticmethod
#     def _sentence_to_dict(word_dict, sent, labels):
#         vocab_words = list()
#         vocab_words += [word_dict['<s>']]
#         vocab_words += SentData._word_seq_to_dict(sent, word_dict)
#         vocab_words += [word_dict['</s>']]
#         vocab_words = np.array(vocab_words, dtype=np.int32)

#         return vocab_words

#     @staticmethod
#     def _word_seq_to_dict(word_seq, word_dict, labels):
#         vocab_words = list()
#         for w_index, w in enumerate(word_seq):
#             if len(w) <= 0:
#                 continue
#             if w in word_dict:
#                 vocab_words.append(word_dict[w])
#             elif "-" in w:
#                 vocab_words += SentData._word_seq_to_dict(w.split("-"), word_dict, labels=[labels[w_index]])
#             elif "/" in w:
#                 vocab_words += SentData._word_seq_to_dict(w.split("/"), word_dict)
#             else:
#                 subword = re.sub('\W+','', w)
#                 if subword in word_dict:
#                     vocab_words.append(word_dict[subword])
#         return vocab_words
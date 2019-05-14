import torch
import torch.nn as nn
import argparse
import math
import os
from tqdm import tqdm
import sys
import json
import pickle
import numpy as np
from glob import glob

from model2 import NLIModel
from model import MultiTaskEncoder
from data import DatasetHandler as data
from mutils import load_model, load_model_from_args, load_args, args_to_params, visualize_tSNE, get_transfer_datasets

from allennlp.modules.elmo import Elmo, batch_to_ids

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# from tensorboardX import SummaryWriter


class WICclassifier(nn.Module):
    def __init__(self, batch_size=64, lstm_dim=400, fc_dim=100, n_classes=1):
        super(WICclassifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(lstm_dim * 2, fc_dim), nn.ReLU(), nn.Linear(fc_dim, n_classes),
                                        nn.Sigmoid())

    def forward(self, inputs):
        return self.classifier(inputs)


class WIC:

    def __init__(self, batch_size=64, lstm_dim=400, fc_dim=100, n_classes=1):
        self.train_dataset, self.val_dataset, self.test_dataset = data.load_WiC_datasets(debug_dataset=False)
        # self.encoder = model[0]
        self.elmo = self.load_elmo()
        self.classifier = WICclassifier(batch_size=batch_size, lstm_dim=lstm_dim, fc_dim=fc_dim)
        self.classifier = self.classifier.to(device)
        self.elmo = self.elmo.to(device)
        self.batch_size = batch_size

    # Testing
    def load_elmo(self):
        """
        Loads medium ELMo model. The files need to be downloaded from https://allennlp.org/elmo
        :return: the ELMo model, which embeds each sentence as a learned 512-dim scalar mix of its three layers' states
        """
        options_path = 'elmo/medium/elmo_2x2048_256_2048cnn_1xhighway_options.json'
        weights_path = 'elmo/medium/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'
        assert os.path.isfile(options_path)
        assert os.path.isfile(weights_path)
        return Elmo(options_path, weights_path, 1, scalar_mix_parameters=[0, 1, 0], dropout=0)

    def get_target_embed(self, embeds, lengths, batch_p1, batch_p2):
        # encode the sentences batch
        sent1_encoded = self.encoder.encode_sentence(embeds[0], lengths[0], word_level=True)[1].detach()
        sent2_encoded = self.encoder.encode_sentence(embeds[1], lengths[1], word_level=True)[1].detach()

        # prepare the input features based on specified word only  
        bins1 = torch.arange(0, sent1_encoded.size(0) * sent1_encoded.size(1), step=sent1_encoded.size(1))
        bins2 = torch.arange(0, sent2_encoded.size(0) * sent2_encoded.size(1), step=sent2_encoded.size(1))

        sent1_encoded = sent1_encoded.view(-1, sent1_encoded.size(2))
        sent2_encoded = sent2_encoded.view(-1, sent2_encoded.size(2))

        u = torch.index_select(sent1_encoded, dim=0, index=bins1 + batch_p1)
        v = torch.index_select(sent2_encoded, dim=0, index=bins2 + batch_p2)

        return u, v

    def threshold_evaluator(self, u_generator, v_generator, batch_labels):

        cos_sim = torch.Tensor()

        n_batches = math.ceil(len(batch_labels) / 64)

        for u, v in tqdm(zip(u_generator, v_generator), total=n_batches, ncols=100, ascii=True):

            u = u.unsqueeze(dim=1)
            v = v.unsqueeze(dim=2)
            cos_sim = torch.cat((cos_sim, torch.bmm(u, v).squeeze()), 0)

        print()
        min_val = torch.min(cos_sim)
        max_val = torch.max(cos_sim)

        best_acc = 0
        best_split = min_val.item()

        print("\nMinimum cosin similarity value = {}".format(min_val.item()))
        print("Maximum cosin similarity value = {}\n".format(max_val.item()))

        for iter in torch.arange(min_val, max_val, step=0.02):
            preds = (cos_sim > iter).float()
            correct = (preds == batch_labels.float()).float().sum()
            acc = correct / len(preds)
            if acc > best_acc:
                best_acc = acc.item()
                best_split = iter
                print("New best accuracy = {:.4f}, Split = {:.4f}".format(best_acc, best_split))
        print("=" * 70 + "\nBest accuracy found = {:.4f}, Best split found = {:.4f}".format(best_acc, best_split)
              + "\n" + "=" * 70)

    def mlp_evaluator(self):
        self.encoder.eval()
        self.classifier.train()

        l_rate = args.lr

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=l_rate)
        criterion = nn.BCELoss(reduction='mean')  # reduction = sum/mean??

        train_dataset = self.train_dataset
        val_dataset = self.val_dataset

        number_batches_train = int(math.ceil(train_dataset.get_num_examples() * 1.0 / self.batch_size))
        number_batches_val = int(math.ceil(val_dataset.get_num_examples() * 1.0 / self.batch_size))

        print("\nNumber of train batches = ", number_batches_train)
        print("Number of val batches = ", number_batches_val)

        preds_list = []
        loss_list = []
        accuracy = []
        dev_accuracy = []
        prev_dev = 0

        for epoch in range(50):
            if l_rate < 1e-6:
                print("Termination conditon reached!")
                break

            self.classifier.train()
            for batch_ind in range(number_batches_train):
                self.classifier.train()

                # get batch of data
                embeds, lengths, batch_labels, batch_p1, batch_p2 = train_dataset.get_batch(self.batch_size,
                                                                                            loop_dataset=False,
                                                                                            toTorch=True)

                # encode the sentences batch
                u, v = self.get_target_embed(embeds, lengths, batch_p1, batch_p2)

                input_features = torch.cat((u, v), dim=1)

                # get predictions
                out = self.classifier(input_features)

                # accuracy calculation
                output = (out > 0.5).float()
                correct = (output.squeeze() == batch_labels.float()).float().sum()
                acc = correct / len(output)
                accuracy.append(acc.item())

                # calculate loss
                loss = criterion(out[:, 0], batch_labels.float())
                loss_list.append(loss.item())
                a = list(self.classifier.parameters())[0].clone()

                # back prop steps
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                b = list(self.classifier.parameters())[0].clone()

            #                 print("Params equal = ", torch.equal(a,b))

            # Evaluate on dev set after every epoch
            for batch_ind in range(number_batches_val):
                self.classifier.eval()

                embeds, lengths, batch_labels, batch_p1, batch_p2 = val_dataset.get_batch(self.batch_size,
                                                                                          loop_dataset=False,
                                                                                          toTorch=True)
                u, v = self.get_target_embed(embeds, lengths, batch_p1, batch_p2)
                input_features = torch.cat((u, v), dim=1)
                out = self.classifier(input_features)
                output = (out > 0.5).float()
                correct = (output.squeeze() == batch_labels.float()).float().sum()
                acc = correct / len(output)
                dev_accuracy.append(acc.item())

            print("Epoch: {}/50,  Train accuracy = {:.4f}, Train Loss = {:.4f}, Val accuracy = {:.4f}".format(epoch,
                                                                                                              sum(
                                                                                                                  accuracy) / len(
                                                                                                                  accuracy) * 100,
                                                                                                              sum(
                                                                                                                  loss_list) / len(
                                                                                                                  loss_list),
                                                                                                              sum(
                                                                                                                  dev_accuracy) / len(
                                                                                                                  dev_accuracy) * 100))
            current_dev = sum(dev_accuracy) / len(dev_accuracy)
            if prev_dev > current_dev:
                l_rate = l_rate * 0.25
                print("Reduced learning rate to: ", l_rate)
            prev_dev = current_dev

        print("Training Done")

    def get_elmo_batch(self, batch, lengths, batch_p, id2word):
        """
        Generator function that yields batches of elmo embeddings
        :param batch: batch of sentences in ID form
        :param lengths: length of each sentence in the batch
        :param batch_p: indices of the words of interest for each sentence
        :param id2word: dictionary mapping word IDs to words
        :return: each batch containing an elmo embedding for the word of interest in the sentence
        """

        # elmo batch size
        batch_size = 64

        batch_start_idx = 0
        elmo_batch = []

        for sentence_idx in range(batch.size()[0]):

            sentence = []
            for word_idx in range(batch.size()[1]):
                w_ix = batch[sentence_idx, word_idx].item()
                if word_idx == 0:
                    sentence.append('<S>')
                elif word_idx == lengths[sentence_idx] - 1:
                    sentence.append('</S>')
                elif word_idx < lengths[sentence_idx]:
                    sentence.append(id2word[w_ix])
                else:
                    break

            elmo_batch.append(sentence)

            if (len(elmo_batch) >= batch_size) or (sentence_idx >= batch.size()[0] - 1):

                # running elmo
                elmo_embeds = self.get_elmo_embeds(elmo_batch)

                wic_embeds = []
                for idx, w_idx in enumerate(batch_p[batch_start_idx:sentence_idx + 1]):
                    wic_embeds.append(elmo_embeds[idx][w_idx + 1])

                yield torch.stack(wic_embeds).detach()

                batch_start_idx = sentence_idx + 1
                elmo_batch = []

    def get_elmo_embeds(self, batch):
        character_ids = batch_to_ids(batch)
        if torch.cuda.is_available():
            character_ids = character_ids.cuda()
        torch.set_default_tensor_type(torch.FloatTensor)
        embeds = self.elmo(character_ids)['elmo_representations'][0]
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        return embeds

    def WIC_main(self, iteration=None, dataset=None, ret_pred_list=False):

        # encoder in eval mode and classifier in train mode
        self.elmo.eval()
        self.classifier.train()

        train_dataset = self.train_dataset
        val_dataset = self.val_dataset

        if args.val_type == 'threshold':

            print("=" * 50 + "\n\t\tTraining set\n" + "=" * 50)

            # getting id2word
            id2word = data.load_id2word_dict()

            embeds, lengths, batch_labels, batch_p1, batch_p2 = train_dataset.get_batch(
                train_dataset.get_num_examples(), loop_dataset=False,
                toTorch=True)

            print("\n--------------Target Embeddings retrieved-------------")
            self.threshold_evaluator(self.get_elmo_batch(embeds[0], lengths[0], batch_p1, id2word),
                                     self.get_elmo_batch(embeds[1], lengths[1], batch_p2, id2word),
                                     batch_labels)

            print("=" * 70 + "\n\t\t\tValidation set\n" + "=" * 70)
            embeds, lengths, batch_labels, batch_p1, batch_p2 = val_dataset.get_batch(train_dataset.get_num_examples(),
                                                                                      loop_dataset=False,
                                                                                      toTorch=True)

            print("\n--------------Target Embeddings retrieved-------------")
            self.threshold_evaluator(self.get_elmo_batch(embeds[0], lengths[0], batch_p1, id2word),
                                     self.get_elmo_batch(embeds[1], lengths[1], batch_p2, id2word),
                                     batch_labels)



        elif args.val_type == 'mlp':

            self.mlp_evaluator()

        else:
            print("Invalid Evaluator!! Choose --mlp-- or --threshold--")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path",
                        help="Folder(name) where checkpoints are saved. If it is a regex expression, all experiments are evaluated",
                        type=str, default='./checkpoints')
    parser.add_argument("--overwrite",
                        help="Whether evaluations should be re-run if there already exists an evaluation file.",
                        action="store_true")
    parser.add_argument("--visualize_embeddings",
                        help="Whether the embeddings of the model should be visualized or not", action="store_true")
    parser.add_argument("--val_type", help="Whether to run threshold or MLP eval (threshold or mlp)", type=str,
                        default='mlp')
    parser.add_argument("--lr", help="Learning rate (for MLP evaluation)", type=float, default=2e-3)
    args = parser.parse_known_args()[0]
    # model_list = sorted(glob(args.checkpoint_path))

    # for model_checkpoint in model_list:

    #     try:
    #         model = load_model_from_args(load_args(args.checkpoint_path), args.checkpoint_path)            

    #     except RuntimeError as e:
    #         print("[!] Runtime error while loading " + model_checkpoint)
    #         print(e)
    #         continue
    wic = WIC()
    wic.WIC_main()

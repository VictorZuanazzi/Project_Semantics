import torch 
import torch.nn as nn
import argparse
import math
import os
import sys
import json
import pickle
import numpy as np
from glob import glob

from model2 import NLIModel
from model import MultiTaskEncoder
from data import DatasetHandler as data 
from mutils import load_model, load_model_from_args, load_args, args_to_params, visualize_tSNE, get_transfer_datasets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from tensorboardX import SummaryWriter


class WICclassifier(nn.Module):
    def __init__(self, batch_size=64, lstm_dim= 400, fc_dim = 100, n_classes = 1):
        super(WICclassifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(lstm_dim*2,fc_dim), nn.ReLU(), nn.Linear(fc_dim,n_classes), nn.Sigmoid())
    def forward(self,inputs):
        return self.classifier(inputs)
        
    

class WIC:

    def __init__(self, model, batch_size=64, lstm_dim= 400, fc_dim = 100, n_classes = 1):
        self.train_dataset, self.val_dataset, self.test_dataset = data.load_WiC_datasets(debug_dataset = False)
        self.encoder = model[0]
        self.classifier = WICclassifier(batch_size= batch_size, lstm_dim=lstm_dim, fc_dim=fc_dim)
        self.classifier = self.classifier.to(device)
        self.encoder = self.encoder.to(device)
        self.batch_size = batch_size
        
        
    def get_target_embed(self, embeds, lengths, batch_p1, batch_p2):
        # encode the sentences batch
        sent1_encoded = self.encoder.encode_sentence(embeds[0], lengths[0], word_level=True)[1].detach()
        sent2_encoded = self.encoder.encode_sentence(embeds[1], lengths[1], word_level=True)[1].detach()

        # prepare the input features based on specified word only  
        bins1 = torch.arange(0, sent1_encoded.size(0)*sent1_encoded.size(1), step = sent1_encoded.size(1))
        bins2 = torch.arange(0, sent2_encoded.size(0)*sent2_encoded.size(1), step = sent2_encoded.size(1))

        sent1_encoded = sent1_encoded.view(-1, sent1_encoded.size(2))
        sent2_encoded = sent2_encoded.view(-1, sent2_encoded.size(2))

        u = torch.index_select(sent1_encoded, dim=0, index = bins1 + batch_p1)
        v = torch.index_select(sent2_encoded, dim=0, index = bins2 + batch_p2)
        
        return u, v

    def train(self, iteration=None, dataset=None, ret_pred_list=False):
        
        # encoder in eval mode and classifier in train mode
        self.encoder.eval()
        self.classifier.train()
        
        l_rate = 2e-3
               
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr = l_rate)
        criterion = nn.BCELoss(reduction = 'mean') # reduction = sum/mean??
        
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset
        
        number_batches_train = int(math.ceil(train_dataset.get_num_examples() * 1.0 / self.batch_size))
        number_batches_val = int(math.ceil(val_dataset.get_num_examples() * 1.0 / self.batch_size))        
        
        print("Number of train batches = ", number_batches_train)
        print("Number of val batches = ", number_batches_val)
        
        preds_list = []
        loss_list = []
        accuracy = []
        dev_accuracy = []
        prev_dev = 0
       
        
        for epoch in range(10):
            for batch_ind in range(number_batches_train):
                self.classifier.train()
               
                # get batch of data
                embeds, lengths, batch_labels, batch_p1, batch_p2 = train_dataset.get_batch(self.batch_size, loop_dataset=False, 
                                                                                            toTorch=True)
                
                
                # encode the sentences batch
                u, v = self.get_target_embed(embeds,lengths, batch_p1, batch_p2)
               
                input_features = torch.cat((u,v), dim=1)
#                 cos_sim = torch.mm(u.permute(0,1), v.permute(1,0)).squeeze()
#                 preds = cos_sim/torch.sum(cos_sim)

                # get predictions
                out = self.classifier(input_features)
          

                # correct preds
                output = (out>0.5).float()
                correct = (output.squeeze() == batch_labels.float()).float().sum()
                acc = correct/len(output)
                accuracy.append(acc.item())

                # calculate loss
                loss = criterion(out[:,0], batch_labels.float())
                loss_list.append(loss.item())
                
#                 a = list(self.classifier.parameters())[0].clone()
                
                # back prop steps
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
#                 b = list(self.classifier.parameters())[0].clone()
                
#                 print("Params equal = ", torch.equal(a,b))
            
            # Evaluate on dev set
            for batch_ind in range(number_batches_val):
                self.classifier.eval()
                
                embeds, lengths, batch_labels, batch_p1, batch_p2 = val_dataset.get_batch(self.batch_size, loop_dataset=False, 
                                                                                            toTorch=True)
                u,v = self.get_target_embed(embeds, lengths, batch_p1, batch_p2)
                input_features = torch.cat((u,v), dim=1)
                out = self.classifier(input_features)
                output = (out>0.5).float()
                correct = (output.squeeze() == batch_labels.float()).float().sum()
                acc = correct/len(output)
                dev_accuracy.append(acc.item())

            
            print("Epoch: {}/10,  Train accuracy = {:.2f}, Train Loss = {:.4f}, Val accuracy = {:.2f}".format(epoch, sum(accuracy)/len(accuracy), sum(loss_list)/len(loss_list), sum(dev_accuracy)/len(dev_accuracy)))
            current_dev = sum(dev_accuracy)/len(dev_accuracy)
            if prev_dev > current_dev:
                l_rate = l_rate*0.3
                print("Reduced learning rate to: ", l_rate)
            prev_dev = current_dev
            
        print("Training Done")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints are saved. If it is a regex expression, all experiments are evaluated", type=str, default = './checkpoints')
    parser.add_argument("--overwrite", help="Whether evaluations should be re-run if there already exists an evaluation file.", action="store_true")
    parser.add_argument("--visualize_embeddings", help="Whether the embeddings of the model should be visualized or not", action="store_true")
    parser.add_argument("--full_senteval", help="Whether to run SentEval with the heavy setting or not", action="store_true")
    args = parser.parse_known_args()[0]
    model_list = sorted(glob(args.checkpoint_path))

    for model_checkpoint in model_list:

        try:
            model = load_model_from_args(load_args(args.checkpoint_path), args.checkpoint_path)            
            
        except RuntimeError as e:
            print("[!] Runtime error while loading " + model_checkpoint)
            print(e)
            continue
        wic = WIC(model)
        wic.train()
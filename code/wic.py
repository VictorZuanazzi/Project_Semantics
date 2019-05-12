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
from model import MultiTaskEncoder, get_device
from data import DatasetHandler as data 
from mutils import load_model, load_model_from_args, load_args, args_to_params, visualize_tSNE, get_transfer_datasets

# from tensorboardX import SummaryWriter


class WICclassifier(nn.Module):
    def __init__(self, batch_size=64, lstm_dim= 400, fc_dim = 100, n_classes = 1):
        super(WICclassifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(lstm_dim*2,fc_dim), nn.ReLU(), nn.Linear(fc_dim,n_classes), nn.Sigmoid())
    def forward(self,inputs):
        return self.classifier(inputs)
        
    

class WIC:

    def __init__(self, model, batch_size=64, lstm_dim= 400, fc_dim = 100, n_classes = 2):
        self.train_dataset, self.val_dataset, self.test_dataset = data.load_WiC_datasets(debug_dataset = False)
        self.encoder = model[0]
        self.classifier = WICclassifier(batch_size= batch_size, lstm_dim=lstm_dim, fc_dim=fc_dim)
        self.batch_size = batch_size
        self.classifier = self.classifier.to(get_device())

    def train(self, iteration=None, dataset=None, ret_pred_list=False):
        
        # encoder in eval mode and classifier in train mode
        self.encoder.eval()
        self.classifier.train()
               
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr = 1e-3)
        criterion = nn.BCELoss(reduction = 'sum') # reduction = sum/mean??
        
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset
        
        number_batches = int(math.ceil(train_dataset.get_num_examples() * 1.0 / self.batch_size))
        correct_preds = []
        preds_list = []
        loss_list = []
        accuracy = []
        
        for epoch in range(10):
            for batch_ind in range(number_batches):
               
                # get batch of data
                embeds, lengths, batch_labels, batch_p1, batch_p2 = train_dataset.get_batch(self.batch_size, loop_dataset=False, 
                                                                                            toTorch=True)

               
                # encode the sentences batch
                with torch.no_grad():
                    sent1_encoded = self.encoder.encode_sentence(embeds[0], lengths[0], word_level=True)[1].detach()
                    sent2_encoded = self.encoder.encode_sentence(embeds[1], lengths[1], word_level=True)[1].detach()
                
                # prepare the input features based on specified word only  
                bins1 = torch.arange(0, sent1_encoded.size(0)*sent1_encoded.size(1), step = sent1_encoded.size(1), device=get_device())
                bins2 = torch.arange(0, sent2_encoded.size(0)*sent2_encoded.size(1), step = sent2_encoded.size(1), device=get_device())
                
                sent1_encoded = sent1_encoded.view(-1, sent1_encoded.size(2))
                sent2_encoded = sent2_encoded.view(-1, sent2_encoded.size(2))
               
                u = torch.index_select(sent1_encoded, dim=0, index = bins1 + batch_p1)
                v = torch.index_select(sent2_encoded, dim=0, index = bins2 + batch_p2)
                
                
#                 u = torch.zeros((64,400), requires_grad=True)
#                 v = torch.zeros((64,400), requires_grad=True)
#                 for i, index in enumerate(batch_p1):
#                     u[i,:] = sent1_encoded[i,index, :]
#                 for i, index in enumerate(batch_p2):
#                     v[i,:] = sent2_encoded[i,index, :]
#                 print(u)

                input_features = torch.cat((u,v), dim=1)
#                 cos_sim = torch.mm(u.permute(0,1), v.permute(1,0)).squeeze()
#                 preds = cos_sim/torch.sum(cos_sim)

                # get predictions
                out = self.classifier(input_features)[:,0]

                # correct preds
#                 correct_preds += torch.squeeze(preds == batch_labels).tolist()
#                 acc = sum(correct_preds)/len(correct_preds)
#                 accuracy.append(acc)

                # calculate loss
                loss = criterion(out, batch_labels.float())
                loss_list.append(loss.item())

                # back prop steps
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Evaluate on dev set
            ########################
            ######## code ##########
            dev_accuracy = 0
            print("Epoch: {}/10,  Train accuracy = {},  Val accuracy = {}".format(epoch, np.sum(accuracy)*100, np.sum(dev_accuracy)*100))
            


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints are saved. If it is a regex expression, all experiments are evaluated", type=str, default = './checkpoints')
    parser.add_argument("--overwrite", help="Whether evaluations should be re-run if there already exists an evaluation file.", action="store_true")
    
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
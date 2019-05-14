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
import tqdm


from model import MultiTaskEncoder
from data import DatasetHandler as data 
from vocab import get_id2word_dict 
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
    
    
    def threshold_evaluator(self,u,v,batch_labels, best_split=0, val =0):
                
        u = u.unsqueeze(dim=1)
        v = v.unsqueeze(dim=2)
        cos_sim = torch.bmm(u,v).squeeze()

        min_val = torch.min(cos_sim)
        max_val = torch.max(cos_sim)
     
        print("\nMinimum cosin similarity value = {:.4f}".format(min_val.item()))
        print("Maximum cosin similarity value = {:.4f}\n".format(max_val.item()))
        
        if val==0:  
            
            # Getting the best split on training data
            best_acc = 0
            for iter in torch.arange(min_val, max_val, step =0.02):
                preds = (cos_sim>iter).float()
                correct = (preds == batch_labels.float()).float().sum()
                acc = correct/len(preds)
                if acc> best_acc:
                    best_acc = acc.item()
                    best_split = iter
                    print("New best accuracy = {:.4f}, Split = {:.4f}".format(best_acc, best_split))
            print("="*70 + "\nBest accuracy found = {:.4f}, Best split found = {:.4f}".format(best_acc, best_split)
                  + "\n"+"="*70)
            return best_split
        
        elif val==1:
            # Evaluating on the best split obtained from training
            print("Best split is: ", best_split.item())
            preds = (cos_sim>best_split).float()
            correct = (preds == batch_labels.float()).float().sum()
            acc = correct/len(preds)
            print("Validation accuracy = ", acc.item()*100)
            
        else:
            print("[!] INVALID CHOICE!! Please enter either 0 or 1 for val argument !!")
    
    

    
    def mlp_evaluator(self):
        self.encoder.eval()
        self.classifier.train()

        l_rate = args.lr

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr = l_rate)
        criterion = nn.BCELoss(reduction = 'mean') # reduction = sum/mean??
        
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset

        number_batches_train = int(math.ceil(train_dataset.get_num_examples() * 1.0 / self.batch_size))
        number_batches_val = int(math.ceil(val_dataset.get_num_examples() * 1.0 / self.batch_size))
        
        print("\nNumber of train batches = ", number_batches_train)
        print("Number of val batches = ", number_batches_val)

        
        loss_list = []
        accuracy = []
        dev_accuracy = []
        prev_dev = 0
        best_dev_acc = 0

        for epoch in range(50):
            if l_rate < 1e-6:
                print("Termination conditon reached!")
                break
            
            self.classifier.train()
            
            for batch_ind in range(number_batches_train):
                self.classifier.train()

                # get batch of data
                embeds, lengths, batch_labels, batch_p1, batch_p2 = train_dataset.get_batch(self.batch_size, loop_dataset=False, 
                                                                                            toTorch=True)

                # changing index to account for <S> token added
                batch_p1 = torch.tensor([indx +1 for indx in batch_p1])
                batch_p2 = torch.tensor([indx +1 for indx in batch_p2])
                
                # Get target word embeddings
                u, v = self.get_target_embed(embeds,lengths, batch_p1, batch_p2)

                # Prepare input for classifier
                input_features = torch.cat((u,v), dim=1)

                # get predictions
                out = self.classifier(input_features)


                # accuracy calculation
                output = (out>0.5).float()
                correct = (output.squeeze() == batch_labels.float()).float().sum()
                acc = correct/len(output)
                accuracy.append(acc.item())

                # calculate loss
                loss = criterion(out[:,0], batch_labels.float())
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

                embeds, lengths, batch_labels, batch_p1, batch_p2 = val_dataset.get_batch(self.batch_size, loop_dataset=False, 
                                                                                            toTorch=True)
                
                batch_p1 = torch.tensor([b+1 for b in batch_p1])
                batch_p2 = torch.tensor([b+1 for b in batch_p2])
                
                u,v = self.get_target_embed(embeds, lengths, batch_p1, batch_p2)
                input_features = torch.cat((u,v), dim=1)
                out = self.classifier(input_features)
                output = (out>0.5).float()
                correct = (output.squeeze() == batch_labels.float()).float().sum()
                acc = correct/len(output)
                dev_accuracy.append(acc.item())
                

            print("Epoch: {}/50,  Train accuracy = {:.4f}, Train Loss = {:.4f}, Val accuracy = {:.4f}".format(epoch, sum(accuracy)/len(accuracy)*100, sum(loss_list)/len(loss_list), sum(dev_accuracy)/len(dev_accuracy)*100))
            current_dev = sum(dev_accuracy)/len(dev_accuracy)
            if prev_dev > current_dev:
                l_rate = l_rate*0.20
                print("Reduced learning rate to: ", l_rate)
            prev_dev = current_dev
            
            if current_dev > best_dev_acc:
                print("NEW best Dev accuracy!!")
                best_dev_acc = current_dev

        print("Training Done!!")
        print("="*70+"\n"+"Best Validation accuracy = {:.4f}%\n".format(current_dev*100)+"="*70)
                
        

    def WIC_main(self, iteration=None, dataset=None, ret_pred_list=False):
        
        # encoder in eval mode and classifier in train mode
        self.encoder.eval()
        self.classifier.train()
       
        
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset
         
        dict_id = get_id2word_dict()
        if args.val_type == 'threshold':
            
            print("="*70 + "\n\t\t\tTraining set\n"+ "="*70)
            embeds, lengths, batch_labels, batch_p1, batch_p2 = train_dataset.get_batch(train_dataset.get_num_examples(), loop_dataset=False, toTorch=True)
            # changing index to account for <S> token added
            batch_p1 = torch.tensor([b+1 for b in batch_p1])
            batch_p2 = torch.tensor([b+1 for b in batch_p2])
            
                
            print("\n---------------Getting Target Embeddings--------------")
            # encode the sentences batch
            u, v = self.get_target_embed(embeds,lengths, batch_p1, batch_p2)
            
#             idxs = []           
#             for i in range(u.size(0)):
#                 if torch.max(norm[i]):
#                     print(i)
#                     idxs.append(i)
            
            
#             errors1 = [dict_id[word.item()] for id in idxs for word in embeds[0][id]]
#             print(errors1)
            
#             errors2 = [dict_id[word.item()] for id in idxs for word in embeds[1][id]]
#             print(errors2)
                        
            # Normalizing the embeddings for stable cosin similarity calculation
            norm = u.norm(p=2, dim=1, keepdim=True).detach()
            u = u.div(norm)
            
            norm = v.norm(p=2, dim=1, keepdim=True).detach()
            v = v.div(norm)

            print("\n---------------Target Embeddings retrieved--------------")
            best_split = self.threshold_evaluator(u,v,batch_labels, val = 0)
            
            print("="*70 + "\n\t\t\tValidation set\n"+ "="*70)
            embeds, lengths, batch_labels, batch_p1, batch_p2 = val_dataset.get_batch(train_dataset.get_num_examples(), loop_dataset=False, 
                                                                                            toTorch=True)
            # changing index to account for <S> token added
            batch_p1 = torch.tensor([b+1 for b in batch_p1])
            batch_p2 = torch.tensor([b+1 for b in batch_p2])
                
            print("\n---------------Getting Target Embeddings--------------")
            # encode the sentences batch
            u, v = self.get_target_embed(embeds,lengths, batch_p1, batch_p2)
            
            # Normalizing the embeddings for stable cosin similarity calculation
            norm = u.norm(p=2, dim=1, keepdim=True).detach()
            u = u.div(norm)
            
            norm = v.norm(p=2, dim=1, keepdim=True).detach()
            v = v.div(norm)
            print("\n---------------Target Embeddings retrieved--------------")
            self.threshold_evaluator(u,v,batch_labels, best_split, val = 1)
            
            
        
        elif args.val_type == 'mlp':
            
            self.mlp_evaluator()
            
        else:
            print("Invalid Evaluator!! Choose --mlp-- or --threshold--")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints are saved. If it is a regex expression, all experiments are evaluated", type=str, default = './checkpoints/mnli_vu_pos_12')
    parser.add_argument("--overwrite", help="Whether evaluations should be re-run if there already exists an evaluation file.", action="store_true")
    parser.add_argument("--visualize_embeddings", help="Whether the embeddings of the model should be visualized or not", action="store_true")
    parser.add_argument("--val_type", help="Whether to run threshold or MLP eval (--threshold-- or --mlp--)", type = str, default = 'mlp')
    parser.add_argument("--lr", help="Learning rate (for MLP evaluation)", type = float, default = 1e-3)
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
        wic.WIC_main()
import os
import math
import itertools
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm

from sparse_linear import sparse_linear
from torch.utils.tensorboard import SummaryWriter
from DRNet import RuleFunction, LabelFunction, Binarization as RuleBinarization


class CancelOut(nn.Module):

    def __init__(self, input_size, *kargs, **kwargs):
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(input_size, requires_grad = True) + 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return (x * self.relu(self.weights.float()))
    
    def regularization(self):
        weights_co = self.relu(self.weights)
        
        return torch.norm(weights_co, 1) / len(weights_co)
    
    
class CancelBinarization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        
        output = (input > 0.000001).float()
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        
        grad_input = grad_output.clone()
        
        grad_input[input <= 0] = 0
        
        return grad_input
    

class R2Ntab(nn.Module):
    def __init__(self, in_features, num_rules, out_features, lr_rules=1e-2, 
              lr_cancel=5e-3, and_lam=1e-2, or_lam=1e-5, cancel_lam=1e-4):

        super(R2Ntab, self).__init__()
        
        self.linear = sparse_linear('l0')
        self.cancelout_layer = CancelOut(in_features)
        self.and_layer = self.linear(in_features, num_rules, linear=RuleFunction.apply)
        self.or_layer = self.linear(num_rules, out_features, linear=LabelFunction.apply)
        
        self.lr_rules = lr_rules
        self.lr_cancel = lr_cancel
        self.and_lam = and_lam
        self.or_lam = or_lam
        self.cancel_lam = cancel_lam

        self.and_layer.bias.requires_grad = False
        self.and_layer.bias.data.fill_(1)
        self.or_layer.weight.requires_grad = False
        self.or_layer.weight.data.fill_(1)
        self.or_layer.bias.requires_grad = False
        self.or_layer.bias.data.fill_(-0.5)
        
    def forward(self, input):
        out = self.cancelout_layer(input)
        out = CancelBinarization.apply(out)
        out = self.and_layer(out)
        out = RuleBinarization.apply(out)
        out = self.or_layer(out)
        
        return out
        
    def reweight_layer(self):
        with torch.no_grad():
            indices = torch.where(self.cancelout_layer.weights < 0)[0]
            for index in indices:
                self.and_layer.weight[:,index] = 0
    
    def regularization(self):
        sparsity = ((self.and_layer.regularization(axis=1)+1) * self.or_layer.regularization(mean=False)).mean()
        
        return sparsity
    
    def statistics(self):
        rule_indices = (self.or_layer.masked_weight() != 0).nonzero()[:, 1]
        
        reweight = self.and_layer.masked_weight().clone()
        indices = torch.where(self.cancelout_layer.weights < 0)     
        for index in indices:
            reweight[:,index] = 0
        
        sparsity = (reweight[rule_indices] == 0).float().mean().item()
        num_rules = rule_indices.size(0)
        return sparsity, num_rules
            
    def extract_rules(self, header=None, print_rules=False):
        self.eval()
        self.to('cpu')

        prune_weights = self.and_layer.masked_weight()
        valid_indices = self.or_layer.masked_weight().nonzero(as_tuple=True)[1]
        rules = np.sign(prune_weights[valid_indices].detach().numpy()) * 0.5 + 0.5

        if header != None:
            rules_exp = []
            for weight in prune_weights[valid_indices]:
                rule = []
                for w, h in zip(weight, header):
                    if w < 0:
                        rule.append('not ' + h)
                    elif w > 0:
                        rule.append(h)
                rules_exp.append(rule)
            rules = rules_exp
            
            if print_rules:
                print("Rulelist:")
                for index, rule in enumerate(rules):
                    if index == 0:
                        print('if', end=' ')
                    else:
                        print('else if', end=' ')

                    print('[', end=' ')
                    for index, condition in enumerate(rule):
                        print(condition, end=' ')
                        if index != len(rule) - 1:
                            print('&&', end=' ')

                    print(']:')        
                    print('  prediction = true')
                print('else')
                print('  prediction = false')

        return rules 

    def predict(self, X, Y):
        X = np.array(X)
        rules = self.extract_rules()
        
        results = []
        for x in X:
            indices = np.where(np.absolute(x - rules).max(axis=1) < 1)[0]
            result = int(len(indices) != 0)
            results.append(result)
            
        Y_pred = np.array(results)
        
        return (Y == Y_pred).mean()

    def fit(self, train_set, test_set, device="cpu", epochs=2000, num_alter=500, batch_size=400, cancel_iterations=500, track_performance=False, dummy_index=None):
        def score(out, y):
            y_labels = (out >= 0).float()
            y_corrs = (y_labels == y.reshape(y_labels.size())).float()

            return y_corrs

        reg_lams = [self.and_lam, self.or_lam]

        optimizers = [optim.Adam(self.and_layer.parameters(), lr=self.lr_rules),
                      optim.Adam(self.or_layer.parameters(), lr=self.lr_rules)]

        optimizer_cancel = optim.Adam(self.cancelout_layer.parameters(), lr=self.lr_cancel)

        criterion = nn.BCEWithLogitsLoss().to(device)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=True)

        dummies, accuracies = [], []

        for epoch in tqdm(range(epochs), ncols=60):
            self.to(device)
            self.train()

            batch_losses = []
            batch_corres = []

            for index, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                out = self(x_batch)

                phase = int((epoch / num_alter) % 2)

                optimizers[phase].zero_grad()
                optimizer_cancel.zero_grad()

                loss = criterion(out, y_batch.reshape(out.size())) + reg_lams[phase] * self.regularization() + self.cancel_lam * self.cancelout_layer.regularization()

                loss.backward()

                optimizers[phase].step()
                if epoch < cancel_iterations:
                    optimizer_cancel.step()

                corr = score(out, y_batch).sum()

                batch_losses.append(loss.item())
                batch_corres.append(corr.item())

            epoch_loss = torch.Tensor(batch_losses).mean().item()
            epoch_accu = torch.Tensor(batch_corres).sum().item() / len(train_set)

            if dummy_index is not None:
                count_dummies = 0
                for idx in range(dummy_index, len(self.cancelout_layer.weights)):
                    count_dummies += (self.cancelout_layer.weights[idx] < 0)

                dummies.append(count_dummies)

            self.to('cpu')
            self.eval()
            with torch.no_grad():
                test_accu = score(self(test_set[:][0]), test_set[:][1]).mean().item()
                sparsity, num_rules = self.statistics()

                accuracies.append(test_accu)

        if track_performance:
            return accuracies

        if dummy_index is not None:
            return dummies
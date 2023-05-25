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

class CancelOut(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        
        output = input.clone()
        
        indices = torch.where(weight < 0)
        
        for index in indices:
            output[:,index] -= 2
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        
        grad_weight = grad_output.clone()
        
        return None, grad_weight
    

class RuleFunction(torch.autograd.Function):
    '''
    The autograd function used in the Rules Layer.
    The forward function implements the equation (1) in the paper.
    The backward function implements the gradient of the forward function.
    '''
    @staticmethod
    def forward(ctx, input, weight, bias):                
        for index in range(input.size(1)):
            if not torch.any(input[:,index] >= 0):
                input[:,index] += 2
                weight[:,index] = 0
            
        ctx.save_for_backward(input, weight, bias)            
            
        output = input.mm(weight.t())    
        output = output + bias.unsqueeze(0).expand_as(output)
        output = output - (weight * (weight > 0)).sum(-1).unsqueeze(0).expand_as(output)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input) - grad_output.sum(0).unsqueeze(1).expand_as(weight) * (weight > 0)
        grad_bias = grad_output.sum(0)
        grad_bias[(bias >= 1) * (grad_bias < 0)] = 0
        
        return grad_input, grad_weight, grad_bias


class Binarization(torch.autograd.Function):
    '''
    The autograd function for the binarization activation in the Rules Layer.
    The forward function implements the equations (2) in the paper. Note here 0.999999 is used to cancel the rounding error.
    The backward function implements the STE estimator with equation (3) in the paper.
    '''
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = (input > 0.999999).float()
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        
        grad_input = grad_output.clone()
        grad_input[(input < 0)] = 0
        grad_input[(input >= 1) * (grad_output < 0)] = 0
        
        return grad_input
    
    
class LabelFunction(torch.autograd.Function):
    '''
    The autograd function used in the OR Layer.
    The forward function implements the equations (4) and (5) in the paper.
    The backward function implements the standard STE estimator.
    '''
    
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
    
        output = input.mm((weight.t() > 0).float())       
        output += bias.unsqueeze(0).expand_as(output)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
    

class R2Ntab(nn.Module):
    def __init__(self, in_features, num_rules, out_features, cancel_rate=5e-3):
        """
        DR-Net: https://arxiv.org/pdf/2103.02826.pdf
        
        Args
            in_features (int): the input dimension.
            num_rules (int): number of hidden neurons, which is also the maximum number of rules.
            out_features (int): the output dimension; should always be 1.
        """
        super(R2Ntab, self).__init__()
        
        self.linear = sparse_linear('l0')
        self.cancel = sparse_linear('cancel')
        self.cancelout_layer = self.cancel(in_features, 1, linear=CancelOut.apply, cancel_rate=cancel_rate)
        self.and_layer = self.linear(in_features, num_rules, linear=RuleFunction.apply)
        self.or_layer = self.linear(num_rules, out_features, linear=LabelFunction.apply)

        self.and_layer.bias.requires_grad = False
        self.and_layer.bias.data.fill_(1)
        self.or_layer.weight.requires_grad = False
        self.or_layer.weight.data.fill_(1)
        self.or_layer.bias.requires_grad = False
        self.or_layer.bias.data.fill_(-0.5)
        
    def forward(self, input):
        out = self.cancelout_layer(input)
        out = self.and_layer(out)
        out = Binarization.apply(out)
        out = self.or_layer(out)
        
        return out
        
    def reweight_layer(self):
        with torch.no_grad():
            indices = torch.where(self.cancelout_layer.weight < 0)[0]
            for index in indices:
                self.and_layer.weight[:,index] = 0
    
    def regularization(self):
        """
        Implements the Sparsity-Based Regularization (equation 7).
        
        Returns
            regularization (float): the regularization term.
        """

        sparsity = ((self.and_layer.regularization(axis=1)+1) * self.or_layer.regularization(mean=False)).mean()
        
        #print('sparsity: ')
        #print(sparsity)
        
        return sparsity
    
    def statistics(self):
        """
        Return the statistics of the network.
        
        Returns
            sparsity (float): sparsity of the rule set.
            num_rules (int): number of unpruned rules.
        """
        
        rule_indices = (self.or_layer.masked_weight() != 0).nonzero()[:, 1]
        
        reweight = self.and_layer.masked_weight().clone()
        indices = torch.where(self.cancelout_layer.weight < 0)     
        for index in indices:
            reweight[:,index] = 0
        
        sparsity = (reweight[rule_indices] == 0).float().mean().item()
        num_rules = rule_indices.size(0)
        return sparsity, num_rules
            
    def get_rules(self, header=None):
        """
        Translate network into rules.
        
        Args
            header (list OR None): the description of each input feature.
        Returns
            rules (np.array OR list): contains a list of rules. 
                If header is None (2-d np.array), each rule is represented by a list of numbers (1: positive feature, 0: negative feature, 0.5: dont' care).
                If header is not None (list of lists): each rule is represented by a list of strings.
        """
        
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
                        rule.append('NOT ' + h)
                    elif w > 0:
                        rule.append(h)
                rules_exp.append(rule)
            rules = rules_exp

        return rules

    def predict(self, X):
        """
        Classifiy the labels of X using rules encoded by the network.
        
        Args
            X (np.array) 2-d np.array of instances with binary features.
        Returns
            results (np.array): 1-d array of labels.
        """
        
        rules = self.get_rules()
        
        results = []
        for x in X:
            indices = np.where(np.absolute(x - rules).max(axis=1) < 1)[0]
            result = int(len(indices) != 0)
            results.append(result)
        return np.array(results)
    
    def save(self, path):
        state = {
            'state_dict': self.state_dict(),
            'parameters': {
                'in_features': self.and_layer.weight.size(1), 
                'num_rules': self.and_layer.bias.size(0), 
                'out_features': self.or_layer.bias.size(0), 
                'and_lam': self.and_lam, 
                'or_lam': self.or_lam,
            }
        }
        
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        torch.save(state, path)
        
    @staticmethod
    def load(path):
        state = torch.load(path)
        model = DRNet(**state['parameters'])
        model.load_state_dict(state['state_dict'])
        
        return model

def train(net, train_set, test_set, device="cuda", epochs=2000, batch_size=2000, lr_rules=1e-2, 
          lr_cancel=5e-3, and_lam=1e-2, or_lam=1e-5, cancel_lam=1e-3, num_alter=500, track_performance=False, dummy_index=None):
    def score(out, y):
        y_labels = (out >= 0).float()
        y_corrs = (y_labels == y.reshape(y_labels.size())).float()
        
        return y_corrs
        
    reg_lams = [and_lam, or_lam]

    optimizers = [optim.Adam(net.and_layer.parameters(), lr=lr_rules),
                  optim.Adam(net.or_layer.parameters(), lr=lr_rules)]
    
    optimizer_cancel = optim.Adam(net.cancelout_layer.parameters(), lr=lr_cancel)

    criterion = nn.BCEWithLogitsLoss().to(device)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=True)
    
    losses, performance_constraint, cancel_constraint, rule_constraint = [], [], [], []
    
    accuracies, rules = [], []
    
    writer = SummaryWriter()
    
    with tqdm(total=epochs, desc="Epoch", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as t:
        for epoch in range(epochs):
            net.to(device)
            net.train()

            batch_losses = []
            batch_corres = []
            
            #if epoch > 100:
                #optimizer_cancel = optim.Adam(net.cancelout_layer.parameters(), lr=lr_cancel*2)
                
            #if epoch > 200:
                #optimizer_cancel = optim.Adam(net.cancelout_layer.parameters(), lr=lr_cancel)
            
            and_layer_weight = net.and_layer.weight
            for index, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                net.and_layer.weight = and_layer_weight

                out = net(x_batch)
                
                phase = int((epoch / num_alter) % 2)
                
                optimizers[phase].zero_grad()
                optimizer_cancel.zero_grad()
                
                performance = criterion(out, y_batch.reshape(out.size()))
                sparsity = net.regularization()
                
                #if epoch < 250 or epoch >= 500:
                loss = performance + reg_lams[phase] * sparsity + cancel_lam * net.cancelout_layer.regularization()

                loss.backward()
                
                losses.append(loss)
                
                optimizers[phase].step()
                if epoch >= 50 and epoch < 500:
                    optimizer_cancel.step()
                    
                performance_constraint.append(performance)
                cancel_constraint.append((net.cancelout_layer.weight < 0).sum().item())
                rule_constraint.append(sparsity)

                corr = score(out, y_batch).sum()

                batch_losses.append(loss.item())
                batch_corres.append(corr.item())
                
            count = (net.cancelout_layer.weight < 0).sum().item()
                
            writer.add_scalar("CancelOut weights over time", count, epoch)
                
            epoch_loss = torch.Tensor(batch_losses).mean().item()
            epoch_accu = torch.Tensor(batch_corres).sum().item() / len(train_set)
            
            if dummy_index is not None:
                count_dummies = 0
                for idx in range(dummy_index, len(net.cancelout_layer.weight)):
                    count_dummies += (net.cancelout_layer.weight[idx] < 0)
                    
                dummies.append(count_dummies)

            net.to('cpu')
            net.eval()
            with torch.no_grad():
                test_accu = score(net(test_set[:][0]), test_set[:][1]).mean().item()
                sparsity, num_rules = net.statistics()
                
                accuracies.append(test_accu)
                #cancel_reg.append(c)
                #rule_reg.append(r)
                #loss_ot.append(loss)
                rules.append(num_rules)
                #cancelled.append(count)
                
            t.update(1)
            t.set_postfix({
                'rules cancelled': count,
                'loss': epoch_loss,
                'epoch accu': epoch_accu,
                'test accu': test_accu,
                'num rules': num_rules,
                'sparsity': sparsity,
            })
            
    net.reweight_layer()
            
    writer.flush()
    
    if track_performance:
        #return performance_constraint, cancel_constraint, rule_constraint
        return accuracies, rules
    
    if dummy_index is not None:
        return dummies
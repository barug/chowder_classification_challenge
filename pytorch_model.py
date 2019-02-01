import numpy as np

import torch 
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from pytorch_metrics import compute_weights_metrics, compute_forward_metrics, compute_backward_metrics

from collections import OrderedDict
from pathlib import Path


class MinMax(nn.Module):
    """Implementation of the MinMax layer described in the Chowder paper"""

    def __init__(self, r):
        super(MinMax, self).__init__()
        self.R = r
    
    def forward(self, x):
        top, _ = torch.topk(x, self.R, sorted=True)
        bottom, _ = torch.topk(x, self.R, largest=False, sorted=True)
        res = torch.cat((top, bottom), dim=2)
        return res


class PytorchChowder:
    """Encapsulate a pytroch implementation of the Chowder algorithm"""

    def __init__(self, r = 5, bs = 10, ep = 30, lr = 0.001, mt = 1000, drp= 0.5 ):
        self.R = r
        self.BATCH_SIZE = bs
        self.EPOCHS = ep
        self.LR = lr
        self.MAX_TILES_NBR = mt

        self.model = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(2048, 1, 1)),
            ('drop1', nn.Dropout(p = drp)),
            ('minmax', MinMax(self.R)),
            ('lin1', nn.Linear(2 * self.R, 200)),
            ('sig1', nn.Sigmoid()),
            ('drop2', nn.Dropout(p = drp)),
            ('lin2', nn.Linear(200, 100)),
            ('sig2', nn.Sigmoid()),
            ('drop3', nn.Dropout(p = drp)),
            ('lin3', nn.Linear(100, 1))
        ])).float()

        self.loss_func = nn.BCEWithLogitsLoss()
        self.init_parameters()
        self.init_optimizer()
        
    def init_parameters(self):
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
        self.model.apply(init_layer)

    def init_optimizer(self):
        # init optimizer and apply L2 regularization to conv parameters
        params_conv = list(self.model[0].parameters())
        params_others = [param for layer in self.model[1:] for param in layer.parameters()]
        self.opt = optim.Adam([
                {'params': params_conv, 'weight_decay': 0.5},
                {'params': params_others}
        ], lr=self.LR)
    
    def get_model_metrics(self, xb, yb):
        # this method can be used to compute metrics on the weights, forward and backward pass of the model (see pytorch_metrics.py) 
        compute_weights_metrics(self.model)
        compute_forward_metrics(self.model, xb)
        compute_backward_metrics(self.model, self.loss_func, xb, yb)
        metrics = OrderedDict()
        for name, layer in self.model.named_children():
            if hasattr(layer, 'my_metrics'):
                metrics[name] = layer.my_metrics
        return metrics

    def train_model(self, dataset):
        print("training model")
        dataloader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        for i in range(self.EPOCHS):
            self.model.train()
            for xb, yb in dataloader:
                self.opt.zero_grad()
                y_pred = self.model(xb)
                loss = self.loss_func(y_pred, yb)
                loss.backward()

                self.opt.step()
            torch.set_printoptions(precision=10)

            self.model.eval()
            with torch.no_grad():
                valid_loss = sum(self.loss_func(self.model(xb), yb) for xb, yb in dataloader)
            
            print("E :", i, "loss :", valid_loss.item() / len(dataloader.dataset))

    def save_model(self, path, prefix):
        # save model with name containing hyperparams
        self.model.eval()
        model_name = '{}_B{}_EP{}_LR{}_R{}.pt'.format(prefix, self.BATCH_SIZE, self.EPOCHS, self.LR, self.R)
        torch.save(self.model, path + "/" + model_name)

    def compute_predictions(self, data):
        # compute class predictions
        self.model.eval()
        data = torch.Tensor(data)
        with torch.no_grad():
            preds = nn.Sigmoid().forward(self.model(data).squeeze()).numpy()
        assert np.max(preds) <= 1.0
        assert np.min(preds) >= 0.0
        return preds

    def compute_tiles_scores(self, data):
        # compute tile tumor likelihood score
        self.model.eval()
        data = torch.Tensor(data)
        with torch.no_grad():
            scores = self.model[0].forward(data)
        return scores.numpy()


class ChowderEnsembler:
    """ Encampsulate an ensemble of n_models of class PytorchChowder"""

    def __init__(self, n_models = 10, r = 5, bs = 10, ep = 30, lr = 0.001, mt = 1000, drp = 0.5):
        self.BATCH_SIZE = bs
        self.model_pool = []
        for i in range(n_models):
            self.model_pool.append(PytorchChowder(r, bs, ep, lr, mt))
    
    # this class mainly iterate on the PytorchChowder class instances to execute the operations

    def train_models(self, dataset):
        features = torch.Tensor(dataset.features)
        labels = torch.Tensor(dataset.labels).view(-1, 1, 1)
        assert len(features) == len(labels)
        
        t_dataset = TensorDataset(features, labels)
        
        for model in self.model_pool:            
            model.train_model(t_dataset)    

    def compute_predictions(self, data):
        preds = []
        for model in self.model_pool:
            pred = model.compute_predictions(data)
            preds.append(pred)

        results = np.sum(preds, axis = 0) / len(self.model_pool)
        return results

    def compute_tiles_scores(self, data):
        scores = []
        for model in self.model_pool:
            scr = model.compute_tiles_scores(data)
            scores.append(scr)

        results = np.sum(scores, axis = 0) / len(self.model_pool)
        return results
    
    def save_ensemble(self, save_dir, ens_prefix):
        for i, model in enumerate(self.model_pool):
            mod_prefix = "{}_M{}".format(ens_prefix, i)
            model.save_model(save_dir, mod_prefix)

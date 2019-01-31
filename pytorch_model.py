import numpy as np

import torch 
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from pytorch_metrics import compute_weights_metrics, compute_forward_metrics, compute_backward_metrics

from collections import OrderedDict


def prepare_dataset(ft_train_np, ft_test_np, lb_train_np, lb_test_np, bs):
    features_train = torch.Tensor(ft_train_np)
    features_test = torch.Tensor(ft_test_np)
    labels_train = torch.Tensor(lb_train_np).view(-1, 1, 1)
    assert len(features_train) == len(labels_train)
    labels_test = torch.Tensor(lb_test_np).view(-1, 1, 1)
    assert len(features_test) == len(labels_test)

    train_dataset = TensorDataset(features_train, labels_train)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True) 
    valid_dataset = TensorDataset(features_test, labels_test)
    valid_dataloader = DataLoader(valid_dataset, batch_size=bs * 2, shuffle=True)

    return train_dataloader, valid_dataloader


class MinMax(nn.Module):

    def __init__(self, r):
        super(MinMax, self).__init__()
        self.R = r
    
    def forward(self, x):
        top, _ = torch.topk(x, self.R, sorted=True)
        bottom, _ = torch.topk(x, self.R, largest=False, sorted=True)
        res = torch.cat((top, bottom), dim=2)
        return res


class PytorchChowder:

    def __init__(self, r = 5, bs = 10, ep = 30, lr = 0.001, mt = 1000):
        self.R = r
        self.BATCH_SIZE = bs
        self.EPOCHS = ep
        self.LR = lr
        self.MAX_TILES_NBR = mt
        print(self.R, self.BATCH_SIZE, self.EPOCHS, self.LR, self.MAX_TILES_NBR)

        self.model = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(2048, 1, 1,bias=False)),
            #('drop1', nn.Dropout(p=0.5)),
            ('minmax', MinMax(self.R)),
            ('lin1', nn.Linear(2 * self.R, 20,bias=False)),
            #('lrelu1', nn.LeakyReLU(0.1)),
            ('sig1', nn.Sigmoid()),
            #('drop2', nn.Dropout(p=0.5)),
            ('lin2', nn.Linear(20, 10,bias=False)),
            #('lrelu2', nn.LeakyReLU(0.1)),
            ('sig2', nn.Sigmoid()),
            #('drop3', nn.Dropout(p=0.5)),
            ('lin3', nn.Linear(10, 1,bias=False))
            #nn.Sigmoid()
        ])).float()

        #loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_ratio)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.init_parameters()
        self.init_optimizer()
        
    def init_parameters(self):
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
        self.model.apply(init_layer)

    def init_optimizer(self):
        params_conv = list(self.model[0].parameters())
        params_others = [param for layer in self.model[1:] for param in layer.parameters()]
        # self.opt = optim.SGD(self.model.parameters(), lr=self.LR)
        self.opt = optim.Adam([
                {'params': params_conv, 'weight_decay': 0.5},
                {'params': params_others}
        ], lr=self.LR)
    
    def get_model_metrics(self, xb, yb):
        compute_weights_metrics(self.model)
        compute_forward_metrics(self.model, xb)
        compute_backward_metrics(self.model, self.loss_func, xb, yb)
        metrics = OrderedDict()
        for name, layer in self.model.named_children():
            if hasattr(layer, 'my_metrics'):
                metrics[name] = layer.my_metrics
        return metrics

    def load_dataset(self, ft_train_np, ft_test_np, lb_train_np, lb_test_np):

        unique, counts = np.unique(lb_train_np, return_counts=True)
        print(unique, counts, counts[1] / (counts[0] + counts[1]))
        pos_ratio = torch.Tensor((counts[0] / counts[1],))
        print('pos_ratio: ' + str(pos_ratio))
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_ratio)
        
        trn_dl, vld_dl = prepare_dataset(ft_train_np, ft_test_np, lb_train_np, lb_test_np, self.BATCH_SIZE)

        self.train_dataloader = trn_dl
        self.valid_dataloader = vld_dl
    
    def set_dataloaders(self, trn_dl, vld_dl):
        self.train_dataloader = trn_dl
        self.valid_dataloader = vld_dl

    def train_model(self):
        print("training model")
        for i in range(self.EPOCHS):
            self.model.train()
            for xb, yb in self.train_dataloader:
                self.opt.zero_grad()
                y_pred = self.model(xb)
                loss = self.loss_func(y_pred, yb)
                loss.backward()

                self.opt.step()
            compute_forward_metrics(self.model, xb)
            torch.set_printoptions(precision=10)

            #print(self.model[0].my_metrics["forward"]["input"])
            # print(self.model[0].my_metrics["forward"]["output"])
            # print(self.model[1].my_metrics["forward"]["output"])
            #exit()
            self.model.eval()
            with torch.no_grad():
                valid_loss = sum(self.loss_func(self.model(xb), yb) for xb, yb in self.valid_dataloader)
            
            print("E :", i, "loss :", valid_loss.item() / len(self.valid_dataloader.dataset))

    def save_model(self, path):    
        torch.save(self.model, path)

    def compute_predictions(self, data):
        self.model.eval()
        data = torch.Tensor(data)
        with torch.no_grad():
            preds = nn.Sigmoid().forward(self.model(data).squeeze()).numpy()
        assert np.max(preds) <= 1.0
        assert np.min(preds) >= 0.0
        # preds[preds > 0.5] = 1
        # preds[preds <= 0.5] = 0
        return preds
        # return preds.astype(int)

    
class ChowderEnsembler:

    def __init__(self, n_models = 10, r = 5, bs = 10, ep = 30, lr = 0.001, mt = 1000):
        self.BATCH_SIZE = bs
        self.ensemble = []
        for i in range(n_models):
            self.ensemble.append(PytorchChowder(r, bs, ep, lr, mt))
    
    def load_dataset(self, ft_train_np, ft_test_np, lb_train_np, lb_test_np):
        # trn_dl, vld_dl = prepare_dataset(ft_train_np, ft_test_np, lb_train_np, lb_test_np, self.BATCH_SIZE)
        # for model in self.ensemble:
        #     model.set_dataloaders(trn_dl, vld_dl)
        features_train = torch.Tensor(ft_train_np)
        features_test = torch.Tensor(ft_test_np)
        labels_train = torch.Tensor(lb_train_np).view(-1, 1, 1)
        assert len(features_train) == len(labels_train)
        labels_test = torch.Tensor(lb_test_np).view(-1, 1, 1)
        assert len(features_test) == len(labels_test)

        train_dataset = TensorDataset(features_train, labels_train)
        valid_dataset = TensorDataset(features_test, labels_test)
        
        for model in self.ensemble:
            train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True) 
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.BATCH_SIZE * 2, shuffle=True)
            model.set_dataloaders(train_dataloader, valid_dataloader)


    def train_models(self):
        for model in self.ensemble:
            model.train_model()    

    def compute_predictions(self, data):
        preds = []
        print(data.shape)
        for model in self.ensemble:
            pred = model.compute_predictions(data)
            print(pred.shape)
            print(pred)
            preds.append(pred)

        results = np.sum(preds, axis = 0) / len(self.ensemble)
        print(results.shape)
        return results
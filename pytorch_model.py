import numpy as np

import torch 
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

torch.set_printoptions(precision=10)


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

        self.model = nn.Sequential(
            nn.Conv1d(2048, 1, 1,bias=False),
            nn.Dropout(p=0.5),
            MinMax(self.R),
            nn.Linear(2 * self.R, 200,bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
            nn.Linear(200, 100,bias=False),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
            nn.Linear(100, 1,bias=False)
            #nn.Sigmoid()
        ).float()

        #loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_ratio)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.init_optimizer()
    
    def init_parameters(self):
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
        self.model.apply(init_layer)

    def init_optimizer(self):
        params_conv = list(self.model[0].parameters())
        params_others = [param for layer in self.model[1:] for param in layer.parameters()]
        self.opt = optim.Adam([
                {'params': params_conv, 'weight_decay': 0.5},
                {'params': params_others}
        ], lr=self.LR)
        
    def prepare_dataset(self, features_train, features_test, labels_train, labels_test):
        self.train_dataset = TensorDataset(features_train, labels_train)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True) 
        self.valid_dataset = TensorDataset(features_test, labels_test)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.BATCH_SIZE * 2, shuffle=True)

    
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
                        
            self.model.eval()
            with torch.no_grad():
                valid_loss = sum(self.loss_func(self.model(xb), yb) for xb, yb in self.valid_dataloader)
            
            print("E : ", i, "loss : ", valid_loss / len(self.valid_dataset))

    def save_model(self, path):    
        torch.save(self.model, path)

    def compute_predictions(self, data):
        self.model.eval()
        with torch.no_grad():
            preds = nn.Sigmoid().forward(self.model(data).squeeze()).numpy()
        preds[preds > 0.5] = 1
        preds[preds <= 0.5] = 0
        assert np.max(preds) <= 1.0
        assert np.min(preds) >= 0.0
        return preds

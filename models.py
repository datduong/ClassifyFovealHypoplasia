import torch
import torch.nn as nn
import geffnet
# from resnest.torch import resnest101
# from pretrainedmodels import se_resnext101_32x4d


class Effnet_Custom(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False, args=None):
        super(Effnet_Custom, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained) # ! make EfficientNet
        # self.dropout_rate=0.5
        # if args is not None: 
        self.dropout_rate=args.dropout
        # !
        self.dropouts = nn.ModuleList([
            nn.Dropout(self.dropout_rate) for _ in range(5)
        ])
        print ('output of vec embed from img network {}'.format(self.enet.classifier.in_features))
        self.in_ch = self.enet.classifier.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]), # ! takes array of features [[1,2,3...], [], ...]
                nn.BatchNorm1d(n_meta_dim[0]),
                nn.SiLU(),
                nn.Dropout(p=self.dropout_rate), # default was 0.3 
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                nn.SiLU(),
            )
            self.in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(self.in_ch, out_dim) # ! simple classifier
        self.enet.classifier = nn.Identity() # ! pass through, no update

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1) ## flatten ?
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts) # ! takes average output after doing many dropout
        return out


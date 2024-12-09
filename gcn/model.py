import importlib

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import GraphConv
import pdb


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std
class FCEncoder(torch.nn.Module):
    def __init__(self, in_channels, feature_channels, **kwargs):
        super(FCEncoder, self).__init__()
        self.fc11 = torch.nn.Linear(int(in_channels*(in_channels-1)/2), 2*feature_channels)
        self.bn1 = torch.nn.BatchNorm1d(2*feature_channels)
        self.fc12 = torch.nn.Linear(2*feature_channels, feature_channels)
        self.bn2 = torch.nn.BatchNorm1d(feature_channels)

        self.fc21 = torch.nn.Linear(feature_channels, feature_channels)
        self.fc22 = torch.nn.Linear(feature_channels, feature_channels)

    def forward(self, data):
        x = data.x.view(data.num_graphs, -1, data.x.size(-1))
        triu_idx = torch.triu_indices(x.size(1), x.size(2), 1)
        x = x[:, triu_idx[0], triu_idx[1]].view(data.num_graphs, -1)

        h1 = F.dropout(F.leaky_relu(self.fc11(x)), p=0.5, training=self.training)
        h1 = self.bn1(h1)
        h1 = F.dropout(F.leaky_relu(self.fc12(h1)), p=0.5, training=self.training)
        h1 = self.bn2(h1)

        h_mean = self.fc21(h1)
        h_logvar = self.fc22(h1)
        return h_mean, h_logvar

class Classifier(torch.nn.Module):
    def __init__(self, out_channels, feature_channels, num_nodes, **kwargs):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(feature_channels, out_channels)
        self.num_nodes = num_nodes
    
    def forward(self, x):
        out = self.fc1(x.view(x.size(0), -1))
        return out

class BRIDGEDecoder(torch.nn.Module):
    def __init__(self, feature_channels, num_nodes, **kwargs):
        super(BRIDGEDecoder, self).__init__()

        self.fc11 = torch.nn.Linear(feature_channels, 1)

        self.fc21 = torch.nn.Linear(feature_channels, num_nodes)
        self.conv21 = GraphConv(1, 8, aggr='mean')
        self.conv22 = GraphConv(8, 16, aggr='mean')
        
        self.num_nodes = num_nodes
        
    def forward(self, x, S, batch=None):
        edge_index, edge_attr = S
        param = F.relu(self.fc11(x))
        h1 = F.leaky_relu(self.fc21(x)).view(x.size(0) * self.num_nodes, -1)
        h1 = F.leaky_relu(self.conv21(h1, edge_index, edge_attr))
        out = F.leaky_relu(self.conv22(h1, edge_index, edge_attr)).view(x.size(0), self.num_nodes, -1)

        epsilon = 1e-10
        k = torch.matmul(out, out.permute(0, 2, 1))
        S = to_dense_adj(S[0], batch, S[1]) + epsilon
        mask = S!=0
        conn_est = F.sigmoid(k + S.log() * param.unsqueeze(-1)) * mask.float()
        return conn_est, param

class GCNDecoder(torch.nn.Module):
    def __init__(self, in_channels, feature_channels, num_nodes, **kwargs):
        super(GCNDecoder, self).__init__()

        self.fc1 = torch.nn.Linear(feature_channels, num_nodes)
        self.conv21 = GraphConv(1, 8, aggr='mean')
        self.conv22 = GraphConv(8, in_channels, aggr='mean')
        
        self.num_nodes = num_nodes
        
    def forward(self, x, S):
        edge_index, edge_attr = S
        h1 = F.dropout(F.leaky_relu(self.fc1(x)).view(x.size(0) * self.num_nodes, -1), p=0.5, training=self.training)
        h1 = F.leaky_relu(self.conv21(h1, edge_index, edge_attr))
        h1 = self.conv22(h1, edge_index, edge_attr).view(x.size(0), self.num_nodes, -1)
        conn_est = F.sigmoid(0.5 * (h1 + torch.transpose(h1, 1, 2)))
        return conn_est, None
class FCDecoder(torch.nn.Module):
    def __init__(self, feature_channels, num_nodes, **kwargs):
        super(FCDecoder, self).__init__()
        self.fc1 = torch.nn.Linear(feature_channels, feature_channels*2)
        self.bn1 = torch.nn.BatchNorm1d(feature_channels*2)
        self.fc2 = torch.nn.Linear(feature_channels*2, feature_channels*4)
        self.bn2 = torch.nn.BatchNorm1d(feature_channels*4)
        self.fc3 = torch.nn.Linear(feature_channels*4, int(num_nodes*(num_nodes-1)/2))
        self.num_nodes = num_nodes

    def forward(self, x):
        h1 = F.dropout(F.leaky_relu(self.fc1(x)), p=0.5, training=self.training)
        h1 = self.bn1(h1)
        h1 = F.dropout(F.leaky_relu(self.fc2(h1)), p=0.5, training=self.training)
        h1 = self.bn2(h1)

        triu_idx = torch.triu_indices(self.num_nodes, self.num_nodes, 1)
        conn_est = torch.zeros(x.size(0), self.num_nodes, self.num_nodes).to(x.device)
        conn_est[:, triu_idx[0], triu_idx[1]] = self.fc3(h1)
        conn_est = F.sigmoid(0.5 * (conn_est + torch.transpose(conn_est, 1, 2)))

        return conn_est, None

class BRIDGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, feature_channels, num_nodes, **kwargs):
        super(BRIDGE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_channels = feature_channels
        self.num_nodes = num_nodes
        self.encoder = FCEncoder(in_channels, feature_channels)
        self.decoder = BRIDGEDecoder(feature_channels, num_nodes)
        self.classifier = Classifier(out_channels, feature_channels, num_nodes)

    def GeneratorVAE(self, S, h_mean, h_logvar=None):
        S = [S[0][:, :S[2]], S[1][:S[2]]]
        if h_logvar is not None:
            h = reparameterize(h_mean, h_logvar)
        else:
            h = h_mean
        conn_est, param = self.decoder(h, S)
        y = self.classifier(h_mean)
        return conn_est, y, param, h

    def ClassifierVAE(self, data, S):
        S = [S[0][:, :S[2]*data.num_graphs], S[1][:S[2]*data.num_graphs]]
        h_mean, _ = self.encoder(data)
        y = self.classifier(h_mean)
        return y

    def Embedding(self, data):
        h_mean, _ = self.encoder(data)
        h_nodal = self.decoder.fc21(h_mean)
        return h_mean, h_nodal

    def forward(self, data, S):
        S = [S[0][:, :S[2]*data.num_graphs], S[1][:S[2]*data.num_graphs]]
        h_mean, h_logvar = self.encoder(data)
        h = reparameterize(h_mean, h_logvar)
        if self.training:
            y = self.classifier(h)
        else:
            y = self.classifier(h_mean)
        conn_est, param = self.decoder(h, S, data.batch)
        return conn_est, y, h_mean, h_logvar, param, h


class GCNVAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, feature_channels, num_nodes, **kwargs):
        super(GCNVAE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_channels = feature_channels
        self.num_nodes = num_nodes
        self.encoder = FCEncoder(in_channels, feature_channels)
        self.decoder = GCNDecoder(in_channels, feature_channels, num_nodes)
        self.classifier = Classifier(out_channels, feature_channels, num_nodes)

    def ClassifierVAE(self, data, S):
        S = [S[0][:, :S[2]*data.num_graphs], S[1][:S[2]*data.num_graphs]]
        h_mean, _ = self.encoder(data)
        y = self.classifier(h_mean)
        return y

    def GeneratorVAE(self, S, h_mean, h_logvar=None):
        S = [S[0][:, :S[2]], S[1][:S[2]]]
        if h_logvar is not None:
            h = reparameterize(h_mean, h_logvar)
        else:
            h = h_mean
        conn_est, param = self.decoder(h, S)
        y = self.classifier(h_mean)
        return conn_est, y, param, h

    def forward(self, data, S):
        S = [S[0][:, :S[2]*data.num_graphs], S[1][:S[2]*data.num_graphs]]
        h_mean, h_logvar = self.encoder(data)
        h = reparameterize(h_mean, h_logvar)
        if self.training:
            y = self.classifier(h)
        else:
            y = self.classifier(h_mean)
        conn_est, param = self.decoder(h, S)
        return conn_est, y, h_mean, h_logvar, param, h

class FCVAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, feature_channels, num_nodes, **kwargs):
        super(FCVAE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_channels = feature_channels
        self.num_nodes = num_nodes
        self.encoder = FCEncoder(in_channels, feature_channels)
        self.decoder = FCDecoder(feature_channels, num_nodes)
        self.classifier = Classifier(out_channels, feature_channels, num_nodes)

    def ClassifierVAE(self, data, S):
        S = [S[0][:, :S[2]*data.num_graphs], S[1][:S[2]*data.num_graphs]]
        h_mean, _ = self.encoder(data)
        y = self.classifier(h_mean)
        return y

    def GeneratorVAE(self, S, h_mean, h_logvar=None):
        if h_logvar is not None:
            h = reparameterize(h_mean, h_logvar)
        else:
            h = h_mean
        conn_est, param = self.decoder(h)
        y = self.classifier(h_mean)
        return conn_est, y, param, h

    def forward(self, data, S):
        h_mean, h_logvar = self.encoder(data)
        h = reparameterize(h_mean, h_logvar)
        if self.training:
            y = self.classifier(h)
        else:
            y = self.classifier(h_mean)
        conn_est, param = self.decoder(h)
        return conn_est, y, h_mean, h_logvar, param, h

class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, feature_channels, num_nodes, **kwargs):
        super(MLP, self).__init__()
        self.fc11 = torch.nn.Linear(int(in_channels*(in_channels-1)/2), 2*feature_channels)
        self.bn1 = torch.nn.BatchNorm1d(2*feature_channels)
        self.fc12 = torch.nn.Linear(2*feature_channels, feature_channels)
        self.bn2 = torch.nn.BatchNorm1d(feature_channels)

        self.classifier = Classifier(out_channels, feature_channels, num_nodes)

    def forward(self, data):
        x = data.x.view(data.num_graphs, -1, data.x.size(-1))
        triu_idx = torch.triu_indices(x.size(1), x.size(2), 1)
        x = x[:, triu_idx[0], triu_idx[1]].view(data.num_graphs, -1)

        h1 = F.dropout(F.leaky_relu(self.fc11(x)), p=0.5, training=self.training)
        h1 = self.bn1(h1)
        h1 = F.dropout(F.leaky_relu(self.fc12(h1)), p=0.5, training=self.training)
        h1 = self.bn2(h1)

        y = self.classifier(h1)
        
        return y
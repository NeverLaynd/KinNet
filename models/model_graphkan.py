import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from models.efficient_kan import KAN
from models.graphkan import kanChebConv
from models import gps_layer

class Drug3DModel(nn.Module):
    def __init__(self,
                 d_vocab=21,hidden_dim=None,pos_enc_dim=None,in_feat_dropout=None,
        d_emb=66,  d_edge=16,
        d_gcn=[166,166,166],
        k_n= [1,2,3]
    ):
        super(Drug3DModel, self).__init__()
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.linear_h = nn.Linear(d_vocab, hidden_dim)
        self.linear_e = nn.Linear(d_edge, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        d_gcn_in = d_gcn[0]
        gcn_layer_sizes = [d_gcn_in] + d_gcn
        self.layers = nn.ModuleList()
        for i in range(len(gcn_layer_sizes) - 1):               
            self.layers.append(gps_layer.GPSLayer(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in, K=k_n[i]))      
        
        self.pool = torch_geometric.nn.global_mean_pool
        self.other=1.0


    def forward(self, data):
        h = data.x
        edge_index = data.edge_index
        batch = data.batch
        e = data.edge_attr

        h = self.linear_h(h)
        x_h = self.in_feat_dropout(h)
        data.x = x_h
        data.edge_attr = self.linear_e(e)

        for mod in self.layers:
            data = mod(data)

        data.x = x_h
        x = self.pool(data.x, data.batch)  # [batch_size, 166]
        return x

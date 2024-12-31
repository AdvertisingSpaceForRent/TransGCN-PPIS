import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class Normalize(nn.Module): 
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class TransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0.2):
        super(TransformerLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()
        self.attn_norm = nn.LayerNorm(num_hidden, eps=1e-6)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(3)])
        self.ffn_norm = nn.LayerNorm(num_hidden, eps=1e-6)

        self.attention = NeighborAttention(num_hidden, num_in, num_heads)
        self.dense1 = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        self.dense2 = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        self.dense3 = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        self.weight1 = Parameter(torch.FloatTensor(128 * 2, 128))
        self.weight2 = Parameter(torch.FloatTensor(128 * 2, 128))
        self.gcn1 = GCN(128)
        self.gcn2 = GCN(128)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(128)
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)

    def forward(self, h_V, h_E,adj,h0,l,mask_V=None, mask_attend=None):    # mask_attend [B, L, K]
        # Concatenate h_V_i to h_E_ij      
        """ Parallel computation of full transformer layer """
        # Self-attention
        # h_V = self.attn_norm(h_V)
        # h_V = self.relu(h_V)
        first_dimension = h_V.shape[0]
        theta = min(1, math.log(1.5 / l + 1))

        dh = self.attention(h_V, h_E, mask_attend)
        h_V = self.norm[0](h_V + self.dropout(dh))
        h_V = self.relu(h_V)

        # # theta = min(1, math.log(1.5 / l + 1))
        dh = self.gcn1(h_V, adj, mask_V)
        support = torch.cat([dh, h0], -1)
        # support = support.view(-1, 64)
        r = (1 - 0.7) * dh + 0.7 * h0
        weight_expanded = self.weight1.unsqueeze(0).expand(first_dimension, 256, 128)
        output = theta * torch.bmm(support, weight_expanded) + (1 - theta) * r
        h_V = self.norm[1](h_V + self.dropout(self.relu(output)))


        dh = self.dense1(h_V)
        h_V = self.norm[2](h_V + self.dropout(dh))


        if mask_V is not None: 
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.relu(self.W_in(h_V))
        h = self.W_out(h)
        return h


class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).cuda())
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend * attend
        return attend

    def forward(self, h_V, h_E, mask_attend=None):
        """ Self-attention, graph-structured O(Nk)
        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, K, N_hidden] 
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]           mask_attend [B, L, K] 
        Returns:
            h_V:            Node update
        """

        # Queries, Keys, Values
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.num_heads

        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_batch, n_nodes, 1, n_heads, 1, d])
        K = self.W_K(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d, 1])
        V = self.W_V(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K).view([n_batch, n_nodes, n_neighbors, n_heads]).transpose(-2,-1)
        attend_logits = attend_logits / np.sqrt(d)
        
        if mask_attend is not None:
            # Masked softmax
            mask = mask_attend.unsqueeze(2).expand(-1,-1,n_heads,-1)
            attend = self._masked_softmax(attend_logits, mask) # [B, L, heads, K]
        else:
            attend = F.softmax(attend_logits, -1)

        # Attentive reduction
        h_V_update = torch.matmul(attend.unsqueeze(-2), V.transpose(2,3)) # [B, L, heads, 1, K] × [B, L, heads, K, d]
        h_V_update = h_V_update.view([n_batch, n_nodes, self.num_hidden])
        h_V_update = self.W_O(h_V_update)
        return h_V_update
class GCN(nn.Module):
    def __init__(self, hidden_size):
        super(GCN, self).__init__()
        self.hidden_size = hidden_size
        self.weight = Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, mask):
        # mask shape: (B, N), where 1 indicates valid nodes and 0 indicates padded nodes
        B, N, D = x.size()

        # Expand weight for batch multiplication
        expanded_weight = self.weight.unsqueeze(0).expand(B, -1, -1)

        # Apply mask to x and adj
        x = x * mask.unsqueeze(-1)
        adj = adj * mask.unsqueeze(-1) * mask.unsqueeze(1)

        # Perform graph convolution
        y = torch.bmm(adj, x)
        y = torch.bmm(y, expanded_weight)

        # Apply mask to output
        y = y * mask.unsqueeze(-1)

        return y
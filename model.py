from self_attention import *
from edge_features import EdgeFeatures

class MVGNN(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, num_encoder_layers=4, k_neighbors=30, augment_eps=0., dropout=0.2):
        super(MVGNN, self).__init__()

        # Hyperparameters
        self.augment_eps = augment_eps

        # Edge featurization layers augment_eps
        self.EdgeFeatures = EdgeFeatures(edge_features, top_k=k_neighbors, augment_eps=augment_eps)
        self.dropout = nn.Dropout(dropout)

        # Embedding layers
        self.act_fn = nn.ReLU()
        self.lin1 = nn.Linear(node_features, 512, bias=True)
        self.lin2 = nn.Linear(512, 256, bias=False)
        self.W_v = nn.Linear(256, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        # self.dropout = dropout
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.W_out1 = nn.Linear(hidden_dim, 64, bias=True)
        self.W_out2 = nn.Linear(64, 1, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, X, V, mask,adj):
        # Prepare node and edge embeddings
        # X is the alpha-C coordinate matrix; V is the pre-computed and normalized features ProtTrans+DSSP
        E, E_idx = self.EdgeFeatures(X, mask) # X [B, L, 3] mask [B, L] => E [B, L, K, d_edge]; E_idx [B, L, K]

        # Data augmentation
        if self.training and self.augment_eps > 0:
            V = V + 0.1 * self.augment_eps * torch.randn_like(V)

        V = self.act_fn(self.lin1(V))
        V = self.act_fn(self.lin2(V))


        h_V = self.W_v(V)
        ho = h_V
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend # mask_attend [B, L, K]
        for i,layer in enumerate(self.encoder_layers):
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV,adj,ho,i+1, mask_V=mask, mask_attend=mask_attend)
        logits = self.act_fn(self.W_out1(h_V))
        logits = self.W_out2(logits).squeeze(-1)
        # [B, L]
        return logits

import torch
from torch_geometric.nn import MessagePassing

class GRUConv(MessagePassing):
    def __init__(self, emb_dim, aggr):
        super(GRUConv, self).__init__(aggr=aggr)
        self.rnn = torch.nn.GRUCell(emb_dim, emb_dim)

    def forward(self, x, edge_index):
        out = self.rnn(self.propagate(edge_index, x=x), x)
        return out

class GRUMLPConv(MessagePassing):
    def __init__(self, emb_dim, mlp_edge, aggr):
        super(GRUMLPConv, self).__init__(aggr=aggr)
        self.rnn = torch.nn.GRUCell(emb_dim, emb_dim)
        self.mlp_edge = mlp_edge

    def forward(self, x, edge_index):
        out = self.rnn(self.propagate(edge_index, x=x), x)
        return out

    def message(self, x_j, x_i):
        concatted = torch.cat((x_j, x_i), dim=1)
        return self.mlp_edge(concatted)
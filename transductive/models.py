import torch
import torch.nn as nn
from torch_scatter import scatter

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)
        # nn.Embedding用于创建一个嵌入矩阵，其维度为(2*n_rel + 1, in_dim)
        # 2*n_rel + 1表示关系的总数（考虑到正反关系），in_dim是每个关系的嵌入向量的维度。

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        sub = edges[:,4]
        rel = edges[:,2]
        obj = edges[:,5]

        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:,0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))  # attention
        # alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr)))) #ablation study remove h_qr
        # 替换为 softmax
        # alpha = torch.softmax(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))), dim=1)

        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new

class MRD_GNN(torch.nn.Module):
    def __init__(self, params, loader, node_embeddings=None):  # node_embeddings: 预训练的节点嵌入
        super(MRD_GNN, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader

        # 如果传入了预训练嵌入，则使用它进行初始化
        if node_embeddings is not None:
            self.node_embeddings = torch.FloatTensor(node_embeddings).cuda()
        else:
            self.node_embeddings = None  # 如果没有传入预训练嵌入

        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
       
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)         # get score
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, subs, rels, mode='train'):
        n = len(subs)
        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()

        # 使用预训练嵌入初始化节点表示
        if self.node_embeddings is not None:
            hidden = self.node_embeddings[q_sub].cuda()
        else:
            hidden = torch.zeros(n, self.hidden_dim).cuda()  # 如果没有预训练嵌入，则初始化为零

        h0 = torch.zeros((1, n,self.hidden_dim)).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        # hidden = torch.zeros(n, self.hidden_dim).cuda()


        scores_all = []
        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)
            hidden = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate (hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()         # non_visited entities have 0 scores
        scores_all[[nodes[:,0], nodes[:,1]]] = scores
        return scores_all




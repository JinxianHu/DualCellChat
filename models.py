from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score


import torch
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from layers import InnerProductDecoder, DirectedInnerProductDecoder
from initializations import reset


EPS        = 1e-15
MAX_LOGSTD = 10


class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.
    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

    

class DirectedGAE(torch.nn.Module):
    def __init__(self, encoder, decoder=None):
        super(DirectedGAE, self).__init__()
        self.encoder = encoder
        self.decoder = DirectedInnerProductDecoder() if decoder is None else decoder
        DirectedGAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    
    def forward(self, data):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        ##### change by huo 
        x, edge_index, edge_weight = data.x, data.train_pos_edge_index, data.edge_weight
        s, t = self.encoder(x, x, edge_index)
        adj_pred = self.decoder.forward_all(s, t)
        return adj_pred

    
    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    
    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    
    def recon_loss(self, s, t, pos_edge_index, neg_edge_index=None):
        # print(self.decoder(s, t, pos_edge_index, sigmoid=True)) # [7155] [0.1920, 0.4232, 0.0068,  ..., 0.1476, 0.1290, 0.4393]
        # print(EPS) #1e-15
        # print(self.decoder(s, t, pos_edge_index, sigmoid=True) + EPS) # [7155] [0.1920, 0.4232, 0.0068,  ..., 0.1476, 0.1290, 0.4393]
        # print(-torch.log(self.decoder(s, t, pos_edge_index, sigmoid=True) + EPS)) # [7155] [1.6504, 0.8599, 4.9894,  ..., 1.9135, 2.0483, 0.8225]

        pos_loss = -torch.log(
            self.decoder(s, t, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, s.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(s, t, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, s, t, pos_edge_index, neg_edge_index):
        # XXX
        pos_y = s.new_ones(pos_edge_index.size(1))
        neg_y = s.new_zeros(neg_edge_index.size(1))
        # print('pos_edge_index: ',pos_edge_index.size())
        # print('neg_edge_index: ',neg_edge_index.size())
        # print('pos_y: ',pos_y)
        # print('neg_y: ',neg_y)
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(s, t, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(s, t, neg_edge_index, sigmoid=True)
        # print('pos_pred: ',pos_pred)
        # print('neg_pred: ',neg_pred)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        # adj_pred = self.decoder.forward_all(s, t) 
        # adj_pred = self.decoder.forward_all(s, t, sigmoid=False)
        adj_pred = self.decoder.forward_all(s, t, sigmoid=True)

        # return roc_auc_score(y, pred), average_precision_score(y, pred), adj_pred, s, t
        ################### modified by jinxian: Calculate metrics
        auc = roc_auc_score(y, pred)
        auprc = average_precision_score(y, pred)  # AUPRC is the same as AP
        # Calculate F1 score with threshold 0.5
        y_pred_binary = (pred >= 0.5).astype(int)
        f1 = f1_score(y, y_pred_binary)
        acc = accuracy_score(y, y_pred_binary)

        return auc, auprc, acc, f1, adj_pred, s, t

    def test_posotive(self, s, t, pos_edge_index):
        # XXX
        pos_y = s.new_ones(pos_edge_index.size(1))
        # neg_y = s.new_zeros(neg_edge_index.size(1))

        # y = torch.cat([pos_y, neg_y], dim=0)
        y = pos_y

        pos_pred = self.decoder(s, t, pos_edge_index, sigmoid=True)
        # neg_pred = self.decoder(s, t, neg_edge_index, sigmoid=True)
        # pred = torch.cat([pos_pred, neg_pred], dim=0)
        pred = pos_pred


        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        ### add by huo
        # adj_pred = self.decoder.forward_all(s, t) 
        # adj_pred = self.decoder.forward_all(s, t, sigmoid=False)
        adj_pred = self.decoder.forward_all(s, t, sigmoid=True)

        return roc_auc_score(y, pred), average_precision_score(y, pred), adj_pred, s, t


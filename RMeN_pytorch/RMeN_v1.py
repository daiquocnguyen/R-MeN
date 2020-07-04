import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from Model import Model
from numpy.random import RandomState
from relational_rnn_general import *

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

use_gpu = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    use_gpu = True

class RMeN(Model):

    def __init__(self, config):
        super(RMeN, self).__init__(config)

        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)  # vectorized quaternion
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)

        self.pos_h = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, self.config.hidden_size)))
        self.pos_r = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, self.config.hidden_size)))
        self.pos_t = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, self.config.hidden_size)))

        self.transformer_rel_rnn = RelationalMemory(
                        mem_slots=self.config.mem_slots, head_size=self.config.head_size,
                        num_heads=self.config.num_heads, input_size=self.config.hidden_size,
                        gate_style=self.config.gate_style, attention_mlp_layers=self.config.attention_mlp_layers,
                        return_all_outputs=True
                        ).to(device)
        self.model_memory = self.transformer_rel_rnn.initial_state(self.config.batch_seq_size).to(device)

        self.dropout = nn.Dropout(self.config.convkb_drop_prob)
        self.fc_layer = nn.Linear(self.transformer_rel_rnn.mem_size, 1)

        self.criterion = nn.Softplus()
        self.init_parameters()

    def init_parameters(self):
        if self.config.use_init_embeddings == False:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        else:
            self.ent_embeddings.weight.data = self.config.init_ent_embs
            self.rel_embeddings.weight.data = self.config.init_rel_embs

        nn.init.xavier_uniform_(self.fc_layer.weight.data)

    def _calc(self, h, r, t):
        if self.config.use_pos:
            h = h + self.pos_h
            r = r + self.pos_r
            t = t + self.pos_t

        h = h.unsqueeze(1) # bs x 1 x dim
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)
        hrt = torch.cat([h, r, t], 1)  # bs x 3 x dim

        # forward pass
        trans_rel_rnn_output, _ = self.transformer_rel_rnn(hrt, self.model_memory) # concatenate outputs (h, r, t) dim 0 --> (3xbs) x (head_size * num_head)
        h, r, t = torch.split(trans_rel_rnn_output, self.config.batch_seq_size, dim=0)

        input_fc = self.dropout(h * r * t) # bs * mem_size
        score = self.fc_layer(input_fc).view(-1)

        return -score

    def loss(self, score, regul):
        return torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        r = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        score = self._calc(h, r, t)

        # regularization
        l2_reg = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(2)

        return self.loss(score, l2_reg)

    def predict(self):
        output = []
        idx = np.arange(len(self.batch_h))
        # Transformer-based relational memory uses a memory of batch_size, not a dynamic batch size
        for i in range(0, len(self.batch_h), self.config.batch_seq_size):
            sampled_idx = idx[i:i+self.config.batch_seq_size]
            if len(sampled_idx) == 0:
                continue
            while len(sampled_idx) < self.config.batch_seq_size:  # for the last split
                sampled_idx = np.append(sampled_idx, sampled_idx[-1])

            h = self.ent_embeddings(self.batch_h[sampled_idx])
            r = self.rel_embeddings(self.batch_r[sampled_idx])
            t = self.ent_embeddings(self.batch_t[sampled_idx])
            score = self._calc(h, r, t)

            output.append(score)

        score = torch.cat(output)[:len(self.batch_h)]
        return score.cpu().data.numpy()

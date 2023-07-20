import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def gate_init(forget_gate):
    """
    Initialize the forget gaste bias to 1
    Args:
        forget_gate: forget gate bias term
    References: https://arxiv.org/abs/1602.02410
    """
    forget_gate.data.fill_(1)


class MinimalRNNCell(nn.Module):
    """A Minimal RNN cell .
    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`

    Inputs: input, hidden
        - input of shape `(batch, input_size)`: input features
        - hidden of shape `(batch, hidden_size)`: initial hidden state

    Outputs: h'
        - h' of shape `(batch, hidden_size)`: next hidden state
    """

    def __init__(self, input_size, hidden_size):
        super(MinimalRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_z = nn.Linear(input_size, hidden_size)
        self.weight_uh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_uz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.weight_uh.data.uniform_(-stdv, stdv)
        self.weight_uz.data.uniform_(-stdv, stdv)
        self.bias_hh.data.uniform_(stdv)

    def forward(self, input, ht_1):
        ut = torch.tanh(self.W_z(input))
        z = torch.addmm(self.bias_hh, ht_1, self.weight_uh)
        ft = torch.addmm(z, ut, self.weight_uz)
        ft = torch.sigmoid(ft)
        return ft * ht_1 + (1 - ft) * ut

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, use_residual=True):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.use_residual = use_residual
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        if self.use_residual:
            q += residual

        q = self.layer_norm(q)

        return q, attn

class RNNModel(torch.nn.Module):
    """
    Recurrent neural network (RNN) base class
    Missing values (i.e. NaN) are filled using model prediction
    """

    def __init__(self, celltype, nb_classes, nb_mri, nb_pet, h_mri, h_pet, h_drop, i_drop, nb_layers1,
                 nb_layers2, alpha, beta, n_head):
        super(RNNModel, self).__init__()
        self.h_ratio = 1. - h_drop
        self.i_ratio = 1. - i_drop

        # self.hid2category_mri = nn.Linear(h_mri, nb_classes)
        self.hid2measures_mri = nn.Linear(h_mri + h_pet, nb_mri)

        # self.hid2category_pet = nn.Linear(h_pet, nb_classes)
        self.hid2category = nn.Linear(h_pet + h_mri, nb_classes)
        self.hid2measures_pet1_1 = nn.Linear(h_mri, h_mri // 2)
        self.hid2measures_pet1_2 = nn.Linear(h_mri // 2, nb_pet)
        self.hid2measures_pet2 = nn.Linear(h_mri + h_pet, nb_pet)

        self.h_mri = h_mri
        self.h_pet = h_pet
        self.alpha = alpha
        self.beta = beta

        self.attention1 = MultiHeadAttention(n_head=n_head, d_model=h_mri,
                                             d_k=int((h_mri) / n_head), d_v= int((h_mri) / n_head))
        self.attention2 = MultiHeadAttention(n_head=n_head, d_model=h_mri * 2,
                                             d_k=int((h_mri) * 2 / n_head), d_v=int((h_mri) * 2 / n_head))

        self.trans_fc = nn.Linear(h_mri * 5 + h_pet * 5, 2)
        self.fc_ac = nn.Softmax()

        self.cells_mri = nn.ModuleList()
        self.cells_mri.append(celltype(nb_mri, h_mri))
        for _ in range(1, nb_layers1):
            self.cells_mri.append(celltype(h_mri, h_mri))

        self.cells_pet = nn.ModuleList()
        self.cells_pet.append(celltype(nb_pet, h_pet))
        if nb_layers1 > 1:
            for _ in range(1, nb_layers2):
                self.cells_pet.append(celltype(h_pet, h_pet))

    def init_hidden_state(self, batch_size):
        raise NotImplementedError

    def dropout_mask_mri(self, batch_size):
        dev = next(self.parameters()).device
        i_mask = torch.ones(
            batch_size, self.hid2measures_mri.out_features, device=dev)
        r_mask = [
            torch.ones(batch_size, cell.hidden_size, device=dev)
            for cell in self.cells_mri
        ]

        if self.training:
            i_mask.bernoulli_(self.i_ratio)
            for mask in r_mask:
                mask.bernoulli_(self.h_ratio)
        # i_mask = i_mask.bool()
        # for i, j in enumerate(r_mask):
        #     r_mask[i] = r_mask[i].bool()
        return i_mask, r_mask

    def dropout_mask_pet(self, batch_size):
        dev = next(self.parameters()).device
        i_mask = torch.ones(
            batch_size, self.hid2measures_pet1_2.out_features, device=dev)
        r_mask = [
            torch.ones(batch_size, cell.hidden_size, device=dev)
            for cell in self.cells_pet
        ]

        if self.training:
            i_mask.bernoulli_(self.i_ratio)
            for mask in r_mask:
                mask.bernoulli_(self.h_ratio)
        # i_mask = i_mask.bool()
        # for i, j in enumerate(r_mask):
        #     r_mask[i] = r_mask[i].bool()
        return i_mask, r_mask

    def forward(self, _mri_seq, _pet_seq):
        out_cat_seq, out_mri_seq = [], []
        pet_s_esti = []
        pet_f_esti = []

        mri_h_out = []
        pet_h_out = []
        attns1 = []

        hidden_mri = self.init_hidden_state(_mri_seq.shape[1])
        masks_mri = self.dropout_mask_mri(_mri_seq.shape[1])

        hidden_pet = self.init_hidden_state_pet(_pet_seq.shape[1])
        masks_pet = self.dropout_mask_pet(_pet_seq.shape[1])

        # cat_seq = _cat_seq.copy()
        mri_seq = _mri_seq.copy()
        pet_seq = _pet_seq.copy()

        for i, j in zip(range(len(mri_seq) + 1), range(1, len(mri_seq) + 1)):
            hidden_mri, mri_h_t = self.predict(mri_seq[i], hidden_mri, masks_mri)
            pet_estimate1 = F.tanh(self.hid2measures_pet1_1(mri_h_t))
            pet_estimate1 = self.hid2measures_pet1_2(pet_estimate1) + hidden_mri[0].new(mri_seq[i])
            # out_cat_seq.append(o_cat)
            # mri_h_out.append(mri_h_t)
            if i == 0:
                pet_s_esti.append(pet_estimate1)
                idx_f = np.isnan(_pet_seq[i])
                pet_seq[i][idx_f] = pet_estimate1.data.cpu().numpy()[idx_f]
            else:
                pet_s_esti.append(pet_estimate1)
                idx_f = np.isnan(_pet_seq[i])
                pet_seq[i][idx_f] = self.alpha * pet_seq[i][idx_f] + \
                                    self.beta * pet_estimate1.data.cpu().numpy()[idx_f]

            hidden_pet, pet_h_t = self.predict_pet(pet_seq[i], hidden_pet, masks_pet)

            o_cat = self.hid2category(torch.cat((mri_h_t, pet_h_t), dim=-1))

            mri_estimate = self.hid2measures_mri(torch.cat((mri_h_t, pet_h_t), dim=-1)) + hidden_mri[0].new(mri_seq[i])
            pet_estimate2 = self.hid2measures_pet2(torch.cat((mri_h_t, pet_h_t), dim=-1)) + hidden_mri[0].new(
                pet_seq[i])

            out_cat_seq.append(o_cat)
            # pet_cat_seq.append(pet_cat)
            # pet_h_out.append(pet_h_t)

            # fill in the missing features of the next timepoint
            if j < len(mri_seq):
                out_mri_seq.append(mri_estimate)
                idx = np.isnan(_mri_seq[j])
                mri_seq[j][idx] = mri_estimate.data.cpu().numpy()[idx]

                pet_f_esti.append(pet_estimate2)
                idx_f = np.isnan(_pet_seq[j])
                pet_seq[j][idx_f] = pet_estimate2.data.cpu().numpy()[idx_f]

            ht_set = torch.cat((mri_h_t.unsqueeze(1), pet_h_t.unsqueeze(1)), dim=1)
            ht_set_new, attn1 = self.attention1(ht_set, ht_set, ht_set)

            _ht_set_new = ht_set_new.view(batch_size, -1)
            o_cat = self.hid2category(_ht_set_new)
            o_cat = self.fc_ac(o_cat)
            out_cat_seq.append(o_cat)
            # pet_cat_seq.append(pet_cat)
            mri_h_out.append(ht_set_new[:, 0, :])
            pet_h_out.append(ht_set_new[:, 1, :])
            attns1.append(attn1)

        mri_h_out = torch.stack(mri_h_out).permute(1, 0, 2)
        pet_h_out = torch.stack(pet_h_out).permute(1, 0, 2)

        h_out = torch.cat((mri_h_out, pet_h_out), dim=-1)
        h_out, attn2 = self.attention2(h_out, h_out, h_out)
        h_out_new = h_out.reshape(batch_size, -1)

        trans_out = self.trans_fc(h_out_new)
        trans_out = self.fc_ac(trans_out)

        return torch.stack(out_cat_seq), torch.stack(out_mri_seq), \
               torch.stack(pet_s_esti), torch.stack(pet_f_esti), trans_out, attns1, attn2


class SingleTimepoint(RNNModel):
    """
    Base class for RNN model with 1 hidden state (e.g. MinimalRNN)
    (in contrast LSTM has 2 hidden states: c and h)
    """

    def init_hidden_state(self, batch_size):
        dev = next(self.parameters()).device
        state = []
        for cell in self.cells_mri:
            state.append(torch.zeros(batch_size, cell.hidden_size, device=dev))
        return state

    def init_hidden_state_pet(self, batch_size):
        dev = next(self.parameters()).device
        state = []
        for cell in self.cells_pet:
            state.append(torch.zeros(batch_size, cell.hidden_size, device=dev))
        return state

    def predict(self, i_val, hid, masks):
        i_mask, r_mask = masks
        h_t = hid[0].new(i_val) * i_mask

        next_hid = []
        for cell, prev_h, mask in zip(self.cells_mri, hid, r_mask):
            h_t = cell(h_t, prev_h * mask)
            next_hid.append(h_t)

        # mri_estimate = self.hid2measures_mri(h_t) + hid[0].new(i_val)
        # pet_estimate1 = self.hid2measures_pet1(h_t) + hid[0].new(i_val)

        return next_hid, h_t

    def predict_pet(self, i_val, hid, masks):
        i_mask, r_mask = masks
        h_t = hid[0].new(i_val) * i_mask

        next_hid = []
        for cell, prev_h, mask in zip(self.cells_pet, hid, r_mask):
            h_t = cell(h_t, prev_h * mask)
            next_hid.append(h_t)

        # mri_estimate = self.hid2measures_pet(h_t) + hid[0].new(i_val)

        return next_hid, h_t


class MinimalRNN(SingleTimepoint):
    """ Minimal RNN """

    def __init__(self, **kwargs):
        super(MinimalRNN, self).__init__(MinimalRNNCell, **kwargs)
        for cell in self.cells_mri:
            gate_init(cell.bias_hh)


class GRU(SingleTimepoint):
    """ Minimal RNN """

    def __init__(self, **kwargs):
        super(GRU, self).__init__(nn.GRUCell, **kwargs)
        for cell in self.cells_mri:
            gate_init(cell.bias_hh)


##############---Discriminator---################
class Discriminator(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.linear1 = nn.Linear(x.shape[1], x.shape[1])
        self.linear2 = nn.Linear(x.shape[1], int(x.shape[1]) // 2)
        self.linear3 = nn.Linear(int(x.shape[1]) // 2, x.shape[1])

    def forward(self, x):
        x1 = F.tanh(self.linear1(x.float()))
        x2 = F.tanh(self.linear2(x1))
        predict_mask = F.sigmoid(self.linear3(x2))
        return predict_mask


if __name__ == '__main__':
    gpu_id = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # val_seq = torch.randn(5, 10, 116)
    # cat_seq = torch.randn(5, 10, 3)
    val_seq = np.random.randn(5, 3, 90)
    pet_seq = np.random.randn(5, 3, 90)
    # cat_seq = np.random.randn(5, 10, 3)
    pet_seq[0] = np.nan
    pet_seq[2,  1:, :] = np.nan
    pet_seq[-1, 0, :] = np.nan
    batch_size = 3
    hidden_size = 64
    model = MinimalRNN(nb_classes=2, nb_mri=90, nb_pet=90, h_mri=64, h_pet=64,
                       h_drop=0.05, i_drop=0.05, nb_layers1=3, nb_layers2=3, alpha=0.5, beta=0.5, n_head=4).to(device)
    out_cat_seq, out_val_seq, pet_s_esti, pet_f_esti, trans_out, attn1, attn2 = model(val_seq, pet_seq)



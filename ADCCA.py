import torch
import torch.nn as nn
import torch.nn.init as init
from math import sqrt
import math
import torch.nn.functional as F

def self_attention(query, key, value, dropout=None, mask=None):
    """
    Compute self-attention scores and apply it to the value.
    Args:
        query (Tensor): Query tensor 'Q'.
        key (Tensor): Key tensor 'K'.
        value (Tensor): Value tensor 'V'.
        dropout (function, optional): Dropout function to be applied to attention scores.
        mask (Tensor, optional): Mask tensor to mask certain positions before softmax.
    Returns:
        Tuple[Tensor, Tensor]: Output after applying self-attention and the attention scores.
    """
    # Dimension of key to scale down dot product values
    d_k = query.size(-1)
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Q,K相似度计算
    # 判断是否要mask，注：mask的操作在QK之后，softmax之前
    # if mask is not None:
    #     """
    #     scores.masked_fill默认是按照传入的mask中为1的元素所在的索引，
    #     在scores中相同的的索引处替换为value，替换值为-1e9，即-(10^9)
    #     """
    #     # mask.cuda()
    #     # 进行mask操作，由于参数mask==0，因此替换上述mask中为0的元素所在的索引
    #
    #   scores = scores.masked_fill(mask == 0, -1e9)

    self_attn_softmax = F.softmax(scores, dim=-1)  # 进行softmax
    # # Apply dropout if provided
    if dropout is not None:
        self_attn_softmax = dropout(self_attn_softmax)

    return torch.matmul(self_attn_softmax, value), self_attn_softmax

# Weight & Bias Initialization
def initialization(net):
    """
        Initialize weights and biases of a neural network.
        Args:
            net (nn.Module): Neural network module.
    """
    if isinstance(net, nn.Linear):
        init.xavier_uniform(net.weight)
        init.zeros_(net.bias)

# ADCCA with 4 Modality
class ADCCA_4_M(nn.Module):
    """
       ADCCA model for four different data modalities.
       Each modality has its own neural network, and the model applies self-attention
       to each modality's output.
    """
    def __init__(self, m1_embedding_list, m2_embedding_list, m3_embedding_list, m4_embedding_list, top_k):
        super(ADCCA_4_M, self).__init__()
        # Embedding List of each modality
        m1_du0, m1_du1, m1_du2, m1_du3 = m1_embedding_list
        m2_du0, m2_du1, m2_du2, m2_du3 = m2_embedding_list
        m3_du0, m3_du1, m3_du2 = m3_embedding_list
        m4_du0, m4_du1, m4_du2, m4_du3 = m4_embedding_list

        # Initialize neural networks for each modality
        self.model1 = nn.Sequential(
            nn.Linear(m1_du0, m1_du1), nn.Tanh(),
            nn.Linear(m1_du1, m1_du2), nn.Tanh(),
            nn.Linear(m1_du2, m1_du3), nn.Tanh())

        self.model2 = nn.Sequential(
            nn.Linear(m2_du0, m2_du1), nn.Tanh(),
            nn.Linear(m2_du1, m2_du2), nn.Tanh(),
            nn.Linear(m2_du2, m2_du3), nn.Tanh())

        self.model3 = nn.Sequential(
            nn.Linear(m3_du0, m3_du1), nn.Tanh(),
            nn.Linear(m3_du1, m3_du2), nn.Tanh())

        self.model4 = nn.Sequential(
            nn.Linear(m4_du0, m4_du1), nn.Tanh(),
            nn.Linear(m4_du1, m4_du2), nn.Tanh(),
            nn.Linear(m4_du2, m4_du3), nn.Tanh())

        # Weight & Bias Initialization
        self.model1.apply(initialization)
        self.model2.apply(initialization)
        self.model3.apply(initialization)
        self.model4.apply(initialization)

        self.top_k = top_k

        # Projection matrix
        self.U = None

        # Softmax Function
        self.softmax = nn.Softmax(dim=1)

    # Compute outputs for each modality with self-attention
    def forward(self, x1, x2, x3, x4):
        output1o = self.model1(x1)
        output2o = self.model2(x2)
        output3o = self.model3(x3)
        output4o = self.model4(x4)
        output1, _ = self_attention(output1o, output1o, output1o, dropout=None, mask=None)
        output2, _ = self_attention(output2o, output2o, output2o, dropout=None, mask=None)
        output3, _ = self_attention(output3o, output3o, output3o, dropout=None, mask=None)
        output4, _ = self_attention(output4o, output4o, output4o, dropout=None, mask=None)

        return output1, output2, output3, output4, output1o, output2o, output3o, output4o

    # Calculate correlation loss
    def cal_loss(self, H_list, train=True):
        """
            Calculate correlation loss for the given list of modality representations.
            This function applies SVD and QR decomposition to compute the loss.
            Args:
                H_list (List[Tensor]): List of modality representations.
                train (bool, optional): Flag indicating if the model is in training phase.
            Returns:
                Tensor: Computed correlation loss.
        """
        eps = 1e-8
        AT_list = []

        for H in H_list:
            assert torch.isnan(H).sum().item() == 0
            m = H.size(1)  # out_dim
            Hbar = H - H.mean(dim=1).repeat(m, 1).view(-1, m)
            assert torch.isnan(Hbar).sum().item() == 0

            A, S, B = Hbar.svd(some=True, compute_uv=True)
            A = A[:, :self.top_k]
            assert torch.isnan(A).sum().item() == 0

            S_thin = S[:self.top_k]
            S2_inv = 1. / (torch.mul(S_thin, S_thin) + eps)
            assert torch.isnan(S2_inv).sum().item() == 0

            T2 = torch.mul(torch.mul(S_thin, S2_inv), S_thin)
            assert torch.isnan(T2).sum().item() == 0

            T2 = torch.where(T2 > eps, T2, (torch.ones(T2.shape) * eps).to(H.device))
            T = torch.diag(torch.sqrt(T2))
            assert torch.isnan(T).sum().item() == 0

            T_unnorm = torch.diag(S_thin + eps)
            assert torch.isnan(T_unnorm).sum().item() == 0

            AT = torch.mm(A, T)
            AT_list.append(AT)

        # Concatenate the A*T matrices obtained from all viewpoints into one large matrix
        M_tilde = torch.cat(AT_list, dim=1)
        assert torch.isnan(M_tilde).sum().item() == 0

        # Perform QR decomposition on the concatenated matrix
        # Perform SVD dimensionality reduction to obtain the G matrix
        Q, R = M_tilde.qr() # QR decomposition
        assert torch.isnan(R).sum().item() == 0
        assert torch.isnan(Q).sum().item() == 0

        U, lbda, _ = R.svd(some=False, compute_uv=True) # SVD on R
        assert torch.isnan(U).sum().item() == 0
        assert torch.isnan(lbda).sum().item() == 0

        G = Q.mm(U[:, :self.top_k])
        assert torch.isnan(G).sum().item() == 0

        U = []  # Projection Matrix

        # Get mapping to shared space
        # Get Projection matrix U
        views = H_list
        F = [H.shape[0] for H in H_list]  # features per view
        for idx, (f, view) in enumerate(zip(F, views)):
            _, R = torch.qr(view)
            Cjj_inv = torch.inverse((R.T.mm(R) + eps * torch.eye(view.shape[1], device=view.device)))
            assert torch.isnan(Cjj_inv).sum().item() == 0
            pinv = Cjj_inv.mm(view.T)

            U.append(pinv.mm(G))

        # If model training -> Change projection matrix
        # Else -> Using projection matrix for calculate correlation loss
        if train:
            self.U = U
            for i in range(len(self.U)):
                self.U[i] = nn.Parameter(torch.tensor(self.U[i]))
        _, S, _ = M_tilde.svd(some=True)

        assert torch.isnan(S).sum().item() == 0
        use_all_singular_values = False
        if not use_all_singular_values:
            S = S.topk(self.top_k)[0]
        corr = torch.sum(S)
        assert torch.isnan(corr).item() == 0

        loss = - corr
        return loss # Negative correlation as loss

    # ADCCA prediction
    # Input: Each modality
    # Output: Soft voting of the label presentation of each modality
    def predict(self, x1, x2, x3, x4):
        # out1 = self.model1(x1)
        # out2 = self.model2(x2)
        # out3 = self.model3(x3)
        output1o = self.model1(x1)
        output2o = self.model2(x2)
        output3o = self.model3(x3)
        output4o = self.model4(x4)

        out1, _ = self_attention(output1o, output1o, output1o, dropout=None, mask=None)
        out2, _ = self_attention(output2o, output2o, output2o, dropout=None, mask=None)
        out3, _ = self_attention(output3o, output3o, output3o, dropout=None, mask=None)
        out4, _ = self_attention(output4o, output4o, output4o, dropout=None, mask=None)

        t1 = torch.matmul(out1, self.U[0])
        t2 = torch.matmul(out2, self.U[1])
        t3 = torch.matmul(out3, self.U[2])
        t4 = torch.matmul(out4, self.U[3])

        y_hat1 = torch.matmul(t1, torch.pinverse(self.U[4]))
        y_hat2 = torch.matmul(t2, torch.pinverse(self.U[4]))
        y_hat3 = torch.matmul(t3, torch.pinverse(self.U[4]))
        y_hat4 = torch.matmul(t4, torch.pinverse(self.U[4]))
        y_ensemble = (y_hat1 + y_hat2 + y_hat3 + y_hat4) / 4

        y_hat1 = self.softmax(y_hat1)
        y_hat2 = self.softmax(y_hat2)
        y_hat3 = self.softmax(y_hat3)
        y_hat4 = self.softmax(y_hat4)
        y_ensemble = self.softmax(y_ensemble)

        return y_hat1, y_hat2, y_hat3, y_hat4, y_ensemble

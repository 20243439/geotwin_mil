import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AdvancedAttentionPooling(nn.Module):
    def __init__(self, feature_dim, mid_dim, out_dim=1, flatten=False, dropout=0., spatial_dim=(224, 224)):
        super(AdvancedAttentionPooling, self).__init__()

        self.feature_dim = feature_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.flatten = flatten

        self.scale1 = nn.AdaptiveAvgPool2d((1, 1))
        self.scale2 = nn.AdaptiveAvgPool2d((2, 2))
        self.scale3 = nn.AdaptiveAvgPool2d((4, 4))

        self.spatial_attention = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1, bias=False)

        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, 1)
            ) for _ in range(3)
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

        self.sigmoid = nn.Sigmoid()
        self.positional_encoding = self._generate_positional_encoding(spatial_dim, feature_dim)

    def _generate_positional_encoding(self, spatial_dim, feature_dim):
        h, w = spatial_dim
        pos_encoding = torch.zeros(h, w, feature_dim)
        for y in range(h):
            for x in range(w):
                pos_encoding[y, x, :] = torch.tensor([
                    y / (h - 1),
                    x / (w - 1),
                ] * (feature_dim // 2))
        return pos_encoding.unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        x_reshaped = x.view(batch_size, feature_dim, int(seq_len**0.5), -1)
        scale1 = self.scale1(x_reshaped).view(batch_size, feature_dim, -1)
        scale2 = self.scale2(x_reshaped).view(batch_size, feature_dim, -1)
        scale3 = self.scale3(x_reshaped).view(batch_size, feature_dim, -1)

        spatial_weights = F.softmax(self.spatial_attention(x_reshaped).view(batch_size, -1), dim=1)
        spatial_features = torch.bmm(spatial_weights.unsqueeze(1), x.view(batch_size, seq_len, feature_dim)).squeeze(1)

        attention_outputs = []
        for attention in self.attention_heads:
            A = attention(x)
            A = F.softmax(A, dim=1)
            A = self.dropout(A)
            attention_features = torch.bmm(A.transpose(1, 2), x)
            attention_outputs.append(attention_features.squeeze(1))

        combined_features = torch.cat([
            scale1.mean(dim=2),
            scale2.mean(dim=2),
            scale3.mean(dim=2),
            spatial_features
        ] + attention_outputs, dim=1)

        pos_enc = self.positional_encoding.to(x.device)
        combined_features += pos_enc[:, :combined_features.size(1), :]

        Y_prob = self.classifier(combined_features)

        return Y_prob


class AttentionPooling(nn.Module):
    def __init__(self, feature_dim, mid_dim, out_dim=1, flatten=False, dropout=0.):
        super(AttentionPooling, self).__init__()

        self.feature_dim = feature_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.flatten = flatten

        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.out_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        A = self.attention(x)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=2)
        A = self.dropout(A)

        M = torch.matmul(A, x)

        if self.flatten:
            M = M.view(M.size(0), -1)

        M = M.mean(dim=1)
        Y_prob = self.classifier(M)

        return Y_prob

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=17,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x):
        
        normed_x = self.norm(x)
        attn_output, attn_weights = self.attn(normed_x, return_attn=True)  # 수정
    
        
        x = x + attn_output
        return x, attn_weights  # Attention Weights 반환

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

class TransMIL(nn.Module):
    def __init__(self, n_classes, input_size):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, feats):
        device = feats.device

        h = feats
        h = self._fc1(h)
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)

      

        h, aw1 = self.layer1(h)
        h = self.pos_layer(h, _H, _W)

        # Layer 2
        h, aw2 = self.layer2(h)
        h = self.norm(h)[:, 0]
        logits = self._fc2(h)
        
        # Return logits and attention weights
        return logits, aw1, aw2
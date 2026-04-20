# -*- coding: utf-8 -*-
"""
对比实验最佳模型架构 - Bert-BiLSTM-CNN
架构: BERT(freeze 8) + BiLSTM(64) + CNN(64*3) + Fusion(128)
最优checkpoint: comprehensive_results/checkpoints/BERT_BiLSTM_CNN_best.pt
"""
import torch
import torch.nn as nn
from transformers import BertModel


class BertBiLSTMCNN(nn.Module):
    """
    Bert-BiLSTM-CNN 对比实验最优架构

    与 flask_app 使用的旧 bert_bilstm_cnn.py 区别:
    - LSTM hidden=64 (旧: 256), 单层 (旧: 2层)
    - CNN n_filters=64 (旧: 256)
    - Fusion hidden=128 (旧: 256)
    - freeze_bert=8 (旧: 全部可训练)
    - CNN直接处理BERT输出 (旧: 处理LSTM输出)
    """
    def __init__(self, num_classes=3, freeze_bert=8, hidden=64, n_filters=64, kernels=[2, 3, 4]):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lstm = nn.LSTM(
            768, hidden, 1,
            bidirectional=True, batch_first=True
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(768, n_filters, k) for k in kernels
        ])
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2 + n_filters * len(kernels), 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(128, num_classes)
        self._freeze(freeze_bert)

    def _freeze(self, n):
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, ids, attention_mask):
        x = self.bert(ids, attention_mask)['last_hidden_state']
        # BiLSTM branch: 双向最后一层隐状态拼接
        _, (h, _) = self.lstm(x)
        lstm_f = torch.cat([h[-2], h[-1]], dim=1)
        # CNN branch: 对BERT序列输出做1D卷积
        cnn_x = x.permute(0, 2, 1)
        cnn_f = torch.cat([
            torch.max(torch.relu(conv(cnn_x)), dim=2)[0]
            for conv in self.convs
        ], dim=1)
        # 特征融合
        fused = self.fusion(torch.cat([lstm_f, cnn_f], dim=1))
        return self.fc(fused)

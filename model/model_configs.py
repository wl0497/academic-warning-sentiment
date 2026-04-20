"""
4种模型架构定义
"""
import torch
import torch.nn as nn
from transformers import BertModel


class BertOnlyModel(nn.Module):
    """BERT-Only: 仅使用BERT [CLS]输出进行分类"""
    
    def __init__(self, num_classes=3, freeze_bert_layers=8):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(768, num_classes)
        
        # 冻结BERT层
        self._freeze_bert(freeze_bert_layers)
    
    def _freeze_bert(self, freeze_layers):
        """冻结BERT前N层"""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"BERT-Only: 冻结前{freeze_layers}层，可训练分类头")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取[CLS] token的输出
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


class BertCNNModel(nn.Module):
    """BERT-CNN: BERT + CNN特征提取"""
    
    def __init__(self, num_classes=3, freeze_bert_layers=8, 
                 num_filters=64, filter_sizes=[2, 3, 4]):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.5)
        
        # CNN层
        self.convs = nn.ModuleList([
            nn.Conv1d(768, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # 分类器
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
        self._freeze_bert(freeze_bert_layers)
    
    def _freeze_bert(self, freeze_layers):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"BERT-CNN: 冻结前{freeze_layers}层，CNN可训练")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [batch, seq_len, hidden] -> [batch, hidden, seq_len]
        x = outputs.last_hidden_state.permute(0, 2, 1)
        
        # 多尺度CNN
        conv_outputs = []
        for conv in self.convs:
            # [batch, num_filters, seq_len - filter_size + 1]
            conv_out = torch.relu(conv(x))
            # Max pooling over time
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        # 拼接多尺度特征
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


class BertBiLSTMModel(nn.Module):
    """BERT-BiLSTM: BERT + BiLSTM序列建模"""
    
    def __init__(self, num_classes=3, freeze_bert_layers=8,
                 lstm_hidden=64, lstm_layers=1):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.5)
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if lstm_layers > 1 else 0
        )
        
        # 分类器 (双向LSTM输出2*hidden)
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)
        
        self._freeze_bert(freeze_layers)
    
    def _freeze_bert(self, freeze_layers):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"BERT-BiLSTM: 冻结前{freeze_layers}层，BiLSTM可训练")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # BiLSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后时刻的隐藏状态（双向拼接）
        # h_n: [num_layers*2, batch, hidden]
        forward_h = h_n[-2]  # 前向最后层
        backward_h = h_n[-1]  # 后向最后层
        x = torch.cat([forward_h, backward_h], dim=1)
        
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


class BertBiLSTMCNNModel(nn.Module):
    """BERT-BiLSTM-CNN: 完整模型（BERT + BiLSTM + CNN融合）"""
    
    def __init__(self, num_classes=3, freeze_bert_layers=8,
                 lstm_hidden=64, lstm_layers=1,
                 num_filters=64, filter_sizes=[2, 3, 4]):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if lstm_layers > 1 else 0
        )
        
        # CNN层（用于捕获局部n-gram特征）
        self.convs = nn.ModuleList([
            nn.Conv1d(768, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # 融合层
        lstm_output_dim = lstm_hidden * 2
        cnn_output_dim = num_filters * len(filter_sizes)
        fusion_dim = lstm_output_dim + cnn_output_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 分类器
        self.classifier = nn.Linear(128, num_classes)
        
        self._freeze_bert(freeze_bert_layers)
    
    def _freeze_bert(self, freeze_layers):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"BERT-BiLSTM-CNN: 冻结前{freeze_layers}层，BiLSTM+CNN可训练")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # [batch, seq_len, 768]
        
        # BiLSTM分支
        lstm_out, (h_n, c_n) = self.lstm(x)
        forward_h = h_n[-2]
        backward_h = h_n[-1]
        lstm_features = torch.cat([forward_h, backward_h], dim=1)  # [batch, 128]
        
        # CNN分支
        x_cnn = x.permute(0, 2, 1)  # [batch, 768, seq_len]
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x_cnn))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        cnn_features = torch.cat(conv_outputs, dim=1)  # [batch, 192]
        
        # 特征融合
        fused = torch.cat([lstm_features, cnn_features], dim=1)  # [batch, 320]
        fused = self.fusion(fused)  # [batch, 128]
        
        logits = self.classifier(fused)
        return logits

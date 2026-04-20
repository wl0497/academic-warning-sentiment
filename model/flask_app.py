# -*- coding: utf-8 -*-
"""
Flask推理服务 - 对比实验最佳模型 BERT-BiLSTM-CNN
端口: 5000
模型: comprehensive_results/checkpoints/BERT_BiLSTM_CNN_best.pt
架构: BERT(freeze8) + BiLSTM(64) + CNN(64*3) + Fusion(128)
性能: Test=97.66%, OOD=23.32%, 鲁棒性=96.80%
"""
import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer
import logging
import time

# 模型路径
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL = os.path.join(MODEL_DIR, 'comprehensive_results', 'checkpoints', 'BERT_BiLSTM_CNN_best.pt')

# 加载自定义模型类
from bert_bilstm_cnn_comparison import BertBiLSTMCNN

# 配置
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL_NAMES = ['正面', '中性', '负面']
LABEL_MAP_INV = {0: '正面', 1: '中性', 2: '负面'}

# Flask配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(MODEL_DIR, 'flask.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 全局变量
model = None
tokenizer = None


def load_model():
    """加载对比实验最优模型"""
    global model, tokenizer

    logger.info(f'Device: {DEVICE}')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    model = BertBiLSTMCNN(num_classes=3, freeze_bert=8, hidden=64, n_filters=64)
    checkpoint = torch.load(BEST_MODEL, map_location=DEVICE, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    # 统计参数量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model loaded: {BEST_MODEL}')
    logger.info(f'Total params: {total:,} / Trainable: {trainable:,}')


def predict_single(text):
    """单条预测"""
    enc = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    ids = enc['input_ids'].to(DEVICE)
    mask = enc['attention_mask'].to(DEVICE)

    with torch.no_grad():
        logits = model(ids, mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    label_name = LABEL_MAP_INV[pred]

    # 预警映射
    warnings = {
        '正面': {'level': '正常', 'color': 'green', 'suggestion': '学业状态良好，继续保持'},
        '中性': {'level': '关注', 'color': 'yellow', 'suggestion': '需关注学业状态，适当引导'},
        '负面': {'level': '预警', 'color': 'red', 'suggestion': '需及时干预，建议辅导员谈话、学习帮扶'}
    }

    return {
        'label': label_name,
        'label_id': int(pred),
        'confidence': round(conf, 4),
        'probabilities': {LABEL_MAP_INV[i]: round(probs[0][i].item(), 4) for i in range(3)},
        'warning': warnings[label_name]
    }


# ============================================================
# 路由
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    total = sum(p.numel() for p in model.parameters()) if model else 0
    return jsonify({
        'status': 'ok',
        'model': 'BERT-BiLSTM-CNN (comparison best)',
        'model_file': 'BERT_BiLSTM_CNN_best.pt',
        'device': str(DEVICE),
        'params': f'{total:,}' if total else 'unknown'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """单条文本预测"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'Missing text field'}), 400

        result = predict_single(text)
        result['text'] = text
        return jsonify(result)
    except Exception as e:
        logger.error(f'Predict error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """批量文本预测"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Missing or invalid texts field'}), 400

        results = []
        for text in texts:
            r = predict_single(text)
            r['text'] = text
            results.append(r)

        # 统计 summary (兼容前端)
        summary = {}
        for r in results:
            lb = r['label']
            summary[lb] = summary.get(lb, 0) + 1

        return jsonify({
            'total': len(results),
            'results': results,
            'summary': summary
        })
    except Exception as e:
        logger.error(f'Batch predict error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """分析接口 - 返回预警统计"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        if not texts:
            return jsonify({'error': 'Missing texts'}), 400

        results = []
        stats = {'正常': 0, '关注': 0, '预警': 0}

        for text in texts:
            r = predict_single(text)
            r['text'] = text
            results.append(r)
            stats[r['warning']['level']] += 1

        return jsonify({
            'total': len(results),
            'results': results,
            'statistics': stats,
            'summary': stats
        })
    except Exception as e:
        logger.error(f'Analyze error: {e}')
        return jsonify({'error': str(e)}), 500


# ============================================================
# 启动
# ============================================================

if __name__ == '__main__':
    print('=' * 55)
    print('  Flask 推理服务 - 对比实验最佳模型')
    print('  模型: BERT-BiLSTM-CNN')
    print('  性能: Test=97.66%, OOD=23.32%, Robust=96.80%')
    print('=' * 55)
    load_model()
    print('  模型加载完成，服务启动于 0.0.0.0:5000')
    print('=' * 55)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

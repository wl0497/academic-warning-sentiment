<template>
  <div class="model-page">

    <!-- 模型基本信息 -->
    <el-card shadow="hover" style="margin-top: 0;">
      <template #header><span>模型架构 &amp; 训练参数</span></template>
      <el-row :gutter="20">
        <el-col :span="12">
          <el-descriptions :column="1" border title="模型架构">
            <el-descriptions-item label="模型名称">BERT-BiLSTM-CNN 混合模型</el-descriptions-item>
            <el-descriptions-item label="预训练模型">bert-base-chinese</el-descriptions-item>
            <el-descriptions-item label="BERT冻结层">8层（前8层冻结）</el-descriptions-item>
            <el-descriptions-item label="BiLSTM隐藏维度">64</el-descriptions-item>
            <el-descriptions-item label="BiLSTM层数">1层</el-descriptions-item>
            <el-descriptions-item label="CNN滤波器数量">64</el-descriptions-item>
            <el-descriptions-item label="CNN卷积核尺寸">[2, 3, 4]</el-descriptions-item>
            <el-descriptions-item label="Dropout">0.5</el-descriptions-item>
            <el-descriptions-item label="融合层维度">128</el-descriptions-item>
            <el-descriptions-item label="分类数量">3（正面/中性/负面）</el-descriptions-item>
            <el-descriptions-item label="最大序列长度">128</el-descriptions-item>
          </el-descriptions>
        </el-col>
        <el-col :span="12">
          <el-descriptions :column="1" border title="训练参数">
            <el-descriptions-item label="批次大小">16</el-descriptions-item>
            <el-descriptions-item label="学习率">2e-5</el-descriptions-item>
            <el-descriptions-item label="优化器">AdamW</el-descriptions-item>
            <el-descriptions-item label="损失函数">CrossEntropyLoss</el-descriptions-item>
            <el-descriptions-item label="训练轮数">15 epochs</el-descriptions-item>
            <el-descriptions-item label="GPU训练">支持CUDA加速</el-descriptions-item>
          </el-descriptions>
        </el-col>
      </el-row>
    </el-card>

    <!-- 模型性能指标 -->
    <el-card shadow="hover" style="margin-top: 20px;">
      <template #header>
        <span>模型评估指标</span>
        <el-tag type="success" style="float:right">对比实验最佳</el-tag>
      </template>
      <el-row :gutter="20">
        <el-col :span="6" v-for="m in metrics" :key="m.label">
          <div class="metric-card" :style="{ borderColor: m.color }">
            <div class="metric-value" :style="{ color: m.color }">{{ m.value }}</div>
            <div class="metric-label">{{ m.label }}</div>
            <div class="metric-sub">{{ m.sub }}</div>
          </div>
        </el-col>
      </el-row>

      <el-divider content-position="left">模型对比</el-divider>
      <el-table :data="comparisonTable" stripe style="width: 100%; margin-top: 10px;" size="small">
        <el-table-column prop="model" label="模型" min-width="140" />
        <el-table-column prop="testAcc" label="测试准确率" align="center" />
        <el-table-column prop="oodAcc" label="OOD泛化" align="center">
          <template #default="{ row }">
            <el-tag :type="row.oodAcc.includes('23.32') ? 'success' : 'info'" size="small">{{ row.oodAcc }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="robust" label="噪声鲁棒性" align="center" />
        <el-table-column prop="params" label="参数量" align="center" />
        <el-table-column prop="best" label="综合评价" align="center">
          <template #default="{ row }">
            <el-tag v-if="row.best" type="success">最佳</el-tag><span v-else>—</span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 架构演示 -->
    <el-card shadow="hover" style="margin-top: 20px;">
      <template #header>
        <span>技术架构流程演示</span>
        <el-tag type="primary" style="float:right">点击下方按钮打开交互演示窗口</el-tag>
      </template>

      <div class="arch-summary">
        <div class="arch-summary-title">BERT-BiLSTM-CNN 模型推理全流程</div>
        <div class="arch-flow">
          <div class="arch-step step-1">
            <div class="arch-step-icon">1</div>
            <div class="arch-step-name">文本输入</div>
            <div class="arch-step-desc">社交评论</div>
          </div>
          <div class="arch-arrow">→</div>
          <div class="arch-step step-2">
            <div class="arch-step-icon">2</div>
            <div class="arch-step-name">BERT编码</div>
            <div class="arch-step-desc">768维向量</div>
          </div>
          <div class="arch-arrow">→</div>
          <div class="arch-step step-3">
            <div class="arch-step-icon">3</div>
            <div class="arch-step-name">BiLSTM+CNN</div>
            <div class="arch-step-desc">双路并行</div>
          </div>
          <div class="arch-arrow">→</div>
          <div class="arch-step step-4">
            <div class="arch-step-icon">4</div>
            <div class="arch-step-name">特征融合</div>
            <div class="arch-step-desc">320→128维</div>
          </div>
          <div class="arch-arrow">→</div>
          <div class="arch-step step-5">
            <div class="arch-step-icon">5</div>
            <div class="arch-step-name">输出分类</div>
            <div class="arch-step-desc">正面/中性/负面</div>
          </div>
        </div>
        <div class="arch-core-note">
          核心优势：BiLSTM 捕获时序依赖 + CNN 捕获 n-gram 局部特征 → 双路互补融合 → OOD泛化最优
        </div>
      </div>

      <div class="demo-trigger-area">
        <div class="demo-visual">
          <div class="mini-flow">
            <span class="mini-node n1">文本</span>
            <span class="mini-arrow">→</span>
            <span class="mini-node n2">BERT</span>
            <span class="mini-arrow">→</span>
            <span class="mini-node n3">BiLSTM</span>
            <span class="mini-sep">|</span>
            <span class="mini-node n4">CNN</span>
            <span class="mini-arrow">→</span>
            <span class="mini-node n5">融合</span>
            <span class="mini-arrow">→</span>
            <span class="mini-node n6">分类</span>
          </div>
        </div>
        <div class="demo-btn-area">
          <div class="demo-desc">交互式动画演示窗口</div>
          <div class="demo-desc-sub">自定义输入文本 · 逐步查看数据流转 · 直观展示模型原理</div>
          <el-button type="primary" size="large" class="demo-open-btn" @click="demoVisible = true">
            打开架构演示窗口
          </el-button>
        </div>
      </div>
    </el-card>

    <!-- 数据集信息 -->
    <el-card shadow="hover" style="margin-top: 20px;">
      <template #header><span>数据集信息</span></template>
      <el-descriptions :column="2" border>
        <el-descriptions-item label="训练数据">27,000条</el-descriptions-item>
        <el-descriptions-item label="验证数据">3,000条</el-descriptions-item>
        <el-descriptions-item label="测试数据（ID）">5,001条</el-descriptions-item>
        <el-descriptions-item label="OOD测试数据">1,998条</el-descriptions-item>
        <el-descriptions-item label="数据集划分">训练80%/验证10%/测试10%</el-descriptions-item>
        <el-descriptions-item label="标签分布">三分类均衡</el-descriptions-item>
        <el-descriptions-item label="数据重复率">0%</el-descriptions-item>
        <el-descriptions-item label="数据来源">社交评论 + 学术预警场景</el-descriptions-item>
      </el-descriptions>
    </el-card>

    <!-- 实验说明 -->
    <el-card shadow="hover" style="margin-top: 20px;">
      <template #header><span>实验说明</span></template>
      <el-form label-width="auto">
        <el-form-item label="实验目标">对比4种BERT变体架构在学术预警情感分析任务上的性能差异</el-form-item>
        <el-form-item label="对比模型">BERT-Only / BERT-CNN / BERT-BiLSTM / BERT-BiLSTM-CNN</el-form-item>
        <el-form-item label="评估维度">测试准确率（ID）、OOD泛化准确率、噪声鲁棒性（15%随机丢词）</el-form-item>
        <el-form-item label="核心发现">
          BERT-BiLSTM-CNN在OOD泛化（23.32%）和噪声鲁棒性（96.80%）上均为最优，
          综合指标最均衡。其BiLSTM时序建模+CNN局部特征提取的混合架构对分布外样本具有最强泛化能力。
        </el-form-item>
        <el-form-item label="部署状态">
          <el-tag type="success">生产运行中</el-tag>
          <span style="margin-left: 12px; color: #909399; font-size: 13px;">模型文件: comprehensive_results/checkpoints/BERT_BiLSTM_CNN_best.pt</span>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 独立演示窗口 -->
    <ArchDemoDialog v-model="demoVisible" />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ArchDemoDialog from '@/components/ArchDemoDialog.vue'

const demoVisible = ref(false)

const metrics = [
  { label: '测试准确率', value: '97.66%', color: '#409eff', sub: 'ID测试集 5001条' },
  { label: 'OOD泛化', value: '23.32%', color: '#67c23a', sub: '真实OOD数据 1998条 · 最高' },
  { label: '噪声鲁棒性', value: '96.80%', color: '#e6a23c', sub: '15%随机丢词 · 最高' },
  { label: '参数量', value: '29.9M', color: '#f56c6c', sub: '含BERT预训练参数' },
]

const comparisonTable = [
  { model: 'BERT-Only', testAcc: '99.76%', oodAcc: '20.22%', robust: '99.00%', params: '28.9M', best: false },
  { model: 'BERT-CNN', testAcc: '95.60%', oodAcc: '10.06%', robust: '93.40%', params: '29.4M', best: false },
  { model: 'BERT-BiLSTM', testAcc: '97.72%', oodAcc: '15.12%', robust: '95.00%', params: '29.4M', best: false },
  { model: 'BERT-BiLSTM-CNN', testAcc: '97.66%', oodAcc: '23.32%', robust: '96.80%', params: '29.9M', best: true },
]
</script>

<style scoped>
.metric-card {
  text-align: center;
  padding: 20px;
  border: 2px solid #eee;
  border-radius: 8px;
  margin: 10px 0;
}
.metric-value { font-size: 30px; font-weight: 700; }
.metric-label { font-size: 14px; color: #606266; margin-top: 5px; }
.metric-sub { font-size: 12px; color: #c0c4cc; margin-top: 3px; }

.arch-summary { margin-bottom: 20px; }
.arch-summary-title {
  font-size: 16px; font-weight: 700; color: #303133;
  margin-bottom: 16px; text-align: center;
}
.arch-flow {
  display: flex; justify-content: center; align-items: center;
  gap: 8px; padding: 0 20px; flex-wrap: wrap;
}
.arch-step {
  display: flex; flex-direction: column; align-items: center; gap: 4px;
  padding: 10px 12px; border-radius: 10px; min-width: 80px; text-align: center;
  border: 2px solid;
}
.step-1 { background: #ecf5ff; border-color: #b3d8ff; }
.step-2 { background: #f0f9eb; border-color: #b3e19d; }
.step-3 { background: #fff5f5; border-color: #fab6b6; }
.step-4 { background: #f4f4f5; border-color: #d3d4d6; }
.step-5 { background: #e8f5e9; border-color: #a5d6a7; }
.arch-step-icon { font-size: 20px; font-weight: 700; color: #303133; }
.arch-step-name { font-size: 12px; font-weight: 700; color: #303133; }
.arch-step-desc { font-size: 10px; color: #909399; }
.arch-arrow { color: #c0c4cc; font-size: 20px; font-weight: 700; }
.arch-core-note {
  margin-top: 12px; text-align: center; font-size: 13px;
  color: #606266; background: #f0f9eb; border-radius: 8px;
  padding: 10px 16px; border: 1px solid #b3e19d;
}

.demo-trigger-area {
  display: flex; gap: 20px; align-items: center;
  padding: 20px;
  background: linear-gradient(135deg, #f5f7fa 0%, #e8f4fd 100%);
  border-radius: 12px; border: 2px dashed #b3d8ff;
}
.demo-visual { flex: 1; }
.mini-flow {
  display: flex; align-items: center; gap: 4px;
  font-size: 12px; justify-content: center; flex-wrap: wrap;
}
.mini-node {
  padding: 6px 10px; border-radius: 8px; text-align: center;
  font-size: 11px; font-weight: 600; min-width: 50px;
}
.n1 { background: #ecf5ff; color: #409eff; }
.n2 { background: #f0f9eb; color: #67c23a; }
.n3 { background: #fffbf0; color: #e6a23c; }
.n4 { background: #fff5f5; color: #f56c6c; }
.n5 { background: #f4f4f5; color: #606266; }
.n6 { background: #e8f5e9; color: #2e7d32; }
.mini-sep { color: #d3d4d6; font-size: 18px; }
.mini-arrow { color: #c0c4cc; font-size: 16px; font-weight: 700; }

.demo-btn-area { text-align: center; min-width: 240px; }
.demo-desc { font-size: 15px; font-weight: 700; color: #303133; margin-bottom: 6px; }
.demo-desc-sub {
  font-size: 12px; color: #909399; margin-bottom: 14px; line-height: 1.6;
}
.demo-open-btn {
  font-size: 15px; padding: 14px 28px; border-radius: 30px;
}
</style>

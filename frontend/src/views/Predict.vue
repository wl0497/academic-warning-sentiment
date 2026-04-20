<template>
  <div class="predict-page">
    <el-card shadow="hover">
      <template #header>
        <div class="card-header">
          <span>🔍 文本情感分析</span>
          <el-tag>基于 Bert-BiLSTM-CNN 融合模型</el-tag>
        </div>
      </template>

      <!-- 输入区域 -->
      <el-input
        v-model="inputText"
        type="textarea"
        :rows="5"
        placeholder="请输入待分析的文本内容（如学生反馈、社交评论等）"
        maxlength="500"
        show-word-limit
      />

      <div class="btn-group">
        <el-button type="primary" @click="handlePredict" :loading="loading" size="large">
          🔍 开始分析
        </el-button>
        <el-button @click="inputText = ''">清空</el-button>
      </div>

      <!-- 快捷示例 -->
      <div class="examples">
        <span class="examples-label">快捷示例：</span>
        <el-tag
          v-for="(ex, i) in examples"
          :key="i"
          class="example-tag"
          effect="plain"
          @click="inputText = ex"
          style="cursor: pointer;"
        >{{ ex.length > 20 ? ex.slice(0, 20) + '...' : ex }}</el-tag>
      </div>
    </el-card>

    <!-- 分析结果 -->
    <el-card v-if="result" shadow="hover" style="margin-top: 20px;">
      <template #header><span>📋 分析结果</span></template>

      <el-descriptions :column="2" border>
        <el-descriptions-item label="输入文本" :span="2">{{ result.text }}</el-descriptions-item>
        <el-descriptions-item label="情感标签">
          <el-tag :type="tagType(result.label)">{{ result.label }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="置信度">
          <el-progress :percentage="Number((result.confidence * 100).toFixed(1))" :color="progressColor(result.label)" />
        </el-descriptions-item>
        <el-descriptions-item label="预警级别">
          <el-tag :type="result.warning?.level === '正常' ? 'success' : result.warning?.level === '关注' ? 'warning' : 'danger'" effect="dark">
            {{ result.warning?.level }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="干预建议">{{ result.warning?.suggestion }}</el-descriptions-item>
      </el-descriptions>

      <!-- 概率分布图 -->
      <div class="prob-chart-wrapper">
        <h4>概率分布</h4>
        <div ref="probChartRef" style="height: 250px;"></div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { predictText } from '@/api'
import * as echarts from 'echarts'

const inputText = ref('')
const loading = ref(false)
const result = ref(null)
const probChartRef = ref(null)
let probChart = null

const examples = [
  '这学期课程太难了，完全跟不上进度，感觉要挂科了',
  '考研复习压力好大，每天都在焦虑中度过',
  '今天听了老师的课，收获很大，对未来充满信心',
  '学了半天也看不进去，真的不想学了'
]

async function handlePredict() {
  if (!inputText.value.trim()) {
    ElMessage.warning('请输入待分析文本')
    return
  }
  loading.value = true
  try {
    const res = await predictText(inputText.value)
    console.log('Predict response:', res)
    console.log('res.data:', res.data)
    
    if (!res.data) {
      ElMessage.error('返回数据为空')
      return
    }
    
    result.value = res.data
    await nextTick()
    
    if (res.data.probabilities) {
      renderProbChart(res.data.probabilities)
    } else {
      console.warn('probabilities is undefined')
    }
  } catch (err) {
    console.error('Predict error:', err)
    ElMessage.error('分析失败：' + (err.message || '模型服务未启动'))
  } finally {
    loading.value = false
  }
}

function renderProbChart(probs) {
  if (!probs) {
    console.warn('renderProbChart: probs is undefined')
    return
  }
  if (!probChartRef.value) {
    console.warn('renderProbChart: chart ref not ready')
    return
  }
  if (probChart) probChart.dispose()
  probChart = echarts.init(probChartRef.value)
  probChart.setOption({
    tooltip: { trigger: 'axis', formatter: '{b}: {c}%' },
    xAxis: {
      type: 'category',
      data: ['正面', '中性', '负面'],
      axisLabel: { fontSize: 14 }
    },
    yAxis: { type: 'value', max: 1, axisLabel: { formatter: '{0}%' } },
    series: [{
      type: 'bar',
      data: [
        { value: (probs['正面'] || 0) * 100, itemStyle: { color: '#67c23a' } },
        { value: (probs['中性'] || 0) * 100, itemStyle: { color: '#e6a23c' } },
        { value: (probs['负面'] || 0) * 100, itemStyle: { color: '#f56c6c' } }
      ],
      barWidth: '40%',
      label: { show: true, position: 'top', formatter: '{c}%' }
    }]
  })
}

function tagType(label) {
  return label === '正面' ? 'success' : label === '中性' ? 'warning' : 'danger'
}

function progressColor(label) {
  return label === '正面' ? '#67c23a' : label === '中性' ? '#e6a23c' : '#f56c6c'
}

onMounted(() => {
  window.addEventListener('resize', () => probChart?.resize())
})

onUnmounted(() => {
  probChart?.dispose()
})
</script>

<style scoped>
.card-header { display: flex; justify-content: space-between; align-items: center; }
.btn-group { margin: 15px 0; }
.examples { margin-top: 10px; }
.examples-label { font-size: 13px; color: #909399; margin-right: 8px; }
.example-tag { margin: 3px; }
.prob-chart-wrapper { margin-top: 20px; text-align: center; }
.prob-chart-wrapper h4 { margin-bottom: 10px; color: #606266; }
</style>

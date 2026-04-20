<template>
  <div class="batch-analysis">
    <el-card shadow="hover">
      <template #header>
        <span>批量情感分析</span>
      </template>

      <el-tabs v-model="activeTab">
        <!-- 文本输入模式 -->
        <el-tab-pane label="文本输入" name="text">
          <el-input
            v-model="batchText"
            type="textarea"
            :rows="8"
            placeholder="每行一条文本，最多100条"
            style="margin-bottom: 16px"
          />
          <el-button type="primary" @click="analyzeTextBatch" :loading="textLoading">
            批量分析
          </el-button>
          <span style="margin-left: 12px; color: #666; font-size: 13px">
            {{ textCount }} 条文本
          </span>
        </el-tab-pane>

        <!-- 文件上传模式 -->
        <el-tab-pane label="文件上传" name="file">
          <el-alert type="info" :closable="false" style="margin-bottom: 12px">
            <template #title>
              支持 CSV/Excel 文件，格式：<strong>学号, 姓名, 学院, 班级, 评论</strong>（5列，必需表头行）
            </template>
          </el-alert>
          <el-upload
            ref="uploadRef"
            :auto-upload="false"
            :limit="1"
            accept=".csv,.xlsx,.xls"
            :on-change="handleFileChange"
          >
            <el-button type="primary">选择文件</el-button>
            <template #tip>
              <div class="el-upload__tip">CSV 或 Excel 文件，单次不超过 500 条</div>
            </template>
          </el-upload>
          <el-button
            type="success"
            style="margin-top: 12px"
            @click="uploadAndAnalyze"
            :loading="fileLoading"
            :disabled="!selectedFile"
          >
            上传并分析
          </el-button>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <!-- 结果展示 -->
    <el-card v-if="results.length > 0" shadow="hover" style="margin-top: 20px">
      <template #header>
        <span>分析结果</span>
      </template>

      <!-- 统计摘要 -->
      <el-row :gutter="16" style="margin-bottom: 16px">
        <el-col :span="6">
          <div class="stat-item">
            <div class="stat-label">总条数</div>
            <div class="stat-num">{{ results.length }}</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="stat-item">
            <div class="stat-label">正面</div>
            <div class="stat-num" style="color: #67c23a">{{ summary['正面'] || 0 }}</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="stat-item">
            <div class="stat-label">中性</div>
            <div class="stat-num" style="color: #e6a23c">{{ summary['中性'] || 0 }}</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="stat-item">
            <div class="stat-label">负面</div>
            <div class="stat-num" style="color: #f56c6c">{{ summary['负面'] || 0 }}</div>
          </div>
        </el-col>
      </el-row>

      <!-- 结果表格 -->
      <el-table :data="results" stripe style="width: 100%">
        <el-table-column prop="name" label="姓名" width="100" />
        <el-table-column prop="college" label="学院" width="140" />
        <el-table-column prop="className" label="班级" width="140" />
        <el-table-column prop="text" label="评论内容" min-width="200" show-overflow-tooltip />
        <el-table-column prop="label" label="情感" width="80">
          <template #default="{ row }">
            <el-tag :type="getSentimentType(row.label)" size="small">
              {{ row.label }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="置信度" width="80">
          <template #default="{ row }">
            {{ (row.confidence * 100).toFixed(0) }}%
          </template>
        </el-table-column>
        <el-table-column label="预警" width="120">
          <template #default="{ row }">
            <template v-if="row.warning">
              <el-tag :type="getWarningType(row.warning.level)" size="small">
                {{ row.warning.level }}
              </el-tag>
            </template>
            <span v-else style="color: #999">—</span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 空状态 -->
    <el-empty v-if="showEmpty" description="请先输入文本或上传文件进行分析" style="margin-top: 40px" />
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { Upload } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const activeTab = ref('text')
const batchText = ref('')
const results = ref([])
const selectedFile = ref(null)
const uploadRef = ref(null)
const textLoading = ref(false)
const fileLoading = ref(false)

const textCount = computed(() => {
  return batchText.value.split('\n').filter(t => t.trim()).length
})

const summary = computed(() => {
  const s = {}
  results.value.forEach(r => {
    s[r.label] = (s[r.label] || 0) + 1
  })
  return s
})

const showEmpty = computed(() => {
  return !textLoading.value && !fileLoading.value && results.value.length === 0
})

async function analyzeTextBatch() {
  const texts = batchText.value.split('\n').filter(t => t.trim()).slice(0, 100)
  if (texts.length === 0) {
    ElMessage.warning('请输入至少一条文本')
    return
  }
  textLoading.value = true
  try {
    const res = await api.post('/batch/predict', { texts })
    results.value = texts.map((text, i) => {
      const r = res.data.results[i] || {}
      return {
        name: '文本' + (i + 1),
        college: '',
        className: '',
        text: text,
        label: r.label || '未知',
        confidence: r.confidence || 0,
        warning: r.warning || null
      }
    })
  } catch (e) {
    ElMessage.error('分析失败：' + (e.response?.data?.error || e.message))
  } finally {
    textLoading.value = false
  }
}

function handleFileChange(file) {
  selectedFile.value = file.raw
}

async function uploadAndAnalyze() {
  if (!selectedFile.value) {
    ElMessage.warning('请选择文件')
    return
  }
  fileLoading.value = true
  results.value = []
  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    const res = await api.post('/batch/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    results.value = (res.data.results || []).map(r => ({
      name: r.name || '未知',
      college: r.college || '',
      className: r.className || '',
      text: r.text || '',
      label: r.label || '未知',
      confidence: r.confidence || 0,
      warning: r.warning || null
    }))
    if (results.value.length === 0) {
      ElMessage.warning('文件中没有找到有效数据')
    } else {
      ElMessage.success('分析完成，共 ' + results.value.length + ' 条')
    }
  } catch (e) {
    ElMessage.error('上传失败：' + (e.response?.data?.error || e.message))
  } finally {
    fileLoading.value = false
  }
}

function getSentimentType(label) {
  const map = { '正面': 'success', '中性': 'warning', '负面': 'danger' }
  return map[label] || 'info'
}

function getWarningType(level) {
  const map = { '正常': 'success', '关注': 'warning', '预警': 'danger' }
  return map[level] || 'info'
}
</script>

<style scoped>
.batch-analysis { padding: 20px }
.stat-item {
  background: #f5f7fa;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}
.stat-label { font-size: 13px; color: #666; margin-bottom: 8px }
.stat-num { font-size: 28px; font-weight: bold; color: #333 }
</style>

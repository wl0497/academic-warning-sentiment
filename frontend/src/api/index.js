import axios from 'axios'

// 前端直连Flask模式（跳过SpringBoot时使用）
// 改为 'http://localhost:8080/api' 即切换到SpringBoot模式
const API_BASE = 'http://localhost:5000'
const USE_SPRINGBOOT = true  // 切换: true=经SpringBoot, false=直连Flask

const request = axios.create({
  baseURL: USE_SPRINGBOOT ? 'http://localhost:8080/api' : API_BASE,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' }
})

// 响应拦截器 - 统一处理响应数据
request.interceptors.response.use(
  response => {
    // 确保返回的数据有data字段
    if (response.data === undefined || response.data === null) {
      console.warn('Empty response data:', response)
    }
    return response
  },
  error => {
    console.error('API Error:', error)
    if (error.response) {
      console.error('Error response:', error.response.data)
    }
    return Promise.reject(error)
  }
)

// ========== 预测接口 ==========
// 单条预测
export function predictText(text) {
  return request.post('/predict', { text })
}

// 批量预测(文本列表)
export function predictBatch(texts) {
  return request.post('/batch/predict', { texts })
}

// 批量预测(文件上传) — 仅SpringBoot模式支持
export function predictBatchFile(file) {
  const formData = new FormData()
  formData.append('file', file)
  const base = USE_SPRINGBOOT ? 'http://localhost:8080/api' : API_BASE
  return axios.post(`${base}/batch/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000
  })
}

// ========== 健康检查 ==========
export function healthCheck() {
  return request.get('/health')
}

// ========== SpringBoot专用接口（直连模式返回mock数据） ==========
export function getStatistics() {
  if (!USE_SPRINGBOOT) return Promise.resolve({ data: { total: 0, sentimentDistribution: { positive: 0, neutral: 0, negative: 0 } } })
  return request.get('/predict/statistics')
}

export function getWarningList(params) {
  if (!USE_SPRINGBOOT) return Promise.resolve({ data: { records: [], total: 0 } })
  return request.get('/warning', { params })
}

export function handleWarning(id, data) {
  if (!USE_SPRINGBOOT) return Promise.resolve({ data: 'ok' })
  return request.post(`/warning/${id}/handle`, data)
}

export function getWarningStats() {
  if (!USE_SPRINGBOOT) return Promise.resolve({ data: { total: 0, pending: 0, handled: 0 } })
  return request.get('/warning/stats')
}

export function getDashboardStats() {
  if (!USE_SPRINGBOOT) return Promise.resolve({ data: { totalPredictions: 0, totalWarnings: 0, pendingWarnings: 0 } })
  return request.get('/dashboard/stats')
}

export function getDashboardTrend(days = 7) {
  if (!USE_SPRINGBOOT) return Promise.resolve({ data: [] })
  return request.get('/dashboard/trend', { params: { days } })
}

export default request

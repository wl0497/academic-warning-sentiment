<template>
  <div class="dashboard">
    <el-row :gutter="20" class="stat-cards">
      <el-col :span="6" v-for="card in statCards" :key="card.title">
        <el-card shadow="hover" class="stat-card" :style="{ borderTop: '3px solid ' + card.color }">
          <div class="stat-value" :style="{ color: card.color }">{{ card.value }}</div>
          <div class="stat-title">{{ card.title }}</div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :span="16">
        <el-card shadow="hover">
          <template #header><span>分析趋势（近7天）</span></template>
          <div ref="trendChartRef" style="width:100%;height:300px"></div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover">
          <template #header><span>情感分布</span></template>
          <div ref="pieChartRef" style="width:100%;height:300px"></div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header><span>学院预警分布</span></template>
          <div ref="barChartRef" style="width:100%;height:300px"></div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header><span>预警等级分布</span></template>
          <div ref="levelChartRef" style="width:100%;height:300px"></div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { getDashboardStats, getDashboardTrend } from '@/api'
import { Loading } from '@element-plus/icons-vue'

const trendChartRef = ref(null)
const pieChartRef = ref(null)
const barChartRef = ref(null)
const levelChartRef = ref(null)

let charts = []
let refreshTimer = null
const loading = ref(false)

const statCards = ref([
  { title: '分析总量', value: 0, color: '#409eff' },
  { title: '预警学生', value: 0, color: '#f56c6c' },
  { title: '已处理', value: 0, color: '#67c23a' },
  { title: '待处理', value: 0, color: '#e6a23c' }
])

function updateStatCards(data) {
  if (!data) return
  statCards.value = [
    { title: '分析总量', value: data.totalPredictions ?? 0, color: '#409eff' },
    { title: '预警学生', value: data.totalWarnings ?? 0, color: '#f56c6c' },
    { title: '已处理', value: data.handledWarnings ?? 0, color: '#67c23a' },
    { title: '待处理', value: data.pendingWarnings ?? 0, color: '#e6a23c' }
  ]
}

function updateTrendChart(trend) {
  if (!trendChartRef.value) return
  let chart = charts[0]
  if (!chart) { chart = echarts.init(trendChartRef.value); charts.push(chart) }

  const dates = trend ? trend.map(t => t.date) : []
  const counts = trend ? trend.map(t => t.count) : []

  chart.setOption({
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: dates, name: '日期' },
    yAxis: { type: 'value', name: '分析量' },
    series: [{
      type: 'bar',
      data: counts,
      itemStyle: { color: (p) => counts[p.dataIndex] > 0 ? '#409eff' : '#e0e0e0' },
      label: { show: true, position: 'top', formatter: '{c}' }
    }]
  }, true)
}

function updatePieChart(data) {
  if (!pieChartRef.value) return
  let chart = charts[1]
  if (!chart) { chart = echarts.init(pieChartRef.value); charts.push(chart) }

  const dist = data?.sentimentDistribution || { positive: 0, neutral: 0, negative: 0 }
  const hasData = dist.positive > 0 || dist.neutral > 0 || dist.negative > 0

  chart.setOption({
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { bottom: 10 },
    series: [{
      type: 'pie', radius: ['40%', '70%'], center: ['50%', '45%'],
      data: hasData ? [
        { value: dist.positive || 0, name: '正面', itemStyle: { color: '#67c23a' } },
        { value: dist.neutral || 0, name: '中性', itemStyle: { color: '#e6a23c' } },
        { value: dist.negative || 0, name: '负面', itemStyle: { color: '#f56c6c' } }
      ] : [{ value: 1, name: '暂无数据', itemStyle: { color: '#e0e0e0' } }],
      label: { formatter: '{b}\n{c} ({d}%)' }
    }]
  }, true)
}

function updateBarChart(data) {
  if (!barChartRef.value) return
  let chart = charts[2]
  if (!chart) { chart = echarts.init(barChartRef.value); charts.push(chart) }

  let colleges, posData, negData, neutralData, totalData
  const dist = data?.collegeDistribution || []
  const hasRealData = dist.length > 0 && dist.some(c => (c.total || 0) > 0)

  if (hasRealData) {
    colleges = dist.map(c => c.name)
    posData = dist.map(c => c.positive || 0)
    neutralData = dist.map(c => c.neutral || 0)
    negData = dist.map(c => c.negative || 0)
    totalData = dist.map(c => c.total || 0)
  } else {
    colleges = ['暂无学院数据，请先上传文件']
    posData = [0]
    neutralData = [0]
    negData = [0]
    totalData = [0]
  }

  chart.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: function(params) {
        const idx = params[0].dataIndex
        const college = hasRealData ? dist[idx].name : '暂无学院数据'
        const total = hasRealData ? (dist[idx].total || 0) : 0
        let html = '<b>' + college + '</b><br/>总人数: ' + total + '<br/>'
        params.forEach(p => {
          if (p.value > 0) html += p.marker + p.seriesName + ': ' + p.value + '<br/>'
        })
        return html
      }
    },
    legend: { data: hasRealData ? ['正面', '中性', '负面'] : [], bottom: 5 },
    grid: { left: '3%', right: '4%', bottom: '12%', top: '8%', containLabel: true },
    xAxis: {
      type: 'category',
      data: colleges,
      axisLabel: {
        rotate: colleges.length > 4 ? 25 : 0,
        fontSize: 11,
        interval: 0
      }
    },
    yAxis: { type: 'value', name: '人数', nameTextStyle: { padding: [0, 0, 0, 30] } },
    series: [
      {
        name: '正面',
        type: 'bar',
        stack: 'total',
        data: posData,
        itemStyle: { color: '#67c23a' },
        barMaxWidth: 50,
        label: { show: true, position: 'inside', formatter: function(p) { return p.value > 0 ? p.value : '' }, fontSize: 11 }
      },
      {
        name: '中性',
        type: 'bar',
        stack: 'total',
        data: neutralData,
        itemStyle: { color: '#e6a23c' },
        barMaxWidth: 50,
        label: { show: true, position: 'inside', formatter: function(p) { return p.value > 0 ? p.value : '' }, fontSize: 11 }
      },
      {
        name: '负面',
        type: 'bar',
        stack: 'total',
        data: negData,
        itemStyle: { color: '#f56c6c' },
        barMaxWidth: 50,
        label: { show: true, position: 'inside', formatter: function(p) { return p.value > 0 ? p.value : '' }, fontSize: 11 }
      }
    ]
  }, true)
}

function updateLevelChart(data) {
  if (!levelChartRef.value) return
  let chart = charts[3]
  if (!chart) { chart = echarts.init(levelChartRef.value); charts.push(chart) }

  const levelDist = data?.warningLevelDistribution || { normal: 0, attention: 0, alert: 0 }
  const hasData = levelDist.normal > 0 || levelDist.attention > 0 || levelDist.alert > 0

  chart.setOption({
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { bottom: 10 },
    series: [{
      type: 'pie', radius: '65%',
      data: hasData ? [
        { value: levelDist.normal || 0, name: '正常', itemStyle: { color: '#67c23a' } },
        { value: levelDist.attention || 0, name: '关注', itemStyle: { color: '#e6a23c' } },
        { value: levelDist.alert || 0, name: '预警', itemStyle: { color: '#f56c6c' } }
      ] : [{ value: 1, name: '暂无数据', itemStyle: { color: '#e0e0e0' } }],
      label: { formatter: '{b}: {c} ({d}%)' },
      emphasis: { itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0,0,0,.5)' } }
    }]
  }, true)
}

async function loadDashboardData() {
  loading.value = true
  try {
    const [statsResult, trendResult] = await Promise.allSettled([
      getDashboardStats(),
      getDashboardTrend(7)
    ])
    const stats = statsResult.status === 'fulfilled' ? statsResult.value.data : null
    const trend = trendResult.status === 'fulfilled' ? (trendResult.value.data || []) : []

    updateStatCards(stats)
    updateTrendChart(trend)
    updatePieChart(stats)
    updateBarChart(stats)
    updateLevelChart(stats)
  } catch (err) {
    console.error('Dashboard refresh error:', err)
  } finally {
    loading.value = false
  }
}

function handleResize() {
  charts.forEach(chart => chart && chart.resize())
}

onMounted(() => {
  loadDashboardData()
  refreshTimer = setInterval(loadDashboardData, 30000)
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  clearInterval(refreshTimer)
  window.removeEventListener('resize', handleResize)
  charts.forEach(chart => chart && chart.dispose())
})
</script>

<style scoped>
.dashboard { padding: 20px }
.stat-cards { margin-bottom: 0 }
.stat-card { text-align: center; margin-bottom: 0 }
.stat-value { font-size: 32px; font-weight: bold; line-height: 1.2 }
.stat-title { font-size: 14px; color: #666; margin-top: 8px }
</style>

<template>
  <div class="emotion-curve" ref="chartRef" style="width: 100%; height: 300px;"></div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  data: { type: Array, default: () => [] }
})

const chartRef = ref(null)
let chart = null

const initChart = () => {
  if (!chartRef.value) return
  chart = echarts.init(chartRef.value)
  updateChart()
}

const updateChart = () => {
  if (!chart) return
  const dates = props.data.map(d => d.date)
  const counts = props.data.map(d => d.count)

  chart.setOption({
    title: { text: '预测趋势', left: 'center', textStyle: { fontSize: 15 } },
    tooltip: { trigger: 'axis' },
    grid: { left: 50, right: 30, bottom: 30, top: 40 },
    xAxis: { type: 'category', data: dates, axisLabel: { fontSize: 11 } },
    yAxis: { type: 'value', name: '预测数', minInterval: 1 },
    series: [{
      type: 'line',
      data: counts,
      smooth: true,
      areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
        { offset: 0, color: 'rgba(64,158,255,0.3)' },
        { offset: 1, color: 'rgba(64,158,255,0.05)' }
      ])},
      lineStyle: { width: 2, color: '#409eff' },
      itemStyle: { color: '#409eff' }
    }]
  })
}

watch(() => props.data, updateChart, { deep: true })
onMounted(initChart)
</script>

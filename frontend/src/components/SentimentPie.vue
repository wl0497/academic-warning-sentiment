<template>
  <div class="sentiment-pie" ref="chartRef" style="width: 100%; height: 350px;"></div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  data: { type: Object, default: () => ({ positive: 0, neutral: 0, negative: 0 }) }
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
  chart.setOption({
    title: { text: '情感分布', left: 'center', textStyle: { fontSize: 15 } },
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { bottom: 10 },
    series: [{
      type: 'pie',
      radius: ['35%', '60%'],
      center: ['50%', '45%'],
      avoidLabelOverlap: true,
      itemStyle: { borderRadius: 6, borderColor: '#fff', borderWidth: 2 },
      label: { show: true, formatter: '{b}\n{d}%' },
      data: [
        { value: props.data.positive, name: '正面', itemStyle: { color: '#67c23a' } },
        { value: props.data.neutral, name: '中性', itemStyle: { color: '#e6a23c' } },
        { value: props.data.negative, name: '负面', itemStyle: { color: '#f56c6c' } }
      ]
    }]
  })
}

watch(() => props.data, updateChart, { deep: true })
onMounted(initChart)
</script>

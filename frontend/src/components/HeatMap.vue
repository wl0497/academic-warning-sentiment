<template>
  <div class="heat-map" ref="chartRef" style="width: 100%; height: 350px;"></div>
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
  // 模拟学业情感热力图：横轴=时间段，纵轴=学院/班级
  const hours = ['8:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00']
  const categories = ['计算机学院', '文学院', '理学院', '经管学院', '外语学院']

  // 生成模拟数据 [x, y, value]
  const heatData = []
  for (let i = 0; i < hours.length; i++) {
    for (let j = 0; j < categories.length; j++) {
      heatData.push([i, j, Math.round(Math.random() * 100)])
    }
  }

  chart.setOption({
    title: { text: '学业情感热力图', left: 'center', textStyle: { fontSize: 15 } },
    tooltip: {
      position: 'top',
      formatter: (p) => `${categories[p.value[1]]} ${hours[p.value[0]]}<br/>负面情感指数: ${p.value[2]}`
    },
    grid: { left: 100, right: 50, bottom: 40, top: 40 },
    xAxis: { type: 'category', data: hours, splitArea: { show: true } },
    yAxis: { type: 'category', data: categories, splitArea: { show: true } },
    visualMap: {
      min: 0, max: 100,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      inRange: { color: ['#67c23a', '#e6a23c', '#f56c6c'] }
    },
    series: [{
      type: 'heatmap',
      data: heatData,
      label: { show: true },
      emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } }
    }]
  })
}

watch(() => props.data, updateChart, { deep: true })
onMounted(initChart)
</script>

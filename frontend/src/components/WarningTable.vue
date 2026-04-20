<template>
  <div class="warning-table">
    <el-table :data="records" stripe style="width: 100%" max-height="400">
      <el-table-column prop="studentName" label="姓名" width="80" />
      <el-table-column prop="studentId" label="学号" width="120" />
      <el-table-column prop="className" label="班级" width="120" />
      <el-table-column prop="inputText" label="输入文本" min-width="200" show-overflow-tooltip />
      <el-table-column prop="warningLevel" label="预警级别" width="90">
        <template #default="{ row }">
          <el-tag :type="levelType(row.warningLevel)" size="small">{{ row.warningLevel }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="confidence" label="置信度" width="80">
        <template #default="{ row }">
          {{ row.confidence ? (row.confidence * 100).toFixed(1) + '%' : '-' }}
        </template>
      </el-table-column>
      <el-table-column prop="handled" label="状态" width="80">
        <template #default="{ row }">
          <el-tag :type="row.handled ? 'success' : 'danger'" size="small">
            {{ row.handled ? '已处理' : '待处理' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="createTime" label="时间" width="160" />
    </el-table>
    <el-pagination
      v-if="total > pageSize"
      layout="prev, pager, next"
      :total="total"
      :page-size="pageSize"
      @current-change="loadData"
      style="margin-top: 16px; text-align: right;"
    />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import api from '../api'

const records = ref([])
const total = ref(0)
const pageSize = 10

const loadData = async (page = 1) => {
  try {
    const res = await api.get('/warning/list', { params: { page, size: pageSize } })
    records.value = res.data.records
    total.value = res.data.total
  } catch (e) {
    console.error('加载预警列表失败', e)
  }
}

const levelType = (level) => {
  const map = { '正常': 'success', '关注': 'warning', '预警': 'danger' }
  return map[level] || 'info'
}

onMounted(() => loadData())
</script>

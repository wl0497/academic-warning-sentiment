<template>
  <div class="warning-page">
    <!-- 筛选栏 -->
    <el-card shadow="hover">
      <el-row :gutter="20" align="middle">
        <el-col :span="5">
          <el-select v-model="filterLevel" placeholder="预警级别" clearable @change="fetchData">
            <el-option label="正常" value="正常" />
            <el-option label="关注" value="关注" />
            <el-option label="预警" value="预警" />
          </el-select>
        </el-col>
        <el-col :span="5">
          <el-select v-model="filterHandled" placeholder="处理状态" clearable @change="fetchData">
            <el-option label="未处理" :value="0" />
            <el-option label="已处理" :value="1" />
          </el-select>
        </el-col>
        <el-col :span="4">
          <el-button type="primary" @click="fetchData">查询</el-button>
          <el-button type="danger" plain @click="confirmClearAll" style="margin-left: 8px;">
            <el-icon><Delete /></el-icon>
            清空记录
          </el-button>
        </el-col>
        <!-- 批量操作工具栏 -->
        <el-col :span="10" style="text-align: right;">
          <transition name="el-fade-in-linear">
            <span v-if="selected.length > 0" class="selection-info">
              <el-badge :value="selected.length" type="primary" class="badge">
                <span>已选中</span>
              </el-badge>
              <el-button type="danger" size="small" plain @click="confirmBatchDelete" style="margin-left: 12px;">
                <el-icon><Delete /></el-icon>
                批量删除
              </el-button>
              <el-button size="small" @click="clearSelection" style="margin-left: 4px;">取消</el-button>
            </span>
          </transition>
        </el-col>
      </el-row>
    </el-card>

    <!-- 数据表格 -->
    <el-card shadow="hover" style="margin-top: 20px;">
      <el-table
        ref="tableRef"
        :data="tableData"
        border
        stripe
        v-loading="loading"
        style="width: 100%;"
        @selection-change="onSelectionChange"
        :row-key="row => row.id"
      >
        <el-table-column type="selection" width="45" :reserve-selection="true" />
        <el-table-column prop="studentName" label="学生姓名" width="100" />
        <el-table-column prop="studentId" label="学号" width="130" />
        <el-table-column prop="college" label="学院" width="120" show-overflow-tooltip />
        <el-table-column prop="inputText" label="分析文本" show-overflow-tooltip />
        <el-table-column prop="label" label="情感标签" width="90" align="center">
          <template #default="{ row }">
            <el-tag :type="tagType(row.label)" size="small">{{ row.label }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="warningLevel" label="预警级别" width="90" align="center">
          <template #default="{ row }">
            <el-tag :type="levelTagType(row.warningLevel)" size="small" effect="dark">{{ row.warningLevel }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="置信度" width="80" align="center">
          <template #default="{ row }">{{ (row.confidence * 100).toFixed(1) }}%</template>
        </el-table-column>
        <el-table-column prop="handled" label="状态" width="80" align="center">
          <template #default="{ row }">
            <el-tag :type="row.handled === 1 ? 'success' : 'info'" size="small">{{ row.handled === 1 ? '已处理' : '待处理' }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="handler" label="处理人" width="100" show-overflow-tooltip />
        <el-table-column prop="handleRemark" label="处理备注" show-overflow-tooltip />
        <el-table-column label="操作" width="150" align="center" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" size="small" link @click="openHandleDialog(row)" :disabled="row.handled === 1">处理</el-button>
            <el-divider direction="vertical" />
            <el-button type="danger" size="small" link @click="confirmDelete(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div style="margin-top: 15px; display: flex; justify-content: flex-end;">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :total="total"
          :page-sizes="[10, 20, 50]"
          layout="total, sizes, prev, pager, next"
          @current-change="fetchData"
          @size-change="fetchData"
        />
      </div>
    </el-card>

    <!-- 处理对话框 -->
    <el-dialog v-model="dialogVisible" title="预警处理" width="500px">
      <el-form :model="handleForm" label-width="80px">
        <el-form-item label="学生">{{ currentRow?.studentName }} ({{ currentRow?.studentId }})</el-form-item>
        <el-form-item label="预警内容">
          <div class="warning-text">{{ currentRow?.inputText }}</div>
        </el-form-item>
        <el-form-item label="处理人">
          <el-input v-model="handleForm.handler" placeholder="请输入处理人姓名" />
        </el-form-item>
        <el-form-item label="处理备注">
          <el-input v-model="handleForm.remark" type="textarea" :rows="3" placeholder="请输入处理措施和备注" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitHandle" :loading="submitLoading">确认处理</el-button>
      </template>
    </el-dialog>

    <!-- 删除确认对话框 -->
    <el-dialog v-model="deleteDialogVisible" title="确认删除" width="420px">
      <div style="display: flex; align-items: center; gap: 12px;">
        <el-icon size="32" color="#f56c6c"><WarningFilled /></el-icon>
        <div>
          <p style="margin: 0 0 4px; font-size: 15px; font-weight: 600;">确定要删除以下记录吗？</p>
          <p v-if="deleteTarget === 'batch'" style="margin: 0; color: #909399; font-size: 13px;">
            已选择 <strong style="color: #f56c6c;">{{ selected.length }} 条</strong> 记录，删除后不可恢复。
          </p>
          <p v-else style="margin: 0; color: #909399; font-size: 13px;">
            学生：<strong>{{ currentRow?.studentName }}</strong>，删除后不可恢复。
          </p>
        </div>
      </div>
      <template #footer>
        <el-button @click="deleteDialogVisible = false">取消</el-button>
        <el-button type="danger" @click="doDelete" :loading="deleteLoading">确认删除</el-button>
      </template>
    </el-dialog>

    <!-- 清空所有确认对话框 -->
    <el-dialog v-model="clearDialogVisible" title="确认清空" width="420px">
      <div style="display: flex; align-items: center; gap: 12px;">
        <el-icon size="32" color="#f56c6c"><WarningFilled /></el-icon>
        <div>
          <p style="margin: 0 0 4px; font-size: 15px; font-weight: 600;">确定要清空所有预警记录吗？</p>
          <p style="margin: 0; color: #909399; font-size: 13px;">将永久删除 <strong style="color: #f56c6c;">{{ total }} 条</strong> 记录，此操作不可恢复！</p>
        </div>
      </div>
      <template #footer>
        <el-button @click="clearDialogVisible = false">取消</el-button>
        <el-button type="danger" @click="doClearAll" :loading="clearLoading">确认清空</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, WarningFilled } from '@element-plus/icons-vue'
import request from '@/api'

const tableRef = ref(null)
const tableData = ref([])
const loading = ref(false)
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)
const filterLevel = ref('')
const filterHandled = ref('')

const selected = ref([])
const deleteDialogVisible = ref(false)
const deleteTarget = ref('single') // 'single' or 'batch'
const deleteLoading = ref(false)

const clearDialogVisible = ref(false)
const clearLoading = ref(false)

const dialogVisible = ref(false)
const currentRow = ref(null)
const handleForm = ref({ handler: '', remark: '' })
const submitLoading = ref(false)

function tagType(label) {
  if (label === '正面') return 'success'
  if (label === '中性') return 'warning'
  return 'danger'
}

function levelTagType(level) {
  if (level === '正常') return 'success'
  if (level === '关注') return 'warning'
  return 'danger'
}

async function fetchData() {
  loading.value = true
  try {
    const params = { page: currentPage.value, size: pageSize.value }
    if (filterLevel.value) params.level = filterLevel.value
    if (filterHandled.value !== '') params.handled = filterHandled.value
    const res = await request.get('/warning', { params })
    tableData.value = res.data?.records || []
    total.value = res.data?.total || 0
  } catch (err) {
    ElMessage.error('获取数据失败')
  } finally {
    loading.value = false
  }
}

function onSelectionChange(rows) {
  selected.value = rows
}

function clearSelection() {
  tableRef.value?.clearSelection()
}

function confirmDelete(row) {
  currentRow.value = row
  deleteTarget.value = 'single'
  deleteDialogVisible.value = true
}

function confirmBatchDelete() {
  deleteTarget.value = 'batch'
  deleteDialogVisible.value = true
}

function confirmClearAll() {
  clearDialogVisible.value = true
}

async function doClearAll() {
  clearLoading.value = true
  try {
    const res = await request.delete('/warning/clear')
    const cleared = res.data?.cleared || 0
    ElMessage.success('已清空 ' + cleared + ' 条记录')
    clearDialogVisible.value = false
    fetchData()
  } catch (err) {
    ElMessage.error('清空失败')
  } finally {
    clearLoading.value = false
  }
}

async function doDelete() {
  deleteLoading.value = true
  try {
    if (deleteTarget.value === 'batch') {
      const ids = selected.value.map(r => r.id)
      await request.delete('/warning/batch', { params: { ids: ids.join(',') } })
      ElMessage.success(`成功删除 ${selected.value.length} 条记录`)
      clearSelection()
    } else {
      await request.delete(`/warning/${currentRow.value.id}`)
      ElMessage.success('删除成功')
    }
    deleteDialogVisible.value = false
    fetchData()
  } catch (err) {
    ElMessage.error('删除失败')
  } finally {
    deleteLoading.value = false
  }
}

function openHandleDialog(row) {
  currentRow.value = row
  handleForm.value = { handler: '', remark: '' }
  dialogVisible.value = true
}

async function submitHandle() {
  if (!handleForm.value.handler) {
    ElMessage.warning('请填写处理人')
    return
  }
  submitLoading.value = true
  try {
    await request.post(`/warning/${currentRow.value.id}/handle`, {
      handler: handleForm.value.handler,
      handleRemark: handleForm.value.remark
    })
    ElMessage.success('处理成功')
    dialogVisible.value = false
    fetchData()
  } catch (err) {
    ElMessage.error('处理失败')
  } finally {
    submitLoading.value = false
  }
}

onMounted(() => { fetchData() })
</script>

<style scoped>
.warning-text {
  background: #f5f7fa;
  padding: 10px;
  border-radius: 4px;
  color: #606266;
  font-size: 13px;
  max-height: 100px;
  overflow-y: auto;
}
.selection-info {
  display: inline-flex;
  align-items: center;
}
.badge { line-height: 1; }
</style>

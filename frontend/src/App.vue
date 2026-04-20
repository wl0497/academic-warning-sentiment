<template>
  <el-container class="app-container">
    <!-- 侧边栏 -->
    <el-aside width="220px" class="aside">
      <div class="logo">
        <h2>🎓 学业预警系统</h2>
        <p>Bert-BiLSTM-CNN</p>
      </div>
      <el-menu
        :default-active="activeMenu"
        router
        background-color="#304156"
        text-color="#bfcbd9"
        active-text-color="#409eff"
      >
        <el-menu-item index="/">
          <span>📊 数据概览</span>
        </el-menu-item>
        <el-menu-item index="/predict">
          <span>🔍 情感分析</span>
        </el-menu-item>
        <el-menu-item index="/batch">
          <span>📁 批量分析</span>
        </el-menu-item>
        <el-menu-item index="/warning">
          <span>⚠️ 预警管理</span>
        </el-menu-item>
        <el-menu-item index="/model">
          <span>🧠 模型信息</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <!-- 主内容区 -->
    <el-container>
      <el-header class="header">
        <span class="title">高校大学生学业预警情感分析系统</span>
        <span class="subtitle">淮南师范学院 · 省级创新训练项目</span>
      </el-header>
      <el-main class="main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()
const activeMenu = computed(() => route.path)
</script>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; font-family: 'Microsoft YaHei', sans-serif; }

.app-container { height: 100vh; }

.aside {
  background-color: #304156;
  overflow-y: auto;
}
.logo {
  text-align: center;
  padding: 20px 10px;
  color: #fff;
  border-bottom: 1px solid #3d4d5e;
}
.logo h2 { font-size: 16px; margin-bottom: 5px; }
.logo p { font-size: 11px; color: #8b9ab5; }

.header {
  background: #fff;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 1px 4px rgba(0,21,41,.08);
  padding: 0 20px;
}
.title { font-size: 18px; font-weight: 600; color: #303133; }
.subtitle { font-size: 12px; color: #909399; }

.main {
  background: #f0f2f5;
  padding: 20px;
  overflow-y: auto;
}
</style>

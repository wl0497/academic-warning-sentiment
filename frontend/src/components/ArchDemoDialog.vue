<template>
  <!-- 触发按钮 -->
  <div class="demo-trigger">
    <el-button type="primary" size="large" @click="openDialog">
      <el-icon><Cpu /></el-icon>
      打开架构演示窗口
    </el-button>
    <div class="demo-hint">交互式动画演示 · 可自定义输入文本</div>
  </div>

  <!-- 演示窗口 -->
  <el-dialog
    v-model="dialogVisible"
    title=""
    :close-on-click-modal="true"
    width="960px"
    top="2vh"
    class="arch-demo-dialog"
    :before-close="onClose"
  >
    <template #header>
      <div class="dialog-header">
        <span class="dialog-title">🔬 BERT-BiLSTM-CNN 模型推理流程演示</span>
        <div class="dialog-subtitle">逐步动画 · 看清每一步数据流转</div>
      </div>
    </template>

    <div class="demo-container">
      <!-- 左侧：控制面板 -->
      <div class="control-panel">
        <!-- 输入文本 -->
        <div class="ctrl-section">
          <div class="ctrl-label">📝 输入文本 <span class="input-hint">(可随时修改)</span></div>
          <el-input
            v-model="inputText"
            type="textarea"
            :rows="3"
            placeholder="输入待分析的评论文本..."
            maxlength="200"
            show-word-limit
          />
        </div>

        <!-- 播放控制 -->
        <div class="ctrl-section">
          <div class="ctrl-label">🎮 播放控制</div>
          <div class="playback-controls">
            <el-button-group>
              <el-button @click="resetDemo" :disabled="currentStep === 0">
                <el-icon><RefreshLeft /></el-icon> 重置
              </el-button>
              <el-button @click="prevStep" :disabled="currentStep === 0">
                <el-icon><ArrowLeft /></el-icon> 上一步
              </el-button>
              <el-button type="primary" @click="togglePlay">
                <el-icon><VideoPlay v-if="!isPlaying" /><VideoPause v-else /></el-icon>
                {{ isPlaying ? '暂停' : '播放' }}
              </el-button>
              <el-button @click="nextStep" :disabled="currentStep >= TOTAL_STEPS">
                下一步 <el-icon><ArrowRight /></el-icon>
              </el-button>
            </el-button-group>
          </div>
        </div>

        <!-- 步骤进度 -->
        <div class="ctrl-section">
          <div class="ctrl-label">📍 当前步骤</div>
          <div class="step-indicator">
            <div
              v-for="(s, i) in steps"
              :key="i"
              class="step-dot"
              :class="{ active: currentStep === i + 1, done: currentStep > i + 1 }"
              @click="goToStep(i + 1)"
            >
              <div class="step-dot-inner">{{ i + 1 }}</div>
              <div class="step-dot-label">{{ s.short }}</div>
            </div>
          </div>
        </div>

        <!-- 步骤详情 -->
        <div class="ctrl-section step-detail" v-if="currentStep > 0">
          <div class="ctrl-label">{{ steps[currentStep - 1].title }}</div>
          <div class="step-desc-text">{{ steps[currentStep - 1].desc }}</div>
          <div class="step-tech">{{ steps[currentStep - 1].tech }}</div>
        </div>
        <div class="ctrl-section step-detail" v-else>
          <div class="ctrl-label">💡 开始演示</div>
          <div class="step-desc-text">在左上角输入文本，然后点击「播放」或「下一步」开始演示</div>
        </div>

        <!-- 结果展示 -->
        <div class="ctrl-section result-section" v-if="currentStep >= 5">
          <div class="ctrl-label">📊 预测结果</div>
          <div class="result-bars">
            <div class="result-row" v-for="r in results" :key="r.label">
              <span class="result-label" :style="{ color: r.color }">{{ r.label }}</span>
              <div class="result-bar-bg">
                <div class="result-bar-fill" :style="{ width: r.pct + '%', background: r.color }"></div>
              </div>
              <span class="result-pct">{{ r.pct.toFixed(1) }}%</span>
            </div>
          </div>
          <div class="result-pred">
            <el-tag :type="results[predictedIdx].tag" size="large" effect="dark">
              预测：{{ results[predictedIdx].label }}
            </el-tag>
          </div>
        </div>
      </div>

      <!-- 右侧：流水线动画 -->
      <div class="pipeline-panel">
        <svg class="pipeline-svg" viewBox="0 0 600 520" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
              <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <filter id="nodeGlow">
              <feGaussianBlur stdDeviation="6" result="blur"/>
              <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
          </defs>

          <!-- 数据粒子 -->
          <g class="data-particles">
            <circle v-for="(p, i) in particles" :key="p.id"
              :cx="p.x" :cy="p.y" r="5"
              :fill="p.color" :opacity="p.opacity"
              filter="url(#glow)"/>
          </g>

          <!-- 连接线 -->
          <g class="connectors">
            <line x1="80" y1="60" x2="280" y2="60" stroke-width="3"
              :stroke="lineColor(1)" class="connector-line"/>
            <line x1="320" y1="60" x2="140" y2="200" stroke-width="3"
              :stroke="lineColor(3)" class="connector-line"/>
            <line x1="320" y1="60" x2="500" y2="200" stroke-width="3"
              :stroke="lineColor(3)" class="connector-line"/>
            <line x1="140" y1="310" x2="140" y2="385" stroke-width="3"
              :stroke="lineColor(4)" class="connector-line"/>
            <line x1="500" y1="310" x2="500" y2="385" stroke-width="3"
              :stroke="lineColor(4)" class="connector-line"/>
            <line x1="320" y1="435" x2="320" y2="490" stroke-width="3"
              :stroke="lineColor(5)" class="connector-line"/>
          </g>

          <!-- 用户输入显示 -->
          <foreignObject x="0" y="0" width="600" height="0" class="user-text-display">
          </foreignObject>

          <!-- 节点：文本输入 -->
          <g :class="['node-group', { 'node-active': currentStep >= 1 }]"
             transform="translate(30, 30)">
            <rect width="120" height="60" rx="10"
              :fill="nodeBg(1)" stroke-width="2"
              :stroke="nodeBorder(1)" filter="url(#glow)"/>
            <text x="60" y="22" text-anchor="middle" class="node-icon">📝</text>
            <text x="60" y="40" text-anchor="middle" class="node-title">原始文本</text>
            <text x="60" y="54" text-anchor="middle" class="node-sub">Step 1</text>
          </g>

          <!-- 节点：BERT -->
          <g :class="['node-group', { 'node-active': currentStep >= 2 }]"
             transform="translate(250, 30)">
            <rect width="140" height="60" rx="10"
              :fill="nodeBg(2)" stroke-width="2"
              :stroke="nodeBorder(2)" filter="url(#glow)"/>
            <text x="70" y="22" text-anchor="middle" class="node-icon">🧠</text>
            <text x="70" y="40" text-anchor="middle" class="node-title">BERT 编码</text>
            <text x="70" y="54" text-anchor="middle" class="node-sub">冻结8层 · 768维</text>
          </g>

          <!-- 节点：BiLSTM -->
          <g :class="['node-group', { 'node-active': currentStep >= 3 }]"
             transform="translate(50, 200)">
            <rect width="180" height="110" rx="12"
              :fill="nodeBg(3)" stroke-width="2"
              :stroke="nodeBorder(3)" filter="url(#glow)"/>
            <text x="90" y="22" text-anchor="middle" class="node-icon">🔄</text>
            <text x="90" y="42" text-anchor="middle" class="node-title">BiLSTM</text>
            <text x="90" y="58" text-anchor="middle" class="node-sub">时序建模 · 64×2维</text>
            <line x1="15" y1="68" x2="165" y2="68" stroke-width="1" stroke="#f3d19e" stroke-dasharray="4,2"/>
            <text x="90" y="83" text-anchor="middle" class="node-sub">← 正向 + 反向 →</text>
            <text x="90" y="98" text-anchor="middle" class="node-tag">捕获词序依赖</text>
          </g>

          <!-- 节点：CNN -->
          <g :class="['node-group', { 'node-active': currentStep >= 3 }]"
             transform="translate(420, 200)">
            <rect width="180" height="110" rx="12"
              :fill="nodeBg(3)" stroke-width="2"
              :stroke="nodeBorder(3)" filter="url(#glow)"/>
            <text x="90" y="22" text-anchor="middle" class="node-icon">🪟</text>
            <text x="90" y="42" text-anchor="middle" class="node-title">CNN</text>
            <text x="90" y="58" text-anchor="middle" class="node-sub">局部特征 · 64×3维</text>
            <line x1="15" y1="68" x2="165" y2="68" stroke-width="1" stroke="#fab6b6" stroke-dasharray="4,2"/>
            <text x="90" y="83" text-anchor="middle" class="node-sub">卷积核 [2, 3, 4]</text>
            <text x="90" y="98" text-anchor="middle" class="node-tag">捕获n-gram模式</text>
          </g>

          <!-- 节点：特征融合 -->
          <g :class="['node-group', { 'node-active': currentStep >= 4 }]"
             transform="translate(240, 385)">
            <rect width="160" height="50" rx="10"
              :fill="nodeBg(4)" stroke-width="2"
              :stroke="nodeBorder(4)" filter="url(#glow)"/>
            <text x="80" y="20" text-anchor="middle" class="node-icon">⚡</text>
            <text x="80" y="38" text-anchor="middle" class="node-title">特征融合</text>
          </g>

          <!-- 节点：Softmax输出 -->
          <g :class="['node-group', { 'node-active': currentStep >= 5 }]"
             transform="translate(240, 490)">
          </g>
        </svg>

        <!-- 用户输入气泡（显示在文本节点下方） -->
        <div class="user-text-bubble" :class="{ 'bubble-active': currentStep >= 1 }">
          <span class="bubble-text">"{{ displayText }}"</span>
        </div>

        <!-- 底部输出结果条 -->
        <transition name="el-fade-in">
          <div class="pipeline-output" v-if="currentStep >= 5">
            <div class="output-row">
              <div class="output-class neg" :class="{ 'output-pred': predictedIdx === 0 }">
                <div class="output-icon">😠</div>
                <div class="output-name">负面</div>
                <div class="output-bar"><div class="out-bar neg-bar" :style="{ width: results[0].pct + '%' }"></div></div>
                <div class="output-pct">{{ results[0].pct.toFixed(1) }}%</div>
              </div>
              <div class="output-class neu" :class="{ 'output-pred': predictedIdx === 1 }">
                <div class="output-icon">😐</div>
                <div class="output-name">中性</div>
                <div class="output-bar"><div class="out-bar neu-bar" :style="{ width: results[1].pct + '%' }"></div></div>
                <div class="output-pct">{{ results[1].pct.toFixed(1) }}%</div>
              </div>
              <div class="output-class pos" :class="{ 'output-pred': predictedIdx === 2 }">
                <div class="output-icon">😊</div>
                <div class="output-name">正面</div>
                <div class="output-bar"><div class="out-bar pos-bar" :style="{ width: results[2].pct + '%' }"></div></div>
                <div class="output-pct">{{ results[2].pct.toFixed(1) }}%</div>
              </div>
            </div>
          </div>
        </transition>

        <!-- 当前步骤标签 -->
        <div class="step-badge" v-if="currentStep > 0">
          {{ steps[currentStep - 1]?.short }}
        </div>
      </div>
    </div>
  </el-dialog>
</template>

<script setup>
import { ref, computed, watch, onUnmounted } from 'vue'
import {
  Cpu, VideoPlay, VideoPause, ArrowLeft, ArrowRight,
  RefreshLeft
} from '@element-plus/icons-vue'

const TOTAL_STEPS = 5
let firstOpen = true

const dialogVisible = ref(false)
const inputText = ref('')
const currentStep = ref(0)
const isPlaying = ref(false)
let timer = null
let animFrame = null
let particleId = 0

const particles = ref([])

function generateResults(text) {
  // 根据用户输入的文本情感倾向生成更真实的结果
  const positive = ['好', '优秀', '棒', '赞', '喜欢', '满意', '有收获', '有帮助'].filter(w => text.includes(w)).length
  const negative = ['难', '差', '听不懂', '焦虑', '压力', '放弃', '担心', '挂'].filter(w => text.includes(w)).length
  const neu = 3 - Math.max(positive, negative)

  const base = [
    Math.max(1, 2 + negative * 15),
    Math.max(1, 45 + neu * 10),
    Math.max(1, 50 + positive * 15)
  ]
  const jitter = () => (Math.random() - 0.5) * 6
  const raw = base.map((v, i) => Math.max(0.1, v + jitter()))
  const total = raw.reduce((a, b) => a + b, 0)
  return raw.map((v, i) => ({
    label: ['负面', '中性', '正面'][i],
    color: ['#f56c6c', '#e6a23c', '#67c23a'][i],
    tag: ['danger', 'warning', 'success'][i],
    pct: (v / total) * 100,
  }))
}

const results = ref(generateResults('默认正面文本'))

const predictedIdx = computed(() => {
  return results.value.reduce((maxIdx, r, idx, arr) =>
    r.pct > arr[maxIdx].pct ? idx : maxIdx, 0)
})

// 用户输入文本的显示截取
const displayText = computed(() => {
  const t = inputText.value || '请输入文本...'
  return t.length > 20 ? t.substring(0, 20) + '...' : t
})

// 动态步骤数据（使用computed使tech字段反映用户输入）
const steps = computed(() => [
  {
    short: '文本输入',
    title: 'Step 1 · 文本输入',
    desc: '将待分析的评论文本作为输入，可以是社交评论、评教反馈等任意中文文本。',
    tech: `输入文本: "${inputText.value || '（请在上方输入）'}"`
  },
  {
    short: 'BERT编码',
    title: 'Step 2 · BERT深层语义编码',
    desc: 'BERT将中文文本编码为768维的语义向量表示，同时捕获上下文信息。冻结前8层保留预训练知识，微调后6层学习任务特征。',
    tech: '输出: [batch, seq_len, 768] 向量'
  },
  {
    short: '双路提取',
    title: 'Step 3 · 双路特征提取（并行）',
    desc: 'BERT输出同时送入BiLSTM和CNN两路并行处理：BiLSTM捕获时序依赖，CNN捕获n-gram局部特征。',
    tech: 'BiLSTM: 64×2=128维 | CNN: 64×3=192维'
  },
  {
    short: '特征融合',
    title: 'Step 4 · 特征融合 + 分类',
    desc: '双路特征拼接后经融合层降维(320→128)，加入Dropout(0.5)防止过拟合，最后通过全连接层输出。',
    tech: '融合: 320维 → ReLU → Dropout(0.5) → 128维'
  },
  {
    short: '结果输出',
    title: 'Step 5 · Softmax分类输出',
    desc: '通过Softmax函数将输出转换为概率分布，选择概率最大的类别作为预测结果。',
    tech: `预测: ${results.value[predictedIdx.value].label} (${results.value[predictedIdx.value].pct.toFixed(1)}%)`
  },
])

// ==================== 控制逻辑 ====================
function openDialog() {
  dialogVisible.value = true
  if (firstOpen) {
    inputText.value = ''  // 首次打开为空，用户自己输入
    firstOpen = false
  }
  // 不重置inputText，允许保留上次输入
  // 重置演示步骤，但不重置用户文本
  currentStep.value = 0
  isPlaying.value = false
  results.value = generateResults(inputText.value || '默认正面文本')
  if (timer) { clearInterval(timer); timer = null }
}

function onClose() {
  stopPlay()
  dialogVisible.value = false
}

function togglePlay() {
  if (isPlaying.value) {
    stopPlay()
  } else {
    if (currentStep.value >= TOTAL_STEPS) {
      currentStep.value = 0
      results.value = generateResults(inputText.value || '默认正面文本')
    }
    isPlaying.value = true
    timer = setInterval(() => {
      if (currentStep.value < TOTAL_STEPS) {
        currentStep.value++
        spawnParticle()
        spawnParticle()
      } else {
        stopPlay()
      }
    }, 2200)
  }
}

function stopPlay() {
  isPlaying.value = false
  if (timer) { clearInterval(timer); timer = null }
}

function resetDemo() {
  stopPlay()
  currentStep.value = 0
  particles.value = []
  results.value = generateResults(inputText.value || '默认正面文本')
}

function prevStep() {
  if (currentStep.value > 0) {
    currentStep.value--
  }
}

function nextStep() {
  if (currentStep.value < TOTAL_STEPS) {
    currentStep.value++
    spawnParticle()
    spawnParticle()
  }
}

function goToStep(n) {
  stopPlay()
  currentStep.value = n
  for (let i = 0; i < 3; i++) setTimeout(() => spawnParticle(), i * 200)
}

// ==================== 视觉函数 ====================
function lineColor(minStep) {
  if (currentStep.value >= minStep + 1) return '#409eff'
  if (currentStep.value === minStep) return '#79bbff'
  return '#e4e7ed'
}

function nodeBg(step) {
  if (currentStep.value < step) return 'rgba(230,230,240,0.4)'
  const colors = ['#ecf5ff', '#f0f9eb', '#fffbf0', '#f4f4f5', '#e8f5e9']
  return colors[step - 1] || '#ecf5ff'
}

function nodeBorder(step) {
  if (currentStep.value < step) return '#dcdfe6'
  const borders = ['#409eff', '#67c23a', '#e6a23c', '#909399', '#67c23a']
  return borders[step - 1] || '#409eff'
}

// 监听inputText变化，更新结果
watch(inputText, (newText) => {
  if (currentStep.value > 0) {
    // 重置演示，重新生成结果
    results.value = generateResults(newText || '默认正面文本')
  }
})

// 监听currentStep变化，更新结果（当进入第5步时重新生成）
watch(currentStep, (newStep) => {
  results.value = generateResults(inputText.value || '默认正面文本')
})

// 粒子动画
function spawnParticle() {
  const paths = [
    { from: [80, 60], to: [280, 60], color: '#409eff' },
    { from: [320, 60], to: [140, 200], color: '#e6a23c' },
    { from: [320, 60], to: [500, 200], color: '#f56c6c' },
    { from: [140, 310], to: [140, 385], color: '#e6a23c' },
    { from: [500, 310], to: [500, 385], color: '#f56c6c' },
    { from: [320, 435], to: [320, 490], color: '#67c23a' },
  ]
  const activePaths = paths.filter((_, i) => {
    if (i === 0) return currentStep.value >= 1
    if (i <= 2) return currentStep.value >= 2
    if (i <= 4) return currentStep.value >= 3
    return currentStep.value >= 4
  })
  if (activePaths.length === 0) return

  const p = activePaths[Math.floor(Math.random() * activePaths.length)]
  const id = ++particleId
  const part = { ...p, opacity: 1, id, x: p.from[0], y: p.from[1] }
  particles.value.push(part)

  const dur = 800 + Math.random() * 400
  const startTime = Date.now()
  const [x1, y1] = p.from
  const [x2, y2] = p.to

  function animate() {
    if (!dialogVisible.value) return
    const elapsed = Date.now() - startTime
    const t = Math.min(elapsed / dur, 1)
    const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t
    const px = x1 + (x2 - x1) * ease
    const py = y1 + (y2 - y1) * ease
    const opacity = t < 0.7 ? 1 : 1 - (t - 0.7) / 0.3

    const idx = particles.value.findIndex(pt => pt.id === id)
    if (idx >= 0) {
      particles.value[idx] = { ...particles.value[idx], x: px, y: py, opacity }
    }
    if (t < 1) {
      animFrame = requestAnimationFrame(animate)
    } else {
      particles.value = particles.value.filter(pt => pt.id !== id)
    }
  }
  animFrame = requestAnimationFrame(animate)
}

onUnmounted(() => {
  stopPlay()
  if (animFrame) cancelAnimationFrame(animFrame)
})
</script>

<style scoped>
.demo-trigger { text-align: center; padding: 20px; }
.demo-trigger .el-button { font-size: 16px; padding: 20px 40px; }
.demo-hint { margin-top: 10px; color: #909399; font-size: 13px; }

:deep(.arch-demo-dialog .el-dialog__header) {
  padding: 0 20px 10px;
  border-bottom: 1px solid #f0f0f0;
}
.dialog-header { text-align: center; }
.dialog-title { font-size: 18px; font-weight: 700; color: #303133; }
.dialog-subtitle { font-size: 13px; color: #909399; margin-top: 4px; }

.demo-container {
  display: flex; gap: 20px; height: 580px;
}

/* 左侧控制面板 */
.control-panel {
  width: 280px; flex-shrink: 0;
  display: flex; flex-direction: column; gap: 12px; overflow-y: auto;
}
.ctrl-section { background: #f9fafb; border-radius: 10px; padding: 12px; }
.ctrl-label {
  font-size: 13px; font-weight: 700; color: #606266; margin-bottom: 8px;
}
.input-hint { font-size: 11px; color: #409eff; font-weight: 400; }

.step-detail { background: #f0f9ff; border: 1px solid #b3d8ff; }
.step-desc-text { font-size: 12px; color: #303133; line-height: 1.7; }
.step-tech {
  margin-top: 6px; font-size: 11px; color: #409eff;
  background: rgba(64,158,255,0.08); padding: 4px 8px;
  border-radius: 4px; font-family: monospace; word-break: break-all;
}

.playback-controls .el-button-group .el-button { padding: 8px 12px; }

/* 步骤指示器 */
.step-indicator { display: flex; gap: 6px; justify-content: space-between; }
.step-dot {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; cursor: pointer; transition: all 0.3s;
}
.step-dot-inner {
  width: 28px; height: 28px; border-radius: 50%;
  background: #e4e7ed; color: #909399;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; transition: all 0.4s;
}
.step-dot.active .step-dot-inner {
  background: #409eff; color: #fff;
  box-shadow: 0 0 12px rgba(64,158,255,0.5); transform: scale(1.1);
}
.step-dot.done .step-dot-inner { background: #67c23a; color: #fff; }
.step-dot-label { font-size: 9px; color: #909399; margin-top: 3px; text-align: center; }
.step-dot.active .step-dot-label { color: #409eff; font-weight: 600; }

/* 结果 */
.result-section { background: #f0f9eb; border: 1px solid #b3e19d; }
.result-bars { display: flex; flex-direction: column; gap: 8px; }
.result-row { display: flex; align-items: center; gap: 8px; font-size: 12px; }
.result-label { width: 32px; font-weight: 700; }
.result-bar-bg { flex: 1; height: 12px; background: #f0f0f0; border-radius: 6px; overflow: hidden; }
.result-bar-fill { height: 100%; border-radius: 6px; transition: width 1.5s ease; }
.result-pct { width: 40px; text-align: right; font-weight: 600; }
.result-pred { text-align: center; margin-top: 10px; }

/* 右侧流水线 */
.pipeline-panel {
  flex: 1; position: relative;
  background: linear-gradient(180deg, #fafbfc 0%, #f0f2f7 100%);
  border-radius: 12px; border: 1px solid #e4e7ed; overflow: hidden;
}
.pipeline-svg { width: 100%; height: 460px; }

/* SVG节点 */
.node-group rect { transition: all 0.5s ease; }
.node-title { font-size: 13px; font-weight: 700; fill: #303133; font-family: sans-serif; }
.node-icon { font-size: 16px; font-family: sans-serif; }
.node-sub { font-size: 10px; fill: #909399; font-family: sans-serif; }
.node-tag { font-size: 9px; fill: #e6a23c; font-family: sans-serif; font-weight: 600; }

.node-active rect {
  filter: url(#nodeGlow);
  animation: nodeGlow 2s ease-in-out infinite;
}
@keyframes nodeGlow {
  0%, 100% { filter: url(#nodeGlow) brightness(1); }
  50% { filter: url(#nodeGlow) brightness(1.15); }
}

.connector-line { transition: stroke 0.5s ease; }

/* 用户输入气泡 */
.user-text-bubble {
  position: absolute;
  top: 92px; left: 35px; right: 35px;
  text-align: center;
  background: #ecf5ff;
  border: 1px solid #b3d8ff;
  border-radius: 8px;
  padding: 4px 12px;
  font-size: 12px;
  color: #303133;
  opacity: 0;
  transform: translateY(-5px);
  transition: all 0.4s ease;
  pointer-events: none;
}
.user-text-bubble.bubble-active {
  opacity: 1;
  transform: translateY(0);
}
.bubble-text {
  font-style: italic;
  color: #409eff;
  font-weight: 600;
  word-break: break-all;
}

/* 底部输出 */
.pipeline-output {
  position: absolute; bottom: 0; left: 0; right: 0;
  background: rgba(255,255,255,0.95);
  border-top: 2px solid #67c23a;
  padding: 10px 16px;
}
.output-row { display: flex; gap: 10px; justify-content: center; }
.output-class {
  flex: 1; max-width: 160px;
  display: flex; flex-direction: column; align-items: center;
  gap: 4px; padding: 6px; border-radius: 8px;
  border: 2px solid transparent; transition: all 0.4s;
}
.output-class.output-pred { border-color: currentColor; transform: scale(1.05); }
.neg { color: #f56c6c; }
.neu { color: #e6a23c; }
.pos { color: #67c23a; }
.output-icon { font-size: 20px; }
.output-name { font-size: 12px; font-weight: 700; }
.output-bar { width: 100%; height: 8px; background: #f0f0f0; border-radius: 4px; overflow: hidden; }
.out-bar { height: 100%; border-radius: 4px; transition: width 1.5s ease; }
.neg-bar { background: linear-gradient(90deg, #f56c6c, #fab6b6); }
.neu-bar { background: linear-gradient(90deg, #e6a23c, #f3d19e); }
.pos-bar { background: linear-gradient(90deg, #67c23a, #b3e19d); }
.output-pct { font-size: 12px; font-weight: 700; }

/* 步骤徽章 */
.step-badge {
  position: absolute; top: 10px; right: 12px;
  background: #409eff; color: #fff;
  font-size: 12px; font-weight: 700;
  padding: 4px 12px; border-radius: 20px;
  box-shadow: 0 2px 8px rgba(64,158,255,0.3);
}
</style>

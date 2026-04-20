import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'Dashboard', component: () => import('@/views/Dashboard.vue') },
  { path: '/predict', name: 'Predict', component: () => import('@/views/Predict.vue') },
  { path: '/batch', name: 'BatchAnalysis', component: () => import('@/views/BatchAnalysis.vue') },
  { path: '/warning', name: 'Warning', component: () => import('@/views/WarningList.vue') },
  { path: '/model', name: 'ModelInfo', component: () => import('@/views/ModelInfo.vue') }
]

export default createRouter({
  history: createWebHistory(),
  routes
})

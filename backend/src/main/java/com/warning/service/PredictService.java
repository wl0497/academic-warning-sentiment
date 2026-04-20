package com.warning.service;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.warning.entity.PredictResult;
import com.warning.entity.WarningRecord;
import com.warning.mapper.PredictResultMapper;
import com.warning.mapper.WarningRecordMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.http.converter.StringHttpMessageConverter;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 预测服务
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class PredictService {

    private final PredictResultMapper predictResultMapper;
    private final WarningRecordMapper warningRecordMapper;

    // Flask服务地址
    private static final String DEFAULT_MODEL_URL = "http://localhost:5000";
    private String modelServiceUrl = DEFAULT_MODEL_URL;

    public void setModelServiceUrl(String url) {
        this.modelServiceUrl = url;
    }

    private RestTemplate createRestTemplate() {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(5000);
        factory.setReadTimeout(30000);
        RestTemplate template = new RestTemplate(factory);
        template.getMessageConverters().add(
            new StringHttpMessageConverter(StandardCharsets.UTF_8));
        return template;
    }

    /**
     * 单条文本预测
     */
    public Map<String, Object> predict(String text) {
        String url = modelServiceUrl + "/predict";
        
        RestTemplate restTemplate = createRestTemplate();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        
        JSONObject body = new JSONObject();
        body.put("text", text);
        
        HttpEntity<String> request = new HttpEntity<>(body.toJSONString(), headers);
        
        try {
            ResponseEntity<String> response = restTemplate.exchange(
                url, HttpMethod.POST, request, String.class);
            
            Map<String, Object> result = JSON.parseObject(response.getBody(), Map.class);
            
            // 保存预测结果到数据库
            savePredictResult(text, result);
            
            return result;
        } catch (HttpClientErrorException e) {
            log.error("Flask API调用失败: {}", e.getMessage());
            Map<String, Object> fallback = new HashMap<>();
            fallback.put("label", "中性");
            fallback.put("confidence", 0.5);
            fallback.put("error", "模型服务暂不可用");
            return fallback;
        }
    }

    /**
     * 批量文本预测
     */

    /**
     * 带置信度的单条预测（供PredictController使用）
     */
    public Map<String, Object> predictWithProbabilities(String text) {
        return predict(text);
    }
    public Map<String, Object> predictBatch(List<String> texts) {
        String url = modelServiceUrl + "/predict/batch";
        
        RestTemplate restTemplate = createRestTemplate();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        
        JSONObject body = new JSONObject();
        body.put("texts", texts);
        
        HttpEntity<String> request = new HttpEntity<>(body.toJSONString(), headers);
        
        try {
            ResponseEntity<String> response = restTemplate.exchange(
                url, HttpMethod.POST, request, String.class);
            
            Map<String, Object> result = JSON.parseObject(response.getBody(), Map.class);
            
            // 保存预测结果到数据库
            List<Map<String, Object>> results = (List<Map<String, Object>>) result.get("results");
            if (results != null) {
                for (int i = 0; i < results.size(); i++) {
                    Map<String, Object> r = results.get(i);
                    savePredictResult(texts.get(i), r);
                }
            }
            
            return result;
        } catch (Exception e) {
            log.error("批量预测失败: {}", e.getMessage());
            Map<String, Object> error = new HashMap<>();
            error.put("error", "批量预测失败: " + e.getMessage());
            return error;
        }
    }

    /**
     * 批量预测（带学生信息，自动创建预警记录）
     * records中的字段：studentId, name, college, className, comment
     */
    public Map<String, Object> predictBatchWithRecords(List<String> texts, List<Map<String, String>> records) {
        String url = modelServiceUrl + "/predict/batch";
        
        RestTemplate restTemplate = createRestTemplate();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        
        JSONObject body = new JSONObject();
        body.put("texts", texts);
        
        HttpEntity<String> request = new HttpEntity<>(body.toJSONString(), headers);
        
        ResponseEntity<String> response = restTemplate.exchange(
            url, HttpMethod.POST, request, String.class);
        
        Map<String, Object> result = JSON.parseObject(response.getBody(), Map.class);
        
        // 自动将负面/预警结果写入预警管理（包含学生信息）
        List<Map<String, Object>> results = (List<Map<String, Object>>) result.get("results");
        int warningCount = 0;
        if (results != null && records != null && results.size() == records.size()) {
            for (int i = 0; i < results.size(); i++) {
                Map<String, Object> item = results.get(i);
                Map<String, String> record = records.get(i);
                
                // 【关键修复】将Excel记录中的学生信息合并到结果中
                item.put("name", record.getOrDefault("name", "未知"));
                item.put("college", record.getOrDefault("college", "未知"));
                item.put("className", record.getOrDefault("className", "未知"));
                
                String label = (String) item.get("label");
                Map<String, Object> warning = (Map<String, Object>) item.get("warning");
                String level = warning != null ? (String) warning.get("level") : "正常";
                
                // 保存每条预测结果到数据库
                savePredictResult(record.get("comment"), item);
                
                // 【修改】所有情感都创建预警记录（用于统计学院分布）
                // 注意：这里不判断label类型，统一创建
                {
                    try {
                        WarningRecord wr = new WarningRecord();
                        // 截断超长字段
                        String name = record.getOrDefault("name", "未知");
                        if (name.length() > 50) name = name.substring(0, 50);
                        wr.setStudentName(name);
                        
                        String studentId = record.getOrDefault("studentId", "BATCH_" + System.currentTimeMillis() + "_" + warningCount);
                        if (studentId.length() > 50) studentId = studentId.substring(0, 50);
                        wr.setStudentId(studentId);
                        
                        // 学院字段（新增）
                        String college = record.getOrDefault("college", "未知");
                        if (college.length() > 100) college = college.substring(0, 100);
                        wr.setCollege(college);
                        
                        String className = record.getOrDefault("className", "未知");
                        if (className.length() > 100) className = className.substring(0, 100);
                        wr.setClassName(className);
                        
                        String comment = record.getOrDefault("comment", "");
                        if (comment.length() > 500) comment = comment.substring(0, 500);
                        wr.setInputText(comment);
                        wr.setLabel(label);
                        wr.setConfidence(((Number) item.get("confidence")).doubleValue());
                        wr.setWarningLevel(level);
                        wr.setSuggestion(warning != null ? (String) warning.get("suggestion") : "建议关注");
                        wr.setHandled(0);
                        wr.setCreateTime(LocalDateTime.now());
                        warningRecordMapper.insert(wr);
                        warningCount++;
                        
                        // 将预警ID写回结果
                        item.put("warningId", wr.getId());
                    } catch (Exception e) {
                        log.error("创建预警记录失败: " + e.getMessage());
                    }
                }
            }
        }
        result.put("warningCreated", warningCount);
        
        return result;
    }

    /**
     * 保存单条预测结果到数据库
     */
    private void savePredictResult(String text, Map<String, Object> result) {
        try {
            PredictResult pr = new PredictResult();
            pr.setText(text.length() > 500 ? text.substring(0, 500) : text);
            pr.setLabel((String) result.get("label"));
            
            // 处理概率分布
            Map<String, Object> probabilities = (Map<String, Object>) result.get("probabilities");
            if (probabilities != null) {
                pr.setPositiveProb(((Number) probabilities.getOrDefault("正面", 0)).doubleValue());
                pr.setNeutralProb(((Number) probabilities.getOrDefault("中性", 0)).doubleValue());
                pr.setNegativeProb(((Number) probabilities.getOrDefault("负面", 0)).doubleValue());
            }
            
            pr.setConfidence(((Number) result.getOrDefault("confidence", 0)).doubleValue());
            
            Map<String, Object> warning = (Map<String, Object>) result.get("warning");
            if (warning != null) {
                pr.setWarningLevel((String) warning.get("level"));
                pr.setWarningColor((String) warning.get("color"));
                pr.setSuggestion((String) warning.get("suggestion"));
            }
            
            pr.setCreateTime(LocalDateTime.now());
            predictResultMapper.insert(pr);
        } catch (Exception e) {
            log.warn("保存预测结果失败（不影响主流程）: " + e.getMessage());
        }
    }

    /**
     * 获取预测统计
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("total", predictResultMapper.selectCount(null));
        return stats;
    }
}

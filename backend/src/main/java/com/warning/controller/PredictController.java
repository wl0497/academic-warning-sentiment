package com.warning.controller;

import com.warning.entity.PredictResult;
import com.warning.service.PredictService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * 预测控制器 — RESTful API
 */
@RestController
@RequestMapping("/api/predict")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class PredictController {

    private final PredictService predictService;

    /**
     * 单条文本预测
     * POST /api/predict
     */
    @PostMapping
    public ResponseEntity<Map<String, Object>> predict(@RequestBody Map<String, String> request) {
        String text = request.get("text");
        if (text == null || text.trim().isEmpty()) {
            return ResponseEntity.badRequest().build();
        }
        Map<String, Object> result = predictService.predictWithProbabilities(text);
        return ResponseEntity.ok(result);
    }

    /**
     * 批量文本预测
     * POST /api/predict/batch
     */
    @PostMapping("/batch")
    public ResponseEntity<Map<String, Object>> predictBatch(@RequestBody Map<String, List<String>> request) {
        List<String> texts = request.get("texts");
        if (texts == null || texts.isEmpty()) {
            return ResponseEntity.badRequest().build();
        }
        Map<String, Object> result = predictService.predictBatch(texts);
        return ResponseEntity.ok(result);
    }

    /**
     * 获取统计概览（大屏数据）
     * GET /api/predict/statistics
     */
    @GetMapping("/statistics")
    public ResponseEntity<Map<String, Object>> getStatistics() {
        Map<String, Object> stats = predictService.getStatistics();
        return ResponseEntity.ok(stats);
    }
}

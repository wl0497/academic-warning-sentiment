package com.warning.controller;

import com.warning.service.WarningService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * 仪表盘数据控制器（大屏数据聚合）
 */
@RestController
@RequestMapping("/api/dashboard")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class DashboardController {

    private final WarningService warningService;

    /**
     * 获取仪表盘统计
     * GET /api/dashboard/stats
     */
    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getStats() {
        return ResponseEntity.ok(warningService.getDashboardStats());
    }

    /**
     * 获取近期趋势
     * GET /api/dashboard/trend?days=7
     */
    @GetMapping("/trend")
    public ResponseEntity<List<Map<String, Object>>> getTrend(
            @RequestParam(defaultValue = "7") Integer days) {
        return ResponseEntity.ok(warningService.getRecentTrend(days));
    }
}

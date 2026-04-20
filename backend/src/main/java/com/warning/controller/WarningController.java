package com.warning.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.warning.entity.WarningRecord;
import com.warning.mapper.WarningRecordMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 预警管理控制器
 */
@RestController
@RequestMapping("/api/warning")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class WarningController {

    private final WarningRecordMapper warningRecordMapper;

    /**
     * 创建预警记录
     * POST /api/warning
     */
    @PostMapping("")
    public ResponseEntity<WarningRecord> create(@RequestBody WarningRecord record) {
        warningRecordMapper.insert(record);
        return ResponseEntity.ok(record);
    }

    /**
     * 分页查询预警记录
     * GET /api/warning?page=1&size=10&level=预警&handled=0
     */
    @GetMapping("")
    public ResponseEntity<Page<WarningRecord>> list(
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String level,
            @RequestParam(required = false) Integer handled) {
        
        Page<WarningRecord> pageParam = new Page<>(page, size);
        LambdaQueryWrapper<WarningRecord> wrapper = new LambdaQueryWrapper<>();
        
        if (level != null && !level.isEmpty()) {
            wrapper.eq(WarningRecord::getWarningLevel, level);
        }
        if (handled != null) {
            wrapper.eq(WarningRecord::getHandled, handled);
        }
        wrapper.orderByDesc(WarningRecord::getCreateTime);
        
        Page<WarningRecord> result = warningRecordMapper.selectPage(pageParam, wrapper);
        return ResponseEntity.ok(result);
    }

    /**
     * 处理预警记录
     * POST /api/warning/{id}/handle
     */
    @PostMapping("/{id}/handle")
    public ResponseEntity<String> handle(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {

        WarningRecord record = warningRecordMapper.selectById(id);
        if (record == null) {
            return ResponseEntity.notFound().build();
        }

        record.setHandled(1);
        record.setHandler((String) request.get("handler"));
        record.setHandleRemark((String) request.get("handleRemark"));
        warningRecordMapper.updateById(record);

        return ResponseEntity.ok("处理成功");
    }

    /**
     * 删除单条预警记录
     * DELETE /api/warning/{id}
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<String> deleteOne(@PathVariable Long id) {
        WarningRecord record = warningRecordMapper.selectById(id);
        if (record == null) {
            return ResponseEntity.notFound().build();
        }
        warningRecordMapper.deleteById(id);
        return ResponseEntity.ok("删除成功");
    }

    /**
     * 批量删除预警记录
     * DELETE /api/warning/batch?ids=1,2,3
     */
    @DeleteMapping("/batch")
    public ResponseEntity<Map<String, Object>> deleteBatch(@RequestParam List<Long> ids) {
        if (ids == null || ids.isEmpty()) {
            Map<String, Object> resp = new HashMap<>();
            resp.put("message", "请选择要删除的记录");
            resp.put("deleted", 0);
            return ResponseEntity.ok(resp);
        }
        int deleted = warningRecordMapper.deleteBatchIds(ids);
        Map<String, Object> resp = new HashMap<>();
        resp.put("message", "删除成功");
        resp.put("deleted", deleted);
        return ResponseEntity.ok(resp);
    }

    /**
     * 清空所有预警记录
     * DELETE /api/warning/clear
     */
    @DeleteMapping("/clear")
    public ResponseEntity<Map<String, Object>> clearAll() {
        long before = warningRecordMapper.selectCount(null);
        warningRecordMapper.delete(null);
        Map<String, Object> resp = new HashMap<>();
        resp.put("message", "已清空 " + before + " 条记录");
        resp.put("cleared", before);
        return ResponseEntity.ok(resp);
    }

    /**
     * 预警统计（大屏）
     * GET /api/warning/stats
     */
    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> stats() {
        Long total = warningRecordMapper.selectCount(null);
        Long pending = warningRecordMapper.selectCount(
            new LambdaQueryWrapper<WarningRecord>().eq(WarningRecord::getHandled, 0)
        );
        Long handled = warningRecordMapper.selectCount(
            new LambdaQueryWrapper<WarningRecord>().eq(WarningRecord::getHandled, 1)
        );
        
        Map<String, Object> stats = new HashMap<>();
        stats.put("total", total);
        stats.put("pending", pending);
        stats.put("handled", handled);
        return ResponseEntity.ok(stats);
    }
}

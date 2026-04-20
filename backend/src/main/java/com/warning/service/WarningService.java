package com.warning.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.warning.entity.PredictResult;
import com.warning.entity.WarningRecord;
import com.warning.mapper.PredictResultMapper;
import com.warning.mapper.WarningRecordMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.*;

@Service
@RequiredArgsConstructor
public class WarningService {
    private final WarningRecordMapper warningRecordMapper;
    private final PredictResultMapper predictResultMapper;
    private final JdbcTemplate jdbcTemplate;

    public Map<String, Object> getDashboardStats() {
        Map<String, Object> stats = new LinkedHashMap<>();

        long totalPredictions = predictResultMapper.selectCount(null);
        stats.put("totalPredictions", totalPredictions);

        long totalWarnings = warningRecordMapper.selectCount(null);
        long pendingWarnings = warningRecordMapper.selectCount(
            new LambdaQueryWrapper<WarningRecord>().eq(WarningRecord::getHandled, 0));
        long handledWarnings = warningRecordMapper.selectCount(
            new LambdaQueryWrapper<WarningRecord>().eq(WarningRecord::getHandled, 1));
        stats.put("totalWarnings", totalWarnings);
        stats.put("pendingWarnings", pendingWarnings);
        stats.put("handledWarnings", handledWarnings);

        long positiveCount = predictResultMapper.selectCount(
            new LambdaQueryWrapper<PredictResult>().eq(PredictResult::getLabel, "正面"));
        long neutralCount = predictResultMapper.selectCount(
            new LambdaQueryWrapper<PredictResult>().eq(PredictResult::getLabel, "中性"));
        long negativeCount = predictResultMapper.selectCount(
            new LambdaQueryWrapper<PredictResult>().eq(PredictResult::getLabel, "负面"));
        Map<String, Object> sentimentDist = new LinkedHashMap<>();
        sentimentDist.put("positive", positiveCount);
        sentimentDist.put("neutral", neutralCount);
        sentimentDist.put("negative", negativeCount);
        stats.put("sentimentDistribution", sentimentDist);

        long normalCount = predictResultMapper.selectCount(
            new LambdaQueryWrapper<PredictResult>().eq(PredictResult::getWarningLevel, "正常"));
        long attentionCount = predictResultMapper.selectCount(
            new LambdaQueryWrapper<PredictResult>().eq(PredictResult::getWarningLevel, "关注"));
        long alertCount = predictResultMapper.selectCount(
            new LambdaQueryWrapper<PredictResult>().eq(PredictResult::getWarningLevel, "预警"));
        Map<String, Object> warningDist = new LinkedHashMap<>();
        warningDist.put("normal", normalCount);
        warningDist.put("attention", attentionCount);
        warningDist.put("alert", alertCount);
        stats.put("warningLevelDistribution", warningDist);

        stats.put("collegeDistribution", getCollegeDistribution());
        return stats;
    }

    private List<Map<String, Object>> getCollegeDistribution() {
        List<Map<String, Object>> dist = new ArrayList<>();
        try {
            String sql = "SELECT college, COUNT(*) as total, " +
                "SUM(CASE WHEN label='正面' THEN 1 ELSE 0 END) as positive, " +
                "SUM(CASE WHEN label='中性' THEN 1 ELSE 0 END) as neutral, " +
                "SUM(CASE WHEN label='负面' THEN 1 ELSE 0 END) as negative, " +
                "SUM(CASE WHEN handled=0 THEN 1 ELSE 0 END) as pending " +
                "FROM warning_record WHERE college IS NOT NULL AND college<>'' AND college<>'未知' " +
                "GROUP BY college ORDER BY total DESC";
            List<Map<String, Object>> rows = jdbcTemplate.queryForList(sql);
            for (Map<String, Object> row : rows) {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("name", row.get("COLLEGE"));
                item.put("total", ((Number)row.get("TOTAL")).intValue());
                item.put("positive", ((Number)row.get("POSITIVE")).intValue());
                item.put("neutral", ((Number)row.get("NEUTRAL")).intValue());
                item.put("negative", ((Number)row.get("NEGATIVE")).intValue());
                item.put("pending", ((Number)row.get("PENDING")).intValue());
                dist.add(item);
            }
        } catch (Exception e) {
        }
        return dist;
    }

    public List<Map<String, Object>> getRecentTrend(int days) {
        List<Map<String, Object>> trend = new ArrayList<>();
        LocalDateTime now = LocalDateTime.now();
        for (int i = days - 1; i >= 0; i--) {
            LocalDateTime dayStart = now.minusDays(i).toLocalDate().atStartOfDay();
            LocalDateTime dayEnd = dayStart.plusDays(1);
            long count = predictResultMapper.selectCount(
                new LambdaQueryWrapper<PredictResult>().ge(PredictResult::getCreateTime, dayStart)
                    .lt(PredictResult::getCreateTime, dayEnd));
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("date", dayStart.toLocalDate().toString());
            item.put("count", count);
            trend.add(item);
        }
        return trend;
    }

    public List<WarningRecord> getWarningList(Integer handled, String keyword, int page, int size) {
        LambdaQueryWrapper<WarningRecord> w = new LambdaQueryWrapper<>();
        if (handled != null) w.eq(WarningRecord::getHandled, handled);
        if (keyword != null && !keyword.isEmpty()) {
            w.and(q -> q
                .like(WarningRecord::getStudentName, keyword)
                .or().like(WarningRecord::getStudentId, keyword)
                .or().like(WarningRecord::getCollege, keyword)
                .or().like(WarningRecord::getClassName, keyword)
                .or().like(WarningRecord::getInputText, keyword));
        }
        w.orderByDesc(WarningRecord::getCreateTime);
        w.last("LIMIT " + size + " OFFSET " + (page - 1) * size);
        return warningRecordMapper.selectList(w);
    }

    public long countWarningList(Integer handled, String keyword) {
        LambdaQueryWrapper<WarningRecord> w = new LambdaQueryWrapper<>();
        if (handled != null) w.eq(WarningRecord::getHandled, handled);
        if (keyword != null && !keyword.isEmpty()) {
            w.and(q -> q
                .like(WarningRecord::getStudentName, keyword)
                .or().like(WarningRecord::getStudentId, keyword)
                .or().like(WarningRecord::getCollege, keyword)
                .or().like(WarningRecord::getClassName, keyword)
                .or().like(WarningRecord::getInputText, keyword));
        }
        return warningRecordMapper.selectCount(w);
    }

    public void handleWarning(Long id, String handler, String remark) {
        WarningRecord wr = warningRecordMapper.selectById(id);
        if (wr != null) {
            wr.setHandled(1);
            wr.setHandler(handler);
            wr.setHandleRemark(remark);
            wr.setUpdateTime(LocalDateTime.now());
            warningRecordMapper.updateById(wr);
        }
    }

    public boolean deleteWarning(Long id) {
        return warningRecordMapper.deleteById(id) > 0;
    }

    public int deleteWarnings(List<Long> ids) {
        return warningRecordMapper.deleteBatchIds(ids);
    }
}

package com.warning.controller;

import com.warning.service.PredictService;
import lombok.RequiredArgsConstructor;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * 批量分析控制器
 */
@RestController
@RequestMapping("/api/batch")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class BatchController {

    private final PredictService predictService;

    /**
     * 批量文本预测
     * POST /api/batch/predict
     * Body: {"texts": ["文本1", "文本2", ...]}
     */
    @PostMapping("/predict")
    public ResponseEntity<Map<String, Object>> batchPredict(
            @RequestBody Map<String, List<String>> request) {
        List<String> texts = request.get("texts");
        if (texts == null || texts.isEmpty()) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "文本列表不能为空");
            return ResponseEntity.badRequest().body(error);
        }
        if (texts.size() > 100) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "单次最多100条文本");
            return ResponseEntity.badRequest().body(error);
        }
        Map<String, Object> result = predictService.predictBatch(texts);
        return ResponseEntity.ok(result);
    }

    /**
     * Excel/CSV文件上传批量预测
     * 支持5列格式：学号、姓名、学院、班级、评论
     * 也兼容4列格式：学号、姓名、班级、评论
     * POST /api/batch/upload
     */
    @PostMapping("/upload")
    public ResponseEntity<Map<String, Object>> batchUpload(
            @RequestParam("file") MultipartFile file) {
        if (file.isEmpty()) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "文件不能为空");
            return ResponseEntity.badRequest().body(error);
        }

        try {
            String filename = file.getOriginalFilename();
            List<Map<String, String>> records = new ArrayList<>();
            
            if (filename != null && (filename.endsWith(".xlsx") || filename.endsWith(".xls"))) {
                records = parseExcel(file.getInputStream(), filename.endsWith(".xlsx"));
            } else {
                records = parseCsv(file.getInputStream());
            }

            if (records.isEmpty()) {
                Map<String, Object> error = new HashMap<>();
                error.put("error", "文件中未找到有效数据");
                return ResponseEntity.badRequest().body(error);
            }

            // 提取评论文本进行预测
            List<String> texts = new ArrayList<>();
            for (Map<String, String> r : records) {
                texts.add(r.get("comment"));
            }

            // 调用预测服务（包含预警自动创建）
            Map<String, Object> result = predictService.predictBatchWithRecords(texts, records);
            result.put("fileCount", records.size());
            return ResponseEntity.ok(result);

        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "文件解析失败: " + e.getMessage());
            return ResponseEntity.internalServerError().body(error);
        }
    }

    /**
     * 解析Excel文件
     * 支持格式：
     *   5列：学号(col0) 姓名(col1) 学院(col2) 班级(col3) 评论(col4)
     *   4列：学号(col0) 姓名(col1) 班级(col2) 评论(col3)  [旧格式兼容]
     */
    private List<Map<String, String>> parseExcel(InputStream inputStream, boolean isXlsx) throws Exception {
        List<Map<String, String>> records = new ArrayList<>();
        Workbook workbook = isXlsx ? new XSSFWorkbook(inputStream) : new HSSFWorkbook(inputStream);
        Sheet sheet = workbook.getSheetAt(0);
        
        boolean isFirstRow = true;
        for (Row row : sheet) {
            if (isFirstRow) {
                isFirstRow = false;
                continue; // 跳过表头
            }
            
            Map<String, String> record = new HashMap<>();
            
            // 读取前5列，先尝试5列格式
            String col0 = getCellValue(row.getCell(0));
            String col1 = getCellValue(row.getCell(1));
            String col2 = getCellValue(row.getCell(2));
            String col3 = getCellValue(row.getCell(3));
            String col4 = getCellValue(row.getCell(4));
            
            if (col4.isEmpty() && !col3.isEmpty()) {
                // 4列格式：学号、姓名、班级、评论
                record.put("studentId", col0.isEmpty() ? "BATCH_" + System.currentTimeMillis() + "_" + records.size() : col0);
                record.put("name", col1.isEmpty() ? "未知" : col1);
                record.put("college", "未知");
                record.put("className", col2.isEmpty() ? "未知" : col2);
                record.put("comment", col3);
            } else {
                // 5列格式：学号、姓名、学院、班级、评论
                record.put("studentId", col0.isEmpty() ? "BATCH_" + System.currentTimeMillis() + "_" + records.size() : col0);
                record.put("name", col1.isEmpty() ? "未知" : col1);
                record.put("college", col2.isEmpty() ? "未知" : col2);
                record.put("className", col3.isEmpty() ? "未知" : col3);
                record.put("comment", col4);
            }
            
            if (!record.get("comment").isEmpty()) {
                records.add(record);
            }
            
            if (records.size() >= 200) break;
        }
        
        workbook.close();
        return records;
    }

    /**
     * 获取单元格值（统一转为字符串）
     */
    private String getCellValue(Cell cell) {
        if (cell == null) return "";
        switch (cell.getCellType()) {
            case STRING:
                return cell.getStringCellValue().trim();
            case NUMERIC:
                if (DateUtil.isCellDateFormatted(cell)) {
                    return cell.getDateCellValue().toString();
                }
                double num = cell.getNumericCellValue();
                if (num == Math.floor(num)) {
                    return String.valueOf((long) num);
                }
                return String.valueOf(num);
            case BOOLEAN:
                return String.valueOf(cell.getBooleanCellValue());
            case FORMULA:
                return cell.getCellFormula();
            default:
                return "";
        }
    }

    /**
     * 解析CSV文件
     * 支持格式：
     *   5列：学号,姓名,学院,班级,评论
     *   4列：学号,姓名,班级,评论  [旧格式兼容]
     */
    private List<Map<String, String>> parseCsv(InputStream inputStream) throws Exception {
        List<Map<String, String>> records = new ArrayList<>();
        BufferedReader reader = new BufferedReader(
            new InputStreamReader(inputStream, StandardCharsets.UTF_8));
        String line;
        boolean isFirstLine = true;

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;

            if (isFirstLine) {
                isFirstLine = false;
                continue;
            }

            String[] parts = parseCsvLine(line);
            Map<String, String> record = new HashMap<>();
            
            if (parts.length >= 5) {
                // 5列格式：学号、姓名、学院、班级、评论
                record.put("studentId", parts[0].trim());
                record.put("name", parts[1].trim());
                record.put("college", parts[2].trim());
                record.put("className", parts[3].trim());
                record.put("comment", parts[4].trim());
            } else if (parts.length >= 4) {
                // 4列格式：学号、姓名、班级、评论
                record.put("studentId", parts[0].trim());
                record.put("name", parts[1].trim());
                record.put("college", "未知");
                record.put("className", parts[2].trim());
                record.put("comment", parts[3].trim());
            } else if (parts.length >= 1) {
                // 只有评论列
                record.put("name", "未知");
                record.put("college", "未知");
                record.put("className", "未知");
                record.put("comment", parts[parts.length - 1].trim());
            }
            
            if (!record.get("comment").isEmpty()) {
                records.add(record);
            }

            if (records.size() >= 200) break;
        }
        reader.close();
        return records;
    }
    
    /**
     * 解析CSV行（处理引号包裹的字段）
     */
    private String[] parseCsvLine(String line) {
        List<String> fields = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;
        
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                fields.add(current.toString());
                current = new StringBuilder();
            } else {
                current.append(c);
            }
        }
        fields.add(current.toString());
        
        return fields.toArray(new String[0]);
    }
}
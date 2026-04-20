package com.warning.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * 预警记录实体
 */
@Data
@TableName("warning_record")
public class WarningRecord {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    /** 学生姓名 */
    private String studentName;
    
    /** 学号 */
    private String studentId;
    
    /** 学院 */
    private String college;
    
    /** 班级 */
    private String className;
    
    /** 输入文本 */
    private String inputText;
    
    /** 预测标签: 正面/中性/负面 */
    private String label;
    
    /** 置信度 */
    private Double confidence;
    
    /** 预警级别: 正常/关注/预警 */
    private String warningLevel;
    
    /** 干预建议 */
    private String suggestion;
    
    /** 是否已处理: 0-未处理, 1-已处理 */
    private Integer handled;
    
    /** 处理人 */
    private String handler;
    
    /** 处理备注 */
    private String handleRemark;
    
    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;
    
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;
}

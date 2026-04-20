package com.warning.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

/**
 * 预测结果实体
 */
@Data
@TableName("predict_result")
public class PredictResult {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    /** 输入文本 */
    private String text;
    
    /** 预测标签 */
    private String label;
    
    /** 正面概率 */
    private Double positiveProb;
    
    /** 中性概率 */
    private Double neutralProb;
    
    /** 负面概率 */
    private Double negativeProb;
    
    /** 置信度 */
    private Double confidence;
    
    /** 预警级别 */
    private String warningLevel;
    
    /** 预警颜色 */
    private String warningColor;
    
    /** 干预建议 */
    private String suggestion;
    
    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;
}

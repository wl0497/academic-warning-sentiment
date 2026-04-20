-- ============================================================
-- 学业预警情感分析系统 — 数据库初始化脚本
-- 数据库: MySQL 8.0+
-- ============================================================

CREATE DATABASE IF NOT EXISTS academic_warning
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE academic_warning;

-- 1. 预警记录表
DROP TABLE IF EXISTS `warning_record`;
CREATE TABLE `warning_record` (
    `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT '主键',
    `student_name` VARCHAR(50) DEFAULT NULL COMMENT '学生姓名',
    `student_id` VARCHAR(30) DEFAULT NULL COMMENT '学号',
    `college` VARCHAR(100) DEFAULT NULL COMMENT '学院',
    `class_name` VARCHAR(100) DEFAULT NULL COMMENT '班级',
    `input_text` TEXT COMMENT '输入文本',
    `label` VARCHAR(20) DEFAULT NULL COMMENT '预测标签: 正面/中性/负面',
    `confidence` DOUBLE DEFAULT NULL COMMENT '置信度',
    `warning_level` VARCHAR(20) DEFAULT NULL COMMENT '预警级别: 正常/关注/预警',
    `suggestion` VARCHAR(500) DEFAULT NULL COMMENT '干预建议',
    `handled` TINYINT DEFAULT 0 COMMENT '是否已处理: 0-未处理, 1-已处理',
    `handler` VARCHAR(50) DEFAULT NULL COMMENT '处理人',
    `handle_remark` VARCHAR(500) DEFAULT NULL COMMENT '处理备注',
    `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`),
    KEY `idx_student_id` (`student_id`),
    KEY `idx_warning_level` (`warning_level`),
    KEY `idx_handled` (`handled`),
    KEY `idx_create_time` (`create_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='预警记录表';

-- 2. 预测结果表
DROP TABLE IF EXISTS `predict_result`;
CREATE TABLE `predict_result` (
    `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT '主键',
    `text` TEXT COMMENT '输入文本',
    `label` VARCHAR(20) DEFAULT NULL COMMENT '预测标签',
    `positive_prob` DOUBLE DEFAULT NULL COMMENT '正面概率',
    `neutral_prob` DOUBLE DEFAULT NULL COMMENT '中性概率',
    `negative_prob` DOUBLE DEFAULT NULL COMMENT '负面概率',
    `confidence` DOUBLE DEFAULT NULL COMMENT '置信度',
    `warning_level` VARCHAR(20) DEFAULT NULL COMMENT '预警级别',
    `warning_color` VARCHAR(20) DEFAULT NULL COMMENT '预警颜色',
    `suggestion` VARCHAR(500) DEFAULT NULL COMMENT '干预建议',
    `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    KEY `idx_label` (`label`),
    KEY `idx_create_time` (`create_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='预测结果表';

-- 3. 用户信息表
DROP TABLE IF EXISTS `sys_user`;
CREATE TABLE `sys_user` (
    `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT '主键',
    `username` VARCHAR(50) NOT NULL COMMENT '用户名',
    `password` VARCHAR(100) NOT NULL COMMENT '密码',
    `real_name` VARCHAR(50) DEFAULT NULL COMMENT '真实姓名',
    `role` VARCHAR(20) DEFAULT 'user' COMMENT '角色: admin/teacher/user',
    `college` VARCHAR(100) DEFAULT NULL COMMENT '学院',
    `phone` VARCHAR(20) DEFAULT NULL COMMENT '手机号',
    `status` TINYINT DEFAULT 1 COMMENT '状态: 0-禁用, 1-启用',
    `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户信息表';

-- 4. 插入默认管理员
INSERT INTO `sys_user` (`username`, `password`, `real_name`, `role`) VALUES
('admin', 'e10adc3949ba59abbe56e057f20f883e', '系统管理员', 'admin');
-- 默认密码: 123456 (MD5)

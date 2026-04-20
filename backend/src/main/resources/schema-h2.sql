-- ============================================================
-- 学业预警情感分析系统 — H2数据库初始化脚本 (开发环境)
-- ============================================================

-- 1. 预警记录表
DROP TABLE IF EXISTS warning_record;
CREATE TABLE warning_record (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    student_name VARCHAR(50) DEFAULT NULL,
    student_id VARCHAR(30) DEFAULT NULL,
    college VARCHAR(100) DEFAULT NULL,
    class_name VARCHAR(100) DEFAULT NULL,
    input_text TEXT,
    label VARCHAR(20) DEFAULT NULL,
    confidence DOUBLE DEFAULT NULL,
    warning_level VARCHAR(20) DEFAULT NULL,
    suggestion VARCHAR(500) DEFAULT NULL,
    handled TINYINT DEFAULT 0,
    handler VARCHAR(50) DEFAULT NULL,
    handle_remark VARCHAR(500) DEFAULT NULL,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 预测结果表
DROP TABLE IF EXISTS predict_result;
CREATE TABLE predict_result (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    text TEXT,
    label VARCHAR(20) DEFAULT NULL,
    positive_prob DOUBLE DEFAULT NULL,
    neutral_prob DOUBLE DEFAULT NULL,
    negative_prob DOUBLE DEFAULT NULL,
    confidence DOUBLE DEFAULT NULL,
    warning_level VARCHAR(20) DEFAULT NULL,
    warning_color VARCHAR(20) DEFAULT NULL,
    suggestion VARCHAR(500) DEFAULT NULL,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. 用户信息表
DROP TABLE IF EXISTS sys_user;
CREATE TABLE sys_user (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(100) NOT NULL,
    real_name VARCHAR(50) DEFAULT NULL,
    role VARCHAR(20) DEFAULT 'user',
    college VARCHAR(100) DEFAULT NULL,
    phone VARCHAR(20) DEFAULT NULL,
    status TINYINT DEFAULT 1,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. 插入默认管理员 (密码: 123456 MD5)
INSERT INTO sys_user (username, password, real_name, role) VALUES
('admin', 'e10adc3949ba59abbe56e057f20f883e', '系统管理员', 'admin');

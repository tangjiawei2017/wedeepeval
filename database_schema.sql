-- 数据集生成平台数据库表结构设计（简化版）
-- 基于流程图设计的核心数据库架构

-- 核心任务表（包含所有必要信息）
CREATE TABLE generation_tasks
(
    id               BIGINT PRIMARY KEY AUTO_INCREMENT,
    task_name        VARCHAR(100) NOT NULL COMMENT '任务名称',
    generation_type  ENUM('document', 'context', 'topic', 'augment') NOT NULL COMMENT '生成方式',
    status           ENUM('pending', 'running', 'completed', 'failed', 'cancelled') DEFAULT 'pending' COMMENT '任务状态',
    total_items      INT       DEFAULT 0 COMMENT '总生成数量',
    completed_items  INT       DEFAULT 0 COMMENT '已完成数量',
    -- 生成结果路径
    output_file_path VARCHAR(500) NULL COMMENT '输出文件路径',
    preview          TEXT NULL COMMENT '结果预览（<=50字符，否则截断加...）',
    -- 时间戳
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    -- 错误信息
    error_message    VARCHAR(1000) NULL COMMENT '错误信息'
);

-- 创建索引
CREATE INDEX idx_tasks_status ON generation_tasks (status);
CREATE INDEX idx_tasks_type ON generation_tasks (generation_type);
CREATE INDEX idx_tasks_created_at ON generation_tasks (created_at);
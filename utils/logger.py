import logging
import logging.handlers
import os
from datetime import datetime
from config import LOG_CONFIG

def setup_logger(name: str = 'wedeepeval') -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志目录
    log_dir = LOG_CONFIG['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_CONFIG['log_level'].upper()))
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        LOG_CONFIG['log_format'],
        datefmt=LOG_CONFIG['date_format']
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器 - 按日期轮转，统一使用 wedeepeval 作为文件名
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(log_dir, f'wedeepeval_{today}.log')
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=LOG_CONFIG['max_bytes'],
        backupCount=LOG_CONFIG['backup_count'],
        encoding=LOG_CONFIG['encoding']
    )
    file_handler.setLevel(getattr(logging, LOG_CONFIG['log_level'].upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 错误日志文件处理器 - 统一使用 wedeepeval_error 作为文件名
    error_log_file = os.path.join(log_dir, f'wedeepeval_error_{today}.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=LOG_CONFIG['max_bytes'],
        backupCount=LOG_CONFIG['backup_count'],
        encoding=LOG_CONFIG['encoding']
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    return logger

def get_logger(name: str = 'deepeval') -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name)
    return logger

# 创建默认日志记录器
default_logger = get_logger() 
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
# 根据环境变量加载对应的配置文件
ENV = os.getenv('ENV', 'production')  # 默认开发环境
env_file = BASE_DIR / f'env.{ENV}'

if os.path.exists(env_file):
    load_dotenv(env_file)
    print(f"✅ 加载环境配置文件: {env_file}")
else:
    print(f"⚠️  环境配置文件 {env_file} 不存在，使用默认配置")

# 基础配置
BASE_CONFIG = {
    'env': ENV,
    'debug': os.getenv('DEBUG', 'false').lower() == 'true',
    'log_level': 'DEBUG' if os.getenv('DEBUG', 'false').lower() == 'true' else 'INFO'
}

# 数据库配置
DATABASE_CONFIG = {
    'host': os.getenv('DATABASE_HOST', '127.0.0.1'),
    'port': int(os.getenv('DATABASE_PORT', 3306)),
    'database': os.getenv('DATABASE_NAME', 'wedeepeval'),
    'user': os.getenv('DATABASE_USER', 'root'),
    'password': os.getenv('DATABASE_PASSWORD', '123456')
}

# API配置
API_CONFIG = {
    'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
    'openai_base_url': os.getenv('OPENAI_BASE_URL', 'https://api.gpt.ge/v1/'),
    'openai_model': os.getenv('OPENAI_MODEL', 'gpt-4-turbo')
}

# Embedding模型配置
EMBEDDING_CONFIG = {
    'embedding_api_key': os.getenv('OPENAI_API_KEY', ''),
    'embedding_base_url': os.getenv('OPENAI_BASE_URL', 'https://api.gpt.ge/v1/'),
    'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')
}

# 服务器配置
SERVER_CONFIG = {
    'host': os.getenv('SERVER_HOST', '0.0.0.0'),
    'port': int(os.getenv('SERVER_PORT', 8092)),
    'reload': os.getenv('RELOAD', 'false').lower() == 'true'
}

# 文件配置
FILE_CONFIG = {
    'output_dir': os.getenv('OUTPUT_DIR', '/tmp'),
    'max_file_size': int(os.getenv('MAX_FILE_SIZE', 10 * 1024 * 1024)),  # 10MB
    'allowed_file_types': ['pdf', 'docx', 'txt', 'csv']
}

# 日志配置
LOG_CONFIG = {
    'log_dir': os.getenv('LOG_DIR', 'logs'),
    'log_level': os.getenv('LOG_LEVEL', 'INFO'),
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'max_bytes': int(os.getenv('LOG_MAX_BYTES', 10 * 1024 * 1024)),  # 10MB
    'backup_count': int(os.getenv('LOG_BACKUP_COUNT', 5)),
    'encoding': 'utf-8'
}

# 生成参数配置
GENERATION_CONFIG = {
    'batch_size': int(os.getenv('GENERATION_BATCH_SIZE', 1)),
    'max_retries': int(os.getenv('GENERATION_MAX_RETRIES', 3)),
    'timeout': int(os.getenv('GENERATION_TIMEOUT', 30)),  # 30秒
    'temperature': float(os.getenv('GENERATION_TEMPERATURE', 0.7)),
    'max_tokens': int(os.getenv('GENERATION_MAX_TOKENS', 1000)),
    'single_batch_threshold': int(os.getenv('SINGLE_BATCH_THRESHOLD', 5)),
    'max_single_batch_size': int(os.getenv('MAX_SINGLE_BATCH_SIZE', 8))
}

# Preview配置
PREVIEW_CONFIG = {
    'max_length': int(os.getenv('PREVIEW_MAX_LENGTH', 60)),  # preview字段最大长度限制
    'document_items': int(os.getenv('PREVIEW_DOCUMENT_ITEMS', 5)),  # 文档生成preview取前几条
    'topic_items': int(os.getenv('PREVIEW_TOPIC_ITEMS', 10)),  # 主题生成preview取前几条
    'augment_items': int(os.getenv('PREVIEW_AUGMENT_ITEMS', 5))  # 扩写生成preview取前几条
}



# 打印当前环境信息
print(f"🌍 当前环境: {ENV}")
print(f"🔧 调试模式: {BASE_CONFIG['debug']}")
print(f"🗄️  数据库: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}")
print(f"🌐 服务器: {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
print(f"OPENAI CONFIG: openai_api_key:{API_CONFIG['openai_api_key']},openai_base_url:{API_CONFIG['openai_base_url']}")
print(f"📝 日志目录: {LOG_CONFIG['log_dir']}") 
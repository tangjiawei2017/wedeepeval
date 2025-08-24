import os
from dotenv import load_dotenv

# 根据环境变量加载对应的配置文件
ENV = os.getenv('ENV', 'development')  # 默认开发环境
env_file = f'env.{ENV}'

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
    'batch_size': int(os.getenv('GENERATION_BATCH_SIZE', 3)),  # 每批生成数量
    'max_retries': int(os.getenv('GENERATION_MAX_RETRIES', 3)),  # 最大重试次数
    'timeout': int(os.getenv('GENERATION_TIMEOUT', 30)),  # API超时时间(秒)
    'temperature': float(os.getenv('GENERATION_TEMPERATURE', 0.7)),  # 生成温度
    'max_tokens': int(os.getenv('GENERATION_MAX_TOKENS', 2000)),  # 最大token数
    'single_batch_threshold': int(os.getenv('SINGLE_BATCH_THRESHOLD', 10)),  # 单批生成阈值
    'max_single_batch_size': int(os.getenv('MAX_SINGLE_BATCH_SIZE', 15)),  # 单批最大生成数量
}

# 打印当前环境信息
print(f"🌍 当前环境: {ENV}")
print(f"🔧 调试模式: {BASE_CONFIG['debug']}")
print(f"🗄️  数据库: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}")
print(f"🌐 服务器: {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
print(f"📝 日志目录: {LOG_CONFIG['log_dir']}") 
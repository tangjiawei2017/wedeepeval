import mysql.connector
from mysql.connector import Error
from datetime import datetime
import json
from typing import List, Dict, Optional
from config import DATABASE_CONFIG
from utils.logger import get_logger

# 获取日志记录器
logger = get_logger('database')

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """连接数据库"""
        try:
            self.connection = mysql.connector.connect(
                host=DATABASE_CONFIG['host'],
                port=DATABASE_CONFIG['port'],
                database=DATABASE_CONFIG['database'],
                user=DATABASE_CONFIG['user'],
                password=DATABASE_CONFIG['password']
            )
            if self.connection.is_connected():
                logger.info("数据库连接成功")
        except Error as e:
            logger.error(f"数据库连接失败: {e}")
    
    def create_task(self, task_name: str, generation_type: str, total_items: int = 0) -> Optional[int]:
        """创建新任务"""
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO generation_tasks (task_name, generation_type, total_items, status)
                VALUES (%s, %s, %s, 'pending')
            """
            cursor.execute(query, (task_name, generation_type, total_items))
            self.connection.commit()
            task_id = cursor.lastrowid
            cursor.close()
            logger.info(f"创建任务成功: {task_name}, ID: {task_id}")
            return task_id
        except Error as e:
            logger.error(f"创建任务失败: {e}")
            return None
    
    def update_task_status(self, task_id: int, status: str, completed_items: int = None, 
                          output_file_path: str = None, error_message: str = None):
        """更新任务状态"""
        try:
            cursor = self.connection.cursor()
            if completed_items is not None and output_file_path is not None:
                query = """
                    UPDATE generation_tasks 
                    SET status = %s, completed_items = %s, output_file_path = %s, updated_at = NOW()
                    WHERE id = %s
                """
                cursor.execute(query, (status, completed_items, output_file_path, task_id))
                logger.info(f"更新任务状态: ID={task_id}, status={status}, completed={completed_items}")
            elif completed_items is not None:
                query = """
                    UPDATE generation_tasks 
                    SET status = %s, completed_items = %s, updated_at = NOW()
                    WHERE id = %s
                """
                cursor.execute(query, (status, completed_items, task_id))
                logger.info(f"更新任务状态: ID={task_id}, status={status}, completed={completed_items}")
            elif error_message is not None:
                query = """
                    UPDATE generation_tasks 
                    SET status = %s, error_message = %s, updated_at = NOW()
                    WHERE id = %s
                """
                cursor.execute(query, (status, error_message, task_id))
                logger.error(f"任务失败: ID={task_id}, error={error_message}")
            else:
                query = """
                    UPDATE generation_tasks 
                    SET status = %s, updated_at = NOW()
                    WHERE id = %s
                """
                cursor.execute(query, (status, task_id))
                logger.info(f"更新任务状态: ID={task_id}, status={status}")
            
            self.connection.commit()
            cursor.close()
        except Error as e:
            logger.error(f"更新任务状态失败: {e}")
    
    def get_all_tasks(self) -> List[Dict]:
        """获取所有任务"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
                SELECT id, task_name, generation_type, status, total_items, completed_items,
                       output_file_path, created_at, updated_at, error_message
                FROM generation_tasks 
                ORDER BY created_at DESC
            """
            cursor.execute(query)
            tasks = cursor.fetchall()
            cursor.close()
            logger.info(f"获取任务列表成功，共 {len(tasks)} 个任务")
            return tasks
        except Error as e:
            logger.error(f"获取任务列表失败: {e}")
            return []
    
    def get_task_by_id(self, task_id: int) -> Optional[Dict]:
        """根据ID获取任务"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
                SELECT id, task_name, generation_type, status, total_items, completed_items,
                       output_file_path, created_at, updated_at, error_message
                FROM generation_tasks 
                WHERE id = %s
            """
            cursor.execute(query, (task_id,))
            task = cursor.fetchone()
            cursor.close()
            if task:
                logger.info(f"获取任务成功: ID={task_id}")
            else:
                logger.warning(f"任务不存在: ID={task_id}")
            return task
        except Error as e:
            logger.error(f"获取任务失败: {e}")
            return None
    
    def close(self):
        """关闭数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("数据库连接已关闭")

# 全局数据库管理器实例
db_manager = DatabaseManager()

def generate_task_name(generation_type: str) -> str:
    """生成任务名称"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    return f"generation_datasets_from_{generation_type}_{timestamp}"

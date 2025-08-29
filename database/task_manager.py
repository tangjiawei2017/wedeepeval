from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional, Dict
from datetime import datetime
from database.base import DatabaseManager
from database.models import GenerationTask
from utils.logger import get_logger

logger = get_logger('task_manager', 'business')

class TaskManager(DatabaseManager):
    """任务数据库管理器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            super().__init__()
            self._initialized = True
    
    def create_task(self, task_name: str, generation_type: str, total_items: int = 0, preview: str = None) -> Optional[int]:
        """创建新任务"""
        try:
            # 验证 generation_type 是否为有效值
            valid_types = ['document', 'context', 'topic', 'augment']
            if generation_type not in valid_types:
                logger.error(f"无效的生成类型: {generation_type}，有效类型: {valid_types}")
                return None
            
            task = GenerationTask(
                task_name=task_name,
                generation_type=generation_type,
                total_items=total_items,
                preview=preview
            )
            
            self.db.add(task)
            self.commit()
            self.db.refresh(task)
            
            logger.info(f"创建任务成功: {task_name}, ID: {task.id}")
            return task.id
            
        except Exception as e:
            self.rollback()
            logger.error(f"创建任务失败: {e}")
            return None
    
    def update_task_status(self, task_id: int, status: str, completed_items: int = None, 
                          total_items: int = None,
                          output_file_path: str = None, error_message: str = None, preview: str = None):
        """更新任务状态"""
        try:
            task = self.db.query(GenerationTask).filter(GenerationTask.id == task_id).first()
            if not task:
                logger.error(f"任务不存在: ID={task_id}")
                return
            
            task.status = status
            task.updated_at = datetime.now()
            
            if completed_items is not None:
                task.completed_items = completed_items
            if total_items is not None:
                task.total_items = total_items
            if output_file_path is not None:
                task.output_file_path = output_file_path
            if error_message is not None:
                task.error_message = error_message
            if preview is not None:
                task.preview = preview
            
            self.commit()
            
            logger.info(f"更新任务状态: ID={task_id}, status={status}, completed={completed_items}/{total_items}")
            
        except Exception as e:
            self.rollback()
            logger.error(f"更新任务状态失败: {e}")
    
    def get_all_tasks(self) -> List[Dict]:
        """获取所有任务"""
        try:
            tasks = self.db.query(GenerationTask).order_by(desc(GenerationTask.created_at)).all()
            result = [task.to_dict() for task in tasks]
            logger.info(f"获取任务列表成功，共 {len(result)} 个任务")
            return result
        except Exception as e:
            logger.error(f"获取任务列表失败: {e}")
            return []
    
    def get_task_by_id(self, task_id: int) -> Optional[Dict]:
        """根据ID获取任务"""
        try:
            task = self.db.query(GenerationTask).filter(GenerationTask.id == task_id).first()
            if task:
                # 减少日志记录，提高性能
                return task.to_dict()
            else:
                logger.warning(f"任务不存在: ID={task_id}")
                return None
        except Exception as e:
            logger.error(f"获取任务失败: {e}")
            return None
    
    def get_tasks_by_status(self, status: str) -> List[Dict]:
        """根据状态获取任务"""
        try:
            tasks = self.db.query(GenerationTask).filter(GenerationTask.status == status).all()
            result = [task.to_dict() for task in tasks]
            logger.info(f"获取状态为 {status} 的任务，共 {len(result)} 个")
            return result
        except Exception as e:
            logger.error(f"获取任务失败: {e}")
            return []
    
    def delete_task(self, task_id: int) -> bool:
        """删除任务"""
        try:
            task = self.db.query(GenerationTask).filter(GenerationTask.id == task_id).first()
            if task:
                self.db.delete(task)
                self.commit()
                logger.info(f"删除任务成功: ID={task_id}")
                return True
            else:
                logger.warning(f"任务不存在: ID={task_id}")
                return False
        except Exception as e:
            self.rollback()
            logger.error(f"删除任务失败: {e}")
            return False
    
    def get_running_tasks(self) -> List[Dict]:
        """获取正在运行的任务"""
        return self.get_tasks_by_status('running')
    
    def get_pending_tasks(self) -> List[Dict]:
        """获取等待中的任务"""
        return self.get_tasks_by_status('pending')
    
    def get_completed_tasks(self) -> List[Dict]:
        """获取已完成的任务"""
        return self.get_tasks_by_status('completed')
    
    def get_failed_tasks(self) -> List[Dict]:
        """获取失败的任务"""
        return self.get_tasks_by_status('failed') 
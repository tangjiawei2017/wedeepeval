from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel
from database import TaskManager
from datetime import datetime

router = APIRouter(prefix="/task", tags=["Task"])

# 创建任务管理器实例
task_manager = TaskManager()

from datetime import datetime
from typing import Optional

class TaskResponse(BaseModel):
    id: int
    task_name: str
    generation_type: str
    status: str
    total_items: int
    completed_items: int
    output_file_path: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    error_message: Optional[str] = None
    preview: Optional[str] = None
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_db_task(cls, task_dict: Dict):
        """从数据库任务字典创建响应模型"""
        # 确保日期时间字段是字符串格式
        if task_dict.get('created_at') and isinstance(task_dict['created_at'], datetime):
            task_dict['created_at'] = task_dict['created_at'].isoformat()
        if task_dict.get('updated_at') and isinstance(task_dict['updated_at'], datetime):
            task_dict['updated_at'] = task_dict['updated_at'].isoformat()
        
        return cls(**task_dict)
    
    @classmethod
    def from_db_task(cls, task_dict: Dict):
        """从数据库任务字典创建响应模型"""
        # 确保日期时间字段是字符串格式
        if task_dict.get('created_at') and isinstance(task_dict['created_at'], datetime):
            task_dict['created_at'] = task_dict['created_at'].isoformat()
        if task_dict.get('updated_at') and isinstance(task_dict['updated_at'], datetime):
            task_dict['updated_at'] = task_dict['updated_at'].isoformat()
        
        return cls(**task_dict)

@router.get("/list", response_model=List[TaskResponse], summary="获取所有任务")
async def get_all_tasks():
    """
    获取所有任务列表
    """
    try:
        tasks = task_manager.get_all_tasks()
        # 使用新的转换方法
        return [TaskResponse.from_db_task(task) for task in tasks]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")

@router.get("/{task_id}", response_model=TaskResponse, summary="获取指定任务")
async def get_task(task_id: int):
    """
    根据任务ID获取任务详情
    """
    try:
        task = task_manager.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        # 使用新的转换方法
        return TaskResponse.from_db_task(task)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务失败: {str(e)}") 
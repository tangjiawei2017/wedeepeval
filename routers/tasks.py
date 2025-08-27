from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict, Optional
from pydantic import BaseModel
from database import TaskManager
from datetime import datetime
import os

router = APIRouter(prefix="/task", tags=["Task"])

# 创建任务管理器实例
task_manager = TaskManager()

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

@router.get("/download/{task_id}", summary="下载任务生成的文件")
async def download_task_file(task_id: int):
    """
    根据任务ID下载生成的文件
    
    Args:
        task_id: 任务ID
        
    Returns:
        文件下载响应
    """
    try:
        # 获取任务信息
        task = task_manager.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        # 检查任务状态
        if task['status'] != 'completed':
            raise HTTPException(status_code=400, detail="任务尚未完成，无法下载文件")
        
        # 检查文件路径
        output_file_path = task.get('output_file_path')
        if not output_file_path:
            raise HTTPException(status_code=404, detail="任务没有生成文件")
        
        # 检查文件是否存在
        if not os.path.exists(output_file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 获取文件名
        filename = os.path.basename(output_file_path)
        
        # 返回文件下载响应
        return FileResponse(
            path=output_file_path,
            filename=filename,
            media_type='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载文件失败: {str(e)}") 
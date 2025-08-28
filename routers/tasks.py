from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict, Optional
from pydantic import BaseModel
from database import TaskManager
from datetime import datetime
import os
from utils.response import success, error

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
    
    def __init__(self, **data):
        # 确保时间字段使用正确的格式
        if 'created_at' in data and data['created_at']:
            if isinstance(data['created_at'], str) and 'T' in data['created_at']:
                # 将 T 格式转换为空格格式
                data['created_at'] = data['created_at'].replace('T', ' ')
        if 'updated_at' in data and data['updated_at']:
            if isinstance(data['updated_at'], str) and 'T' in data['updated_at']:
                # 将 T 格式转换为空格格式
                data['updated_at'] = data['updated_at'].replace('T', ' ')
        super().__init__(**data)
    
    def dict(self, *args, **kwargs):
        """重写 dict 方法，确保时间格式正确"""
        result = super().dict(*args, **kwargs)
        if result.get('created_at') and isinstance(result['created_at'], str) and 'T' in result['created_at']:
            result['created_at'] = result['created_at'].replace('T', ' ')
        if result.get('updated_at') and isinstance(result['updated_at'], str) and 'T' in result['updated_at']:
            result['updated_at'] = result['updated_at'].replace('T', ' ')
        return result

@router.get("/list", summary="获取所有任务")
async def get_all_tasks():
    """
    获取所有任务列表
    """
    try:
        tasks = task_manager.get_all_tasks()
        
        # 处理每个任务的时间格式
        for task in tasks:
            if task.get('created_at') and isinstance(task['created_at'], str) and 'T' in task['created_at']:
                task['created_at'] = task['created_at'].replace('T', ' ')
            if task.get('updated_at') and isinstance(task['updated_at'], str) and 'T' in task['updated_at']:
                task['updated_at'] = task['updated_at'].replace('T', ' ')
        
        items = [TaskResponse(**task).dict() for task in tasks]
        return success({"items": items, "pagination": None})
    except Exception as e:
        return error(message=f"获取任务列表失败: {str(e)}", code=500)

@router.get("/{task_id}", summary="获取指定任务")
async def get_task(task_id: int):
    """
    根据任务ID获取任务详情
    """
    try:
        task = task_manager.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        # 处理时间格式
        if task.get('created_at') and isinstance(task['created_at'], str) and 'T' in task['created_at']:
            task['created_at'] = task['created_at'].replace('T', ' ')
        if task.get('updated_at') and isinstance(task['updated_at'], str) and 'T' in task['updated_at']:
            task['updated_at'] = task['updated_at'].replace('T', ' ')
        
        return success(TaskResponse(**task).dict())
    except HTTPException as e:
        # FastAPI 会用全局异常处理；此处保持原样抛出
        raise e
    except Exception as e:
        return error(message=f"获取任务失败: {str(e)}", code=500)

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
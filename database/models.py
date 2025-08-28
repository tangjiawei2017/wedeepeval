from sqlalchemy import Column, Integer, String, Text, DateTime, Enum
from datetime import datetime
from database.base import Base

class GenerationTask(Base):
    """生成任务模型"""
    __tablename__ = "generation_tasks"
    
    id = Column(Integer, primary_key=True, index=True, comment="任务ID")
    task_name = Column(String(255), nullable=False, comment="任务名称")
    generation_type = Column(String(50), nullable=False, comment="生成类型")
    status = Column(Enum('pending', 'running', 'completed', 'failed', 'cancelled'), 
                   default='pending', comment="任务状态")
    total_items = Column(Integer, default=0, comment="总项目数")
    completed_items = Column(Integer, default=0, comment="已完成项目数")
    output_file_path = Column(String(500), nullable=True, comment="输出文件路径")
    preview = Column(Text, nullable=True, comment="结果预览（<=50字符，否则截断加...）")
    error_message = Column(Text, nullable=True, comment="错误信息")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")
    
    def __repr__(self):
        return f"<GenerationTask(id={self.id}, task_name='{self.task_name}', status='{self.status}')>"
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'id': self.id,
            'task_name': self.task_name,
            'generation_type': self.generation_type,
            'status': self.status,
            'total_items': self.total_items,
            'completed_items': self.completed_items,
            'output_file_path': self.output_file_path,
            'preview': self.preview,
            'error_message': self.error_message,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None
        }

# 可以在这里添加其他模型
# class User(Base):
#     __tablename__ = "users"
#     ...

# class Dataset(Base):
#     __tablename__ = "datasets"
#     ... 
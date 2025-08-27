from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.datasets import router as datasets_router
from routers.tasks import router as tasks_router
from database import create_tables
import uvicorn
from config import SERVER_CONFIG, BASE_CONFIG
from utils.logger import get_logger
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# 获取日志记录器
logger = get_logger('main')

app = FastAPI(
    title="WeDeepEval Dataset APIs", 
    version="0.1.0", 
    description="根据文档/上下文/主题/扩写生成数据集的接口集合"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局请求校验异常处理（避免对二进制请求体进行 UTF-8 解码）
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    logger.error(f"参数校验失败: {exc.errors()}")
    return JSONResponse(
        status_code=400,
        content={
            "detail": exc.errors(),
            "message": "参数校验失败，请检查表单字段与文件是否符合要求"
        },
    )

app.include_router(datasets_router)
app.include_router(tasks_router)


@app.get("/", tags=["Health"])
async def root():
    logger.info("健康检查接口被调用")
    return {"message": "WeDeepEval dataset service is running", "docs": "/docs"}


if __name__ == "__main__":
    # 创建数据库表
    create_tables()
    
    logger.info(f"启动服务器: {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
    uvicorn.run(
        "main:app", 
        host=SERVER_CONFIG['host'], 
        port=SERVER_CONFIG['port'], 
        reload=SERVER_CONFIG['reload']
    )

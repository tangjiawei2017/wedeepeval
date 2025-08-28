from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from routers.datasets import router as datasets_router
from routers.tasks import router as tasks_router
from database import create_tables
import uvicorn
from config import SERVER_CONFIG, BASE_CONFIG
from utils.logger import get_logger
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from utils.response import success, error
import os
import time

# 获取日志记录器 - 使用系统日志类型
logger = get_logger('main', 'app')
# 获取访问日志记录器
access_logger = get_logger('access', 'access')

app = FastAPI(
    title="WeDeepEval Dataset APIs", 
    version="0.1.0", 
    description="根据文档/上下文/主题/扩写生成数据集的接口集合",
    docs_url=None,
    redoc_url=None
)

# 访问日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # 记录访问日志
    access_logger.info(
        f'{request.client.host}:{request.client.port} - "{request.method} {request.url.path} HTTP/{request.scope.get("http_version", "1.1")}" {response.status_code} - {process_time:.3f}s'
    )
    
    return response

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
    return JSONResponse(status_code=400, content=error(
        message="参数校验失败，请检查表单字段与文件是否符合要求",
        code=400,
        data={"detail": exc.errors()}
    ))

app.include_router(datasets_router)
app.include_router(tasks_router)





# 使用本地离线文档页面，避免外网依赖
@app.get("/docs", include_in_schema=False)
async def offline_docs():
    project_root = os.getcwd()
    docs_path = os.path.join(project_root, "openapi.html")
    if not os.path.exists(docs_path):
        # 如果缺失，则退化为直接返回 openapi.json
        return FileResponse(os.path.join(project_root, "openapi.json"), media_type="application/json")
    return FileResponse(docs_path, media_type="text/html")


if __name__ == "__main__":
    # 注意：启动时不再强制连接数据库，避免本地未启动 MySQL 导致服务无法启动
    logger.info(f"启动服务器: {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
    
    uvicorn.run(
        "main:app", 
        host=SERVER_CONFIG['host'], 
        port=SERVER_CONFIG['port'], 
        reload=SERVER_CONFIG['reload'],
        access_log=False  # 禁用默认访问日志，使用自定义中间件
    )

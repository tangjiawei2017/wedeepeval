# 数据集生成平台 - Shell 脚本使用说明

本目录包含了数据集生成平台的服务管理脚本。

## 脚本说明

### 1. start.sh - 启动服务
```bash
./sh/start.sh [环境名称]
```
- **功能**: 启动数据集生成平台服务
- **参数**: 
  - `环境名称`: 可选，默认为 `development`
  - 支持的环境: `development`, `production`, `testing`
- **示例**:
  ```bash
  ./sh/start.sh development  # 启动开发环境
  ./sh/start.sh production   # 启动生产环境
  ./sh/start.sh             # 使用默认环境(development)
  ```

### 2. stop.sh - 停止服务
```bash
./sh/stop.sh
```
- **功能**: 停止所有相关的Python和Uvicorn进程
- **特点**: 
  - 先尝试优雅停止（TERM信号）
  - 等待3秒后强制停止（KILL信号）
  - 支持多种Python命令格式

## 使用流程

### 首次使用
1. 确保脚本有执行权限：
   ```bash
   chmod +x sh/*.sh
   ```

2. 检查环境配置文件是否存在：
   ```bash
   ls env.*
   ```

### 日常使用
```bash
# 启动服务
./sh/start.sh development

# 停止服务
./sh/stop.sh
```

## 环境配置

确保在项目根目录下有对应的环境配置文件：
- `env.development` - 开发环境配置
- `env.production` - 生产环境配置
- `env.testing` - 测试环境配置

## 注意事项

1. **端口冲突**: 如果端口被占用，脚本会自动处理
2. **进程清理**: `stop.sh` 会清理所有相关进程
3. **优雅停止**: 优先使用TERM信号，避免数据丢失
4. **环境变量**: 脚本会自动设置 `ENV` 环境变量

## 故障排除

### 服务无法启动
1. 检查环境配置文件是否存在
2. 检查端口是否被占用
3. 查看错误日志

### 服务无法停止
1. 检查进程是否存在
2. 手动使用 `kill -9` 强制停止
3. 检查脚本权限

### 端口被占用
```bash
# 查看端口占用
lsof -i :8092

# 手动释放端口
kill -9 $(lsof -ti:8092)
``` 
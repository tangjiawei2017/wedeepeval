# DeepEval 本地源码

这个目录包含了 DeepEval 的本地源码副本，用于项目开发和自定义修改。

## 目录结构

```
deepeval_source/
├── deepeval/          # DeepEval 主要源码
│   ├── synthesizer/   # 数据集生成器（我们主要使用的模块）
│   ├── dataset/       # 数据集处理
│   ├── models/        # 模型定义
│   └── ...
└── README.md         # 本文件
```

## 使用方法

### 1. 导入本地 DeepEval

在项目代码中，我们已经配置了自动导入本地 DeepEval 源码：

```python
# 在 utils/deepeval_generator.py 中
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepeval_source'))

from deepeval.synthesizer import Synthesizer
```

### 2. 修改 DeepEval 源码

你可以直接修改 `deepeval_source/deepeval/` 目录下的任何文件：

- **主要修改位置**：`deepeval/synthesizer/` - 数据集生成相关
- **模型配置**：`deepeval/models/` - 各种模型实现
- **数据集处理**：`deepeval/dataset/` - 数据集格式和转换

### 3. 依赖管理

本地 DeepEval 源码仍然需要以下依赖包：

```bash
# 核心依赖（已在 requirements.txt 中）
openai==1.99.9
anthropic
google-genai
requests
rich
pydantic

# 其他可选依赖
aiohttp
grpcio
opentelemetry-api
opentelemetry-sdk
```

## 开发建议

### 1. 备份原始代码

在修改之前，建议备份原始代码：

```bash
cp -r deepeval_source/deepeval deepeval_source/deepeval_backup
```

### 2. 版本控制

将修改后的代码纳入版本控制：

```bash
git add deepeval_source/
git commit -m "Add local DeepEval source code"
```

### 3. 测试修改

每次修改后，重启服务并测试功能：

```bash
./sh/stop.sh
./sh/start.sh development
```

## 常见修改场景

### 1. 修改生成策略

编辑 `deepeval/synthesizer/synthesizer.py`：

```python
# 修改生成逻辑
async def a_generate_goldens_from_contexts(self, ...):
    # 你的自定义逻辑
    pass
```

### 2. 添加新的模型支持

编辑 `deepeval/models/` 目录下的相应文件。

### 3. 修改数据集格式

编辑 `deepeval/dataset/` 目录下的文件。

## 注意事项

1. **依赖兼容性**：确保修改后的代码与现有依赖兼容
2. **API 变更**：如果修改了公共 API，需要同步更新项目中的调用代码
3. **测试覆盖**：修改后请确保功能测试通过
4. **文档更新**：如有重大修改，请更新相关文档

## 故障排除

### 导入错误

如果遇到导入错误，检查：

1. Python 路径是否正确设置
2. 依赖包是否已安装
3. 源码文件是否完整

### 功能异常

如果功能异常，检查：

1. 修改的代码是否有语法错误
2. API 调用是否正确
3. 配置参数是否有效 
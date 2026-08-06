# 视觉多模态 RAG 智能体 Baseline

**语言 / Language:** [中文](README.md) | [English](README_EN.md)

这是一个视觉多模态 RAG 智能体的第一版 baseline。项目支持上传知识文档、上传图片进行 YOLO 目标检测，然后结合“视觉检测结果 + 文档检索证据”生成可追溯回答。

> 注意：仓库不包含 YOLO 权重文件。你需要准备自己的训练模型，例如 `best.pt`，并在 `.env` 中配置 `YOLO_MODEL`。

## 功能

- FastAPI 后端接口
- YOLO 图片目标检测，返回类别、置信度和 bbox
- 文档上传、文本抽取和 chunk 切分
- Milvus Lite 向量检索，本地 JSON fallback
- OpenAI-compatible Chat API，默认适配 DeepSeek
- 简单 Web 页面，支持上传文档、上传图片和提问
- 回答包含视觉检测结果和文档引用片段

## 项目结构

```text
app/
  main.py              FastAPI 入口
  config.py            环境变量和路径配置
  schemas.py           API 数据结构
  services/
    vision.py          YOLO 检测
    documents.py       文档读取和切分
    retriever.py       Milvus Lite / 本地向量检索
    llm.py             Chat API 和本地 embedding fallback
    rag.py             RAG 问答流程
web/                   演示页面
samples/               示例规则文档
data/                  本地运行数据，默认不提交
```

## 快速开始

```powershell
cd <PROJECT_ROOT>
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

在 `.env` 中填写你的 API 和 YOLO 模型路径：

```env
OPENAI_API_KEY=你的DeepSeek_API_Key
OPENAI_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-chat
EMBEDDING_MODEL=local

YOLO_MODEL=C:\path\to\your\best.pt
YOLO_CONFIDENCE=0.25
```

启动后端：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

打开页面：

```text
http://127.0.0.1:8000
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

## 演示流程

1. 上传 `samples\safety_rules.txt`。
2. 上传一张测试图片。
3. 点击图片检测，查看目标类别、置信度和 bbox。
4. 输入问题，例如：`图片中是否存在安全问题？依据是什么？`
5. 点击生成回答，查看 RAG 回答、检测结果和引用片段。

## YOLO 模型说明

项目不会上传或内置权重文件。请自行训练或准备 YOLO 模型，并在 `.env` 中配置：

```env
YOLO_MODEL=C:\path\to\your\best.pt
```

如果是安全帽检测场景，建议模型类别至少包含：

```text
head
helmet
person
```

通用 `yolov8n.pt` 只能识别 COCO 类别，通常不能准确识别安全帽。

## 常用命令

检查后端状态：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

关闭 8000 端口后端：

```powershell
$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    $conn | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
        Stop-Process -Id $_
    }
}
```

检查 YOLO 模型类别：

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; m=YOLO(r'C:\path\to\your\best.pt'); print(m.names)"
```

## 版本规划

- `v0.1`：baseline，完成视觉检测、RAG 检索、API 问答和 Web 演示。
- `v0.2`：优化提示词、前端展示、错误处理和评估脚本。
- `v0.3`：接入 LangGraph、Neo4j 和更完整的回答溯源。

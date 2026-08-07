# 视觉多模态 RAG 智能体

**语言 / Language：** [中文](README.md) | [English](README_EN.md)

这是一个可在本地运行的通用视觉多模态 RAG 工作台。用户可以配置任意 OpenAI-compatible Chat API，选择自己的 YOLO `.pt` 模型，上传规则文档和图片，再根据“YOLO 检测结果 + 文档检索证据”进行可追溯问答。

当前版本：`v0.2.0`

> 仓库不包含 YOLO 权重。请使用自己训练或可信来源的模型，例如 `best.pt`。PyTorch `.pt` 文件可能执行反序列化代码，只应在本地加载可信文件。

## 功能

- FastAPI 后端与单页面 Web 工作台
- 运行时填写 API Key、Base URL、Chat Model 和 Embedding Model
- API Key 仅保存在后端进程内存中，不写入 `.env`
- 选择本机 YOLO 模型路径，或上传可信 `.pt` 模型
- 返回检测类别、置信度、bbox、类别计数和模型类别列表
- 上传 `.txt`、`.md`、`.pdf` 规则文档并自动切分、去重和入库
- Milvus Lite 向量检索，并保留 Local JSON fallback
- 不同 embedding 维度使用独立 Milvus 集合，切换模型时自动同步已有本地记录
- 图片检测结果通过 `image_id` 复用，避免检测后问答时重复上传和重复推理
- 回答使用安全的 Markdown 渲染，包含视觉依据、规则引用、来源、相似度和不确定性说明

## v0.2.0 更新

- 网页运行时配置任意 OpenAI-compatible Chat API，配置仅保存在进程内存
- 通过本机路径或可信文件上传动态切换 YOLO `.pt` 模型
- Prompt 和问答流程通用化，不再绑定安全帽检测场景
- 文档内容去重、乱码拦截、Milvus Lite 索引修复和 Local JSON fallback
- 检测结果缓存复用、类别计数、引用片段和结构化 Markdown 回答
- 增加上传大小、PDF 页数、文本量、图片像素和问题长度限制
- 默认限制为本机访问，并增加同源校验、浏览器安全响应头和 CDN 完整性校验
- 增加 Ruff、单元测试、前端语法检查和依赖漏洞审计 CI

## 快速开始

```powershell
cd <PROJECT_ROOT>
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

已有 `.venv` 时无需重新创建。打开：

- Web 工作台：`http://127.0.0.1:8000`
- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

复制 `.env.example` 为 `.env` 可以设置启动默认值，但不是必需步骤；API 和模型也可以直接在网页中配置。

## API 配置

`.env` 可提供启动默认值，也可以在网页“运行配置”中临时填写：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-api-provider-base-url
CHAT_MODEL=your-chat-model
EMBEDDING_MODEL=local
```

项目不绑定特定服务商。任何兼容 OpenAI Chat Completions 格式的服务均可使用，例如：

```env
# DeepSeek 示例
OPENAI_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-chat

# OpenAI 示例
OPENAI_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4o-mini
```

默认推荐 `EMBEDDING_MODEL=local`，因为部分 Chat API 服务不提供 embeddings 接口。填写远程 embedding 模型后，项目会调用 `<BASE_URL>/embeddings`；调用失败时会明确警告并回退本地向量。

网页显示“API 已填写”只代表配置已进入后端内存。API Key、Base URL 和模型是否有效，会在第一次问答请求时验证。

## YOLO 模型

网页支持两种加载方式：

- 输入本机模型路径，例如 `C:\path\to\your\best.pt`
- 上传可信 `.pt` 文件到本地运行目录 `data/models/`

不同模型可服务于施工安全、交通监控、货架巡检、工业质检等场景。模型类别决定“看到了什么”，上传的规则文档决定“如何判断”，代码不绑定安全帽场景。

## 演示流程

1. 填写 API 配置；也可留空使用本地 fallback 回答。
2. 选择自己的 YOLO `.pt` 模型，确认页面显示模型类别。
3. 上传规则文档，例如 `samples\safety_rules.txt`。
4. 上传图片并点击检测，检查检测框和类别计数。
5. 输入问题并点击运行。
6. 查看回答、视觉依据、引用片段、来源和 warning。

重复上传内容相同的文档时，系统会跳过重复向量入库。切换 YOLO 模型后，旧图片检测缓存会失效，需要重新检测图片。

## 本地测试

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m compileall -q app
.\.venv\Scripts\ruff.exe check app tests
node --check web\app.js
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip_audit -r requirements.txt
```

测试覆盖文档处理、chunk 主键唯一性、Milvus 多 chunk 检索、embedding 维度切换和文档去重。GitHub Actions 配置位于 `.github/workflows/ci.yml`。

## 数据与限制

- 文档、图片、模型和向量数据保存在 `data/`，并已被 Git 忽略。
- 默认上传限制：文档 20 MB、图片 20 MB、模型 500 MB，可在 `.env` 修改。
- 默认内容限制：文档 200 万字符、PDF 500 页、图片 4000 万像素、问题 8000 字符。
- 当前运行时配置和图片检测缓存位于单个后端进程内存中，重启后恢复 `.env` 默认值。
- 本地 hashing embedding 用于 baseline 演示，中文语义召回质量不等同于专业 embedding 模型。

## 安全说明

- 默认 `ALLOW_REMOTE_ACCESS=false`，只接受本机请求；请使用 `127.0.0.1` 启动服务。
- 当前版本没有用户认证、权限管理和多租户隔离，不应直接暴露到公网。
- `.env`、上传内容、向量数据和 `.pt` 权重均被 Git 忽略；提交前仍建议运行密钥扫描。
- `.pt` 权重可能包含可执行反序列化内容，只加载自己训练或明确可信来源的模型。
- Base URL 只允许 HTTP/HTTPS，且禁止嵌入用户名、密码、查询参数或 fragment。
- 前端 Markdown 在插入页面前经过 DOMPurify 清理；第三方脚本固定版本并启用 SRI 校验。

## 关闭后端

运行服务的终端按 `Ctrl+C`。也可以在 PowerShell 中执行：

```powershell
$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    $conn | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
        Stop-Process -Id $_
    }
}
```

## 版本规划

- `v0.1`：视觉检测、RAG 检索、API 问答和 Web 演示 baseline。
- `v0.2.0`：通用运行配置、YOLO 模型切换、文档去重、检索修复、安全加固和通用规则问答。
- `v0.3`：LangGraph 编排、Neo4j 知识图谱、会话管理和更完整的评测体系。

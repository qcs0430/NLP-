# NLP-
NLP多功能智能问答系统项目

本人：戚传帅
主要负责豆包API接入与NLP模型整合:获取API Key并封装调用逻辑(doubao_api.py);
整合文本分类、情感分析、机器翻译等本地模型(nlp_services.py);
设计API异常处理与降级机制;完成核心NLP功能测试。

同组人员的任务分配
张庆举(50%)	负责项目整体架构设计与统筹协调；核心模块设计与开发(本地聊天模型、NLP模型、大模型api的服务化接口封装、前后端数据对接协调)； 前端UI界面的重构设计优化(index.html)；后端Flask开发（统一路由、模式分发、数据对接）；后台管理系统的设计与开发(admin.html)；整合文本分类、情感分析、机器翻译等本地模型(nlp_services.py)；协助完成豆包及deepseek的API接入与NLP模型的训练，大模型调用逻辑的封装(doubao_api.py、deepseek_api.py)；最终系统整合(本地部署验证测试)与文档汇总。
戚传帅(25%)	负责豆包API接入与NLP模型整合:获取API Key并封装调用逻辑(doubao_api.py);整合文本分类、情感分析、机器翻译等本地模型(nlp_services.py);设计API异常处理与降级机制;完成核心NLP功能测试。
王若宸(25%)	负责Web服务与应用部署:基于Flask开发Web服务(app.py),实现用户注册/登录、聊天历史管理功能;开发桌面应用与启动器(desktop_app.py/launcher.py);配置conda环境与依赖包安装;完成本地部署与桌面端测试。

📁 项目结构

```
NLP/
├── AI_QuestionAnswering/              # 智能问答系统主项目
│   ├── code/                          # 核心代码目录
│   │   ├── Seq2Seq.py                 # Seq2Seq 模型定义（Encoder、Decoder、Attention）
│   │   ├── app.py                     # Flask 后端主程序（统一路由、API 接口）
│   │   ├── execute.py                 # 模型训练脚本（数据加载、训练循环）
│   │   ├── nlp_services.py            # NLP 服务接口（文本分类、情感分析、翻译）
│   │   ├── doubao_api.py              # 豆包 API 调用模块
│   │   ├── deepseek_api.py            # DeepSeek API 调用模块
│   │   ├── desktop_app.py             # PyWebView 桌面应用封装
│   │   ├── launcher.py                # 统一启动入口
│   │   ├── data_utls.py               # 数据预处理工具
│   │   ├── build.bat                  # PyInstaller 打包脚本
│   │   ├── 启动桌面版.bat              # 桌面版启动器（激活 conda 环境）
│   │   ├── 启动问答系统.bat            # Web 版启动器
│   │   ├── desktop_config.example.json # 桌面应用配置模板
│   │   ├── templates/                 # HTML 模板
│   │   │   ├── index.html             # 前端主界面（聊天、模式切换、用户管理）
│   │   │   └── admin.html             # 后台管理页面
│   │   ├── static/                    # 静态资源
│   │   │   ├── css/                   # 样式文件
│   │   │   ├── js/                    # JavaScript 脚本
│   │   │   └── scss/                  # SCSS 源文件
│   │   ├── nlp_models/                # 模型相关文件
│   │   │   └── cnews.vocab.txt        # 新闻分类词汇表
│   │   └── logs/                      # 日志文件目录
│   ├── data/                          # 数据目录
│   │   ├── dialog/                    # 对话数据集（训练语料）
│   │   │   ├── one.txt ~ five.txt     # 多组对话数据
│   │   ├── ids/                       # 预处理后的数据
│   │   │   ├── all_dict.txt           # 词典文件
│   │   │   ├── source.txt             # 输入序列（30条）
│   │   │   └── target.txt             # 目标序列（30条）
│   │   ├── users.json                 # 用户账户数据
│   │   └── chats.json                 # 聊天历史记录
│   └── tmp/                           # 临时文件与模型参数
│       ├── model/                     # 模型检查点
│       │   ├── checkpoint             # 检查点索引文件
│       │   ├── ckpt-1 ~ ckpt-6.data   # 模型参数文件
│       │   └── ckpt-1 ~ ckpt-6.index  # 模型索引文件
│       ├── all_dict.txt               # 词典副本
│       ├── source.txt                 # 数据副本
│       └── target.txt                 # 数据副本
│
├── nlp_deeplearn/                     # NLP 深度学习实验项目
│   ├── code/                          # 实验代码
│   │   ├── 10_3_1.py                  # 文本分类（LSTM）
│   │   ├── 10_3_2.py                  # 情感分析（LSTM）
│   │   └── 10_4.py                    # 机器翻译（Seq2Seq）
│   ├── data/                          # 实验数据集
│   │   └── cnews.vocab.txt            # 中文新闻词汇表
│   └── tmp/                           # 训练输出
│       └── training_checkpoints/      # 模型检查点
│
├── doubao/                            # 豆包 API 测试
│   └── 1.py                           # 测试脚本
│
├── 系统功能演示.mp4                    # 项目演示视频
├── 自然语言处理与应用课程设计-张庆举.docx  # 技术文档
└── README.md                          # 项目说明文档
```

---

## 🚀 快速开始

### 1. 环境要求
- Python 3.11+
- Conda（推荐使用 Anaconda）
- TensorFlow 2.17.0
- Flask、jieba、pywebview 等依赖

### 2. 安装依赖
```bash
conda activate ziranyuyan
pip install -r requirements.txt
```

### 3. 启动方式

**桌面版（推荐）**：
```bash
cd AI_QuestionAnswering/code
启动桌面版.bat
```

**Web 版**：
```bash
cd AI_QuestionAnswering/code
启动问答系统.bat
# 访问 http://127.0.0.1:8808
```

**开发模式**：
```bash
python app.py
```

---

## 🎯 核心功能

### 智能对话
- **豆包 AI**：接入豆包大模型 API，提供智能对话
- **DeepSeek**：接入 DeepSeek V3，高质量回答
- **本地 Seq2Seq**：基于 GRU+Attention 的对话模型

### NLP 任务
- **文本分类**：中文新闻分类（豆包 API + 本地 LSTM）
- **情感分析**：正负面情感识别
- **智能翻译**：中英互译（豆包 API + 本地模型）
- **综合分析**：多维度文本分析

### 系统功能
- 用户登录/注册
- 聊天历史管理（自动保存、搜索、删除）
- 模式自动切换（切换模式自动创建新聊天）
- 后台用户管理
- 深色模式切换

---

## 🏗️ 技术架构

### 后端
- **Flask**：Web 框架，统一 `/message` 接口
- **TensorFlow**：深度学习框架
- **Seq2Seq + Attention**：对话生成模型
- **LSTM**：文本分类、情感分析

### 前端
- **HTML5 + CSS3**：响应式界面设计
- **JavaScript + jQuery**：交互逻辑
- **AJAX**：异步请求

### 桌面封装
- **PyWebView**：将 Web 应用封装为桌面软件
- **PyInstaller**：打包为 `.exe` 可执行文件

---

## 📊 模型信息

### Seq2Seq 模型
- **Encoder**：Embedding + GRU（512 hidden units）
- **Decoder**：Embedding + GRU + Bahdanau Attention
- **词汇量**：144 词
- **训练轮数**：501 epochs（每 100 轮保存检查点）
- **优化器**：Adam
- **损失函数**：SparseCategoricalCrossentropy

### 文本分类模型
- **架构**：Embedding → LSTM → Dense
- **数据集**：THUCNews 中文新闻
- **类别**：10 类新闻分类

---

## 📝 API 配置

### 豆包 API
编辑 `doubao_api.py`，配置 ARK API Key：
```python
self.api_key = "YOUR_ARK_API_KEY"
```

### DeepSeek API
编辑 `deepseek_api.py`，配置 API Key：
```python
self.api_key = "YOUR_DEEPSEEK_API_KEY"
```

---

## 🔧 开发说明

### 路径配置
项目使用动态路径构建，避免运行目录影响：
```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
data_path = os.path.join(PROJECT_ROOT, 'data', 'ids')
```

### 依赖环境
- 必须在 `ziranyuyan` conda 环境中运行
- TensorFlow 仅在该环境中可用

### 打包发布
```bash
cd AI_QuestionAnswering/code
build.bat
# 生成 dist/智能问答系统/智能问答系统.exe
```

---

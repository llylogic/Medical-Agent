# Medical-Agent
# 🏥 Medical-Agent: 基于 LangGraph 的医疗全科动态调度智能体
## 📑 项目描述 (Project Overview)
- **业务场景**：在真实的临床环境中，医生不仅需要查阅医学指南，更需要结合患者病历、排查用药禁忌，甚至执行挂号等实际操作。传统的 RAG 系统仅能“被动查资料”，缺乏常识风控与系统交互能力。本项目旨在打造一个闭环的“全科主治医师”，通过大模型自主调度异构工具链，实现从“病史核验 -> 知识检索 -> 安全风控 -> 医疗决策 -> 系统写入”的端到端复杂多跳推理。
- **核心能力**：基于 LangGraph 白盒引擎，横向挂载了结构化病历查询（Text2SQL）、RAG 知识库、风控知识图谱（Skill 插件）、联网搜索及挂号写入等 5 大工具。系统具备全链路动作可观测、工具结果动态压缩、SQL 异常自我纠错，以及三层立体的智能体记忆架构。

## 👨‍💻 核心技术贡献 (Core Contributions)
*系统全链路技术决策与优化*

**1. 【调度中枢层】构建白盒 ReAct 状态机与全链路动作可观测 (Observability)**
- **技术实践**：摒弃了高度封装的黑盒 Agent 框架，基于 LangGraph 构建了包含 `Reason` (LLM Node) 与 `Act` (Tool Node) 的有向无环图。在前端交互层面，创新性地利用生成器流式机制（Yield）加入了实时的「Agent 动作追踪 (Graph Trace)」面板。
- **工程收益**：实现了 100% 的白盒透明化。大模型在后台的“内心思考 (Thought) -> 调度工具 (Action) -> 工具返回 (Observation)”全过程被动态解析与前端渲染，赋予了黑盒推理过程极强的可解释性与问题溯源能力。

**2. 【工具基建层】异构工具链调度与 Observation 动态摘要压缩**
- **技术实践**：突破纯检索局限，封装了读/写/逻辑验证的 5 大异构工具。针对复杂环境设计了两大容错机制：
  - **SQL 异常反射**：Text2SQL 语法报错时，系统截获 Traceback 传回图节点，触发大模型“自主反思并重写 SQL”，使跨表查询成功率提升超 40%。
  - **工具观测值压缩**：当外部联网工具（Web Search）抓取到超长网页源数据时，利用旁路轻量级子模型前置介入，将其提炼为高密度短文本后再注入 ReAct 上下文，彻底杜绝了外部脏数据导致的主模型 Token 溢出与系统崩溃。

**3. 【记忆管理层】设计三级立体记忆架构解决“失忆与混淆”痛点**
- **技术实践**：参考认知科学理论，为智能体构建了原生的三级记忆流转体系，完美支撑长周期、复杂多跳的医疗诊断推理：
  - **瞬时记忆**：捕获前端登录态，将 user_iD和user_name 隐式注入底层 System Prompt。实现“零感知”身份锚定，大模型在生成 SQL 查表时可全自动依赖该状态进行主键溯源。
  - **短期记忆**：结合 LangGraph 的 `Append-only Message Tree` 与 **滚动压缩机制 (Context Compression)**。在维持多轮 ReAct 推理状态的同时，当对话触及长度阈值，异步拉起子模型对远古对话进行高密度实体摘要替换，解决深层循环中的 Token 溢出与“注意力稀释”问题。
  - **长期记忆**：实施结构化与非结构化的双轨持久化方案。通过 SQLite/MySQL 物理表，实现了对患者客观病历与历史挂号轨迹的持久化（情景记忆）；同时结合底层的 ChromaDB 向量引擎，实现了核心医学指南与非结构化偏好的跨周期唤醒（语义记忆），打造了真正的伴随式专属智能体。

**4. 【意图路由与风控层】基于 Adapter 模式的外部知识图谱接入与动态降级调度**
- **技术实践**：
  - **防工具滥用**：重构意图路由网络，对日常闲聊实施“懒加载”直接回复，仅在触发医疗处方决策时才拉起全链路深度校验。
  - **风控解耦**：采用 Adapter（适配器）模式，将医疗配伍禁忌规则抽离为独立的 Knowledge Graph API 插件（Skill），彻底脱离底层 Prompt 硬编码。
- **工程收益**：大幅降低了无效的 Token 算力损耗。插件化架构实现了 Agent 调度中枢与底层风控图谱的完全物理解耦，未来可无缝迁移至真正的分布式 MCP（Model Context Protocol）协议集群。

## 📂 目录结构 (Project Structure)

```text
Medical-Agent/
├── db_service.py         # 【数据持久层】SQLite 初始化、表结构解耦 (3NF) 与链接池管理
├── auth_service.py       # 【鉴权服务层】用户注册、密码核验与底层 User_ID 映射
├── prompts.py            # 【提示词工程层】System Prompt 动态生成与意图路由规则声明
├── agent_tools.py        # 【能力基建层】5 大异构工具定义、异常反射及网络降级拦截器
├── agent_workflow.py     # 【核心调度层】LangGraph 状态机构建、ReAct 流转边 (Edge) 约束
└── main_web.py           # 【前端表现层】Gradio 多模态交互、状态机 UI Trace 实时追踪
```
## 🛠️ 操作步骤与使用说明 (Quick Start)
- **1. 环境准备 (Environment Setup)**
请确保本地 Python 版本为 3.10+，建议使用虚拟环境：
`pip install -r requirements.txt`
- **2. 配置 API 密钥 (Configuration)**
在项目根目录下新建 .env 文件，并填入您的 API 密钥：`DEEPSEEK_API_KEY="sk-xxxxxx"`
- **3. 启动全科诊疗工作台 (Launch UI)**
：`python main_web.py`
执行命令，系统将自动在当前目录生成包含 Patients、Medical_Records 等 1:N 关系型业务表的 hospital.db，并启动 Web 服务。
启动成功后，在浏览器中访问控制台输出的本地地址`（默认：http://0.0.0.0:7861）`。

## 👨‍⚕️ 核心测试用例指南 (Test Cases)
为验证 Agent 的“多跳推理与纠错能力”，请在系统内使用内置测试账号 lisi (密码 123) 登录，并发送以下极具挑战性的混合 Prompt：
`"医生我今天有点低烧和剧烈咳嗽。另外我的抑郁症一直没好。你能给我直接开点止咳药吗？"`
您将能在前端的后台状态机追踪 面板中，清晰观察到 Agent 的惊艳表现：
- Thought: 捕获开药意图，自主拼接隐式 UID 决定查表。
- Action: 触发 patient_ehr_query，查出患者在服药物“氟西汀”。
- Action: 触发 medical_rag_search，检索出止咳指南推荐药“右美沙芬”。
- Action: 触发 drug_safety_skill，进行风控撞库，引发红色致命警报（5-羟色胺综合征）。
- Thought: 判定高度危险，推翻开药决策，自主触发 appointment_booking 写入挂号记录，并输出最终安抚拒答。

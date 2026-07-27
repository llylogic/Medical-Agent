# agent_tools.py
import sys
import os,re
import sqlite3
import urllib.request
import urllib.parse
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import requests
from langchain_community.tools.pubmed.tool import PubmedQueryRun

# 1. 跨目录依赖注入保护
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
rag_dir = os.path.join(parent_dir, "rag_interview_project")
if parent_dir not in sys.path: sys.path.append(parent_dir)
if rag_dir not in sys.path: sys.path.append(rag_dir)

load_dotenv()

from rag_interview_project.core.embedding_ops import Embedder
from rag_interview_project.core.vector_store import VectorStore
from rag_interview_project.core.retriever import Retriever
from rag_interview_project.core.reranker import Reranker
from db_service import get_db_connection  # 导入刚才写的数据库层

# 2. 实例化第三方库引擎
embedder = Embedder(api_key=os.getenv("SILICONFLOW_API_KEY"))
vector_store = VectorStore(persist_dir="rag_interview_project/chroma_db")
retriever = Retriever(vector_store, embedder, bm25_path="rag_interview_project/cache/bm25_index.pkl")
reranker = Reranker(api_key=os.getenv("SILICONFLOW_API_KEY"))
web_search_tool = DuckDuckGoSearchRun()


# 2.5 新增一个专用的“摘要小模型”
compress_llm = ChatOpenAI(
    model="deepseek-v4-flash",  
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.0
)

def compress_observation(text: str, max_length: int = 1000) -> str:
    """【内部拦截器】：观测摘要 (Observation Summarization) 算法"""
    if len(text) <= max_length:
        return text
    print(f"\n✂️ [内存保护] 拦截到超长返回 ({len(text)}字)，正在执行动态压缩...")
    prompt = f"请将以下冗长的工具检索结果压缩到 {max_length} 字以内。要求：必须保留所有核心的医学实体、数据、政策要求及最终结论，去除无关的广告和废话。\n\n【原始文本】：\n{text}"
    compressed_text = compress_llm.invoke(prompt).content
    return compressed_text

# 3. 具体 Tool 的定义
@tool
def patient_ehr_query(sql_query: str) -> str:
    """【核心工具】：用于查询患者真实的电子病历(EHR)。输入参数必须是合法 SQLite 语句。"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        conn.close()
        return str(result) if result else "未查询到记录"
    except Exception as e:
        return f"SQL执行失败，报错信息: {str(e)}。请自我检查语法并重试！"

@tool
def medical_rag_search(query: str) -> str:
    """【知识库工具】：查询疾病症状、推荐用药、发病机制等医学指南。"""
    retrieved_docs = retriever.search(query=query, top_k=20, strategy="hybrid")
    reranked_docs = reranker.rerank(query=query, docs=retrieved_docs, top_n=5, threshold=0.0)
    return "\n---\n".join(reranked_docs) if reranked_docs else "知识库未找到相关方案。"

@tool
def drug_safety_skill(drug_a: str, drug_b: str) -> str:
    """
    【真实风控 Skill】：接入美国国立卫生研究院 (NIH) RxNorm 真实医药数据库。
    在开具任何新药前，必须调用此插件检查新药与患者正在服用的药物是否冲突。
    支持输入中英文药物通用名（如：氟西汀、阿司匹林、Fluoxetine）。
    """
    try:
        # 第一步：调用 NIH API，将药物自然语言名称转换为标准的 RxCUI 国际代码
        def get_rxcui(drug_name):
            url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}"
            # NIH 限制频率，加上标准请求头
            headers = {"User-Agent": "Mozilla/5.0 Medical-Agent-POC"}
            response = requests.get(url, headers=headers, timeout=5).json()
            if "idGroup" in response and "rxnormId" in response["idGroup"]:
                return response["idGroup"]["rxnormId"][0]
            return None

        cui_a = get_rxcui(drug_a)
        cui_b = get_rxcui(drug_b)
        
        if not cui_a or not cui_b:
            return f"⚠️ 【系统提示】未能在 NIH 国际权威药物数据库中识别到 '{drug_a}' 或 '{drug_b}' 的标准编号。出于安全考虑，请建议患者遵线下医嘱。"

        # 第二步：调用 NIH 药物相互作用 (Interaction) 真实 API
        interaction_url = f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={cui_a}+{cui_b}"
        interaction_res = requests.get(interaction_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5).json()

        # 第三步：解析真实的国家级医学反馈数据
        if "fullInteractionTypeGroup" in interaction_res:
            # 解析极其复杂的嵌套 JSON，提取真实的医学警告
            try:
                description = interaction_res["fullInteractionTypeGroup"][0]["fullInteractionType"][0]["interactionPair"][0]["description"]
                return f"🚨 【NIH 数据库真实红色预警】检测到严重配伍禁忌：\n- 权威医学机制：{description}\n- 处置建议：严禁开具！必须立即要求患者面诊换药！"
            except:
                return "🚨 【NIH 数据库预警】检测到潜在药物相互作用，请谨慎开具。"
        
        return "✅ 【NIH 权威审核通过】国际数据库未查及这两种药物存在明确配伍禁忌，可正常处方。"
        
    except Exception as e:
        return f"风控系统底层网络调用异常: {str(e)}。出于医疗绝对安全考虑，请暂缓开药并转诊线下！"
    
@tool
def appointment_booking(user_id: str, department: str, date: str) -> str:
    """【写操作工具】：为患者写入挂号记录。"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Appointments (user_id, department, appointment_date) VALUES (?, ?, ?)", (user_id, department, date))
        conn.commit()
        conn.close()
        return f"挂号成功：患者 {user_id} 已预约 {date} {department}。"
    except Exception as e:
        return f"挂号失败: {str(e)}"


# =====================================================================
# 🌟 真正的即插即用生态 Skill：社区维护的 PubMed 国际医学文献检索
# =====================================================================
pubmed_skill = PubmedQueryRun()

# 为了让大模型知道怎么用它，我们可以给它套一层极简的壳，或者直接放进列表里。
# 这里我们给它套个壳，强化一下中文说明书（因为官方自带的说明书是纯英文的，怕大模型犯迷糊）
@tool
def international_medical_literature_skill(query: str) -> str:
    """
    【国际顶级医学文献 Skill】：这是一个由开源社区封装好的 PubMed 检索插件。
    当本地知识库无法解决疑难杂症，或者用户明确要求查阅“国际最新医学研究、医学前沿论文”时，调用此插件。
    输入参数 query 必须是英文医学关键词（如：Fanconi syndrome treatment）。
    """
    try:
        #旁路模型压缩
        raw_result = pubmed_skill.invoke(query)
        clean_result = compress_observation(raw_result)
        return f"【PubMed 国际前沿文献(已压缩提炼)】: {clean_result}"
    except Exception as e:
        return f"PubMed 插件调用失败: {str(e)}"


# 暴漏聚合后的工具列表
tools_list = [patient_ehr_query, medical_rag_search, drug_safety_skill, appointment_booking, international_medical_literature_skill]

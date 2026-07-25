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
    【风控 Skill】：模拟接入外部医药配伍禁忌知识图谱 (Knowledge Graph)。
    在开具任何新药前，必须调用此插件检查与患者正在服用的药物是否冲突。
    """
    # 模拟外部微服务 API 的返回格式
    danger_pairs = {
        frozenset(["右美沙芬", "氟西汀"]): "【极度高危】氟西汀与右美沙芬同服极易引发致命的‘5-羟色胺综合征’，严禁开具！",
        frozenset(["阿司匹林", "布洛芬"]): "【高危】增加胃肠道出血及心血管不良事件风险，避免联合使用。",
        frozenset(["头孢克肟", "酒精"]): "【致命】双硫仑样反应（面部潮红、严重可致休克），用药期间严禁饮酒！",
        frozenset(["西柚汁", "阿托伐他汀"]): "【中危】西柚汁抑制CYP3A4酶，导致他汀类血药浓度升高，增加横纹肌溶解风险。",
        frozenset(["左旋多巴", "维生素B6"]): "【药效降低】维生素B6加速左旋多巴在外周的脱羧代谢，导致进入中枢的药量减少。"
    }
    query_pair = frozenset([drug_a, drug_b])
    if query_pair in danger_pairs:
        return f"🚨 风控系统拦截：{danger_pairs[query_pair]}"
    return "✅ 【风控审核通过】未在知识图谱中发现明确的药物相互作用。"
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

@tool
def web_news_search(query: str) -> str:
    """【外部搜索工具】：突破本地知识库时间限制，获取最新医保政策、突发疫情等资讯。"""
    try:
        raw_result = web_search_tool.invoke(query)
        # 👇👇👇 核心修改：返回前强制过一遍压缩器！
        clean_result = compress_observation(raw_result)
        return f"【互联网检索结果(已高度压缩提炼)】: {clean_result}"
    except Exception as e:
       # 面试高光：网络熔断降级机制，转用原生 Python 直接爬取国内必应！
        print(f"⚠️ 节点 1 超时 ({e})，正在自动切换至国内容错节点...")
        try:
            url = 'https://cn.bing.com/search?q=' + urllib.parse.quote(query)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            html = urllib.request.urlopen(req, timeout=5).read().decode('utf-8')
            # 极简正则提取网页摘要
            snippets = re.findall(r'<p class="b_parring">(.*?)</p>', html) or re.findall(r'<div class="b_caption"><p>(.*?)</p></div>', html)
            text = re.sub(r'<[^>]+>', '', " ".join(snippets))
            return f"【国内节点检索结果】: {text[:1000]}" if text else "全网均未检索到相关资讯。"
        except Exception as inner_e:
            return f"双节点联网均失败，请提示用户检查网络环境。"

# 暴漏聚合后的工具列表
tools_list = [patient_ehr_query, medical_rag_search, drug_safety_skill, appointment_booking, web_news_search]
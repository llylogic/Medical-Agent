# main_web.py
import gradio as gr
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import time
import os
# 从我们重构的模块中引入核心组件！
from db_service import init_sqlite_db
from prompts import build_system_prompt
from agent_workflow import medical_agent_app
from langchain_openai import ChatOpenAI
from auth_service import verify_login, register_account

# 1. 系统启动前，确保数据库和表结构就绪
init_sqlite_db()

# 2. 前端事件与 UI (之前完美运行的逻辑直接搬过来)
custom_css = """
body { background-color: #f9fafb; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }

/* 顶部纯白横幅，带微弱高级阴影 */
.header-banner { 
    background: #ffffff; 
    color: #1f2937; 
    padding: 20px; 
    border-radius: 12px; 
    text-align: center; 
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03); 
    margin-bottom: 15px; 
    border: 1px solid #f3f4f6;
}
.header-banner h1 { margin: 0; font-size: 26px; font-weight: 600; letter-spacing: -0.5px; }
.header-banner p { margin: 8px 0 0 0; font-size: 14px; color: #6b7280; }

/* 聊天气泡文字排版 */
.gradio-container .prose { font-size: 15px !important; line-height: 1.6 !important; }

/* 纯白追踪控制台 - 干净通透 */
.scroll-box { 
    height: 550px; overflow-y: auto; 
    border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; 
    background-color: #ffffff; /* 纯白背景 */
    color: #374151; /* 深灰文字，绝对清晰 */
    font-family: 'Consolas', 'Courier New', monospace; 
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
}
.scroll-box h3 { color: #2563eb !important; margin-top: 0; }
.scroll-box strong { color: #3b82f6 !important;
.profile-bar { display: flex; align-items: center; justify-content: space-between; padding: 10px 15px; background: white; border: 1px solid #e5e7eb; border-radius: 8px; margin-bottom: 15px; }
"""


def add_user_msg(user_msg, history):
    # 【新增：清洗 Gradio 多模态输入格式】
    if isinstance(user_msg, list):
        user_msg = " ".join([item.get("text", "") for item in user_msg if isinstance(item, dict) and "text" in item])
    history = history or []
    history.append({"role": "user", "content": user_msg})
    return "", history

def agent_chat(history, user_id, user_name):
    # 【新增：清洗历史记录中的多模态脏数据】
    for msg in history:
        if isinstance(msg.get("content"), list):
            msg["content"] = " ".join([item.get("text", "") for item in msg["content"] if isinstance(item, dict) and "text" in item])

    user_msg = history[-1]["content"] 
    history.append({"role": "assistant", "content": "🤔 Agent 正在思考并调度工具..."})
    yield history, "🚀 启动 Agent 工作流..."

    # ================= 核心亮点：滚动压缩机制 (Context Compression) =================
    # 提取过去所有的纯聊天记录（排除掉最后这一轮）
    past_chat = history[:-2] 
    memory_summary = ""
    
    # 如果聊天记录超过了 30 条（即 15 轮对话），触发压缩机制！
    if len(past_chat) > 30:
        yield history, "🗜️ 检测到历史对话过长，正在执行记忆压缩..."
        # 把远古对话拼起来
        ancient_chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in past_chat[:-4]])
        
        # 临时借用一个小模型做总结
        summary_llm = ChatOpenAI(model="deepseek-v4-flash", api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", temperature=0.0)
        summary_prompt = f"请将以下用户的早期对话记录提炼成 1000 字以内的摘要，只需保留患者的核心病史、已查明的客观事实和既往诉求：\n{ancient_chat_text}"
        memory_summary = summary_llm.invoke(summary_prompt).content
        
        # 物理截断：真正的 past_chat 只保留最近的 4 条（2轮对话）
        past_chat = past_chat[-4:]

    # ================= 组装注入了“记忆摘要”的系统提示词 =================
    base_system_prompt = build_system_prompt(user_id, user_name)
    if memory_summary:
        base_system_prompt += f"\n\n【早期对话记忆摘要（极其重要）】：\n{memory_summary}"
        
    messages = [SystemMessage(content=base_system_prompt)]
    
    # 装载最近的短期记忆
    for msg in past_chat: 
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
            
    messages.append(HumanMessage(content=user_msg))
    
    final_answer = ""
    trace_log = ""
    
    for event in medical_agent_app.stream({"messages": messages}, stream_mode="values"):
        last_msg = event["messages"][-1]
        
        if last_msg.type == "ai":
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                if last_msg.content:
                    trace_log += f"🧠 **Agent 思考**: \n> {last_msg.content}\n\n"
                for tc in last_msg.tool_calls:
                    trace_log += f"🛠️ **调用工具**: `{tc['name']}`\n- 入参: `{tc['args']}`\n\n"
                    history[-1]["content"] = f"正在调度工具: {tc['name']}..."
                yield history, trace_log
                
            elif not last_msg.tool_calls:
                final_answer = last_msg.content
                
        elif last_msg.type == "tool":
            obs_content = last_msg.content[:200] + "..." if len(last_msg.content) > 200 else last_msg.content
            trace_log += f"👀 **工具返回**: \n```text\n{obs_content}\n```\n\n"
            yield history, trace_log
            
    # 👇👇👇 核心修改：模拟真实人类打字的流式输出！
    trace_log += "🎯 **工作流执行完毕。**"
    temp_text = ""
    for char in final_answer:
        temp_text += char
        history[-1]["content"] = temp_text
        yield history, trace_log
        time.sleep(0.015)  # 控制打字速度，越小越快


# ================= 构建高可用双视图 Web UI =================

def handle_login_ui(username, password):
    """UI层的登录桥接器"""
    success, user_id, name, msg = verify_login(username, password)
    if success:
        # 登录成功：隐藏 Auth面板，显示 Chat面板，保存全局状态
        # 【新增】：生成带头像的 HTML 个人名片
        profile_html = f"""
        <div style='display: flex; align-items: center; gap: 12px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/3135/3135715.png' style='width: 35px; height: 35px; border-radius: 50%;'>
            <div>
                <div style='font-size: 16px; font-weight: bold; color: #1f2937;'>{name}</div>
                <div style='font-size: 12px; color: #6b7280;'>ID: {user_id}</div>
            </div>
        </div>
        """
        return gr.update(visible=False), gr.update(visible=True), user_id, name, msg
    else:
        # 登录失败：维持原状
        return gr.update(visible=True), gr.update(visible=False), None, None, msg

# 【新增】：彻底的注销清理逻辑
def handle_logout():
    """彻底清空状态，恢复登录页，实现重置新对话"""
    return (
        gr.update(visible=True),   # 显示登录注册面板
        gr.update(visible=False),  # 隐藏聊天面板
        None,                      # 清空底层 UserID
        None,                      # 清空底层 UserName
        "",                        # 清空登录成功/失败提示
        "",                        # 清空头像名片 UI
        [],                        # 彻底清空大模型聊天记录 []
        "<span style='color:gray;'>系统就绪，等待触发调度...</span>" # 清空后台追踪日志
    )

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.HTML("""
    <div class="header-banner">
        <h1>🏥 Medical-Agent 智能专家诊疗控制台</h1>
        <p>基于 LangGraph 状态机驱动 · 融合 RAG 与 5 大异构工具链调度</p>
    </div>
    """)
    
    # 系统的隐秘状态变量
    state_user_id = gr.State()
    state_user_name = gr.State()
    
    # -------- 视图 1：Auth 鉴权中心面板 --------
    with gr.Column(visible=True) as auth_panel:
        with gr.Tabs():
            # 标签页 A：用户登录
            with gr.TabItem("🔐 用户登录"):
                gr.Markdown("<br>测试账号：`lisi`，密码：`123`")
                log_user = gr.Textbox(label="登录账号", placeholder="请输入账号 / 手机号")
                log_pwd = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
                login_btn = gr.Button("立即登录", variant="primary")
                login_msg = gr.Markdown("")
                
            # 标签页 B：新患者注册
            with gr.TabItem("📝 新患者建档"):
                gr.Markdown("<br>请完善基础病历资料建档：")
                with gr.Row():
                    reg_user = gr.Textbox(label="注册账号")
                    reg_pwd = gr.Textbox(label="登录密码", type="password")
                with gr.Row():
                    reg_name = gr.Textbox(label="真实姓名")
                    reg_gender = gr.Dropdown(choices=["男", "女", "其他"], label="性别", value="男")
                    reg_age = gr.Number(label="年龄", value=25)
                reg_btn = gr.Button("提交建档", variant="secondary")
                reg_msg = gr.Markdown("")
                
    # -------- 视图 2：诊疗工作台面板 (默认隐藏) --------
    with gr.Column(visible=False) as chat_panel:
        with gr.Row(elem_classes="profile-bar"):
            user_profile_ui = gr.HTML("") # 用于动态接收头像和姓名
            logout_btn = gr.Button("🚪 安全退出", variant="stop", size="sm", scale=0)

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 🩺 诊疗室 (AI 主治医师接诊中...)")
                chatbot = gr.Chatbot(height=500, avatar_images=["https://cdn-icons-png.flaticon.com/512/3135/3135715.png", "https://cdn-icons-png.flaticon.com/512/3304/3304260.png"])
                with gr.Row():
                    msg_input = gr.Textbox(placeholder="请输入您的病情描述，按 Enter 键发送...", show_label=False, scale=8)
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                    
            with gr.Column(scale=3):
                gr.Markdown("### ⚙️ 后台状态机追踪 (Graph Trace)")
                trace_box = gr.Markdown("<span style='color:gray;'>系统就绪，等待触发调度...</span>", elem_classes="scroll-box")

    # ================= 绑定业务事件 =================
    # 登录事件
    login_btn.click(
        handle_login_ui, 
        inputs=[log_user, log_pwd], 
        outputs=[auth_panel, chat_panel, state_user_id, state_user_name, login_msg]
    )
    
    # 注册事件 (直接调外部封装的鉴权服务)
    reg_btn.click(
        register_account,
        inputs=[reg_user, reg_pwd, reg_name, reg_gender, reg_age],
        #outputs=[reg_msg],
        outputs=[auth_panel, chat_panel, state_user_id, state_user_name, login_msg, user_profile_ui]
    )

# 2. 【新增】：在下方加上登出按钮的事件绑定
    logout_btn.click(
        handle_logout,
        inputs=[],
        outputs=[auth_panel, chat_panel, state_user_id, state_user_name, login_msg, user_profile_ui, chatbot, trace_box]
    )
    
    # 对话事件
    msg_input.submit(add_user_msg, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot], queue=False).then(
        agent_chat, inputs=[chatbot, state_user_id, state_user_name], outputs=[chatbot, trace_box]
    )
    send_btn.click(add_user_msg, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot], queue=False).then(
        agent_chat, inputs=[chatbot, state_user_id, state_user_name], outputs=[chatbot, trace_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
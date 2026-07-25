# agent_workflow.py
import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv

from tools import tools_list  # 引入解耦后的工具层

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.0
)
llm_with_tools = llm.bind_tools(tools_list)

def call_model(state: AgentState):
    response = llm_with_tools.invoke(state['messages'])
    return {"messages": [response]}

def execute_tools(state: AgentState):
    last_message = state['messages'][-1]
    tool_responses = []
    
    for tool_call in last_message.tool_calls:
        tool_name, tool_args = tool_call["name"], tool_call["args"]
        tool_func = {t.name: t for t in tools_list}.get(tool_name)
        
        try:
            result = tool_func.invoke(tool_args) if tool_func else f"不存在此工具: {tool_name}"
        except Exception as e:
            result = f"工具内部异常: {str(e)}"
            
        tool_responses.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        
    return {"messages": tool_responses}

def should_continue(state: AgentState) -> str:
    if state['messages'][-1].tool_calls:
        return "continue"
    return "end"

# 画图并编译
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", execute_tools)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent")

medical_agent_app = workflow.compile()
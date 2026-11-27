import json

from langchain_core.messages import HumanMessage,AIMessage
from langgraph.channels import topic
from langgraph.graph import StateGraph,START,END,MessagesState
from langgraph.prebuilt import ToolNode

import langgraph_agent as base
import memory as mem_store
from refs import rebuild_refs

from langgraph_multi import reviewer_node
bad = """# 测试报告

## 一、现状
据了解，大部分企业都在用 AI [99]，效果非常好 [100]。

## 二、结论
就这样。"""
state = {"messages": [AIMessage(content=bad)], "passed": False, "issues": [], "rounds": 0}
print(reviewer_node(state))
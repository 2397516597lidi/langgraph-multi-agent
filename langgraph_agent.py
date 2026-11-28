# langgraph_agent.py —— LangGraph 版单 Agent（多 Agent 实验的基线）
import os
import json
import requests
from datetime import datetime, timezone
import time
from pathlib import Path

TRACE=Path(__file__).parent/"traces"/"trace.jsonl"

#记录日志文件
def _log(kind,**kw):
    TRACE.parent.mkdir(exist_ok=True)
    rec={"ts":round(time.time(),1),"kind":kind,**kw}
    with open(TRACE,"a",encoding="utf-8") as f:
        f.write(json.dumps(rec,ensure_ascii=False)+'\n')

from langchain_core.tools import tool
from langchain_openai import  ChatOpenAI
from langgraph.graph import StateGraph,START,END,MessagesState
from langgraph.prebuilt import ToolNode,tools_condition

from search_layer import search as search_raw,filter_results,text_of
from refs import rebuild_refs
import memory as mem_store

DS_KEY = os.getenv("DEEPSEEK_API_KEY")
CHAT_URL = "https://api.deepseek.com/chat/completions"

MAX_PAGE_CHARS=2000
POOL={}
_next_idx=0
CURRENT={"topic":"","mem":None}

SYSTEM="""你是调研助手，给定一个主题，你自己搜索、阅读，最后写出一份调研报告。
要求：
1. 先把主题拆成 2~3 个关键词方向，分别调用 search_web
2. 对关键的 3~5 条调用 read_page 取正文
3. 资料够了就停止调用工具，直接输出报告
4. 报告用 markdown：第一行 # 标题，之后每节用 ## 开头，至少 3 节
5. 每个论断后用 [编号] 标出处，编号只能用 search_web 返回的 idx
6. 末尾用 `## 参考来源` 逐条列出用到的编号和 URL

只能写搜索结果里有的信息，不要用你自己的知识补充。
"""

@tool
def search_web(query: str, count: int = 6) -> str:
    """搜索网页，返回标题、摘要（约600字）、发布日期、URL 和编号。想看全文需再用 read_page。"""
    global _next_idx
    t0=time.time()
    mem = CURRENT["mem"]

    kept, _ = filter_results(search_raw(query, count=count * 2))

    if mem:
        seen = mem_store.known_urls(mem)
        kept = [r for r in kept if r.get("url") not in seen]

    kept = kept[:count]
    if not kept:
        return json.dumps({"msg": "没有符合条件的结果，换个关键词试试"}, ensure_ascii=False)

    out = []
    for r in kept:
        _next_idx += 1
        POOL[_next_idx] = r
        out.append({
            "idx": _next_idx,
            "title": r.get("title"),
            "summary": (r.get("summary") or "")[:600],
            "date": (r.get("date") or "")[:10],
            "url": r.get("url"),
        })

    _log("tool",name="search_web",ms=round((time.time()-t0)*1000),n=len(out))
    return json.dumps(out, ensure_ascii=False)

@tool
def read_page(idx: int, start: int = 0) -> str:
    """按编号读取某条搜索结果的正文，单次最多返回 2000 字，超出需用 start 翻页。"""
    t0=time.time()
    r = POOL.get(int(idx))
    if not r:
        return json.dumps({"error": f"编号 {idx} 不存在，请先用 search_web"}, ensure_ascii=False)

    if CURRENT["mem"]:
        mem_store.add_page(CURRENT["mem"], r.get("url"), r.get("title"), r.get("summary"))

    content = text_of(r)
    part = content[start:start + MAX_PAGE_CHARS]

    _log("tool",name="read_page",ms=round((time.time()-t0)*1000),idx=idx)
    return json.dumps({
        "idx": idx,
        "title": r.get("title"),
        "url": r.get("url"),
        "total_chars": len(content),
        "has_more": start + MAX_PAGE_CHARS < len(content),
        "content": part,
    }, ensure_ascii=False)

TOOLS_LIST=[search_web,read_page]

llm=ChatOpenAI(
    model="deepseek-flash",
    api_key=DS_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,
).bind_tools(TOOLS_LIST)

def agent_node(state:MessagesState):
    """调模型：给它 system + 全部历史，它决定调工具还是写报告"""
    # resp=llm.invoke([("system",SYSTEM)]+state["messages"])
    # return {"messages":[resp]}
    t0=time.time()
    resp=llm.invoke([("system",SYSTEM)]+state["messages"])
    u=getattr(resp,"usage_metadata",None) or {}
    _log("llm",
         ms=round((time.time()-t0)*1000),
         in_tok=u.get("input_tokens",0),
         out_tok=u.get("output_tokens",0),
         tool_calls=len(getattr(resp,"tool_calls",None) or []))
    return{"messages":[resp]}

def build_graph():
    g=StateGraph(MessagesState)
    g.add_node("agent",agent_node)
    g.add_node("tools",ToolNode(TOOLS_LIST,handle_tool_errors=True))

    g.add_edge(START,"agent")
    #模型要调工具就去tools,写完报告（无tool_calls）就结束
    g.add_conditional_edges("agent",tools_condition,{"tools":"tools",END:END})
    g.add_edge("tools","agent")

    return g.compile()

def run(topic):
    """和 agent.py 的 run 同签名，方便对比"""
    global _next_idx
    POOL.clear()
    _next_idx=0

    m=mem_store.load(topic)
    CURRENT["topic"]=topic
    CURRENT["mem"]=m

    if m["pages"]:
        print(f"[记忆] 加载{len(m['pages'])}篇历史资料")

    app=build_graph()
    out=app.invoke({"messages":[("user",f"请调研这个主题并写出报告：{topic}")]})

    msgs=out["messages"]
    report=msgs[-1].content or ""
    report,missing=rebuild_refs(report,POOL)
    if missing:
        print(f"  ⚠️正文引用了不存在的编号：{missing}")

    m['runs']=m.get('runs',0)+1
    mem_store.save(topic,m)

    steps=sum(1 for x in msgs if type(x).__name__=="AIMessage")
    return report,{"steps":steps,"msgs":len(msgs)}


if __name__=="__main__":
    topic="企业级 AI Agent 在 2026 年的落地现状与主要阻碍"
    report,stats=run(topic)
    print(f"\n统计：{stats}\n")
    print("="*70)
    print(report)
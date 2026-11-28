# langgraph_multi.py —— 双 Agent：研究员 + 审稿人（对比 langgraph_agent.py）
import json

from langchain_core.messages import HumanMessage,AIMessage
from langgraph.channels import topic
from langgraph.graph import StateGraph,START,END,MessagesState
from langgraph.prebuilt import ToolNode

import langgraph_agent as base
import memory as mem_store
from refs import rebuild_refs

MAX_REWRITE=2 #最多重写两轮，防止死循环

# REVIEW_PROMPT="""你是审稿人。检查下面这份调研报告，只做判断不改写。
#
# 检查三项：
# 1.正文里出现的每个[编号]，在“参考来源”里都有对应条目
# 2.正文含具体数据或机构名，不是通篇空话
# 3.结构完整：至少3个 ## 小节，且有参考来源
#
# 只输出JSON，不要任何解释：
# {"pass":true或false,"issues":["问题1","问题2"]}
#
# 报告：
# ---
# __REPORT__
# ---"""

REVIEW_PROMPT = """你是审稿人。检查下面这份调研报告，只做判断不改写。
如果这份内容是「调研无法完成」的说明而非正式报告，直接判 pass=true，不要检查格式。

以下任意一条成立，pass 必须为 false：
- 正文里出现了「参考来源」列表中没有的 [编号]
- 只有不到 3 个 ## 小节
- 没有「参考来源」或「参考资料」列表
- 通篇没有具体数字、机构名或时间，全是空话

只有全部满足，pass 才能为 true。

只输出 JSON，不要任何解释：
{"pass": true 或 false, "issues": ["问题1", "问题2"]}

报告：
---
__REPORT__
---"""

class MultiState(MessagesState):
    """在消息之外加三个字段：审稿是否通过、意见、已重写轮数"""
    passed:bool
    issues:list
    rounds:int

def reviewer_node(state:MultiState):
    """审稿：读最后一份报告，判断是否通过；不通过就把意见塞回消息里"""
    report=""
    for m in reversed(state["messages"]):
        if isinstance(m,AIMessage) and not getattr(m,"tool_calls",None):
            report=m.content or ""
            break

    resp=base.llm.invoke([("user",REVIEW_PROMPT.replace("__REPORT__",report[:8000]))])

    text=(resp.content or "").strip()
    data={}
    if "{" in text and "}" in text:
        try:
            data=json.loads(text[text.find("{"):text.find("}")+1])
        except json.JSONDecodeError:
            data={}

    passed=bool(data.get("pass",False)) # 解析失败就当不通过
    issues=data.get("issues") or []
    rounds=state.get("rounds",0)+1 if not passed else state.get("rounds",0)

    print(f"  [审稿]{'通过' if passed else '不通过'}"
          +("" if passed else f":{issues[:2]}"))

    if passed:
        return {"passed":True,"issues":[]}

    feedback=HumanMessage(
        content="审稿意见（请据此修改后重新输出完整报告）：\n"
                +'\n'.join(f"-{i}" for i in issues)
    )
    return {"passed":False,"issues":issues,"rounds":rounds,"messages":[feedback]}

def route_after_agent(state:MultiState):
    """有工具调用就继续查，否则说明写完了，送去审稿"""
    last=state["messages"][-1]
    if getattr(last,"tool_calls",None):
        return "tools"

    return "reviewer"

def route_after_review(state:MultiState):
    """通过就结束；超过重写上限也结束（避免死循环）"""
    if state.get("passed"):
        return "done"
    if state.get("rounds",0)>=MAX_REWRITE:
        print(f"  【审稿】已达重写上限{MAX_REWRITE}轮，强制结束")
        return "done"

    return "redo"

def build_graph():
    g=StateGraph(MultiState)
    g.add_node("agent",base.agent_node) #复用研究员
    g.add_node("tools",ToolNode(base.TOOLS_LIST))
    g.add_node("reviewer",reviewer_node)#新增审稿人

    g.add_edge(START,"agent")
    g.add_conditional_edges("agent",route_after_agent,
                            {"tools":"tools","reviewer":"reviewer"})
    g.add_edge("tools","agent")
    g.add_conditional_edges("reviewer",route_after_review,
                            {"done":END,"redo":"agent"})

    return g.compile()

def run(topic):
    base.POOL.clear()
    base._next_idx=0

    m=mem_store.load(topic)
    base.CURRENT["topic"]=topic
    base.CURRENT["mem"]=m

    app=build_graph()
    out=app.invoke({
        "messages":[("user",f"请调研这个主题并写出报告:{topic}")],
        "passed":False,"issues":[],"rounds":0,
    })

    msgs=out["messages"]
    report=""
    for x in reversed(msgs):
        if isinstance(x,AIMessage) and not getattr(x,"tool_calls",None):
            report=x.content or ""
            break

    report,missing=rebuild_refs(report,base.POOL)
    if missing:
        print(f"  ⚠ 正文引用了不存在的编号：{missing}")

    m['runs']=m.get('runs',0)+1
    mem_store.save(topic,m)

    return report,{
        "passed":out.get('passed'),
        "rewrites":out.get('rounds',0),
        "msgs":len(msgs),
    }


if __name__=="__main__":
    topic="企业级 AI Agent 在 2026 年的落地现状与主要阻碍"
    report,stats=run(topic)
    print(f"\n统计：{stats}\n")
    print("="*70)
    print(report)
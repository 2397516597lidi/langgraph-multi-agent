# eval_multi.py —— 双 Agent 批量评测，重点看打回率
import os
import shutil
import time

if os.path.exists("memory"):
    shutil.rmtree("memory")

from langgraph_multi import run

TOPICS = [
    "企业级 AI Agent 在 2026 年的落地现状与主要阻碍",
    "RAG 技术在企业知识库问答中的应用与局限",
    "AI Agent 记忆机制的主流技术方案对比",
    "小模型替代大模型在垂直场景中的可行性",
    "多智能体协作系统的主要失败原因",
    "2026 年 AI 编程助手工具的格局与差异",
    "MCP（模型上下文协议）的采用现状与企业实践",
    "AI 客服系统在真实业务中的效果与问题",
    "医疗影像 AI 产品落地的主要障碍",
    "向量数据库的选型对比与性能瓶颈",
]

rows=[]
for i,topic in enumerate(TOPICS,1):
    print(f"\n[{i}/{len(TOPICS)}] {topic}")
    t0=time.time()
    report,st=run(topic)
    secs=round(time.time()-t0,1)
    rows.append((topic,st['passed'],st['rewrites'],st['msgs'],secs))
    print(f"  => passed={st['passed']} rewrites={st['rewrites']} "
          f"msgs={st['msgs']} {secs}s")

print("\n"+"="*70+"\n汇总\n")
print(f"{'结果':<6}{'重写':<6}{'msgs':<6}{'耗时':<8}主题")
for topic,p,r,m,s in rows:
    print(f"{'通过' if p else '未过':<6}{r:<6}{m:<6}{str(s)+'s':<8}{topic[:20]}")

passed=sum(1 for r in rows if r[1])
total_rw=sum(r[2] for r in rows)
avg_msgs=sum(r[3] for r in rows)/len(rows)
print(f"\n通过率：{passed}/{len(rows)}")
print(f"\n总重写次数：{total_rw}")
print(f"\n平均消息数：{avg_msgs:.1f}")
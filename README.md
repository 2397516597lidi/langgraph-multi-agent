# LangGraph 多 Agent 调研系统

用 LangGraph 把「自动调研 Agent」重写为显式工作流，并做一个对比实验：**单 Agent vs 双 Agent（研究员 + 审稿人），多一个 Agent 到底值不值。**&#8203;

## 结论先说

| 指标 | 单 Agent | 双 Agent（+审稿） |
|---|---|---|
| 格式通过率 | 10 / 10 | 9 / 10 |
| 平均耗时 | 35.1s | 31.1s |
| 平均消息数 | ~16 | 22.4 |
| **引用问题检出** | **0（没查）**&#8203; | **3 篇** |
| 重写次数 | 0 | 4 |

**核心发现：单 Agent 那个 10/10 是不可信的。**&#8203;
审稿人揪出 3 篇引用不一致（正文 `[3]` 在列表里没有 / 报告截断无来源列表 / 编号与列表系统性错位），而这三篇在单 Agent 的格式检查里全部「通过」——因为那套检查只数长度、条数、章节数，根本拦不住引用造假。
**加一个审稿节点，多花约 3 步，换来 3 篇问题报告被拦下或修好。**&#8203; 这笔账是否划算，取决于业务能不能接受引用错误——对调研报告来说，不能。

## 两个版本

| 文件 | 内容 |
|---|---|
| `langgraph_agent.py` | 单 Agent：`agent ⇄ tools`，写完即结束 |
| `langgraph_multi.py` | 双 Agent：多一个 `reviewer` 节点，不通过退回重写 |

```
单 Agent：
START → agent ⇄ tools → END

双 Agent：
START → agent ⇄ tools → reviewer → pass? → END
                           ↑ fail（带意见）
                           └──── agent 重写
```

**`agent` 节点两个版本完全一样**，只改了图——这是 LangGraph 相比手写 `while` 循环最直观的好处：改流程不用动业务代码。

## 关键设计
### 1. 三个内置件替掉 60 行手写循环

| 用的 | 替掉了什么 |
|---|---|
| `MessagesState` | 自己维护 `messages` 列表 |
| `ToolNode` | 手写解析 `tool_calls` 并执行 |
| `tools_condition` | 手写「有工具调用就去执行，没有就结束」的判断 |

### 2. 审稿不通过时，必须把意见塞回消息里

```python
feedback = HumanMessage(content="审稿意见（请据此修改）：\n- ...")
return {"passed": False, "issues": issues, "messages": [feedback]}
```

**不塞的话模型不知道要改什么，会原样重写一遍。**&#8203;
### 3. 用 `MAX_REWRITE` 防止「审→改→审」死循环
```python
MAX_REWRITE = 2
```

没有上限时，审稿人和研究员可能无限来回。这是多 Agent 最典型的事故模式。

### 4. 引用重排仍然需要程序兜底

`refs.py` 从正文提取编号、查 `POOL` 取 URL、重建来源列表。

**换框架不改变模型习惯**——它照样会在列来源列表时自己从 1 重排。框架管流程，管不了输出格式。


## 踩过的坑

| 问题 | 现象 | 根因 |
|---|---|---|
| `format` 与 JSON 冲突 | `KeyError: '"pass"'` | `str.format()` 把 JSON 的 `{"pass": ...}` 当成占位符，改用 `replace` |
| **审稿人是摆设** | 引用不存在的编号、只有 2 节、无来源列表，照样判通过 | ① `data.get("pass", True)` 默认放过；② 提示词没写「不满足必须 false」 |
| 模型改不动 | 报告被截断后再审 2 次仍未补上来源列表 | 撞 `MAX_REWRITE` 上限，只能接受不合格输出 |

**第二类最危险：不报错、不崩，只是把所有东西都放过。**&#8203; 只有主动造一份「明显该打回」的报告去测才会现形。

## 局限

- **只做同主题一次会话**，不做跨会话的长期记忆（记忆机制在同目录的 `memory.py`，见调研 Agent 项目）
- **审稿只覆盖格式与引用一致性**，不判断内容真伪
- **10 个主题的样本太小**，单题波动就有 20s，耗时数据不可当结论
- 没有 token 统计，成本只能用消息数间接估计

## 文件

| 文件 | 作用 |
|---|---|
| `langgraph_agent.py` | 单 Agent（LangGraph） |
| `langgraph_multi.py` | 双 Agent（+审稿） |
| `search_layer.py` | 搜索层：接口调用 + 三道筛 |
| `refs.py` | 引用重排 |
| `memory.py` | 跨任务记忆 |
| `eval_multi.py` | 双 Agent 批量评测 |

## 运行

```bash
pip install langgraph langchain-openai
set DEEPSEEK_API_KEY=sk-...

python langgraph_agent.py    # 单 Agent
python langgraph_multi.py    # 双 Agent
python eval_multi.py         # 批量评测，看打回率
```
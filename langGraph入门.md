# LangGraph入门

[TOC]

## uv介绍

‌**UV 是 Astral 公司开发的超高性能 Python 包管理工具**‌，基于 Rust 构建，旨在替代传统工具（如 pip、pip-tools、Poetry 等），提供统一的依赖管理、虚拟环境控制和项目初始化功能，其速度比 pip 快 10-100 倍]()

- 安装方式

  通过官方脚本（推荐）或包管理器安装：

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh  # 官方脚本
  pipx install uv  # 通过 pipx 安装
  ```



- 添加源

  >[[index]]
  >url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/"
  >default = true 

  ​


- 常用命令

  ```bash
  1. 项目初始化
  	uv init project-name  or uv init --package test_2 --python 3.10.9
  2. 虚拟环境
  	uv venv --python 3.10.9
  3. 激活虚拟环境
  	# macos & linux
  	source .venv/bin/activate
  	# windows
  	.venv\Scripts\activate
  4. 添加/删除依赖
  	uv add/remove langgraph
  5. 同步环境
  	uv sync
  6. 查看当前项目依赖
  	uv tree	
  	
  ```



- 参考:https://juejin.cn/post/7485575064899174415#heading-2

  ​


## LangGraph

[LangGraph官方文档](https://langchain-ai.github.io/langgraph/)

### 环境准备

直接复制到uv的pyproject.toml中，执行uv sync

```json
dependencies = [
    "dateparser>=1.2.1",
    "fastapi>=0.115.12",
    "langchain-community>=0.3.24",
    "langchain[openai]>=0.3.25",
    "langgraph>=0.4.5",
    "loguru>=0.7.3",
    "mysql-connector-python==9.0",
    "pandas>=2.2.3",
    "pymysql>=1.1.1",
    "redis>=6.1.0",
    "sse-starlette>=2.3.5",
    "uvicorn>=0.34.2",
]
```



### 核心概念

LangGraph 通过 **有向图**（Directed StateGraph） 建模流程：

- 节点（Node） ：执行具体操作的单元（如调用LLM、工具函数、状态更新）。
- 边（Edge） ：定义节点间的执行顺序（单向或条件分支）。
- 状态（State） ：在图中流动的数据载体（如用户输入、中间结果、最终输出）。
- ​

添加节点

- **方法 **：`add_node(node_name, node_function)`

- 参数 

  - `node_name`：节点唯一标识符（字符串）。
  - `node_function`：处理函数，输入为当前状态（字典），返回更新后的状态。

- 示例 

  ```python
  def process_input(state):

      user_input = input("Enter something: ")

      return {"user_input": user_input, "next_step": "analyze"}

  graph.add_node("process_input", process_input)

  ```

  ​

连接节点（边)

- **方法 **：`add_edge(from_node, to_node)`

- **用途 **：定义节点间的固定执行顺序。

- 示例 

  ```python
  graph.add_edge("process_input", "analyze")  # process_input完成后执行analyze
  ```



条件分支（动态边）

- **方法 **：`add_conditional_edges(from_node, condition_function)`

- **用途 **：根据条件动态选择下一个节点。

- 示例 

  ```python
  def route_based_on_age(state):
      age = state.get("age", 0)
      if age >= 18:
          return "adult_path"
      else:
          return "minor_path"
  graph.add_conditional_edges("ask_age", route_based_on_age,["adult_path","minor_path"])

  ```

  ​

设置入口/终止点

- 方法 
  - `set_entry_point(node_name)`：定义图的起始节点。
  - `set_finish_point(node_name)`：定义图的终止节点。
- **注意 **：终止节点会结束流程，返回最终状态。




#### 状态图

>`StateGraph`是要使用的主要图形类。它由用户定义的`State`对象参数化。
>
>构建图，首先要定义[状态](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)，然后添加[节点](https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes)和[边](https://langchain-ai.github.io/langgraph/concepts/low_level/#edges)，最后进行编译。编译时会对图的结构进行一些基本检查（例如，没有孤立节点等）。你还可以在其中指定运行时参数，例如[检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/)和断点。只需调用以下`.compile`方法即可编译图：

```python
graph = graph_builder.compile(...)
```



#### 状态

>定义图时，首先要定义`State` 。`State`由[图的模式](https://langchain-ai.github.io/langgraph/concepts/low_level/#schema)以及指定如何将更新应用于状态的[`reducer`函数](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)组成。图的模式可以是**TypedDict**，也可以是**Pydantic**

```python
# TypedDict
from typing_extensions import TypedDict
class State(TypedDict):
    user_id: Annotated[str, lambda a, b: b]
    name: Annotated[str, lambda a, b: b]
    age: Annotated[int, lambda a, b: b]
    other_info: Annotated[dict, lambda a, b: b]
# Pydantic    
def get_app_5():
    from pydantic import BaseModel, Field
    class PydanticState(BaseModel):
        subjects: Annotated[list, operator.add] = Field(default=[], description="主题列表")
        user_name: str = Field(..., description="用户姓名")

    def node_a(state: PydanticState, config: RunnableConfig):
        print("step:node_a")
        state.subjects.append("node_a")
        return state

    work_flow: StateGraph = StateGraph(PydanticState)
    work_flow.add_node("node_a", node_a)
    work_flow.add_edge(START, "node_a")
    work_flow.add_edge("node_a", END)
    app = work_flow.compile()
    print(app.invoke(input=PydanticState(user_name="张三"), config=runnableConfig))
```



状态的更新

```python
from langchain_core.messages import AIMessage

## 在节点中手动更新状态，不使用函数
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict


class State(TypedDict):
    messages: list[AnyMessage]
    extra_field: int
def node(state: State):
    messages = state["messages"]
    new_message = AIMessage("Hello!")
    return {"messages": messages + [new_message], "extra_field": 10}
graph = StateGraph(State).add_node(node).add_edge(START, "node").compile()

result = graph.invoke({"messages": [HumanMessage("Hi")]})

for message in result["messages"]:
    message.pretty_print()
#=============================================================================
# 使用 Reducer 进行状态更新
from typing_extensions import Annotated
def add(left, right):
    return left + right

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    extra_field: int
        
def node(state: State):
    new_message = AIMessage("Hello!")
    return {"messages": [new_message], "extra_field": 10}
graph = StateGraph(State).add_node(node).add_edge(START, "node").compile()

result = graph.invoke({"messages": [HumanMessage("Hi")]})

for message in result["messages"]:
    message.pretty_print()
```



#### 节点

- 普通节点：通常是一个python函数

```python
    def node_a(state: PydanticState, config: RunnableConfig):
        print("step:node_a")
        state.subjects.append("node_a")
        return state
```



- START节点：`START`是一个特殊节点，表示将用户输入发送到图的节点。引用此节点的主要目的是确定应首先调用哪些节点。

```python
from langgraph.graph import START

graph.add_edge(START, "node_a")
```



- END节点：`END`是一个特殊节点，表示终端节点。当需要指示哪些边在完成后没有操作时，可以引用此节点。

```python
from langgraph.graph import END

graph.add_edge("node_a", END)
```



#### 边

- 普通边：直接从一个节点到下一个节点。
- 条件边：调用一个函数来确定下一步要去哪个节点。
- 入口点：当用户输入到达时首先调用哪个节点。
- 条件入口点：调用一个函数来确定当用户输入到达时首先调用哪个节点。

1. 普通边

   表示由A节点到B节点直接使用[add_edge](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph.add_edge)这个方法

   >graph.add_edge("node_a", "node_b")

2. 条件边

   **选择性的**路由到一条或多条边（或选择性地终止），可以使用[add_conditional_edges](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph.add_conditional_edges)方法。此方法接受节点名称以及执行该节点后要调用的“路由函数“。路由函数表示需要从节点路由到哪些节点

   ```python
   def router(state: State, config: RunnableConfig):
       if state["age"] < 18:
           return "node_a"
       else:
           return "node_b"
   graph.add_conditional_edges("node", routing_function，["node_a","node_b"])    
   ```

3. 条件入口点

   可以使用[`add_conditional_edges`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph.add_conditional_edges)虚拟[`START`](https://langchain-ai.github.io/langgraph/reference/constants/#langgraph.constants.START)节点来实现这一点。参考上面的条件边



#### send

应用中，上游节点可能生成一个数据列表，下游节点需要消费这些列表，而每次的状态都是独立的。

`Send`接受两个参数：第一个是节点的名称，第二个是传递给该节点的状态

代码示例：send_graph.py

![send_graph](send_graph.png)



#### Command

在同一个节点中同时执行状态更新和决定下一步要转到哪个节点。LangGraph 提供了一种方法，它通过[`Command`](https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.Command)从节点函数返回一个对象来实现：

> 在节点函数中返回时`Command`，必须添加返回类型注释，其中包含节点路由到的节点名称列表，例如`Command[Literal["my_other_node"]]`。这对于图形渲染是必需的，它告诉 LangGraph `my_node`可以导航到`node_a`。

```python
def my_node(state: State) -> Command[Literal["node_a"]]:
    return Command(
        # state update
        update={"foo": "bar"},
        # control flow
        goto="node_a",
        graph=Command.PARENT
    )

```

>设置`graph`为`Command.PARENT`将导航到最近的父图



#### 图的递归限制

在图单次运行过程中，限制图的运行最大步骤，避免进入死循环。默认值是25值，通常在调用过程需要按照实际情况来设置。通过Runnable来进行设置

```python
RunnableConfig(configurable={"thread_id": "1"},recursion_limit=3)
```



#### 延迟执行

| 图一                             | 图二                                       |
| ------------------------------ | ---------------------------------------- |
| ![branch_1](branch_1.png)      | ![branch_2](branch_2.png)                |
| 由于b,c节点在同一个步骤中，所以b,c执行完之后才会执行d | 由于b节点有一个延伸节点，b,c不处于同一个步骤，d节点会执行两次。<br />通过设置节点时控制执行： builder.add_node(d, defer=True) |

> 设置`defer=True`,节点`d` 会在所有待处理任务完成后才会执行



#### 图-线程持久化

> LangGraph 内置了持久层，通过检查点实现。当您使用检查点编译图时，检查点会`checkpoint`在每个超级步骤中保存图状态的副本。这些检查点保存到 中`thread`，可在图执行后访问。
>
> 使用 LangGraph API 时，您无需手动实现或配置检查点。API 会在后台为您处理所有持久化基础架构。
>
> [线程](https://langchain-ai.github.io/langgraph/concepts/persistence/#threads)
>
> 线程是分配给检查点程序保存的每个检查点的唯一 ID 或[线程标识符。使用检查点程序调用图表时，](https://langchain-ai.github.io/langgraph/concepts/persistence/#threads)**必须**在配置部分`thread_id`中指定：`configurable`
>
> ```python
> {"configurable": {"thread_id": "1"}}
> ```



##### 检查点

[文档](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpoints)

检查点是在每个超级步骤中保存的图形状态的快照，并由`StateSnapshot`具有以下关键属性的对象表示：

- `config`：与此检查点相关的配置。
- `metadata`：与此检查点相关的元数据。
- `values`：此时状态通道的值。
- `next` ：图中接下来要执行的节点名称的元组。
- `tasks``PregelTask`：包含后续待执行任务信息的对象元组。如果之前尝试过此步骤，则将包含错误信息。如果图在节点内部被[动态中断，则任务将包含与中断相关的其他数据。](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/#dynamic-breakpoints)

##### 获取状态

> 与已保存的图表状态交互时，**必须**指定[线程标识符](https://langchain-ai.github.io/langgraph/concepts/persistence/#threads)。您可以通过调用来查看图表的*最新*`graph.get_state(config)`状态。这将返回一个`StateSnapshot`对象，该对象对应于与配置中提供的线程 ID 关联的最新检查点，或与线程的检查点 ID 关联的检查点（如果提供）

```python
# 获取最新的
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)

# 获取指定id的
config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
graph.get_state(config)

# 获取历史记录
config = {"configurable": {"thread_id": "1"}}
list(graph.get_state_history(config))


```



##### 添加持久化

参考官方文档：https://langchain-ai.github.io/langgraph/how-tos/persistence/#add-short-term-memory



示例-利用检测点实现人机交互

示例代码：demo_2.py

![demo2_1](demo2_1.png)



#### 图节点的流式输出

| 模式                                       | 描述                                       |
| ---------------------------------------- | ---------------------------------------- |
| [`values`](https://langchain-ai.github.io/langgraph/how-tos/streaming/#stream-graph-state) | 在图的每个步骤之后流式传输状态的完整值。                     |
| [`updates`](https://langchain-ai.github.io/langgraph/how-tos/streaming/#stream-graph-state) | 将图的每个步骤之后的更新流式传输到状态。如果在同一步骤中进行了多个更新（例如，运行了多个节点），则这些更新将分别流式传输。 |
| [`custom`](https://langchain-ai.github.io/langgraph/how-tos/streaming/#stream-custom-data) | 从图形节点内部流式传输自定义数据。                        |
| [`messages`](https://langchain-ai.github.io/langgraph/how-tos/streaming/#messages) | 从调用 LLM 的任何图形节点流式传输 2 元组（LLM 令牌、元数据）。    |
| [`debug`](https://langchain-ai.github.io/langgraph/how-tos/streaming/#debug) | 在整个图表执行过程中传输尽可能多的信息。                     |

```python
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)

# 异步
async for chunk in graph.astream(inputs, stream_mode="updates"):
    print(chunk)
```



> 可以传递一个列表作为`stream_mode`参数来同时传输多种模式
>
> 流输出将是流模式名称和`(mode, chunk)`该模式流式传输的数据的三元组。

```python
for mode, chunk in graph.stream(inputs, stream_mode=["updates", "custom"]):
    print(chunk)

async for mode, chunk in graph.astream(inputs, stream_mode=["updates", "custom"]):
    print(chunk)    
```



> [要将子图](https://langchain-ai.github.io/langgraph/concepts/subgraphs/)的输出包含在流输出中，您可以在父图的方法`subgraphs=True`中进行设置`.stream()`。这将同时从父图和任何子图流输出

```python
for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True, 
    stream_mode="updates",
):
    print(chunk)
```



> 按节点过滤：要仅从特定节点流式传输令牌，请使用流元数据中定义的mode来进行节点匹配

```python
for msg, metadata in graph.stream( 
    inputs,
    stream_mode="messages",
):
    if msg.content and metadata["langgraph_node"] == "some_node_name": 
        print(msg.content)
```

##### 流式传输自定义数据

1. 用于`get_stream_writer()`访问流写入器并发出自定义数据。
2. `stream_mode="custom"`调用`.stream()`或时设置，`.astream()`用于获取流中的自定义数据。您可以组合多种模式（例如`["updates", "custom"]`），但至少必须有一种`"custom"`。

> Python < 3.11 版本中不支持`get_stream_writer()`async
>
> 在 Python 3.11 以下版本上运行的异步代码`get_stream_writer()`将无法正常工作。
> 请改为`writer`向节点或工具添加参数并手动传递。有关使用示例，
> 请参阅[Python 3.11 以下版本的异步代码。](https://langchain-ai.github.io/langgraph/how-tos/streaming/#async)

```python
from typing import TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START

class State(TypedDict):
    query: str
    answer: str

def node(state: State):
    writer = get_stream_writer()  
    writer({"custom_key": "Generating custom data inside node"}) 
    return {"answer": "some data"}

graph = (
    StateGraph(State)
    .add_node(node)
    .add_edge(START, "node")
    .compile()
)

inputs = {"query": "example"}

# Usage
for chunk in graph.stream(inputs, stream_mode="custom"):  
    print(chunk)
```



### 示例

#### 图构建示例

示例代码:demo_1.py

------

| 图片                                       | 方法                  |
| ---------------------------------------- | ------------------- |
| ![demo1_1](demo1_1.png)                  | get_app_1           |
| ![demo1_2](demo1_2.png)                  | get_app_2           |
| ![demo1_3](demo1_3.png)                  | get_app_3           |
| ![demo1_4](demo1_4.png)                  | get_app_4           |
| ![react-agent-demo](react-agent-demo.png) | react_agent_demo.py |


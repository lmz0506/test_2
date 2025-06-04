from datetime import datetime

from fastapi import APIRouter, Request,HTTPException
import json
from src.Config import get_global_config
from sse_starlette.sse import EventSourceResponse

from src.model.ChatModel import ChatRequest, Message, ChatResponse
from src.service.StatisticsLangGraph_02 import statistics_chat_stream
from src.service.sqlgraph.SqlGraph import sql_graph

import uuid

from src.service.sqlgraph.TaskUtil import AsyncTaskManager

api_chat = APIRouter()

config = get_global_config()
task_manager = AsyncTaskManager()


def get_time():
    # 获取当前时间
    now = datetime.now()
    # 转换为 yyyy-MM-dd HH:mm:ss 格式
    return now.strftime("%Y-%m-%d %H:%M:%S")


async def llm_stream(chatRequest: ChatRequest, request: Request = None):
    # 安全获取第一条消息内容
    msgs: list[Message] = chatRequest.messages
    message_content = msgs[0].content if msgs else "你好！"
    llm = config.llm

    try:
        async for chunk in llm.astream(message_content):
            if await request.is_disconnected():
                print("客户端已断开连接，停止生成")
                break

            # 安全获取 finish_reason
            finish_reason = chunk.response_metadata.get('finish_reason') if hasattr(chunk,
                                                                                    'response_metadata') else None
            if finish_reason == "stop" or chunk.content:
                response = ChatResponse(
                    content=chunk.content,
                    created_at=get_time(),
                    is_done=finish_reason == "stop",
                    message_id=chunk.id,
                    message_type="text",
                    thread_id=chatRequest.thread_id
                )
                yield response.model_dump_json()

    except Exception as e:
        error_response = {
            "error": str(e),
            "is_done": True
        }
        yield json.dumps(error_response, ensure_ascii=False)


@api_chat.post("/chat.do", description="sse模型聊天接口", response_model=ChatResponse)
async def post_test(chatRequest: ChatRequest, request: Request, ):
    return EventSourceResponse(llm_stream(request=request, chatRequest=chatRequest))


@api_chat.post("/chat2.do", description="sse模型聊天接口", response_model=ChatResponse)
async def post_test(chatRequest: ChatRequest, request: Request):
    chatRequest.message_id = str(uuid.uuid4())
    return EventSourceResponse(sql_graph(chatRequest=chatRequest, request=request))


@api_chat.post("/chatTest.do", description="sse模型聊天接口-测试", response_model=ChatResponse)
async def chat_test(chatRequest: ChatRequest, request: Request, ):
    return EventSourceResponse(statistics_chat_stream(request=request, chatRequest=chatRequest))



@api_chat.post("/cancel_task/{task_id}")
async def cancel_task(task_id: str):
    success = await task_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or already completed")
    return {"status": "cancelled"}

@api_chat.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    task_info = task_manager.get_task_info(task_id)
    if task_info is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_info
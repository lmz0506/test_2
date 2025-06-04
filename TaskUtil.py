import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Awaitable, TypeVar
from dataclasses import dataclass
import uuid

T = TypeVar('T')  # 定义泛型类型


@dataclass
class TaskInfo:
    """任务信息类"""
    task: asyncio.Task
    create_time: datetime
    task_type: str
    metadata: Dict[str, Any]


class AsyncTaskManager:
    """
    异步任务管理器
    支持任务的添加、取消、状态查询等功能
    """

    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._logger = logging.getLogger(__name__)

    async def create_task(
            self,
            coro: Awaitable[T],
            task_type: str,
            task_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        创建并注册一个新的异步任务

        Args:
            coro: 要执行的协程
            task_type: 任务类型标识
            task_id: 可选的任务ID，如果不提供则自动生成
            metadata: 可选的任务元数据

        Returns:
            task_id: 任务ID
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        if task_id in self._tasks:
            raise ValueError(f"Task with ID {task_id} already exists")

        wrapped_coro = self._wrap_task(coro, task_id)
        task = asyncio.create_task(wrapped_coro)

        self._tasks[task_id] = TaskInfo(
            task=task,
            create_time=datetime.utcnow(),
            task_type=task_type,
            metadata=metadata or {}
        )

        self._logger.info(f"Created task {task_id} of type {task_type}")
        return task_id

    async def _wrap_task(self, coro: Awaitable[T], task_id: str) -> T:
        """
        包装任务协程，添加清理逻辑
        """
        try:
            return await coro
        except asyncio.CancelledError:
            self._logger.info(f"Task {task_id} was cancelled")
            raise
        except Exception as e:
            self._logger.error(f"Task {task_id} failed with error: {str(e)}")
            raise
        finally:
            if task_id in self._tasks:
                del self._tasks[task_id]

    async def cancel_task(self, task_id: str) -> bool:
        """
        取消指定的任务

        Args:
            task_id: 要取消的任务ID

        Returns:
            bool: 是否成功取消任务
        """
        if task_id not in self._tasks:
            self._logger.warning(f"Task {task_id} not found")
            return False

        task_info = self._tasks[task_id]
        if task_info.task.done():
            return False

        task_info.task.cancel()
        try:
            await task_info.task
        except asyncio.CancelledError:
            self._logger.info(f"Successfully cancelled task {task_id}")
            return True
        except Exception as e:
            self._logger.error(f"Error while cancelling task {task_id}: {str(e)}")
            return False

    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务信息

        Args:
            task_id: 任务ID

        Returns:
            Optional[Dict]: 任务信息字典，如果任务不存在则返回None
        """
        if task_id not in self._tasks:
            return None

        task_info = self._tasks[task_id]
        return {
            "task_id": task_id,
            "task_type": task_info.task_type,
            "create_time": task_info.create_time.isoformat(),
            "status": "running" if not task_info.task.done() else
            "cancelled" if task_info.task.cancelled() else
            "completed",
            "metadata": task_info.metadata
        }

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有任务的信息

        Returns:
            Dict: 任务ID到任务信息的映射
        """
        return {
            task_id: self.get_task_info(task_id)
            for task_id in self._tasks.keys()
        }

    def cleanup_completed_tasks(self) -> int:
        """
        清理已完成的任务

        Returns:
            int: 清理的任务数量
        """
        completed_tasks = [
            task_id for task_id, task_info in self._tasks.items()
            if task_info.task.done()
        ]

        for task_id in completed_tasks:
            del self._tasks[task_id]

        return len(completed_tasks)

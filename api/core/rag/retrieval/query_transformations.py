import logging
from typing import Optional

from core.model_manager import ModelManager
from core.model_runtime.entities.message_entities import UserPromptMessage
from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.errors.invoke import InvokeError


class QueryTransformation:
    def __init__(self, query: str, tenant_id: Optional[str] = None):
        self._origin_query: str = query
        self._hyde_query: Optional[str] = query
        self._sub_queries: list[str] = [
            query
        ]  # Breaks down the complex query into sub questions.
        self._multi_queries: list[str] = [
            query
        ]  # LLM generated queries that are similar to the user input
        self.tenant_id: Optional[str] = tenant_id

    @property
    def origin_query(self) -> str:
        return self._origin_query

    @property
    def hyde_query(self) -> str:
        if self.tenant_id:
            return self._get_hyde_query(self.tenant_id)
        return self._hyde_query

    @property
    def sub_queries(self) -> list[str]:
        return self._sub_queries

    @property
    def multi_queries(self) -> list[str]:
        return self._multi_queries

    @classmethod
    def hyde(cls, tenant_id: Optional[str], query: str) -> str:
        if not tenant_id:
            return query

        model_manager = ModelManager()

        model_instance = model_manager.get_model_instance(
            provider="openai_api_compatible",
            tenant_id=tenant_id,
            model_type=ModelType.LLM,
            model="Qwen1.5-14B-Chat",
        )

        prompt = f"请你认真思考后简短的回答这个问题：{query}，50个字左右。"  # 后续把 prompt 做封装。

        prompts = [UserPromptMessage(content=prompt)]
        try:
            response = model_instance.invoke_llm(
                prompt_messages=prompts,
                model_parameters={"max_tokens": 70, "temperature": 1},
                stream=False,
            )
            answer = response.message.content
        except InvokeError:
            answer = query
        except Exception as e:
            logging.exception(e)
            answer = query

        return answer

    def _get_hyde_query(self) -> str:
        self._hyde_query = self.hyde(self.tenant_id, self._origin_query)
        return self._hyde_query

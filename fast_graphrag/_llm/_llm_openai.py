"""LLM Services module."""

import asyncio
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, List, Optional, Tuple, Type, cast

import boto3
import instructor
import numpy as np
from openai import APIConnectionError, AsyncOpenAI, RateLimitError, OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fast_graphrag._exceptions import LLMServiceNoResponseError
from fast_graphrag._types import BTResponseModel, GTResponseModel
from fast_graphrag._utils import TOKEN_TO_CHAR_RATIO, logger, throttle_async_func_call

from ._base import BaseEmbeddingService, BaseLLMService

TIMEOUT_SECONDS = 180.0


@dataclass
class OpenAILLMService(BaseLLMService):
    """LLM Service for OpenAI LLMs."""

    model: Optional[str] = field(default="gpt-4o-mini")
    CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    # CHAT_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    # CHAT_MODEL_ID = "meta.llama3-1-405b-instruct-v1:0"

    def __post_init__(self):
        # logger.debug("Initialized OpenAILLMService with patched OpenAI client.")
        # self.llm_async_client: instructor.AsyncInstructor = instructor.from_openai(
        #     AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=TIMEOUT_SECONDS)
        # )
        if self.model == "claude-3.5-sonnet":
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name="us-west-2",
            )
        elif self.model == "gpt-4o-mini":
            self.client = OpenAI()
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    @throttle_async_func_call(
        max_concurrent=256, stagger_time=0.001, waitting_time=0.001
    )
    async def send_message(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        response_model: Type[GTResponseModel] | None = None,
        **kwargs: Any,
    ) -> Tuple[str, list[dict[str, str]]]:
        """Send a message to the language model and receive a response.

        Args:
            prompt (str): The input message to send to the language model.
            model (str): The name of the model to use. Defaults to the model provided in the config.
            system_prompt (str, optional): The system prompt to set the context for the conversation. Defaults to None.
            history_messages (list, optional): A list of previous messages in the conversation. Defaults to empty.
            response_model (Type[T], optional): The Pydantic model to parse the response. Defaults to None.
            **kwargs: Additional keyword arguments that may be required by specific LLM implementations.

        Returns:
            str: The response from the language model.
        """
        # logger.debug(f"Sending message with prompt: {prompt}")
        # model = model or self.model
        # if model is None:
        #     raise ValueError("Model name must be provided.")
        # messages: list[dict[str, str]] = []

        # if system_prompt:
        #     messages.append({"role": "system", "content": system_prompt})
        #     logger.debug(f"Added system prompt: {system_prompt}")

        # if history_messages:
        #     messages.extend(history_messages)
        #     logger.debug(f"Added history messages: {history_messages}")

        # messages.append({"role": "user", "content": prompt})

        # llm_response: GTResponseModel = await self.llm_async_client.chat.completions.create(
        #     model=model,
        #     messages=messages,  # type: ignore
        #     response_model=response_model.Model
        #     if response_model and issubclass(response_model, BTResponseModel)
        #     else response_model,
        #     **kwargs,
        #     max_retries=AsyncRetrying(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)),
        # )

        # if not llm_response:
        #     logger.error("No response received from the language model.")
        #     raise LLMServiceNoResponseError("No response received from the language model.")

        # messages.append(
        #     {
        #         "role": "assistant",
        #         "content": llm_response.model_dump_json() if isinstance(llm_response, BaseModel) else str(llm_response),
        #     }
        # )
        # logger.debug(f"Received response: {llm_response}")

        try:
            if self.model == "claude-3.5-sonnet":
                messages, system = [], []
                if system_prompt:
                    system.append({"text": system_prompt})

                messages.append({"role": "user", "content": [{"text": prompt}]})

                response = self.client.converse(
                    modelId=self.CHAT_MODEL_ID, messages=messages, system=system
                )

                return response["output"]["message"]["content"][0]["text"], messages
            elif self.model == "gpt-4o-mini":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, stream=False
                )
                return response.choices[0].message.content, messages
            else:
                raise ValueError(f"Unsupported model: {self.model}")
        except Exception as e:
            logger.error(f"Error in sending message: {e}")
            return "Sorry, I am not able to provide an answer to that question.", []


@dataclass
class OpenAIEmbeddingService(BaseEmbeddingService):
    """Base class for Language Model implementations."""

    embedding_dim: int = field(default=1536)
    max_request_tokens: int = 8000
    model: Optional[str] = field(default="text-embedding-3-small")
    embedding_model: SentenceTransformer = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # def __post_init__(self):
    #     self.embedding_async_client: AsyncOpenAI = AsyncOpenAI(
    #         base_url=self.base_url, api_key=self.api_key, timeout=TIMEOUT_SECONDS
    #     )
    #     logger.debug("Initialized OpenAIEmbeddingService with OpenAI client.")

    async def encode(
        self, texts: list[str], model: Optional[str] = None
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Get the embedding representation of the input text.

        Args:
            texts (str): The input text to embed.
            model (str, optional): The name of the model to use. Defaults to the model provided in the config.

        Returns:
            list[float]: The embedding vector as a list of floats.
        """
        logger.debug(f"Getting embedding for texts: {texts}")
        # model = model or self.model
        # if model is None:
        #     raise ValueError("Model name must be provided.")

        # Chunk the requests to size limits
        # max_chunk_length = self.max_request_tokens * TOKEN_TO_CHAR_RATIO
        # text_chunks: List[List[str]] = []

        # current_chunk: List[str] = []
        # current_chunk_length = 0
        # for text in texts:
        #     text_length = len(text)
        #     if text_length + current_chunk_length > max_chunk_length:
        #         text_chunks.append(current_chunk)
        #         current_chunk = []
        #         current_chunk_length = 0
        #     current_chunk.append(text)
        #     current_chunk_length += text_length
        # text_chunks.append(current_chunk)

        # response = await asyncio.gather(*[self._embedding_request(chunk, model) for chunk in text_chunks])

        embeddings = self.embedding_model.encode(
            texts, normalize_embeddings=True, batch_size=32, device="cpu"
        )

        # data = chain(*[r.data for r in response])
        # embeddings = np.array([dp.embedding for dp in data])
        logger.debug(f"Received embedding response: {len(embeddings)} embeddings")

        return embeddings

    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential(multiplier=1, min=4, max=10),
    #     retry=retry_if_exception_type((RateLimitError, APIConnectionError, TimeoutError)),
    # )
    # async def _embedding_request(self, input: List[str], model: str) -> Any:
    #     return await self.embedding_async_client.embeddings.create(model=model, input=input, encoding_format="float")

"""This module implements a Graph-based Retrieval-Augmented Generation (GraphRAG) system."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, Union

from fast_graphrag._llm import BaseLLMService, format_and_send_prompt
from fast_graphrag._llm._base import BaseEmbeddingService
from fast_graphrag._policies._base import BaseEdgeUpsertPolicy, BaseGraphUpsertPolicy, BaseNodeUpsertPolicy
from fast_graphrag._prompt import PROMPTS
from fast_graphrag._services._chunk_extraction import BaseChunkingService
from fast_graphrag._services._information_extraction import BaseInformationExtractionService
from fast_graphrag._services._state_manager import BaseStateManagerService
from fast_graphrag._storage._base import BaseGraphStorage, BaseIndexedKeyValueStorage, BaseVectorStorage
from fast_graphrag._types import GTChunk, GTEdge, GTEmbedding, GTHash, GTId, GTNode, TContext, TDocument, TQueryResponse
from fast_graphrag._utils import TOKEN_TO_CHAR_RATIO, get_event_loop, logger


@dataclass
class InsertParam:
    pass


@dataclass
class QueryParam:
    with_references: bool = False
    only_context: bool = False
    entities_max_tokens: int = 50000
    relationships_max_tokens: int = 50000
    chunks_max_tokens: int = 0


@dataclass
class BaseGraphRAG(Generic[GTEmbedding, GTHash, GTChunk, GTNode, GTEdge, GTId]):
    """A class representing a Graph-based Retrieval-Augmented Generation system."""

    working_dir: str = field()
    domain: str = field()
    example_queries: str = field()
    entity_types: List[str] = field()
    n_checkpoints: int = field(default=0)

    llm_service: BaseLLMService = field(init=False, default_factory=lambda: BaseLLMService())
    chunking_service: BaseChunkingService[GTChunk] = field(init=False, default_factory=lambda: BaseChunkingService())
    information_extraction_service: BaseInformationExtractionService[GTChunk, GTNode, GTEdge, GTId] = field(
        init=False,
        default_factory=lambda: BaseInformationExtractionService(
            graph_upsert=BaseGraphUpsertPolicy(
                config=None,
                nodes_upsert_cls=BaseNodeUpsertPolicy,
                edges_upsert_cls=BaseEdgeUpsertPolicy,
            )
        ),
    )
    state_manager: BaseStateManagerService[GTNode, GTEdge, GTHash, GTChunk, GTId, GTEmbedding] = field(
        init=False,
        default_factory=lambda: BaseStateManagerService(
            workspace=None,
            graph_storage=BaseGraphStorage[GTNode, GTEdge, GTId](config=None),
            entity_storage=BaseVectorStorage[GTId, GTEmbedding](config=None),
            chunk_storage=BaseIndexedKeyValueStorage[GTHash, GTChunk](config=None),
            embedding_service=BaseEmbeddingService(),
            node_upsert_policy=BaseNodeUpsertPolicy(config=None),
            edge_upsert_policy=BaseEdgeUpsertPolicy(config=None),
        ),
    )

    def insert(
        self,
        content: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True,
    ) -> Tuple[int, int, int]:
        return get_event_loop().run_until_complete(self.async_insert(content, metadata, params, show_progress))

    async def async_insert(
        self,
        content: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True,
    ) -> Tuple[int, int, int]:
        """Insert a new memory or memories into the graph.

        Args:
            content (str | list[str]): The data to be inserted. Can be a single string or a list of strings.
            metadata (dict, optional): Additional metadata associated with the data. Defaults to None.
            params (InsertParam, optional): Additional parameters for the insertion. Defaults to None.
            show_progress (bool, optional): Whether to show the progress bar. Defaults to True.
        """
        if params is None:
            params = InsertParam()

        if isinstance(content, str):
            content = [content]
        if isinstance(metadata, dict):
            metadata = [metadata]

        if metadata is None or isinstance(metadata, dict):
            data = (TDocument(data=c, metadata=metadata or {}) for c in content)
        else:
            data = (TDocument(data=c, metadata=m or {}) for c, m in zip(content, metadata))

        try:
            await self.state_manager.insert_start()
            # Chunk the data
            chunked_documents = await self.chunking_service.extract(data=data)

            # Filter the chunks checking for duplicates
            new_chunks_per_data = await self.state_manager.filter_new_chunks(chunks_per_data=chunked_documents)

            # Extract entities and relationships from the new chunks only
            subgraphs = self.information_extraction_service.extract(
                llm=self.llm_service,
                documents=new_chunks_per_data,
                prompt_kwargs={
                    "domain": self.domain,
                    "example_queries": self.example_queries,
                    "entity_types": ",".join(self.entity_types),
                },
                entity_types=self.entity_types,
            )
            if len(subgraphs) == 0:
                logger.info("No new entities or relationships extracted from the data.")

            # Update the graph with the new entities, relationships, and chunks
            await self.state_manager.upsert(
                llm=self.llm_service, subgraphs=subgraphs, documents=new_chunks_per_data, show_progress=show_progress
            )

            # Commit the changes if all is successful
            await self.state_manager.insert_done()

            # Return the total number of entities, relationships, and chunks
            return (
                await self.state_manager.get_num_entities(),
                await self.state_manager.get_num_relations(),
                await self.state_manager.get_num_chunks(),
            )
        except Exception as e:
            logger.error(f"Error during insertion: {e}")
            raise e

    def query(
        self, query: str, id_mapping: dict, params: Optional[QueryParam] = None
    ) -> Tuple[TQueryResponse[GTNode, GTEdge, GTHash, GTChunk], int, int, int]:
        async def _query() -> TQueryResponse[GTNode, GTEdge, GTHash, GTChunk]:
            tic = time.time()
            await self.state_manager.query_start()
            print(f"Query start time: {time.time() - tic}")
            try:
                tic = time.time()
                answer, token_len = await self.async_query(query, id_mapping, params)
                duration = time.time() - tic
                print(f"******Query time: {duration}")
                return answer, duration, token_len, 1
            except Exception as e:
                logger.error(f"Error during query: {e}")
                raise e
            finally:
                await self.state_manager.query_done()

        return get_event_loop().run_until_complete(_query())

    async def async_query(
        self, query: str, id_mapping: dict, params: Optional[QueryParam] = None
    ) -> TQueryResponse[GTNode, GTEdge, GTHash, GTChunk]:
        """Query the graph with a given input.

        Args:
            query (str): The query string to search for in the graph.
            id_mapping (dict): A mapping of entity IDs to their corresponding names.
            params (QueryParam, optional): Additional parameters for the query. Defaults to None.

        Returns:
            TQueryResponse: The result of the query (response + context).
        """
        if params is None:
            params = QueryParam()

        # Extract entities from query
        # tic = time.time()
        # extracted_entities = await self.information_extraction_service.extract_entities_from_query(
        #     llm=self.llm_service, query=query, prompt_kwargs={}
        # )
        # print(f"******Extract entity time: {time.time() - tic}")

        # Retrieve relevant state
        tic = time.time()
        relevant_state = await self.state_manager.get_context(query=query, entities=id_mapping)
        if relevant_state is None:
            return TQueryResponse[GTNode, GTEdge, GTHash, GTChunk](
                response=PROMPTS["fail_response"], context=TContext([], [], [])
            )
        print(f"******Get context time: {time.time() - tic}")
        print(
            f"Number of entities: {len(relevant_state.entities)}, Number of relationships: {len(relevant_state.relationships)}"
        )

        # Ask LLM
        tic = time.time()
        if params.only_context:
            llm_response = ""
        else:
            (llm_response, _), token_len = await format_and_send_prompt(
                prompt_key="generate_response_query_with_references"
                if params.with_references
                else "generate_response_query_no_references",
                llm=self.llm_service,
                format_kwargs={
                    "query": query,
                    "context": relevant_state.to_str(
                        {
                            "entities": params.entities_max_tokens * TOKEN_TO_CHAR_RATIO,
                            "relationships": params.relationships_max_tokens * TOKEN_TO_CHAR_RATIO,
                            "chunks": params.chunks_max_tokens * TOKEN_TO_CHAR_RATIO,
                        }
                    ),
                },
                response_model=str,
            )
        print(f"******LLM response time: {time.time() - tic}")

        return TQueryResponse[GTNode, GTEdge, GTHash, GTChunk](response=llm_response, context=relevant_state), token_len

    def save_graphml(self, output_path: str) -> None:
        """Save the graph in GraphML format."""

        async def _save_graphml() -> None:
            await self.state_manager.query_start()
            try:
                await self.state_manager.save_graphml(output_path)
            finally:
                await self.state_manager.query_done()

        get_event_loop().run_until_complete(_save_graphml())

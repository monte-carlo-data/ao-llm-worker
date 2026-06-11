"""ClickHouse client for llm_inputs and llm_results tables."""

import logging
from dataclasses import dataclass
from uuid import UUID

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from llm_worker.config import ClickHouseConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMInput:
    batch_id: UUID
    row_id: UUID
    model_id: str
    prompt: str
    params: str
    tool_config: str


@dataclass
class LLMResult:
    batch_id: UUID
    row_id: UUID
    response: str
    status: str
    error: str


class ClickHouseClient:
    def __init__(
        self,
        config: ClickHouseConfig | None = None,
        client: Client | None = None,
        pending_batch_limit: int = 100,
        batch_page_size: int = 100,
    ):
        self._pending_batch_limit = pending_batch_limit
        self._batch_page_size = batch_page_size
        if client:
            self._client = client
        elif config:
            self._client = clickhouse_connect.get_client(
                host=config.host,
                port=config.port,
                username=config.user,
                password=config.password,
                database=config.database,
                secure=bool(config.ca_cert),
                verify=config.ca_cert or True,
            )
        else:
            raise ValueError("Either config or client must be provided")

    def get_pending_batches(self) -> list[UUID]:
        result = self._client.query(
            """
            SELECT batch_id
            FROM llm_batches
            GROUP BY batch_id
            HAVING
                countIf(status = 'pending') > 0
                AND countIf(status = 'complete') = 0
            ORDER BY batch_id
            LIMIT {pending_batch_limit:UInt32}
        """,
            parameters={"pending_batch_limit": self._pending_batch_limit},
        )
        return [row[0] for row in result.result_rows]

    def get_batch_rows(self, batch_id: UUID) -> list[LLMInput]:
        result = self._client.query(
            """
            SELECT
                batch_id,
                row_id,
                model_id,
                prompt,
                params,
                tool_config
            FROM llm_inputs
            WHERE batch_id = {batch_id:UUID}
              AND row_id NOT IN (
                  SELECT row_id
                  FROM llm_results
                  WHERE batch_id = {batch_id:UUID}
              )
            ORDER BY row_id
            LIMIT {page_size:UInt32}
        """,
            parameters={"batch_id": batch_id, "page_size": self._batch_page_size},
        )
        return [
            LLMInput(
                batch_id=row[0],
                row_id=row[1],
                model_id=row[2],
                prompt=row[3],
                params=row[4],
                tool_config=row[5],
            )
            for row in result.result_rows
        ]

    def write_results(self, results: list[LLMResult]) -> None:
        if not results:
            return
        data = [[r.batch_id, r.row_id, r.response, r.status, r.error] for r in results]
        self._client.insert(
            "llm_results",
            data,
            column_names=["batch_id", "row_id", "response", "status", "error"],
            settings={"async_insert": 1, "wait_for_async_insert": 1},
        )
        logger.info("results_written", extra={"count": len(results)})

    def get_batch_counts(self, batch_id: UUID) -> tuple[int, int, int]:
        result = self._client.query(
            """
            SELECT
                (SELECT count()
                 FROM llm_inputs
                 WHERE batch_id = {batch_id:UUID}) AS total_rows,
                countIf(status = 'complete') AS completed_rows,
                countIf(status = 'failed') AS failed_rows
            FROM llm_results
            WHERE batch_id = {batch_id:UUID}
        """,
            parameters={"batch_id": batch_id},
        )
        row = result.result_rows[0]
        return row[0], row[1], row[2]

    def write_batch_status(
        self,
        batch_id: UUID,
        status: str,
        total_rows: int = 0,
        completed_rows: int = 0,
        failed_rows: int = 0,
    ) -> None:
        self._client.insert(
            "llm_batches",
            [[batch_id, status, total_rows, completed_rows, failed_rows]],
            column_names=[
                "batch_id",
                "status",
                "total_rows",
                "completed_rows",
                "failed_rows",
            ],
            settings={"async_insert": 0},
        )

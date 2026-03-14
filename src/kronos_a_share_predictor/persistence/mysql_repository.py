from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CHAR,
    DATE,
    DATETIME,
    DECIMAL,
    INT,
    TEXT,
    VARCHAR,
    Column,
    ForeignKey,
    MetaData,
    Table,
    create_engine,
    insert,
    update,
)
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT


class MysqlRepository:
    def __init__(self, db_url: str) -> None:
        self._engine = create_engine(db_url, future=True)
        self._metadata = MetaData()
        self.prediction_runs = Table(
            "prediction_runs",
            self._metadata,
            Column("id", MYSQL_BIGINT(unsigned=True), primary_key=True, autoincrement=True),
            Column("run_uuid", CHAR(36), nullable=False, unique=True),
            Column("recommendation_date", DATE, nullable=False),
            Column("starttime", DATE, nullable=False),
            Column("endtime", DATE, nullable=False),
            Column("lookback_days", INT, nullable=False),
            Column("pred_len", INT, nullable=False),
            Column("model_name", VARCHAR(128), nullable=False),
            Column("tokenizer_name", VARCHAR(128), nullable=False),
            Column("stock_count_raw", INT, nullable=False),
            Column("stock_count_dedup", INT, nullable=False),
            Column("stock_count_success", INT, nullable=False, default=0),
            Column("stock_count_failed", INT, nullable=False, default=0),
            Column("status", VARCHAR(32), nullable=False),
            Column("error_message", TEXT),
            Column("started_at", DATETIME, nullable=False),
            Column("finished_at", DATETIME),
        )
        self.stock_predictions = Table(
            "stock_predictions",
            self._metadata,
            Column("id", MYSQL_BIGINT(unsigned=True), primary_key=True, autoincrement=True),
            Column("run_id", MYSQL_BIGINT(unsigned=True), ForeignKey("prediction_runs.id"), nullable=False),
            Column("stock_code", VARCHAR(16), nullable=False),
            Column("history_start_date", DATE, nullable=False),
            Column("history_end_date", DATE, nullable=False),
            Column("prediction_date", DATE, nullable=False),
            Column("pred_open", DECIMAL(18, 6), nullable=False),
            Column("pred_high", DECIMAL(18, 6), nullable=False),
            Column("pred_low", DECIMAL(18, 6), nullable=False),
            Column("pred_close", DECIMAL(18, 6), nullable=False),
            Column("pred_volume", DECIMAL(18, 6), nullable=False),
            Column("pred_amount", DECIMAL(18, 6), nullable=False),
            Column("created_at", DATETIME, nullable=False, default=datetime.utcnow),
        )
        self.backtest_runs = Table(
            "backtest_runs",
            self._metadata,
            Column("id", MYSQL_BIGINT(unsigned=True), primary_key=True, autoincrement=True),
            Column("run_uuid", CHAR(36), nullable=False, unique=True),
            Column("recommendation_date", DATE, nullable=False),
            Column("backtest_start_date", DATE, nullable=False),
            Column("backtest_end_date", DATE, nullable=False),
            Column("pred_len", INT, nullable=False),
            Column("candidate_context_lengths", TEXT, nullable=False),
            Column("success_mape_threshold", DECIMAL(10, 6), nullable=False),
            Column("best_context_length", INT),
            Column("best_success_rate", DECIMAL(10, 6)),
            Column("total_sample_count", INT, nullable=False, default=0),
            Column("total_result_count", INT, nullable=False, default=0),
            Column("csv_summary_path", VARCHAR(255)),
            Column("csv_detail_path", VARCHAR(255)),
            Column("status", VARCHAR(32), nullable=False),
            Column("error_message", TEXT),
            Column("started_at", DATETIME, nullable=False),
            Column("finished_at", DATETIME),
        )
        self.backtest_results = Table(
            "backtest_results",
            self._metadata,
            Column("id", MYSQL_BIGINT(unsigned=True), primary_key=True, autoincrement=True),
            Column("backtest_run_id", MYSQL_BIGINT(unsigned=True), ForeignKey("backtest_runs.id"), nullable=False),
            Column("stock_code", VARCHAR(16), nullable=False),
            Column("evaluation_date", DATE, nullable=False),
            Column("context_length", INT, nullable=False),
            Column("history_start_date", DATE, nullable=False),
            Column("history_end_date", DATE, nullable=False),
            Column("baseline_close", DECIMAL(18, 6), nullable=False),
            Column("prediction_date_1", DATE, nullable=False),
            Column("prediction_date_2", DATE, nullable=False),
            Column("prediction_date_3", DATE, nullable=False),
            Column("pred_close_1", DECIMAL(18, 6), nullable=False),
            Column("pred_close_2", DECIMAL(18, 6), nullable=False),
            Column("pred_close_3", DECIMAL(18, 6), nullable=False),
            Column("actual_close_1", DECIMAL(18, 6), nullable=False),
            Column("actual_close_2", DECIMAL(18, 6), nullable=False),
            Column("actual_close_3", DECIMAL(18, 6), nullable=False),
            Column("day3_mape", DECIMAL(10, 6), nullable=False),
            Column("close_mae", DECIMAL(18, 6), nullable=False),
            Column("close_rmse", DECIMAL(18, 6), nullable=False),
            Column("direction_correct", INT, nullable=False),
            Column("is_success", INT, nullable=False),
            Column("created_at", DATETIME, nullable=False, default=datetime.utcnow),
        )

    def create_schema(self) -> None:
        self._metadata.create_all(self._engine)

    def create_run(
        self,
        *,
        run_uuid: str,
        recommendation_date,
        starttime,
        endtime,
        lookback_days: int,
        pred_len: int,
        model_name: str,
        tokenizer_name: str,
        stock_count_raw: int,
        stock_count_dedup: int,
    ) -> int:
        statement = insert(self.prediction_runs).values(
            run_uuid=run_uuid,
            recommendation_date=recommendation_date,
            starttime=starttime,
            endtime=endtime,
            lookback_days=lookback_days,
            pred_len=pred_len,
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            stock_count_raw=stock_count_raw,
            stock_count_dedup=stock_count_dedup,
            stock_count_success=0,
            stock_count_failed=0,
            status="running",
            error_message=None,
            started_at=datetime.utcnow(),
            finished_at=None,
        )
        with self._engine.begin() as connection:
            result = connection.execute(statement)
            return int(result.inserted_primary_key[0])

    def complete_run(self, run_id: int, success_count: int, failed_count: int, status: str, error_message: str | None) -> None:
        statement = (
            update(self.prediction_runs)
            .where(self.prediction_runs.c.id == run_id)
            .values(
                stock_count_success=success_count,
                stock_count_failed=failed_count,
                status=status,
                error_message=error_message,
                finished_at=datetime.utcnow(),
            )
        )
        with self._engine.begin() as connection:
            connection.execute(statement)

    def save_predictions(self, run_id: int, prediction_rows: list[dict]) -> None:
        if not prediction_rows:
            return
        rows = [dict(row, run_id=run_id, created_at=datetime.utcnow()) for row in prediction_rows]
        statement = insert(self.stock_predictions)
        with self._engine.begin() as connection:
            connection.execute(statement, rows)

    def create_backtest_run(
        self,
        *,
        run_uuid: str,
        recommendation_date,
        backtest_start_date,
        backtest_end_date,
        pred_len: int,
        candidate_context_lengths: str,
        success_mape_threshold: float,
    ) -> int:
        statement = insert(self.backtest_runs).values(
            run_uuid=run_uuid,
            recommendation_date=recommendation_date,
            backtest_start_date=backtest_start_date,
            backtest_end_date=backtest_end_date,
            pred_len=pred_len,
            candidate_context_lengths=candidate_context_lengths,
            success_mape_threshold=success_mape_threshold,
            best_context_length=None,
            best_success_rate=None,
            total_sample_count=0,
            total_result_count=0,
            csv_summary_path=None,
            csv_detail_path=None,
            status="running",
            error_message=None,
            started_at=datetime.utcnow(),
            finished_at=None,
        )
        with self._engine.begin() as connection:
            result = connection.execute(statement)
            return int(result.inserted_primary_key[0])

    def complete_backtest_run(
        self,
        backtest_run_id: int,
        *,
        best_context_length: int | None,
        best_success_rate: float | None,
        total_sample_count: int,
        total_result_count: int,
        csv_summary_path: str | None,
        csv_detail_path: str | None,
        status: str,
        error_message: str | None,
    ) -> None:
        statement = (
            update(self.backtest_runs)
            .where(self.backtest_runs.c.id == backtest_run_id)
            .values(
                best_context_length=best_context_length,
                best_success_rate=best_success_rate,
                total_sample_count=total_sample_count,
                total_result_count=total_result_count,
                csv_summary_path=csv_summary_path,
                csv_detail_path=csv_detail_path,
                status=status,
                error_message=error_message,
                finished_at=datetime.utcnow(),
            )
        )
        with self._engine.begin() as connection:
            connection.execute(statement)

    def save_backtest_results(self, backtest_run_id: int, result_rows: list[dict]) -> None:
        if not result_rows:
            return
        rows = [
            dict(
                row,
                backtest_run_id=backtest_run_id,
                direction_correct=1 if row["direction_correct"] else 0,
                is_success=1 if row["is_success"] else 0,
                created_at=datetime.utcnow(),
            )
            for row in result_rows
        ]
        statement = insert(self.backtest_results)
        with self._engine.begin() as connection:
            connection.execute(statement, rows)


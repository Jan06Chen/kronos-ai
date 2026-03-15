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
            Column("min_history_points", INT, nullable=False),
            Column("max_context", INT, nullable=False),
            Column("context_length", INT),
            Column("model_name", VARCHAR(128), nullable=False),
            Column("tokenizer_name", VARCHAR(128), nullable=False),
            Column("device", VARCHAR(64)),
            Column("temperature", DECIMAL(10, 6), nullable=False),
            Column("top_p", DECIMAL(10, 6), nullable=False),
            Column("sample_count", INT, nullable=False),
            Column("inference_verbose", INT, nullable=False, default=0),
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
            Column("id", MYSQL_BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="主键 ID"),
            Column("backtest_run_id", MYSQL_BIGINT(unsigned=True), ForeignKey("backtest_runs.id"), nullable=False, comment="关联 backtest_runs.id 的回测运行 ID"),
            Column("stock_code", VARCHAR(16), nullable=False, comment="股票代码"),
            Column("evaluation_date", DATE, nullable=False, comment="回测评估日期，即使用该日期及之前历史数据进行预测"),
            Column("context_length", INT, nullable=False, comment="本次回测使用的历史上下文长度"),
            Column("history_start_date", DATE, nullable=False, comment="本次样本历史窗口起始日期"),
            Column("history_end_date", DATE, nullable=False, comment="本次样本历史窗口结束日期"),
            Column("baseline_close", DECIMAL(18, 6), nullable=False, comment="评估基准收盘价，通常为历史窗口最后一个交易日收盘价"),
            Column("prediction_date_1", DATE, nullable=False, comment="第 1 个预测交易日日期"),
            Column("prediction_date_2", DATE, nullable=False, comment="第 2 个预测交易日日期"),
            Column("prediction_date_3", DATE, nullable=False, comment="第 3 个预测交易日日期"),
            Column("pred_open_1", DECIMAL(18, 6), nullable=False, comment="第 1 个预测交易日预测开盘价"),
            Column("pred_open_2", DECIMAL(18, 6), nullable=False, comment="第 2 个预测交易日预测开盘价"),
            Column("pred_open_3", DECIMAL(18, 6), nullable=False, comment="第 3 个预测交易日预测开盘价"),
            Column("pred_high_1", DECIMAL(18, 6), nullable=False, comment="第 1 个预测交易日预测最高价"),
            Column("pred_high_2", DECIMAL(18, 6), nullable=False, comment="第 2 个预测交易日预测最高价"),
            Column("pred_high_3", DECIMAL(18, 6), nullable=False, comment="第 3 个预测交易日预测最高价"),
            Column("pred_low_1", DECIMAL(18, 6), nullable=False, comment="第 1 个预测交易日预测最低价"),
            Column("pred_low_2", DECIMAL(18, 6), nullable=False, comment="第 2 个预测交易日预测最低价"),
            Column("pred_low_3", DECIMAL(18, 6), nullable=False, comment="第 3 个预测交易日预测最低价"),
            Column("pred_close_1", DECIMAL(18, 6), nullable=False, comment="第 1 个预测交易日预测收盘价"),
            Column("pred_close_2", DECIMAL(18, 6), nullable=False, comment="第 2 个预测交易日预测收盘价"),
            Column("pred_close_3", DECIMAL(18, 6), nullable=False, comment="第 3 个预测交易日预测收盘价"),
            Column("pred_volume_1", DECIMAL(18, 6), nullable=False, comment="第 1 个预测交易日预测成交量"),
            Column("pred_volume_2", DECIMAL(18, 6), nullable=False, comment="第 2 个预测交易日预测成交量"),
            Column("pred_volume_3", DECIMAL(18, 6), nullable=False, comment="第 3 个预测交易日预测成交量"),
            Column("pred_amount_1", DECIMAL(18, 6), nullable=False, comment="第 1 个预测交易日预测成交额"),
            Column("pred_amount_2", DECIMAL(18, 6), nullable=False, comment="第 2 个预测交易日预测成交额"),
            Column("pred_amount_3", DECIMAL(18, 6), nullable=False, comment="第 3 个预测交易日预测成交额"),
            Column("actual_close_1", DECIMAL(18, 6), nullable=False, comment="第 1 个预测交易日实际收盘价"),
            Column("actual_close_2", DECIMAL(18, 6), nullable=False, comment="第 2 个预测交易日实际收盘价"),
            Column("actual_close_3", DECIMAL(18, 6), nullable=False, comment="第 3 个预测交易日实际收盘价"),
            Column("day3_mape", DECIMAL(10, 6), nullable=False, comment="第 3 个预测交易日收盘价 MAPE"),
            Column("close_mae", DECIMAL(18, 6), nullable=False, comment="三天预测收盘价 MAE"),
            Column("close_rmse", DECIMAL(18, 6), nullable=False, comment="三天预测收盘价 RMSE"),
            Column("direction_correct", INT, nullable=False, comment="第 3 个预测交易日涨跌方向是否预测正确"),
            Column("is_success", INT, nullable=False, comment="本条回测结果是否满足成功标准"),
            Column("created_at", DATETIME, nullable=False, default=datetime.utcnow, comment="记录创建时间"),
            comment="回测结果明细表，保存每只股票、每个评估日的三日预测结果与评估指标",
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
        min_history_points: int,
        max_context: int,
        context_length: int | None,
        model_name: str,
        tokenizer_name: str,
        device: str | None,
        temperature: float,
        top_p: float,
        sample_count: int,
        inference_verbose: bool,
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
            min_history_points=min_history_points,
            max_context=max_context,
            context_length=context_length,
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            device=device,
            temperature=temperature,
            top_p=top_p,
            sample_count=sample_count,
            inference_verbose=1 if inference_verbose else 0,
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
        allowed_columns = {
            "stock_code",
            "evaluation_date",
            "context_length",
            "history_start_date",
            "history_end_date",
            "baseline_close",
            "prediction_date_1",
            "prediction_date_2",
            "prediction_date_3",
            "pred_open_1",
            "pred_open_2",
            "pred_open_3",
            "pred_high_1",
            "pred_high_2",
            "pred_high_3",
            "pred_low_1",
            "pred_low_2",
            "pred_low_3",
            "pred_close_1",
            "pred_close_2",
            "pred_close_3",
            "pred_volume_1",
            "pred_volume_2",
            "pred_volume_3",
            "pred_amount_1",
            "pred_amount_2",
            "pred_amount_3",
            "actual_close_1",
            "actual_close_2",
            "actual_close_3",
            "day3_mape",
            "close_mae",
            "close_rmse",
            "direction_correct",
            "is_success",
        }
        rows = [
            dict(
                {key: value for key, value in row.items() if key in allowed_columns},
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


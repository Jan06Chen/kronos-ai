# Kronos A-Share Predictor

这个项目按 Kronos README 的基础推理方式接入 NeoQuasar/Kronos-base，完成以下流程：

1. 从 recommendations 接口获取指定日期的推荐股票。
2. 提取并去重 stock_code。
3. 按每只股票请求 starttime 到 endtime 的 K 线数据，默认 endtime 向前回溯 200 天。
4. 把本地接口字段转换成 Kronos 需要的 OHLCV/amount 格式。
5. 使用 Kronos-base 预测未来 3 个交易日。
6. 把每次运行和预测明细写入 MySQL。

## 目录

- src/kronos_a_share_predictor: 业务代码
- sql/schema.sql: MySQL 建表脚本
- recommand.json: recommendations 响应样例
- date_format/k_line.json: K 线响应样例

## 依赖准备

1. 创建 Python 3.10+ 环境。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 克隆 Kronos 仓库到本地 vendor 目录：

```bash
git clone https://github.com/shiyu-coder/Kronos.git vendor/Kronos
```

说明：Kronos 仓库不是标准 pip 包结构，这个项目通过本地仓库路径导入其中的 model 包。

4. 复制环境变量模板并修改：

```bash
cp .env.example .env
```

重点变量：

- KRONOS_STARTTIME: 可为空。为空时自动按 endtime 向前推 200 天。
- KRONOS_ENDTIME: 历史窗口结束日期。
- KRONOS_RECOMMENDATION_DATE: 推荐股票抓取日期。
- KRONOS_DB_URL: MySQL 连接串。
- KRONOS_REPO_PATH: 本地 Kronos 仓库路径。

## 建表

先在 MySQL 中创建数据库，再执行：

```bash
mysql -u root -p kronos_predict < sql/schema.sql
```

## 运行

```bash
PYTHONPATH=src python main.py
```

也可以覆盖日期参数：

```bash
PYTHONPATH=src python main.py --recommendation-date 2026-03-13 --endtime 2026-03-14
```

## 运行结果

程序会向两张表写入数据：

- prediction_runs: 一次批处理运行的元数据。
- stock_predictions: 每只股票未来 3 天的预测明细。

## 回测与 Context Length 调优

项目提供独立回测入口，用最近两个月的历史窗口做滚动三日预测回测，并扫描候选 context length。

默认口径：

- 候选 context length: 30,60,90,120,150,180,200,300,400,500
- 成功率定义: 第 3 天收盘方向正确，且第 3 天收盘 MAPE <= 8%
- 输出: 控制台汇总 + MySQL + CSV

执行方式：

```bash
python backtest.py --recommendation-date 2026-03-13 --backtest-end-date 2026-03-14
```

可选参数：

- --backtest-start-date: 手工指定回测起始日期，不指定时默认按结束日期回推两个月。
- --context-lengths: 逗号分隔的候选窗口列表。
- --success-mape-threshold: 成功率阈值，默认 0.08。
- --report-output-dir: CSV 报表输出目录。

新增表：

- backtest_runs: 每次回测的参数、最佳窗口、最佳成功率和报表路径。
- backtest_results: 每个样本点、每个 context length 的预测值、真实值和误差指标。

CSV 输出：

- *_summary.csv: 各 context length 的成功率汇总。
- *_details.csv: 样本明细，包含预测与真实对比。

## 当前实现说明

- 未来时间轴使用“工作日”近似规则生成，会跳过周末，但不会自动跳过法定节假日。
- 如果某只股票历史长度不足，会被跳过，不影响其他股票预测。
- 批量推理前会把所有可用股票裁剪到共同历史长度，且长度不超过 200。
- 回测时严格使用评估日及之前的数据构造历史窗口，真实标签始终来自之后 3 个交易日，避免未来数据泄漏。

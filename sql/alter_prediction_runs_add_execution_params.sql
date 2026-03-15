ALTER TABLE prediction_runs
    ADD COLUMN min_history_points INT NOT NULL DEFAULT 0 AFTER pred_len,
    ADD COLUMN max_context INT NOT NULL DEFAULT 512 AFTER min_history_points,
    ADD COLUMN context_length INT NULL AFTER max_context,
    ADD COLUMN device VARCHAR(64) NULL AFTER tokenizer_name,
    ADD COLUMN temperature DECIMAL(10, 6) NOT NULL DEFAULT 1.000000 AFTER device,
    ADD COLUMN top_p DECIMAL(10, 6) NOT NULL DEFAULT 0.900000 AFTER temperature,
    ADD COLUMN sample_count INT NOT NULL DEFAULT 1 AFTER top_p,
    ADD COLUMN inference_verbose TINYINT(1) NOT NULL DEFAULT 0 AFTER sample_count;
# Benchmark Outputs

这个目录只保留当前仍有分析价值的结果。

## 保留目录

- `perf_report_20260620_181529/`
  - 当前 Alpamayo 真实数据 merged-request 性能报告的数据源。
  - 覆盖原图输入 `OMP_NUM_THREADS=1/4/8` 和 `enable_mm_embeds + OMP_NUM_THREADS=8`。
  - 关键汇总见 `combined_summary.csv` / `combined_summary.md`，报告见 `../docs/benchmark/qwen35_enable_mm_embeds_perf_report.md`。
- `benchmark_qwen35_mm_bs_sweep_global_unique_r2/`
  - 历史 synthetic 普通 offline batch sweep。
  - 覆盖 `bs=4,8,16,32,40,48,64`。
  - 所有 row 都有 `request_ids_unique_global=True`。
- `benchmark_qwen35_mm_bs_refine_global_unique_r3/`
  - 历史 synthetic 关键区间 refine sweep。
  - 覆盖 `bs=24,32,40,48,56,64`。
  - 所有 row 都有 `request_ids_unique_global=True`。
- `benchmark_qwen35_mm_bs_global_unique_combined/`
  - 上面两轮的合并分析结果。
  - synthetic controlled sweep 推荐 `bs=48`，不代表真实图片端到端路径的最终结论。
- `benchmark_qwen35_mm_merged_request_smoke/`
  - 合并 logical batch 为单个 engine request 的 smoke 结果。
  - 用于验证 `benchmark_qwen35_mm_merged_request.py` 的多图单请求路径。

## 清理策略

旧 smoke、旧 analyzer 输出、没有唯一 UUID 校验字段的过期 sweep 已清理。后续临时实验建议写到独立目录，确认有价值后再保留。

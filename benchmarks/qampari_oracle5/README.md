# QAMPARI Oracle-5

This task evaluates list QA on the 962 official QAMPARI test questions for which five
distinct proof-backed answers are representable by the benchmark's comma-separated
output protocol. Each question is paired with one gold proof passage for each of those
answers, so it measures the reader rather than retrieval. It must therefore be reported
as **QAMPARI Oracle-5**, not as full open-domain QAMPARI.

The primary metric is ALCE-style `qampari_f1_top5`; precision, full recall/F1, recall@5,
and the average number of predicted answers are also reported. Prepare the data with:

```bash
python benchmarks/qampari_oracle5/prepare_data.py
```

# GuideBench

## Official source and protocol

Audited 2026-10-05. Source: [Dlxxx/GuideBench](https://github.com/Dlxxx/GuideBench),
commit `78c5bfa42facee34db31e4ba03ad2c3b5a04bbbc`.

The public release contains **1,042 Chinese examples in six categories**:
price (442), relevance (192), math (52), chat (180), summary (58), hallu (118).
The paper describes seven categories/1,272 examples, but audit data is not
published at this revision. This adapter does not claim that unreleased set.

`n=1, max_new_tokens=4096, temperature=0, top_p=1` matches the reference generator.
The four prompt templates are copied verbatim from `generate_resp.py` under
the upstream MIT license. Gold answers and reference analyses are excluded
from candidate prompts. String formatting of the Guidelines list is preserved.

We match `evaluate.py`'s exact label/option comparison and count-weighted
accuracy across the six available categories. Malformed JSON is incorrect,
not excluded. Numeric strings are not converted into numbers. JSON booleans
are rejected as labels (upstream Python equality would accept `true == 1`).
Accuracy and parsed rate are reported along with per-category accuracy.
No LLM judge or API key is required. The summarization category is multiple
choice, not open-ended summarization or a writing-quality evaluation.

## Prepare and run

```bash
python benchmarks/init.py guidebench
aethereval --model /path/to/model --tasks guidebench --output-dir outputs
```

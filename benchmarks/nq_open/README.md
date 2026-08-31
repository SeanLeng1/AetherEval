# NQ-Open

This task uses the 3,610-example public NQ-Open development set and normalized exact
match over the complete submitted answer string. The original test labels are not
public, so reports should call this split
**NQ-Open dev**, rather than NQ test. Token F1 is included as a secondary diagnostic.

```bash
python benchmarks/nq_open/prepare_data.py
```

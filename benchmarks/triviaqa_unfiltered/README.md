# TriviaQA Unfiltered

This task uses the public `unfiltered.nocontext` validation split and reports normalized
exact match over the complete submitted answer string, with token F1 as a secondary
diagnostic. The public validation split is used because labeled test answers are not
part of the normal reproducible evaluator.

```bash
python benchmarks/triviaqa_unfiltered/prepare_data.py
```

If the shared Hugging Face cache is full, pass a writable temporary directory such as
`--cache-dir /tmp/aethereval-hf-cache`.

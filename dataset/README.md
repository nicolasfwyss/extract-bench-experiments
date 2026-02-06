# ExtractBench Dataset

A benchmark for evaluating structured information extraction from PDF documents using large language models. ExtractBench consists of 35 PDF documents across 5 domains, each paired with a JSON schema defining the target extraction structure and a human-annotated gold standard JSON extraction.

## Dataset Structure

```
dataset/
  academic/research/          # 6 research papers
  finance/10kq/               # 7 SEC 10-K/Q filings
  finance/credit_agreement/   # 10 credit agreements
  hiring/resume/              # 7 professional resumes
  sport/swimming/             # 5 competition results
```

Each domain directory contains:
- `<domain>-schema.json` -- JSON schema defining the target extraction structure
- `pdf+gold/` -- paired PDF documents and gold standard extractions (`<name>.pdf`, `<name>.gold.json`)

## Domains

| Domain | Documents | Pages | Schema Keys | Gold Values |
|---|---|---|---|---|
| SEC 10-K/Qs | 7 | 422 | 369 | 9,036 |
| Professional Resumes | 7 | 21 | 31 | 721 |
| Credit Agreements | 10 | 1,368 | 13 | 257 |
| Sports Results | 5 | 15 | 12 | 136 |
| Research Papers | 6 | 250 | 16 | 153 |
| **Total** | **35** | **2,076** | **441** | **10,303** |

*Schema Keys* counts evaluatable leaf nodes per schema. *Gold Values* counts all evaluatable leaf values across the gold annotations, including expanded array items.

## Schema Format

Schemas are standard JSON Schema with optional `evaluation_config` blocks specifying how fields should be evaluated:

- `string_similarity` -- fuzzy string matching
- `integer_exact_match` -- exact numeric match
- `number_tolerance` -- numeric match within a tolerance (e.g., `{"tolerance": 0.001}`)
- `llm_judge` -- LLM-based semantic evaluation for arrays and complex fields

## Citation

```bibtex
@inproceedings{extractbench2025,
  title={ExtractBench: A Benchmark for Structured Extraction from PDFs},
  year={2025}
}
```

## License

TBD

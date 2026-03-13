| Path | Metric | Score | Passed | Reason |
|------|--------|-------|--------|--------|
| ids | string_exact | 1.000 | Yes |  |
| number_of_pages | integer_exact | 1.000 | Yes |  |
| publication_type | string_exact | 1.000 | Yes |  |
| publication_date | string_semantic | 1.000 | Yes |  |
| venue | string_semantic | 1.000 | Yes |  |
| keywords | array_llm | 0.000 | No |  |
| abstract | string_semantic | 1.000 | Yes |  |
| title | string_semantic | 1.000 | Yes |  |
| authors | array_llm | 0.667 | No |  |
| citations | error_1776577846272 | 0.000 | No |  |
| citations.items.ids | string_exact | 1.000 | Yes | both_null |
| citations.items.year | integer_exact | 1.000 | Yes | both_null |
| citations.items.title | string_semantic | 1.000 | Yes | both_null |
| citations.items.venue | string_semantic | 1.000 | Yes | both_null |
| citations.items.authors | array_llm | 1.000 | Yes | both_null |
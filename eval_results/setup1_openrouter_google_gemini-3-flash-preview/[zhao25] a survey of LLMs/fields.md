| Path | Metric | Score | Passed | Reason |
|------|--------|-------|--------|--------|
| ids | string_exact | 0.000 | No |  |
| number_of_pages | integer_exact | 1.000 | Yes |  |
| publication_type | string_exact | 1.000 | Yes |  |
| title | string_semantic | 1.000 | Yes |  |
| publication_date | string_semantic | 1.000 | Yes |  |
| keywords | array_llm | 1.000 | Yes |  |
| abstract | string_semantic | 0.700 | Yes |  |
| venue | string_semantic | 0.800 | Yes |  |
| authors | array_llm | 1.000 | Yes |  |
| citations | error_2352189912208 | 0.000 | No |  |
| citations.items.ids | string_exact | 1.000 | Yes | both_null |
| citations.items.year | integer_exact | 1.000 | Yes | both_null |
| citations.items.title | string_semantic | 1.000 | Yes | both_null |
| citations.items.venue | string_semantic | 1.000 | Yes | both_null |
| citations.items.authors | array_llm | 1.000 | Yes | both_null |
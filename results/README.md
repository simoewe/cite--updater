# Citation Validation Results

This folder contains citation validation results organized by category.

## File Structure

- `matched.json` - Citations that matched correctly with DBLP
- `parsing_errors.json` - Citations with parsing errors in author names
- `first_names.json` - Citations with first name mismatches
- `last_names.json` - Citations with last name mismatches
- `accents_missing.json` - Citations with missing accents/diacritics
- `author_not_found.json` - Citations where authors were not found in DBLP
- `author_order_wrong.json` - Citations with correct authors but wrong order
- `empty_list.json` - Citations with empty author lists
- `title_mismatches.json` - Citations with title similarity below threshold
- `no_dblp_match.json` - Citations not found in DBLP database
- `errors.json` - Citations that caused processing errors
- `skipped.json` - Citations that were skipped (e.g., non-academic references)
- `summary.json` - Summary statistics for all categories

## Statistics

| Category | Count |
|----------|-------|
| author_not_found | 120 |
| errors | 29 |
| first_names | 17 |
| last_names | 29 |
| matched | 1340 |
| no_dblp_match | 2 |
| parsing_errors | 189 |
| title_mismatches | 336 |

**Total Results:** 2046

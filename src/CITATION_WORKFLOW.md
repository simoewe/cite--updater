# Citation Validation and Analysis Workflow

## Overview

The citation validation system consists of two main scripts that work together:

```
┌─────────────────────────────────┐
│  1. validate_citations.py        │
│     (VALIDATION)                 │
│                                  │
│  Input:  Parsed JSON files      │
│          (from GROBID)            │
│                                  │
│  Process:                        │
│  - Query DBLP for each citation  │
│  - Compare authors               │
│  - Classify mismatches           │
│                                  │
│  Output: citation_validation_   │
│          results.json            │
└──────────────┬──────────────────┘
               │
               │ (feeds into)
               ▼
┌─────────────────────────────────┐
│  2. analyze_citations.py        │
│     (ANALYSIS)                  │
│                                  │
│  Input:  citation_validation_  │
│          results.json            │
│                                  │
│  Process:                        │
│  - Statistical analysis          │
│  - Parsing error detection       │
│  - Result reorganization         │
│                                  │
│  Output: Analysis reports &     │
│          categorized files       │
└─────────────────────────────────┘
```

## Detailed Comparison

### `validate_citations.py` - VALIDATION Script

**What it does:**
- Performs the actual validation work
- Reads citation data from parsed JSON files
- Queries DBLP database for each citation
- Compares author lists
- Generates validation results

**Input:**
- Directory of parsed JSON files (`data/parsed_jsons/`)
- DBLP XML file (`data/dblp.xml`)

**Output:**
- `citation_validation_results.json` - Contains validation results for each citation

**When to use:**
- When you want to validate citations against DBLP
- First step in the validation workflow

**Example:**
```bash
python src/validate_citations.py \
  --input-dir data/parsed_jsons \
  --dblp-xml data/dblp.xml \
  --output citation_validation_results.json \
  --num-files 50
```

---

### `analyze_citations.py` - ANALYSIS Script

**What it does:**
- Analyzes the validation results
- Performs statistical analysis
- Detects parsing errors
- Reorganizes results into categories

**Input:**
- Validation results JSON file (output from `validate_citations.py`)

**Output:**
- Statistical analysis reports
- Parsing error detection reports
- Categorized result files (matched.json, parsing_errors.json, etc.)

**When to use:**
- After running `validate_citations.py`
- When you want to understand patterns in validation results
- When you want to reorganize results for easier analysis

**Example:**
```bash
# Statistical analysis
python src/analyze_citations.py \
  --input citation_validation_results.json \
  --mode stats \
  --output validation_analysis.json

# Reorganize into categories
python src/analyze_citations.py \
  --input citation_validation_results.json \
  --mode reorganize \
  --output-dir .
```

---

## Typical Workflow

1. **Validate citations:**
   ```bash
   python src/validate_citations.py \
     --input-dir data/parsed_jsons \
     --dblp-xml data/dblp.xml \
     --output citation_validation_results.json
   ```

2. **Analyze results:**
   ```bash
   python src/analyze_citations.py \
     --input citation_validation_results.json \
     --mode all
   ```

3. **Review categorized results:**
   - Check `results/matched.json` for correctly matched citations
   - Check `results/parsing_errors.json` for parsing issues
   - Check `results/first_names.json` for first name mismatches
   - etc.

---

## Key Takeaway

- **`validate_citations.py`** = Does the validation (checks citations)
- **`analyze_citations.py`** = Analyzes the validation results (produces reports)

You always run `validate_citations.py` first, then `analyze_citations.py` on its output.


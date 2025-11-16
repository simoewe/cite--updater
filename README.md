# Academic Paper Processing Toolkit

A comprehensive toolkit for downloading, processing, and analyzing academic papers from arXiv and conference proceedings.

## Overview

This toolkit provides automated workflows for:
- **PDF Collection**: Downloading conference papers from arXiv using DBLP metadata
- **Citation Processing**: Extracting citations and author information using GROBID
- **Citation Validation**: Validating citations against DBLP database with detailed error classification
- **LLM Classification**: Using language models to categorize citation discrepancies
- **Metadata Management**: Creating structured datasets from processed papers

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd cite-updater

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Conference PDFs from arXiv

```bash
# Activate virtual environment
source .venv/bin/activate

# Download PDFs from all supported conferences
python3 src/download_arxiv_pdfs.py
```

**Outputs:**
- `data/arxiv_pdfs/conference/year/` - Organized PDF files
- `data/arxiv_papers_metadata.json` - Download metadata and statistics
- `logs/arxiv_fetcher.log` - Detailed progress logs

### 3. Process PDFs with GROBID (Optional)

```bash
# Start GROBID server
sudo docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.0

# Process PDFs to extract citations and metadata
python src/run_grobid.py
```

**Outputs:** XML files in TEI format stored in `data/xml_files/`

### 4. Extract Structured Metadata

```bash
# Convert GROBID XML output to CSV format
python3 src/parse_grobid_to_csv.py
```

**Output:** `data/arxiv_metadata.csv` (tab-separated)

### 5. Validate Citations

```bash
# Validate citations against DBLP database
python src/validate_citations.py --input-dir data/parsed_jsons --num-files 50
```

**Output:** Creates `validation_results/` directory with categorized results

### 6. Classify Citation Mismatches (Optional)

```bash
# Classify mismatches using LLM
python src/vllm_classifier.py \
  --input_file validation_results/validation_results.json \
  --output_file validation_results/classified_results.json \
  --max_samples 100 \
  --gpu_memory_utilization 0.7
```

## Directory Structure

```
cite-updater/
├── src/                    # Main source code
│   ├── download_arxiv_pdfs.py      # arXiv PDF downloader
│   ├── run_grobid.py                # GROBID processing
│   ├── parse_grobid_to_csv.py      # Metadata extraction
│   ├── validate_citations.py       # Citation validation
│   ├── vllm_classifier.py          # LLM-based classification
│   ├── scrape_dblp_conferences.py  # DBLP scraper utility
│   ├── analyze_matches.py           # Author name matching utilities
│   ├── parser/                      # DBLP parser module
│   └── archive/                     # Deprecated scripts (see src/archive/README.md)
├── data/                   # Data files
│   ├── arxiv_pdfs/         # Raw PDF files by conference/year
│   ├── parsed_jsons/       # Processed citation data by conference/year
│   ├── dblp_conferences/   # DBLP conference metadata
│   ├── xml_files/          # GROBID XML outputs
│   ├── outputs/            # Processing outputs
│   ├── arxiv_papers_metadata.json  # Download metadata
│   └── dblp.xml            # DBLP database XML
├── validation_results/     # Citation validation results
│   ├── validation_results.json    # Complete validation data
│   ├── classified_results.json     # LLM-classified results
│   ├── results/            # Categorized result files
│   └── test_results/       # Test/experimental results
├── logs/                   # Log files
├── config/                 # Configuration files
├── tests/                  # Test files
├── docs/                   # Documentation and figures
├── archive/                # Archived files and old work
└── task/                   # Task-specific files and examples
```

## Core Scripts

### Citation Validation (`src/validate_citations.py`)

Validates citations by checking them against DBLP database and organizes results into categorized files.

**Features:**
- Cross-references citations with DBLP for accuracy validation
- Handles complex author name variations (initials, accents, compound names)
- Classifies mismatch types (accents_missing, first_name_mismatch, parsing_error, etc.)
- Filters non-academic references (Wikipedia, etc.)
- Processes thousands of citations efficiently
- Automatically organizes results into categorized files

**Usage:**
```bash
# Validate citations (uses defaults: data/dblp.xml, validation_results/ output folder)
python src/validate_citations.py --input-dir data/parsed_jsons --num-files 50

# With custom output directory
python src/validate_citations.py \
  --input-dir data/parsed_jsons \
  --output-dir my_validation_results \
  --num-files 50

# With custom DBLP XML path
python src/validate_citations.py \
  --input-dir data/parsed_jsons \
  --dblp-xml /path/to/dblp.xml \
  --output-dir validation_results

# For full-scale validation (time-intensive)
python src/validate_citations.py \
  --input-dir data/parsed_jsons \
  --output-dir full_validation_results
```

**Output:** Creates an output directory (default: `validation_results/`) containing:
- `validation_results.json` - Complete validation data with all results
- `results/` subfolder with categorized JSON files:
  - `matched.json` - Correctly matched citations
  - `parsing_errors.json` - Parsing issues
  - `first_names.json` - First name mismatches
  - `last_names.json` - Last name mismatches
  - `accents_missing.json` - Accent/diacritic issues
  - `author_not_found.json` - Authors not found in DBLP
  - `title_mismatches.json` - Title similarity below threshold
  - `summary.json` - Summary statistics
  - And more...

### LLM-Based Citation Classification (`src/vllm_classifier.py`)

Classifies author name mismatches using large language models to categorize citation discrepancies with detailed reasoning.

**Features:**
- Uses Qwen3-4B-Instruct-2507 model for accurate classification
- Classifies mismatches into 14 categories (TYPO, NICKNAME, DEADNAME, PARSER_ERROR, etc.)
- Provides reasoning and confidence levels for each classification
- Preserves all original validation data fields
- Handles batch processing for efficient inference
- Optimized sampling parameters following Qwen3 best practices

**Usage:**
```bash
# Activate virtual environment first
source .venv/bin/activate

# Classify 100 instances (recommended for testing)
python src/vllm_classifier.py \
  --input_file validation_results/validation_results.json \
  --output_file validation_results/classified_results.json \
  --max_samples 100 \
  --gpu_memory_utilization 0.7

# Classify all instances (remove --max_samples to process entire dataset)
python src/vllm_classifier.py \
  --input_file validation_results/validation_results.json \
  --output_file validation_results/classified_results.json \
  --gpu_memory_utilization 0.7
```

**Command Arguments:**
- `--input_file`: Path to validation results JSON file (default: `validation_results/validation_results.json`)
- `--output_file`: Path where classified results will be saved (default: `validation_results/classified_results.json`)
- `--max_samples`: Maximum number of samples to process (optional, omit to process all)
- `--batch_size`: Number of samples per batch (default: 8, increase if GPU memory allows)
- `--model_name`: HuggingFace model name (default: `Qwen/Qwen3-4B-Instruct-2507`)
- `--gpu_memory_utilization`: GPU memory usage fraction 0-1 (default: 0.9, use 0.7 if OOM errors)
- `--hf_token`: HuggingFace authentication token (default: uses token from code)

**Classification Categories:**
- **PARSER_ERROR** - Text extraction or OCR errors from PDF parsing (COMMON - truncation, diacritics corrupted, names split/merged)
- **NICKNAME** - Informal/short forms of same legal name OR cultural name variants (e.g., "Bill" → "William", "Eric" → "Yunxuan")
- **MIDDLE_NAME** - Different ordering of multi-part given names
- **INITIAL_VS_FULL** - Abbreviation level differs (e.g., "R Chen" → "Ricky T. Q. Chen")
- **TRANSLITERATION** - Different romanization of non-Latin names
- **DEADNAME** - Previous name no longer used by author (VERY RARE - CRITICAL severity)
- **NAME_CHANGE_OTHER** - Non-trans name changes (marriage, conversion, etc.)
- **WRONG_PERSON** - Completely different individuals (RARE)
- **TYPO** - Single character spelling error (VERY RARE - use conservatively)
- **AUTHOR_ORDER_ERROR** - Same authors, different sequence
- **LAST_NAME_ERROR** - Surname parsing/formatting issues (compound surnames)
- **AUTHOR_MISSING** - Authors omitted from citation
- **AMBIGUOUS** - Cannot confidently determine

**Important Classification Rules:**
- "Author not found" mismatches → ALWAYS AUTHOR_MISSING (never TYPO)
- "author_order_wrong" mismatches → ALWAYS AUTHOR_ORDER_ERROR (never TYPO)
- Truncation patterns (e.g., "Brocke" vs "Brockett") → PARSER_ERROR (not TYPO)
- Check parser errors FIRST before assuming typos

**Requirements:**
- GPU with CUDA support (recommended)
- HuggingFace authentication token
- vLLM library installed
- Sufficient GPU memory (model requires ~7-8GB)

### PDF Downloader (`src/download_arxiv_pdfs.py`)

Downloads conference papers from arXiv using DBLP conference metadata.

**Features:**
- Multi-conference support (AAAI, ICML, ICLR, FACCT, NEURIPS)
- Fuzzy title matching with configurable thresholds
- Automatic author name cleaning (removes DBLP numeric suffixes)
- Resume capability for interrupted downloads
- Progress tracking and detailed logging

**Usage:**
```bash
# Download all conferences
python3 src/download_arxiv_pdfs.py

# Download with custom settings
python3 src/download_arxiv_pdfs.py --max-papers 10 --match-threshold 90 --delay 5
```

**Outputs:**
- `data/arxiv_pdfs/conference/year/` - Organized PDF files
- `data/arxiv_papers_metadata.json` - Download metadata and statistics
- `logs/arxiv_fetcher.log` - Detailed progress logs

### GROBID Processing (`src/run_grobid.py`)

Processes PDFs with GROBID to extract citations, authors, and metadata.

**Requirements:** Running GROBID Docker container
```bash
# For GPU acceleration (recommended)
sudo docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.0

# For CPU-only
sudo docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.0
```

**Outputs:** XML files in TEI format stored in `data/xml_files/`

### Metadata Extraction (`src/parse_grobid_to_csv.py`)

Converts GROBID XML outputs to structured CSV format.

**Features:**
- Extracts paper titles, authors, and affiliations
- Filters to main paper authors (excludes citations)
- Handles missing data gracefully
- Processes thousands of XML files efficiently

**Output:** `data/arxiv_metadata.csv` (tab-separated)

### DBLP Conference Scraper (`src/scrape_dblp_conferences.py`)

Scrapes paper lists from major AI conferences on DBLP.

**Features:**
- Cross-referencing against arXiv, DBLP, Semantic Scholar
- Fuzzy author name matching
- Confidence scoring for matches
- Rate limiting and error handling

### Multi-Database API Caller (`api_caller.py`)

Searches for papers across multiple databases (DBLP, arXiv, Semantic Scholar) with proper rate limiting and Open-Access filtering.

**Features:**
- Parallel database searches for efficiency
- Thread-safe rate limiting (DBLP: 3s, arXiv: 3s, Semantic Scholar: 1s)
- Open-Access filtering (only returns accessible papers)
- Fuzzy title matching using Fuzzywuzzy
- Batch processing support for multiple titles
- Comprehensive metadata extraction (title, authors, year, DOI, URLs)

**Usage:**
```python
from api_caller import search_papers_by_title, search_multiple_titles

# Search for a single paper
result = search_papers_by_title("Attention Is All You Need", similarity_threshold=80)

# Search for multiple papers in parallel
titles = ["Paper Title 1", "Paper Title 2"]
results = search_multiple_titles(titles, max_workers=5)
```

**Rate Limits:**
- DBLP: 1 request per 3 seconds
- arXiv: 1 request per 3 seconds
- Semantic Scholar: 1 request per second (5 requests per 5 seconds with API key)

**API Key Configuration (Optional):**
For higher Semantic Scholar rate limits, configure your API key:
1. Copy `.env.example` to `.env`: `cp .env.example .env`
2. Add your API key to `.env`: `SEMANTIC_SCHOLAR_API_KEY=your_key_here`
3. Get your API key from: https://www.semanticscholar.org/product/api

The `.env` file is automatically ignored by git (already in `.gitignore`).

## Supported Conferences

- **AAAI** (2015-2025): Association for the Advancement of Artificial Intelligence
- **ICML** (2015-2024): International Conference on Machine Learning
- **ICLR** (2015-2025): International Conference on Learning Representations
- **FACCT** (2021-2025): ACM Conference on Fairness, Accountability, and Transparency
- **NEURIPS** (2014-2024): Neural Information Processing Systems

## Configuration

Edit `config/config.json` for GROBID settings:

```json
{
    "grobid_server": "http://localhost:8070",
    "batch_size": 1000,
    "sleep_time": 5,
    "timeout": 60,
    "coordinates": ["persName", "figure", "ref", "biblStruct", "formula", "s"]
}
```

## Dependencies

- `arxiv==1.4.8` - arXiv API access
- `nameparser==1.1.3` - Author name parsing
- `fuzzywuzzy==0.18.0` - Fuzzy string matching
- `requests>=2.31.0` - HTTP requests
- `backoff==2.2.1` - Retry logic
- `tqdm>=4.66.0` - Progress bars
- `vllm>=0.11.0` - High-throughput LLM inference (for classification)
- `torch` - PyTorch (required by vLLM)
- `transformers` - HuggingFace transformers (for model loading)
- `huggingface_hub` - HuggingFace model access

## Limitations

- **API Rate Limits**: arXiv and DBLP APIs have request limits
- **Matching Accuracy**: Success depends on title similarity thresholds
- **Coverage**: Only papers with arXiv versions can be downloaded
- **Processing Time**: Large-scale processing can take several hours
- **GROBID**: Requires Docker and significant computational resources
- **LLM Classification**: Requires GPU with CUDA support and sufficient VRAM (~8GB+)
- **Model Size**: Qwen3-4B-Instruct model requires significant disk space (~8GB)

## Archive

Deprecated or unused scripts are kept in `src/archive/` for reference. See `src/archive/README.md` for details about archived files.

**Note:** Archived scripts should not be used in new workflows. Use the current scripts documented above instead.

## License

MIT License

---

## Development Diary

### 2025-01-XX - Multi-Database API Caller with API Key Support

Added comprehensive API caller module for searching papers across multiple databases:

- **New module `api_caller.py`**: Provides unified interface for searching DBLP, arXiv, and Semantic Scholar
  - Thread-safe rate limiting for each API source
  - Parallel database searches using ThreadPoolExecutor
  - Open-Access filtering (arXiv: all papers, Semantic Scholar: openAccessPdf field, DBLP: accessible URLs)
  - Fuzzy title matching with configurable similarity threshold
  - Batch processing support for multiple titles
  - Comprehensive metadata extraction (title, authors, year, DOI, URLs, PDF links)

- **API Key Support**:
  - Secure API key management via environment variables or `.env` file
  - Semantic Scholar API key support for higher rate limits (5 requests per 5 seconds with key)
  - `.env.example` template file for easy configuration
  - Automatic `.env` file exclusion from git (already in `.gitignore`)
  - Graceful fallback to public rate limits if no key is provided

- **Key features**:
  - Respects API rate limits (DBLP: 3s, arXiv: 3s, Semantic Scholar: 1s or 5/5s with key)
  - Efficient parallel processing for batch searches
  - Only returns Open-Access papers
  - Results ranked by title similarity score
  - JSON export functionality for results

- **Use cases**:
  - Identifying papers with similar titles across databases
  - Verifying paper metadata from multiple sources
  - Finding Open-Access versions of papers
  - Batch processing of citation lists

The API caller enables efficient cross-database paper searches while respecting rate limits and focusing on Open-Access content. API key support allows for higher throughput when processing large batches of papers.

### 2025-11-13 (5) - Advanced Author Name Matching and Output Organization

Refined citation validation to handle complex name variations and improve result organization:

- **Enhanced Initial Matching**: Fixed initial matching to properly handle accented characters (e.g., "Ł Kaiser" now matches "Lukasz Kaiser")
- **Improved Accent Detection**: Better detection of accent differences when one name uses initials (e.g., "A Hyvarinen" vs "Aapo Hyvärinen" now classified as accents_missing)
- **Flexible Initial-Name Matching**: Authors with initials now properly match full names starting with the same letter (e.g., "S Corff" matches "Sylvain Le Corff")
- **Output Organization**: Parsing errors are now sorted to the bottom of mismatch results for better prioritization

**Technical Changes:**
- Modified `get_initials()` in `analyze_matches.py` to normalize accented characters before extracting initials
- Enhanced accent detection logic in validation to handle initial-full name combinations
- Added initial matching logic for both first and last name comparisons
- Implemented sorting of mismatch results to prioritize non-parsing errors

**Impact:** More accurate author matching for citations using various naming conventions, better error classification, and improved result organization for easier analysis of validation issues.

### 2025-11-13 (2) - Parsing Error Detection in Citation Validation

Enhanced the citation validation system to detect and flag systematic parsing errors where author names are shifted or mixed up:

- **Added parsing error detection to `src/validate_citations.py`**:
  - Detects when first names and last names are mixed up between adjacent authors
  - Checks if reference author's last name matches DBLP author's first name (or vice versa)
  - Flags entire reference as `parsing_error` when this pattern is detected
  - Short-circuits further error analysis when parsing error is found (since all other errors are likely consequences of the parsing issue)

- **Reorganized output structure**:
  - Output JSON now separates `mismatches` (author_mismatch status) and `matches` (matched status) into distinct top-level sections
  - Mismatches appear first in the output file for easier inspection
  - Added counts to analysis section: `mismatch_count` and `match_count`

- **Test results** (20 files, 825 references):
  - **50 parsing errors detected** - systematic name shifting issues
  - 105 author_not_found errors
  - 35 first_name_mismatch errors
  - 27 last_name_mismatch errors
  - 14 accents_missing errors
  
- **Example parsing errors found**:
  - BERT paper: "Kenton", "Lee Kristina" parsed instead of "Kenton Lee", "Kristina Toutanova"
  - "William Yang", "Wang" split instead of "William Yang Wang"
  - "Christophe Hoa T Le" parsed instead of "Hoa T. Le" (first author's first name becoming previous author's last name)

The parsing error detection helps identify systematic issues in PDF parsing that affect entire reference lists, making it easier to prioritize which citation extraction errors need fixing at the parser level vs. minor variations in author name formatting.

### 2025-11-13 (1) - Enhanced Citation Validation System

Improved the citation validation system with better error detection and classification:

- **Enhanced `src/validate_citations.py`** with new features:
  - **Title similarity filtering**: Only considers DBLP matches if string similarity >= 90% (configurable via `--title-similarity-threshold`)
  - **Minimum author list comparison**: Compares only the minimum length of reference and DBLP author lists, handling cases where authors don't include full author lists
  - **Error classification**: Categorizes mismatches into specific types:
    - `first_name_mismatch`: First names differ
    - `last_name_mismatch`: Last names differ
    - `accents_missing`: Names differ only by accents/diacritics
    - `author_order_wrong`: Authors match but order differs
    - `author_not_found`: Author not found in DBLP list
  - Uses `rapidfuzz` for accurate title similarity calculation
  - Tracks title similarity scores in results for analysis

- **Enhanced `src/analyze_validation_results.py`**:
  - Analyzes error classifications and patterns
  - Identifies common mistakes (low title similarity, order issues, accent problems)
  - Provides statistics on author list length differences
  - Generates detailed analysis JSON with examples

- **Key improvements**:
  - More accurate matching by filtering low-similarity titles
  - Better handling of partial author lists (common in citations)
  - Detailed error classification for easier debugging
  - Comprehensive analysis tools for understanding validation results

The enhanced validation system provides more accurate citation validation and better insights into citation errors.

### 2025-01-11 - Citation Validation System

Added a comprehensive citation validation system to check author citations in parsed JSON files against DBLP database:

- **New script `src/validate_citations.py`**: Processes JSON files from `data/parsed_jsons/` and validates each reference by:
  - Querying DBLP database using paper titles
  - Comparing author lists between references and DBLP entries
  - Using existing name matching logic from `analyze_matches.py` to handle variations (initials, reversed names, etc.)
  - Flagging incorrect citations with detailed mismatch information

- **Analysis tool `src/analyze_validation_results.py`**: Provides detailed statistics and insights:
  - Overall match/mismatch rates
  - Categorization of mismatch types (missing authors, extra authors, list differences)
  - Examples of problematic citations
  - Files with most mismatches

- **Key features**:
  - Processes multiple JSON files (default: 20 files for testing)
  - Uses DBLP parser with BM25 search for efficient title matching
  - Leverages existing author name normalization and matching functions
  - Generates comprehensive JSON output with validation results
  - Handles edge cases (missing titles, parsing errors, etc.)

- **Initial test results** (20 files, 952 references):
  - 53.2% match rate (correct citations)
  - 46.8% mismatch rate (potential errors or variations)
  - Detects real errors (e.g., wrong author names) as well as minor variations (abbreviations, middle initials)

The validation system helps identify citation errors in academic papers, enabling quality control and data cleaning workflows.

### 2025-11-11 - README Organization and Cleanup

Completely reorganized and cleaned up the README to reflect the current state of the project:

- **Removed outdated content**: Eliminated redundant sections, confusing explanations, and outdated file paths
- **Simplified structure**: Streamlined from verbose documentation to clear, actionable information
- **Updated data organization**: Documented the current clean folder structure with `parsed_jsons/`, organized outputs, etc.
- **Focused on core functionality**: Emphasized the main workflows (PDF downloading, GROBID processing, metadata extraction)
- **Added clear data structure diagram**: Shows how files are organized across the project
- **Removed development diary**: Consolidated into a single, clean document rather than maintaining separate diary entries

The README now serves as a clear guide for users to understand and use the toolkit effectively.
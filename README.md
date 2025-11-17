# Academic Paper Processing Toolkit

A comprehensive toolkit for downloading, processing, and analyzing academic papers from arXiv and conference proceedings.

## Overview

This toolkit provides automated workflows for:
- **PDF Collection**: Downloading conference papers from arXiv using DBLP metadata
- **Citation Processing**: Extracting citations and author information using GROBID
- **Metadata Management**: Creating structured datasets from processed papers
- **Data Organization**: Maintaining clean, organized file structures

## Quick Start

### 1. Download Conference PDFs from arXiv

```bash
# Activate virtual environment
source .venv/bin/activate

# Download PDFs from all supported conferences
python3 src/download_arxiv_pdfs.py
```

### 2. Process PDFs with GROBID (Optional)

```bash
# Start GROBID server
sudo docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.0

# Process PDFs to extract citations and metadata
python src/run_grobid.py
```

### 3. Extract Structured Metadata

```bash
# Convert GROBID XML output to CSV format
python3 src/parse_grobid_to_csv.py
```

## Data Organization

The toolkit maintains a clean, organized data structure:

```
data/
├── arxiv_pdfs/           # Raw PDF files by conference/year
│   ├── aaai/2015/...
│   ├── icml/2023/...
│   └── neurips/2024/...
├── parsed_jsons/         # Processed citation data by conference/year
│   ├── aaai/2015/...
│   ├── icml/2023/...
│   └── neurips/2024/...
├── dblp_conferences/     # DBLP conference metadata
│   ├── AAAI/
│   ├── ICML/
│   └── NEURIPS/...
├── xml_files/            # GROBID XML outputs
├── outputs/
│   ├── legacy/           # Archived processing outputs
│   └── xml_files/        # Current XML processing results
└── arxiv_metadata.csv    # Structured metadata (optional)
```

## Installation

1. **Clone the repository**
2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Install GROBID for citation processing**
   ```bash
   # For GPU acceleration (recommended)
   sudo docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.0

   # For CPU-only
   sudo docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.0
   ```

## Core Scripts

### File Organization

The `src/` directory contains the main scripts for the toolkit:
- **Active scripts**: Core functionality scripts documented below
- **Archive folder** (`src/archive/`): Deprecated or unused scripts kept for reference

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


### Recommended Workflow

After processing PDFs with GROBID and extracting structured data, follow this validation workflow:

#### Step 1: Sample Validation (Recommended First)
```bash
# Validate a representative sample to test system and estimate results
source .venv/bin/activate
python src/validate_citations.py \
  --input-dir data/parsed_jsons \
  --output-dir sample_validation_results \
  --num-files 50 \
  --title-similarity-threshold 95.0
```

#### Step 2: Review Categorized Results
Check the output directory (default: `validation_results/`):
- Review `results/parsing_errors.json` for systematic parsing issues
- Check `results/first_names.json` and `results/last_names.json` for name mismatches
- Examine `results/matched.json` to verify correct matches
- Check `results/summary.json` for overall statistics

#### Step 3: Full-Scale Validation (Optional - Time Intensive)
```bash
# Process all 100K+ files (may take several hours/days)
python src/validate_citations.py \
  --input-dir data/parsed_jsons \
  --output-dir full_validation_results \
  --title-similarity-threshold 95.0
```

#### Step 4: Improve Based on Results
Based on validation results, consider:
- Adjusting title similarity thresholds
- Improving author name parsing
- Adding new error classifications
- Updating the GROBID processing pipeline

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
- `arxiv_papers_metadata.json` - Download metadata and statistics
- `arxiv_download_progress.log` - Detailed progress logs

### GROBID Processing (`src/run_grobid.py`)

Processes PDFs with GROBID to extract citations, authors, and metadata.

**Requirements:** Running GROBID Docker container
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
- Supports multiple conferences (ICLR, NeurIPS, ICML, AAAI, FACCT, etc.)
- Fetches conference proceedings in JSON format
- Organizes data by conference and year

**Note:** This is a utility script for gathering DBLP conference metadata. The main workflow uses pre-scraped DBLP data.

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

## Limitations

- **API Rate Limits**: arXiv and DBLP APIs have request limits
- **Matching Accuracy**: Success depends on title similarity thresholds
- **Coverage**: Only papers with arXiv versions can be downloaded
- **Processing Time**: Large-scale processing can take several hours
- **GROBID**: Requires Docker and significant computational resources

## License

MIT License

---

## Archive

Deprecated or unused scripts are kept in `src/archive/` for reference. See `src/archive/README.md` for details about archived files.

**Note:** Archived scripts should not be used in new workflows. Use the current scripts documented above instead.

## Development Diary

### 2025-01-XX - Source Code Reorganization

Reorganized the `src/` directory to improve clarity and remove unused files:

- **Created `src/archive/` folder** for deprecated/unused scripts:
  - Moved `citation_pipeline.py` (superseded by `validate_citations.py`)
  - Moved `sample_and_process.py` and `sample_analysis.py` (test scripts)
  - Moved `parse_citations.py` (unused, redundant functionality)
  - Moved `download_pdf.py` (ACL Anthology downloader, not part of main workflow)

- **Active scripts remain in `src/`:**
  - `validate_citations.py` - Main citation validation system
  - `analyze_citations.py` - Citation analysis tool
  - `analyze_matches.py` - Author name matching utilities (used by validate_citations.py)
  - `download_arxiv_pdfs.py` - arXiv PDF downloader
  - `run_grobid.py` - GROBID processing
  - `parse_grobid_to_csv.py` - Metadata extraction
  - `scrape_dblp_conferences.py` - DBLP conference scraper utility

- **Benefits:**
  - Cleaner `src/` directory with only active scripts
  - Clear separation between current and deprecated code
  - Easier to understand the main workflow
  - Archived scripts preserved for reference

### 2025-11-17 - Citation Results Reorganization System

Added a comprehensive system to reorganize citation validation results into a structured folder format:

- **Result reorganization** (via `src/analyze_citations.py --mode reorganize`): Takes large validation JSON files and splits them into categorized files:
  - `matched.json` - Citations that matched correctly with DBLP (1,210 results)
  - `parsing_errors.json` - Citations with parsing errors in author names (167 results)
  - `first_names.json` - Citations with first name mismatches (59 results)
  - `last_names.json` - Citations with last name mismatches (42 results)
  - `accents_missing.json` - Citations with missing accents/diacritics (40 results)
  - `author_not_found.json` - Citations where authors were not found in DBLP (68 results)
  - `author_order_wrong.json` - Citations with correct authors but wrong order (2 results)
  - `title_mismatches.json` - Citations with title similarity below threshold (352 results)
  - `no_dblp_match.json` - Citations not found in DBLP database (2 results)
  - `errors.json` - Citations that caused processing errors (25 results)
  - `skipped.json` - Citations that were skipped (8 results)
  - `summary.json` - Summary statistics for all categories

- **Features implemented**:
  - Automatic creation of `results/` folder structure
  - Deduplication of results based on reference ID and title
  - Comprehensive error type categorization
  - Auto-generated README.md with file descriptions and statistics
  - Support for all validation status types and error classifications

- **Test results** (sample validation results with 1,948 unique citations):
  - Successfully categorized all citations into appropriate files
  - Generated comprehensive documentation with statistics table
  - Maintained data integrity and proper JSON formatting

This reorganization makes it much easier to analyze specific types of citation errors and understand validation patterns across large datasets.

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

- **Enhanced analysis tool** (`src/analyze_citations.py`):
  - Statistical analysis: error classifications, title similarities, author list lengths, common mistakes
  - Parsing error detection: identifies cascading names, split multi-part names, name swapping
  - Result reorganization: organizes results into categorized files

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

- **Analysis tool `src/analyze_citations.py`**: Provides comprehensive analysis with multiple modes:
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

### 2025-11-17 - Enhanced Name Matching and Validation Fixes

Fixed several validation issues to improve accuracy of citation validation:

- **Accent normalization**: Fixed handling of accented characters (e.g., "Kuebler" vs "Kübler" now correctly identified as the same name)
  - Added `normalize_name_for_comparison()` function using `unidecode` for consistent accent handling
  - Added `names_match_with_accents()` helper function for accent-aware matching

- **Middle initial handling**: Fixed matching when one name has a middle initial and the other doesn't (e.g., "Ed Chi" vs "Ed H. Chi" now correctly match)
  - Added `handle_middle_initial_match()` function to handle middle initial variations
  - Checks if first names match when middle initials are present/absent

- **Compound last names with prefixes**: Fixed handling of last names with prefixes like "De", "Van", "Von" (e.g., "De Choudhury" vs "Choudhury" now correctly identified as the same)
  - Added `normalize_last_name_with_prefixes()` function to handle common name prefixes
  - Normalizes last names by removing prefixes for comparison while preserving originals

- **Improved parsing error detection**: Enhanced detection of single-word fragments and shifted author names
  - Better detection of single-word fragments (e.g., "Wilson", "Veličković" as last names without first names)
  - Improved detection of author name shifting/mixing issues

- **Enhanced first-pass matching**: Added accent-aware and prefix-aware matching in the initial matching phase
  - Checks for accent-normalized matches before error classification
  - Checks for compound name matches with prefix handling
  - Reduces false positives in mismatch detection

**Results after fixes** (20 files, 868 references):
- 517 matched (correct citations)
- 113 mismatches (down from previous runs)
- 85 parsing errors detected
- 8 last name mismatches (many previously false positives now correctly matched)
- 7 first name mismatches

The validation system now more accurately distinguishes between real citation errors and legitimate name variations. Results are automatically categorized into separate JSON files for easy analysis.

### 2025-11-11 - README Organization and Cleanup

Completely reorganized and cleaned up the README to reflect the current state of the project:

- **Removed outdated content**: Eliminated redundant sections, confusing explanations, and outdated file paths
- **Simplified structure**: Streamlined from verbose documentation to clear, actionable information
- **Updated data organization**: Documented the current clean folder structure with `parsed_jsons/`, organized outputs, etc.
- **Focused on core functionality**: Emphasized the main workflows (PDF downloading, GROBID processing, metadata extraction)
- **Added clear data structure diagram**: Shows how files are organized across the project
- **Removed development diary**: Consolidated into a single, clean document rather than maintaining separate diary entries

The README now serves as a clear guide for users to understand and use the toolkit effectively.

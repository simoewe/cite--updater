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

### Citation Analysis (`src/citation_pipeline.py`)

Validates and cross-references citation authors against multiple databases.

**Features:**
- Cross-referencing against arXiv, DBLP, Semantic Scholar
- Fuzzy author name matching
- Confidence scoring for matches
- Rate limiting and error handling

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

## Development Diary

### 2025-11-11 - README Organization and Cleanup

Completely reorganized and cleaned up the README to reflect the current state of the project:

- **Removed outdated content**: Eliminated redundant sections, confusing explanations, and outdated file paths
- **Simplified structure**: Streamlined from verbose documentation to clear, actionable information
- **Updated data organization**: Documented the current clean folder structure with `parsed_jsons/`, organized outputs, etc.
- **Focused on core functionality**: Emphasized the main workflows (PDF downloading, GROBID processing, metadata extraction)
- **Added clear data structure diagram**: Shows how files are organized across the project
- **Removed development diary**: Consolidated into a single, clean document rather than maintaining separate diary entries

The README now serves as a clear guide for users to understand and use the toolkit effectively.
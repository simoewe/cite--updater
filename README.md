# Academic Citation Author Matcher & arXiv PDF Downloader

A comprehensive toolkit for academic paper processing that includes:
- **Citation Author Matching**: Validate and normalize author information across multiple scholarly databases
- **arXiv PDF Downloader**: Automatically download PDFs from arXiv using DBLP conference data

## Objective

This tool helps researchers and publishers by:
1. **Citation Processing**: Extracting citations and author information from academic PDFs using GROBID
2. **Author Validation**: Cross-referencing papers against multiple scholarly databases (arXiv, DBLP, Semantic Scholar)
3. **Name Normalization**: Normalizing author names across different citation formats
4. **PDF Collection**: Automatically downloading conference papers from arXiv based on DBLP data
5. **Metadata Management**: Creating comprehensive metadata with cleaned author names and file organization
6. **Quality Assurance**: Detecting and reporting potential author name mismatches with confidence scores

## Prerequisites

- Python 3.8+
- Docker for running GROBID (for citation processing)
- NVIDIA GPU (optional, but recommended for better GROBID performance)
- Internet connection (for arXiv API access)

## Installation

1. Clone this repository
2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start GROBID server (for citation processing):

```bash
# Pull and run GROBID with GPU support
sudo docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.0
```

## Quick Start

### Download Conference PDFs from arXiv

```bash
# Activate environment
source .venv/bin/activate

# Download all conference papers from arXiv (2-3 hours)
python3 src/download_arxiv_pdfs.py
```

### Process Citations from PDFs

```bash
# Process PDFs with GROBID (requires Docker)
python src/run_grobid.py

# Run citation author matching
python src/citation_pipeline.py --input data/outputs/processed.xml
```

### Extract Metadata to CSV

```bash
# Extract paper metadata (ID, Title, Authors, Affiliations) from GROBID XML files
python3 src/parse_grobid_to_csv.py
```

This script processes all GROBID TEI XML files matching the pattern `2025.*.grobid.tei.xml` and creates a tab-separated CSV file with:
- **ID**: Paper identifier (extracted from filename)
- **Title**: Main paper title
- **Authors**: Semicolon-separated list of authors (forename + surname)
- **Affiliations**: Semicolon-separated list of author affiliations

The output CSV is saved to `data/arxiv_metadata.csv`.

## Detailed Usage

### 1. Prepare Your PDFs
Place your academic PDFs in the `data/pdfs/` directory.

### 2. Process PDFs with GROBID

```python
from grobid_client.grobid_client import GrobidClient

client = GrobidClient(config_path="./config/config.json")
client.process('processReferences', 'data/pdfs', output='data/outputs', consolidate_citations=False, verbose=True)
```

### 3. Run the Citation Pipeline

```bash
python citation_pipeline.py [options]

Options:
  --dry-run          Run with a random sample of papers
  --sample N         Number of papers to sample in dry run (default: 5)
  --input FILE       Input XML file path
  --output FILE      Output JSON file path
  --threshold N      Title match threshold (default: 80)
  --dblp-delay N     Delay between DBLP API calls in seconds (default: 1)
  --arxiv-delay N    Delay between arXiv API calls in seconds (default: 0.5)
```

### 4. Analyze Results

```bash
python analyze_matches.py
```

## arXiv PDF Downloader

Automatically download conference papers from arXiv using DBLP conference data with intelligent matching and metadata creation.

### Features

- **Multi-Conference Support**: Processes AAAI, ICML, ICLR, FACCT, and NEURIPS conferences
- **Intelligent Matching**: Fuzzy title matching with configurable similarity thresholds
- **Organized Storage**: PDFs organized in `conference/year/` folder structure
- **Clean Metadata**: Comprehensive metadata with cleaned author names (removes "0001" suffixes)
- **Resume Capability**: Automatically resumes from interruptions
- **Progress Tracking**: Real-time progress bars and detailed logging

### Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Download PDFs from all conferences
python3 src/download_arxiv_pdfs.py
```

### Advanced Usage

```bash
# Process all conferences (default behavior)
python3 src/download_arxiv_pdfs.py

# Limit papers per conference for testing
python3 src/download_arxiv_pdfs.py --max-papers 10

# Start fresh (don't resume from previous run)
python3 src/download_arxiv_pdfs.py --no-resume

# Custom matching and timing
python3 src/download_arxiv_pdfs.py --match-threshold 90 --delay 5

# Custom metadata file location
python3 src/download_arxiv_pdfs.py --metadata-file my_papers_metadata.json
```

### Output Structure

```
data/arxiv_pdfs/
├── aaai/
│   ├── 2024/
│   │   └── 2306.15222v2.pdf
│   └── 2025/
│       └── 2412.13333v2.pdf
├── icml/
│   └── 2023/
│       └── 2304.01203v7.pdf
└── neurips/
    └── 2024/
        └── 2311.15864v4.pdf
```

### Metadata Format

The tool creates `arxiv_papers_metadata.json` with detailed information:

```json
{
  "2306.15222v2": {
    "arxiv_id": "2306.15222v2",
    "title": "Learning to Rank in Generative Retrieval",
    "authors": ["Hansi Zeng", "Hamed Zamani", "Donald Metzler"],
    "conference": "AAAI",
    "year": 2024,
    "file_path": "aaai/2024/2306.15222v2.pdf",
    "download_date": "2025-11-02T18:22:04.153508",
    "match_score": 99
  }
}
```

### Author Name Cleaning

The tool automatically cleans author names by removing DBLP-style numeric suffixes:
- `"John Doe 0001"` → `"John Doe"`
- `"Jane Smith 1234"` → `"Jane Smith"`

### Supported Conferences

- **AAAI** (2015-2025): Association for the Advancement of Artificial Intelligence
- **ICML** (2015-2024): International Conference on Machine Learning
- **ICLR** (2015-2025): International Conference on Learning Representations
- **FACCT** (2021-2025): ACM Conference on Fairness, Accountability, and Transparency
- **NEURIPS** (2020-2024): Neural Information Processing Systems

## Configuration

### GROBID Configuration
Edit `config/config.json` to configure GROBID settings:

```json
{
    "grobid_server": "http://localhost:8070",
    "batch_size": 1000,
    "sleep_time": 5,
    "timeout": 60,
    "coordinates": ["persName", "figure", "ref", "biblStruct", "formula", "s"]
}
```

## Output Format

The tool generates a JSON file containing:

```json
{
  "title": "Paper Title",
  "parsed_authors": [
    {
      "first_name": "First",
      "middle_name": "Middle",
      "last_name": "Last",
      "suffix": "",
      "title": "",
      "original": "Original Full Name"
    }
  ],
  "matched_authors": [],
  "source": "arxiv|dblp|semantic_scholar",
  "mismatches": ["List of detected mismatches"]
}
```

## Output Files

### Citation Pipeline
- **`author_matches.json`**: Citation validation results with author matching information
- **`citation_pipeline.log`**: Detailed processing logs

### arXiv PDF Downloader
- **`data/arxiv_pdfs/conference/year/`**: Organized PDF files by conference and year
- **`arxiv_papers_metadata.json`**: Comprehensive metadata with cleaned author information
- **`arxiv_download_summary.json`**: Processing statistics and summary
- **`arxiv_download_progress.log`**: Detailed download progress and error logs

### GROBID Metadata Extractor
- **`data/arxiv_metadata.csv`**: Tab-separated CSV file with paper metadata (ID, Title, Authors, Affiliations) extracted from GROBID TEI XML files

## Error Handling

### Citation Pipeline
- Exponential backoff for API failures
- Rate limiting to respect API constraints
- Comprehensive error logging
- Graceful handling of missing or malformed data

### arXiv PDF Downloader
- Intelligent retry logic for failed downloads
- Rate limiting with configurable delays (default 3 seconds)
- Resume capability from any interruption point
- Detailed error logging with progress tracking
- Graceful handling of API timeouts and network issues

## Name Parsing Logic

The tool uses the `nameparser` library to handle complex author names, including:
- Multiple given names
- Compound surnames
- Academic titles
- Suffixes and honorifics

## Limitations

### Citation Processing
1. Fuzzy matching accuracy depends on title similarity threshold
2. API rate limits affect processing speed
3. Requires active internet connection for database queries
4. GROBID parsing quality affects overall results

### arXiv PDF Downloader
1. **Matching Accuracy**: Success depends on title similarity (default 85% threshold)
2. **API Rate Limits**: arXiv API has request limits (3-second delays implemented)
3. **Coverage**: Only downloads papers that have arXiv versions (typically ~50-60% match rate)
4. **File Organization**: Requires sufficient disk space for large PDF collections
5. **Resume Capability**: Large interruptions may require manual intervention for very long runs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License

---

## Development Diary

### 2025-11-10 - GROBID Metadata Extraction to CSV

Added a new script `src/parse_grobid_to_csv.py` that extracts structured metadata from GROBID TEI XML files and converts them to CSV format. The script:

- Processes all XML files matching the pattern `2025.*.grobid.tei.xml` in `data/outputs/arxiv_pdfs/`
- Extracts paper metadata including:
  - **ID**: Extracted from filename (removes `.grobid.tei.xml` extension)
  - **Title**: Main paper title from `<title>` elements
  - **Authors**: Only extracts authors from the `<analytic>` section (main paper authors), not from citations/references
  - **Affiliations**: Extracts organization names from author affiliations in the analytic section
- Outputs a tab-separated CSV file to `data/arxiv_metadata.csv`

**Key Implementation Details:**
- Uses XML namespace-aware parsing (`http://www.tei-c.org/ns/1.0`)
- Restricts author/affiliation extraction to `<sourceDesc>/<biblStruct>/<analytic>` section to avoid including citation authors
- Handles missing data gracefully (empty strings for missing titles/authors/affiliations)
- Processes thousands of XML files efficiently

This tool is useful for creating structured datasets from GROBID-processed academic papers for further analysis or database import.
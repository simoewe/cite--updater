# Academic Citation Author Matcher

A tool for validating and normalizing author information in academic papers by cross-referencing multiple scholarly databases (arXiv, DBLP, and Semantic Scholar).

## Objective

This tool helps researchers and publishers by:
1. Extracting citations and author information from academic PDFs using GROBID
2. Cross-referencing papers against multiple scholarly databases
3. Normalizing author names across different citation formats
4. Detecting and reporting potential author name mismatches
5. Providing confidence scores for paper matches

## Prerequisites

- Python 3.8+
- Docker for running GROBID
- NVIDIA GPU (optional, but recommended for better GROBID performance)

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

4. Start GROBID server:

```bash
# Pull and run GROBID with GPU support
sudo docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.0
```

## Usage

### 1. Prepare Your PDFs
Place your academic PDFs in the `pdfs/` directory.

### 2. Process PDFs with GROBID

```python
from grobid_client.grobid_client import GrobidClient

client = GrobidClient(config_path="./config.json")
client.process('processReferences', 'pdfs', output='output', consolidate_citations=False, verbose=True)
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

## Configuration

### GROBID Configuration
Edit `config.json` to configure GROBID settings:

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

## Error Handling

The pipeline includes:
- Exponential backoff for API failures
- Rate limiting to respect API constraints
- Comprehensive error logging
- Graceful handling of missing or malformed data

## Name Parsing Logic

The tool uses the `nameparser` library to handle complex author names, including:
- Multiple given names
- Compound surnames
- Academic titles
- Suffixes and honorifics

## Limitations

1. Fuzzy matching accuracy depends on title similarity threshold
2. API rate limits affect processing speed
3. Requires active internet connection for database queries
4. GROBID parsing quality affects overall results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License
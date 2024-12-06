# Citation Author Matcher

This tool processes academic papers from GROBID-parsed XML files and matches them against arXiv and DBLP databases to verify and normalize author information.

## Features

- Parses GROBID-generated XML files containing academic paper metadata
- Matches papers against arXiv and DBLP using fuzzy title matching
- Normalizes author names across different sources
- Handles rate limiting and retries for API calls
- Outputs results in JSON format with detailed author information

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up GROBID using Docker:

```bash
# Pull the GROBID image
docker pull lfoppiano/grobid:0.7.3

# Run GROBID container
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.3
```

## Usage

### 1. Process PDFs with GROBID

First, process your PDF files using GROBID to extract references:

```python
from grobid_client.grobid_client import GrobidClient

client = GrobidClient(config_path="./config.json")
# Process references in pdfs folder, saving the output in the output folder
client.process('processReferences', 'pdfs', output='output', consolidate_citations=False, verbose=True)
```

### 2. Run the Citation Pipeline

The pipeline consists of two main scripts:

#### citation_pipeline.py
This script processes the GROBID-parsed XML files and attempts to match papers with arXiv and DBLP:

```bash
python citation_pipeline.py
```

The script will:
1. Parse author information from the XML
2. Try to match each paper with arXiv first
3. If no match is found in arXiv, try DBLP
4. Save the results in `author_matches.json`

#### analyze_matches.py
This script analyzes the matching results and provides statistics:

```bash
python analyze_matches.py
```

## Configuration

The following parameters can be adjusted in `citation_pipeline.py`:

- `TITLE_MATCH_THRESHOLD`: Minimum fuzzy match score (default: 80)
- `XML_FILE_PATH`: Path to the GROBID XML file
- `OUTPUT_FILE`: Path for the output JSON file
- `DBLP_RATE_LIMIT_DELAY`: Delay between DBLP API calls (default: 1s)
- `ARXIV_RATE_LIMIT_DELAY`: Delay between arXiv API calls (default: 0.5s)

## Output Format

The output JSON file contains an array of paper entries, each with:

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
  "matched_authors": [...],  // Same format as parsed_authors
  "source": "arxiv|dblp|null"
}
```

## Dependencies

- arxiv: For querying arXiv database
- nameparser: For consistent author name parsing
- fuzzywuzzy: For fuzzy string matching
- requests: For DBLP API calls
- backoff: For API retry logic
- python-Levenshtein: For improved fuzzy matching performance

## Error Handling

The pipeline includes:
- Exponential backoff for API failures
- Rate limiting to respect API constraints
- Comprehensive error logging
- Graceful handling of missing or malformed data

## Limitations

- Relies on fuzzy matching for titles, which may occasionally produce false positives/negatives
- Dependent on external API availability (arXiv and DBLP)
- Processing speed limited by API rate limits
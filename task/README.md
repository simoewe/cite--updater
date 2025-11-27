# Citation Verification Task

## Overview

This task requires you to verify the accuracy of author names in academic paper citations. You will be given a JSON file containing extracted citations from research papers, and your task is to query external databases (arXiv, DBLP, or Semantic Scholar) to verify if the author names are correct.


## Provided Files

- **`citations.json`**: A JSON file containing ~3,000 citations extracted from academic papers. Each citation includes:
  - `title`: Paper title
  - `authors`: List of author names (may be incomplete or incorrect)
  - `proceedings`: Conference/journal name
  - `year`: Publication year

- **`sample/`**: Directory containing the original XML files from which citations were extracted (for reference)

## Task Description

For each citation in `citations.json`, you need to:

1. **Query external databases** using the paper title to find the correct author information
2. **Compare** the authors listed in the JSON file with the authors found in the databases
3. **Identify discrepancies** such as:
   - Missing authors
   - Incorrect author names (spelling errors, name variations)
   - Wrong author order
   - Completely incorrect author lists

4. **Generate a report** documenting:
   - Which citations were verified
   - Which databases were used for each citation
   - Any discrepancies found
   - Corrected author information

## API Resources

You can use any combination of the following APIs to verify citations:

### 1. arXiv API

**Documentation**: https://arxiv.org/help/api/user-manual

**Python Library**: `arxiv` (install via `pip install arxiv`)

**Example Usage**:
```python
import arxiv

# Search for a paper by title
client = arxiv.Client()
search = arxiv.Search(
    query="Adversarial examples are a natural consequence of test error in noise",
    max_results=5,
    sort_by=arxiv.SortCriterion.Relevance
)

for result in client.results(search):
    print(f"Title: {result.title}")
    print(f"Authors: {[author.name for author in result.authors]}")
    print(f"Year: {result.published.year}")
    break
```

**Rate Limits**: Be respectful - add delays between requests (e.g., 1-3 seconds)

### 2. DBLP API

**Documentation**: https://dblp.org/faq/How+to+use+the+dblp+search+API.html

**Base URL**: `https://dblp.org/search/publ/api`

**Example Usage**:
```python
import requests
from urllib.parse import quote

# Search DBLP by title
title = "Statistical approaches to computer-assisted translation"
query = quote(title)
url = f"https://dblp.org/search/publ/api?q={query}&format=json&h=5"

response = requests.get(url)
data = response.json()

# Parse results
if 'result' in data and 'hits' in data['result']:
    for hit in data['result']['hits']['hit']:
        info = hit['info']
        print(f"Title: {info.get('title', 'N/A')}")
        print(f"Authors: {info.get('authors', {}).get('author', [])}")
        print(f"Year: {info.get('year', 'N/A')}")
```

**Rate Limits**: Add delays (1-2 seconds) between requests

### 3. Semantic Scholar API

**Documentation**: https://api.semanticscholar.org/api-docs/

**Base URL**: `https://api.semanticscholar.org/graph/v1`

**Example Usage**:
```python
import requests

# Search by title
title = "Adversarial examples are a natural consequence of test error in noise"
url = "https://api.semanticscholar.org/graph/v1/paper/search"
params = {
    "query": title,
    "limit": 5,
    "fields": "title,authors,year"
}

response = requests.get(url, params=params)
data = response.json()

# Parse results
if 'data' in data:
    for paper in data['data']:
        print(f"Title: {paper.get('title', 'N/A')}")
        authors = [author.get('name', '') for author in paper.get('authors', [])]
        print(f"Authors: {authors}")
        print(f"Year: {paper.get('year', 'N/A')}")
```

**Rate Limits**: 
- Free tier: 100 requests per 5 minutes
- Consider using API key for higher limits (optional)

## Expected Output Format

Create a JSON file `verification_results.json` with the following structure:

```json
[
  {
    "original_citation": {
      "title": "Paper title",
      "authors": ["Author 1", "Author 2"],
      "proceedings": "Conference name",
      "year": 2019
    },
    "verified_authors": {
      "source": "arxiv|dblp|semantic_scholar",
      "authors": ["Verified Author 1", "Verified Author 2"],
      "match_score": 0.95,
      "discrepancies": [
        {
          "type": "missing_author|incorrect_name|wrong_order",
          "original": "Author Name",
          "corrected": "Correct Author Name",
          "explanation": "Brief explanation"
        }
      ]
    },
    "verification_status": "verified|discrepancy_found|not_found"
  }
]
```

## Implementation Guidelines

### 1. Code Structure

Organize your code into modules:
- `parser.py`: Parse the citations.json file
- `api_clients.py`: Functions to query arXiv, DBLP, and Semantic Scholar
- `verifier.py`: Main logic to compare authors and identify discrepancies
- `main.py`: Entry point that orchestrates the verification process

### 2. Error Handling

- Handle network errors gracefully
- Implement retry logic for failed API requests
- Handle cases where papers are not found in any database
- Handle rate limiting appropriately

### 3. Author Name Matching

Consider these challenges when comparing author names:
- **Name variations**: "John Smith" vs "J. Smith" vs "John A. Smith"
- **Order differences**: Author order may vary
- **Missing authors**: Some authors might be missing from the original citation
- **Spelling errors**: Typos in names

You may want to use fuzzy string matching (e.g., `fuzzywuzzy` library) to handle name variations.

### 4. Progress Tracking

- Save progress periodically (e.g., after every 100 citations)
- Implement resume functionality in case the script is interrupted
- Log which citations were processed and which failed

### 5. Performance Considerations

- Use appropriate delays between API calls to respect rate limits
- Consider parallel processing (but be mindful of rate limits)
- Cache results to avoid re-querying the same papers

## Example Workflow

1. **Load citations**: Read `citations.json` and parse all citations
2. **For each citation**:
   - Extract the paper title
   - Query one or more APIs (start with arXiv, fallback to DBLP or Semantic Scholar)
   - Find the best matching paper (use title similarity)
   - Extract author information from the matched paper
   - Compare authors with the original citation
   - Document any discrepancies
3. **Generate report**: Create `verification_results.json` with all findings
4. **Create summary**: Generate a summary report (markdown or PDF) with statistics




## Tips

- **Start small**: Test your code on a subset of citations first (e.g., first 10-20)
- **Use multiple sources**: If one API doesn't have the paper, try another
- **Handle edge cases**: Papers with very common titles, papers not in any database, etc.
- **Document your approach**: Explain why you chose certain matching strategies
- **Be respectful**: Add appropriate delays between API calls

## Getting Started

1. Install required dependencies:
   ```bash
   cd task
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Read the `citations.json` file to understand the data structure

3. Start with a simple script that queries arXiv for one citation

4. Gradually build up to handle all citations with proper error handling

## Example Starter Script

The `example_starter.py` script provides a working example implementation that:

- **Loads citations** from `citations.json`
- **Queries local arXiv metadata** from `arxiv_papers_metadata.json` for title matching
- **Compares authors** using advanced fuzzy matching that handles:
  - Initial expansions (F → Filipe)
  - Middle name variations (ignored as non-discrepancies)
  - Nickname variations (Nicolas → Nic)
  - Order changes
  - Missing/extra authors
- **Identifies discrepancies** including order changes, name variants, and missing authors
- **Saves results** to `example_results.json`

### Running the Example Script

```bash
cd task
source venv/bin/activate
python example_starter.py
```

The script processes the first 100 citations by default (configurable in the `main()` function). Results are saved to `example_results.json` with detailed comparison information for each citation.

### Key Features

- **Title Matching**: Uses fuzzywuzzy for fuzzy string matching (minimum 80% similarity threshold)
- **Author Comparison**: Sophisticated matching algorithm that:
  - Normalizes names (removes punctuation, converts to lowercase)
  - Handles name variations and initials
  - Ignores middle name differences as non-discrepancies
  - Uses greedy matching to find optimal author pairings
- **Discrepancy Classification**:
  - Order changes (when authors are in different positions)
  - Name changes (when same person has different name variant)
  - Missing authors (in original but not in verified)
  - Extra authors (in verified but not in original)

### Output Format

Each result contains:
- `original`: The original citation from `citations.json`
- `verification`: Dictionary with:
  - `status`: "verified", "discrepancy_found", or "not_found"
  - `title`: Matched paper title
  - `source`: "arxiv" or "placeholder_for_dblp_semantic_scholar"
  - `match_score`: Title similarity score (0-100)
  - `verified_authors`: List of verified author names
  - `comparison`: Detailed comparison results with discrepancies

## Main Pipeline (Main_Pipeline.py)

Die `Main_Pipeline.py` ist die zentrale Ausführungsdatei für die Citation Verification Pipeline. Sie verwendet die moderne API-Caller-Architektur und durchsucht automatisch DBLP, arXiv und Semantic Scholar.

### Ausführung

1. **Virtual Environment aktivieren** (falls noch nicht aktiv):
   ```bash
   cd "/Users/mowe/Library/CloudStorage/GoogleDrive-derdersimon@gmx.net/Andere Computer/MOEWEsTower/Uni WI/3. Semester/Studie/Pranav Studie/cite--updater/cite--updater-1"
   source .venv/bin/activate
   ```

2. **In das task-Verzeichnis wechseln**:
   ```bash
   cd task
   ```

3. **Pipeline ausführen**:
   ```bash
   python Main_Pipeline.py
   ```

### Konfiguration

Die Pipeline kann über Konfigurationsvariablen am Anfang der `Main_Pipeline.py` angepasst werden:

- **`CITATIONS_FILE`**: Pfad zur Eingabe-JSON-Datei (Standard: `'citations.json'`)
- **`OUTPUT_FILE`**: Pfad zur Ausgabe-JSON-Datei (Standard: `'verification_results.json'`)
- **`CITATION_LIMIT`**: Anzahl der zu verarbeitenden Zitate (Standard: `None` = alle)
  - Beispiel: `CITATION_LIMIT = 100` verarbeitet nur die ersten 100 Zitate
- **`SIMILARITY_THRESHOLD`**: Mindest-Ähnlichkeitsschwelle für Titel-Matching (0-100, Standard: 80)
- **`MAX_RESULTS_PER_SOURCE`**: Maximale Anzahl Ergebnisse pro Datenquelle (Standard: 10)

### Ausgabe

Die Pipeline erstellt:
- **`verification_results.json`**: Vollständige Verifikationsergebnisse mit allen Details
- **`citation_verification.log`**: Log-Datei mit detaillierten Informationen über den Verarbeitungsprozess

### Beispiel-Ausgabe

Die Pipeline gibt während der Ausführung Fortschrittsinformationen aus:
- Verarbeitete Zitate (z.B. `[1/100] Processing citation 1`)
- Gefundene Matches mit Ähnlichkeitsscore
- Zusammenfassung am Ende mit Statistiken:
  - Anzahl verifizierter Zitate
  - Anzahl gefundener Diskrepanzen
  - Anzahl nicht gefundener Zitate
  - Durchschnittlicher Match-Score
  - Verwendete Datenquellen

### Tipps

- **Testlauf**: Setze `CITATION_LIMIT = 10` für einen schnellen Test
- **Interrupt**: Die Pipeline kann mit `Ctrl+C` unterbrochen werden
- **Logs**: Prüfe `citation_verification.log` für detaillierte Informationen

## Testing

Die Pipeline verfügt über eine umfassende Test-Suite in `test_Main_Pipeline.py`, die alle Funktionen und Edge Cases abdeckt.

### Tests ausführen

```bash
# Alle Tests ausführen
python test_Main_Pipeline.py

# Mit pytest (falls installiert)
pytest test_Main_Pipeline.py -v
```

### Test-Abdeckung

Die Test-Suite umfasst:

- **`verify_citation()` Tests**:
  - Erfolgreiche Verifikation mit übereinstimmenden Autoren
  - Diskrepanzen gefunden
  - Kein Match gefunden
  - Leerer Titel
  - Fehlerbehandlung
  - Verschiedene Datenquellen (arXiv, DBLP, Semantic Scholar)
  - Verschiedene Ähnlichkeitsschwellen

- **`process_citations()` Tests**:
  - Basis-Verarbeitung mehrerer Zitate
  - Verarbeitung mit Limit
  - Leere Liste
  - Einzelnes Zitat
  - Fortschritts-Logging

- **`generate_summary()` Tests**:
  - Leere Ergebnisse
  - Gemischte Status
  - Parsing-Fehler
  - Alle verifiziert
  - Fehlende Felder

- **`main()` Tests**:
  - Erfolgreiche Ausführung
  - Fehlende Citations-Datei
  - Ungültige Ähnlichkeitsschwelle

- **Edge Cases**:
  - Keine Autoren
  - Sehr langer Titel
  - Sonderzeichen im Titel
  - Limit überschreitet Gesamtanzahl

- **Integration Tests**:
  - Vollständige Pipeline-Szenarien

Die Tests verwenden Mocking, um echte API-Calls zu vermeiden und können daher beliebig oft wiederholt werden, ohne externe Abhängigkeiten oder Rate Limits zu beeinträchtigen.



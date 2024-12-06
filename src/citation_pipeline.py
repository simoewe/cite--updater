"""
Citation Author Matcher Pipeline

This script processes academic papers from GROBID-parsed XML files and matches them against
arXiv and DBLP databases to verify and normalize author information.

Usage:
    python citation_pipeline.py [options]
    
Options:
    --dry-run              Run with a random sample of papers
    --sample N            Number of papers to sample in dry run (default: 5)
    --input FILE          Input XML file path
    --output FILE         Output JSON file path
    --threshold N         Title match threshold (default: 80)
    --dblp-delay N        Delay between DBLP API calls in seconds (default: 1)
    --arxiv-delay N       Delay between arXiv API calls in seconds (default: 0.5)
"""

import xml.etree.ElementTree as ET
import arxiv
import json
from nameparser import HumanName
from fuzzywuzzy import fuzz
import requests
from urllib.parse import urlencode
import time
from typing import Dict, List, Optional, Any
import backoff
import re
import random
import argparse
import os

def normalize_author_name(name: str) -> Dict[str, str]:
    """
    Normalize author names into a consistent format using nameparser.
    Removes both 4-digit suffixes and DBLP-style numeric suffixes.
    
    Args:
        name (str): Full author name string
        
    Returns:
        Dict[str, str]: Normalized name components
    """
    # Remove 4-digit suffixes and DBLP-style numeric suffixes
    cleaned_name = re.sub(r'\s+\d{4}(?:\s|$)', '', name)
    cleaned_name = re.sub(r'\s+\d{4,}$', '', cleaned_name)  # Remove trailing numbers like 0001
    cleaned_name = re.sub(r'\s+\d{4,}\s+', ' ', cleaned_name)  # Remove internal numbers
    
    # Parse the cleaned name
    parsed = HumanName(cleaned_name)
    
    return {
        'first_name': parsed.first or '',
        'middle_name': parsed.middle or '',
        'last_name': parsed.last or '',  # Last name should now be clean of numeric suffixes
        'suffix': parsed.suffix or '',
        'title': parsed.title or '',
        'original': name  # Keep the original name for reference
    }

def parse_xml(file_path: str) -> Dict[str, List[Dict[str, str]]]:
    """Parse XML and return normalized author information."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        publications = {}
        
        for biblStruct in root.findall('.//tei:biblStruct', ns):
            title = biblStruct.find('.//tei:title[@type="main"]', ns)
            title_text = title.text if title is not None else "No title available"
            
            authors = []
            for author in biblStruct.findall('.//tei:author/tei:persName', ns):
                forename = author.find('tei:forename', ns)
                surname = author.find('tei:surname', ns)
                if forename is not None and surname is not None:
                    full_name = f"{forename.text} {surname.text}"
                    authors.append(normalize_author_name(full_name))
            
            publications[title_text] = authors
        return publications
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        return {}

def query_arxiv(title: str, match_threshold: int, delay: float) -> Optional[List[Dict[str, str]]]:
    """Query arXiv and return normalized author information."""
    try:
        print(f"\nQuerying arXiv for: {title[:100]}...")
        client = arxiv.Client()
        search = arxiv.Search(
            query=title,
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for result in client.results(search):
            title_similarity = fuzz.ratio(result.title.lower(), title.lower())
            print(f"arXiv match score: {title_similarity}")
            if title_similarity > match_threshold:
                return [normalize_author_name(author.name) for author in result.authors]
        return None
    except Exception as e:
        print(f"Error querying arXiv: {e}")
        return None
    finally:
        time.sleep(delay)

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError),
    max_tries=5,
    max_time=300
)
def query_dblp(title: str, match_threshold: int, delay: float) -> Optional[List[Dict[str, str]]]:
    """Query DBLP with exponential backoff retry."""
    try:
        print(f"Querying DBLP for: {title[:100]}...")
        options = {
            'q': title,
            'format': 'json',
            'h': 1
        }
        response = requests.get(f'https://dblp.org/search/publ/api?{urlencode(options)}')
        response.raise_for_status()
        
        data = response.json()
        hits = data.get('result', {}).get('hits', {}).get('hit', [])
        
        if hits:
            info = hits[0].get('info', {})
            db_title = info.get('title', '')
            
            title_similarity = fuzz.ratio(db_title.lower(), title.lower())
            print(f"DBLP match score: {title_similarity}")
            
            if title_similarity > match_threshold:
                authors = info.get('authors', {}).get('author', [])
                # Clean the author names before normalizing
                return [normalize_author_name(re.sub(r'\s+\d{4,}$', '', author['text'])) 
                       for author in authors 
                       if isinstance(author, dict) and 'text' in author]
        return None
    except Exception as e:
        print(f"Error querying DBLP: {e}")
        return None
    finally:
        time.sleep(delay)

def check_name_match(parsed_name: Dict[str, str], matched_name: Dict[str, str]) -> Optional[str]:
    """
    Check if two author names match according to specified rules.
    Returns mismatch description if names don't match, None otherwise.
    """
    # Compare first names
    if parsed_name['first_name'] and matched_name['first_name']:
        if parsed_name['first_name'] != matched_name['first_name']:
            return f"First name mismatch: {parsed_name['first_name']} vs {matched_name['first_name']}"
            
    # Compare last names
    if parsed_name['last_name'] != matched_name['last_name']:
        # Special case: handle numeric suffixes in last names (e.g., "0001")
        if not matched_name['last_name'].isdigit() and parsed_name['last_name'] != matched_name['last_name']:
            return f"Last name mismatch: {parsed_name['last_name']} vs {matched_name['last_name']}"
    
    # Compare middle initials if both exist
    if parsed_name['middle_name'] and matched_name['middle_name']:
        if parsed_name['middle_name'][0] != matched_name['middle_name'][0]:
            return f"Middle initial mismatch: {parsed_name['middle_name']} vs {matched_name['middle_name']}"
            
    return None

def process_publications(
    input_file: str,
    output_file: str,
    match_threshold: int,
    dblp_delay: float,
    arxiv_delay: float,
    dry_run: bool = False,
    sample_size: int = 5
) -> List[Dict]:
    """
    Main processing function that coordinates the entire pipeline.
    
    Args:
        input_file: Path to input XML file
        output_file: Path to output JSON file
        match_threshold: Minimum score for title matching
        dblp_delay: Delay between DBLP API calls
        arxiv_delay: Delay between arXiv API calls
        dry_run: If True, process only a random sample
        sample_size: Number of papers to process in dry run
    """
    print(f"\nProcessing publications from {input_file}")
    publications = parse_xml(input_file)
    if not publications:
        print("No publications found in XML file")
        return []
    
    if dry_run:
        titles = list(publications.keys())
        sample_size = min(sample_size, len(titles))
        sampled_titles = random.sample(titles, sample_size)
        publications = {title: publications[title] for title in sampled_titles}
        print(f"\nDRY RUN: Processing {sample_size} random publications")
        
    print(f"Found {len(publications)} publications to process")
    results = []
    matched_count = 0
    
    for i, (title, parsed_authors) in enumerate(publications.items(), 1):
        print(f"\nProcessing publication {i}/{len(publications)}")
        print(f"Title: {title}")
        
        # Try Arxiv first
        if arxiv_authors := query_arxiv(title, match_threshold, arxiv_delay):
            # Check for name mismatches
            mismatches = []
            for idx, (parsed, matched) in enumerate(zip(parsed_authors, arxiv_authors)):
                if mismatch := check_name_match(parsed, matched):
                    mismatches.append(f"Author {idx + 1}: {mismatch}")
                    
            results.append({
                'title': title,
                'parsed_authors': parsed_authors,
                'matched_authors': arxiv_authors,
                'source': 'arxiv',
                'mismatches': mismatches if mismatches else []
            })
            matched_count += 1
        # Fall back to DBLP if no Arxiv match
        elif dblp_authors := query_dblp(title, match_threshold, dblp_delay):
            # Check for name mismatches
            mismatches = []
            for idx, (parsed, matched) in enumerate(zip(parsed_authors, dblp_authors)):
                if mismatch := check_name_match(parsed, matched):
                    mismatches.append(f"Author {idx + 1}: {mismatch}")
                    
            results.append({
                'title': title,
                'parsed_authors': parsed_authors,
                'matched_authors': dblp_authors,
                'source': 'dblp',
                'mismatches': mismatches if mismatches else []
            })
            matched_count += 1
        else:
            print("No matches found in either source")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults have been written to {output_file}")
        print(f"Found matches for {matched_count} out of {len(publications)} publications")
    except Exception as e:
        print(f"Error writing results to file: {e}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Match paper authors against arXiv and DBLP.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output options
    parser.add_argument('--input', type=str, 
                       default='pdfs/2024.lrec-main.320.grobid.tei.xml',
                       help='Input XML file path')
    parser.add_argument('--output', type=str,
                       help='Output JSON file path')
    
    # Matching options
    parser.add_argument('--threshold', type=int, default=80,
                       help='Title match threshold (0-100)')
    
    # Rate limiting options
    parser.add_argument('--dblp-delay', type=float, default=1.0,
                       help='Delay between DBLP API calls in seconds')
    parser.add_argument('--arxiv-delay', type=float, default=0.5,
                       help='Delay between arXiv API calls in seconds')
    
    # Dry run options
    parser.add_argument('--dry-run', action='store_true',
                       help='Run with a random sample of papers')
    parser.add_argument('--sample', type=int, default=5,
                       help='Number of papers to sample in dry run')
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output:
        base_name = 'dry_run_matches.json' if args.dry_run else 'author_matches.json'
        args.output = base_name
    
    # Validate arguments
    if not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")
    
    if args.threshold < 0 or args.threshold > 100:
        parser.error("Threshold must be between 0 and 100")
    
    if args.dblp_delay < 0 or args.arxiv_delay < 0:
        parser.error("Delays must be non-negative")
    
    if args.sample < 1:
        parser.error("Sample size must be positive")
    
    # Run the pipeline
    results = process_publications(
        input_file=args.input,
        output_file=args.output,
        match_threshold=args.threshold,
        dblp_delay=args.dblp_delay,
        arxiv_delay=args.arxiv_delay,
        dry_run=args.dry_run,
        sample_size=args.sample
    )
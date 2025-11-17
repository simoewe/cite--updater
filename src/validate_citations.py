"""
Citation Author Validation Script

This script validates citation authors in parsed JSON files against DBLP database.
For each reference in the JSON files, it queries DBLP by title and compares
the authors to detect incorrect citations.

Usage:
    python src/validate_citations.py [options]
    
Options:
    --input-dir DIR       Directory containing parsed JSON files (default: data/parsed_jsons)
    --dblp-xml FILE       Path to DBLP XML file (default: data/dblp.xml)
    --output FILE         Output JSON file path (default: citation_validation_results.json)
    --num-files N         Number of JSON files to process (default: 20)
    --threshold N         Title match threshold for DBLP search (default: 5.0)
"""

import json
import os
import logging
import argparse
from typing import List, Dict, Optional, Any
from pathlib import Path
from collections import Counter
import random

# Setup logging BEFORE importing other modules
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('citation_validation.log'),
        logging.StreamHandler()
    ]
)

# Reduce verbosity of external libraries BEFORE importing them
logging.getLogger('retriv').setLevel(logging.ERROR)
logging.getLogger('dblp_parser').setLevel(logging.ERROR)
# Also try to suppress all logging from retriv submodules
logging.getLogger('retriv.SparseRetriever').setLevel(logging.ERROR)

# Import existing matching functions
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_matches import (
    is_name_match,
    check_author_lists,
    initial_matches,
    is_compound_initial
)
from parser.dblp_parser import DblpParser
from nameparser import HumanName
from rapidfuzz import fuzz
from unidecode import unidecode
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calculate_title_similarity(title1: str, title2: str) -> float:
    """
    Calculate string similarity between two titles using rapidfuzz.
    
    Args:
        title1: First title string
        title2: Second title string
        
    Returns:
        Similarity score between 0 and 100
    """
    # Normalize titles: lowercase and remove extra whitespace
    t1 = ' '.join(title1.lower().split())
    t2 = ' '.join(title2.lower().split())
    
    # Use ratio for overall similarity
    return fuzz.ratio(t1, t2)


def check_author_with_minimum_lists(ref_authors: List[Dict[str, str]], 
                                    dblp_authors: List[Dict[str, str]],
                                    title: str, max_authors: int = 10) -> Dict[str, Any]:
    """
    Check author lists by comparing only the first N authors (default: 10).
    This handles cases where authors don't include the full author list.
    
    Args:
        ref_authors: Normalized authors from reference
        dblp_authors: Normalized authors from DBLP
        title: Paper title for context
        max_authors: Maximum number of authors to compare (default: 10)
        
    Returns:
        Dictionary with:
            - matches: Boolean indicating if authors match
            - mismatches: List of mismatch descriptions
            - error_classifications: List of error types found
    """
    result = {
        'matches': False,
        'mismatches': [],
        'error_classifications': []
    }
    
    # Take first max_authors from both lists (or fewer if lists are shorter)
    ref_subset = ref_authors[:max_authors]
    dblp_subset = dblp_authors[:max_authors]

    if not ref_subset:
        result['mismatches'].append('Empty author list')
        result['error_classifications'].append('empty_list')
        return result

    # Try to match authors - allow for order differences and length differences
    matched_ref_indices = set()
    matched_dblp_indices = set()
    matches = []

    # First pass: exact matches using is_name_match (handles initials, reversed names, etc.)
    # Compare all reference authors against all DBLP authors (within max_authors limit)
    for i, ref_author in enumerate(ref_subset):
        for j, dblp_author in enumerate(dblp_subset):
            if j in matched_dblp_indices:
                continue
            if is_name_match(ref_author, dblp_author):
                matches.append((i, j, ref_author, dblp_author))
                matched_ref_indices.add(i)
                matched_dblp_indices.add(j)
                break
    
    # Check for unmatched authors and classify errors
    unmatched_ref = [ref_subset[i] for i in range(len(ref_subset)) if i not in matched_ref_indices]
    unmatched_dblp = [dblp_subset[j] for j in range(len(dblp_subset)) if j not in matched_dblp_indices]
    
    if not unmatched_ref and not unmatched_dblp:
        result['matches'] = True
        return result
    
    # First, check for systematic parsing errors (names shifted/mixed up)
    # This happens when first names of one author become last names of another
    parsing_error_detected = False

    # Check for names that got split across multiple reference authors
    for j, dblp_author in enumerate(unmatched_dblp):
        dblp_full = f"{dblp_author.get('first_name', '')} {dblp_author.get('middle_name', '')} {dblp_author.get('last_name', '')}".strip()
        dblp_parts = dblp_full.lower().split()

        # Check if this DBLP author matches parts scattered across consecutive reference authors
        for i in range(len(unmatched_ref) - 1):
            ref_author1 = unmatched_ref[i]
            ref_author2 = unmatched_ref[i + 1]

            ref1_full = f"{ref_author1.get('first_name', '')} {ref_author1.get('last_name', '')}".strip().lower()
            ref2_full = f"{ref_author2.get('first_name', '')} {ref_author2.get('last_name', '')}".strip().lower()

            combined = f"{ref1_full} {ref2_full}".replace('  ', ' ').strip()

            # Check if combined reference authors match the DBLP author
            if combined and dblp_full.lower() == combined:
                parsing_error_detected = True
                break

            # Check if parts of DBLP author are split between reference authors
            ref1_parts = ref1_full.split()
            ref2_parts = ref2_full.split()

            # If DBLP has 3+ parts and reference authors together have the same parts
            if len(dblp_parts) >= 3 and len(ref1_parts + ref2_parts) >= len(dblp_parts):
                if all(part in ref1_parts + ref2_parts for part in dblp_parts):
                    parsing_error_detected = True
                    break

        if parsing_error_detected:
            break

    # Also check the original parsing error detection
    if not parsing_error_detected:
        for i, ref_author in enumerate(unmatched_ref):
            ref_last = ref_author.get('last_name', '').lower().strip()
            ref_first = ref_author.get('first_name', '').lower().strip()

            for j, dblp_author in enumerate(unmatched_dblp):
                dblp_last = dblp_author.get('last_name', '').lower().strip()
                dblp_first = dblp_author.get('first_name', '').lower().strip()

                # Check if ref's last_name matches dblp's first_name (or vice versa)
                # This indicates names are mixed up/shifted
                if ref_last and dblp_first and ref_last == dblp_first:
                    parsing_error_detected = True
                    break
                if ref_first and dblp_last and ref_first == dblp_last:
                    parsing_error_detected = True
                    break

            if parsing_error_detected:
                break
    
    # If parsing error detected, mark all unmatched authors as parsing error
    if parsing_error_detected:
        result['error_classifications'].append('parsing_error')
        result['mismatches'].append(
            'Parsing error detected: author names appear to be shifted or mixed up'
        )
        return result
    
    # Classify errors for unmatched reference authors
    for ref_author in unmatched_ref:
        best_match = None
        best_match_idx = None
        best_match_type = None
        
        for j, dblp_author in enumerate(dblp_subset):
            if j in matched_dblp_indices:
                continue
            
            # Check different types of mismatches
            # Extract first_name and last_name from normalized author dictionaries
            # These are created by normalize_author_name() which uses nameparser
            # Location: ref_author['first_name'] and ref_author['last_name'] 
            # (normalized from the original author string)
            ref_last = ref_author.get('last_name', '').lower().strip()
            dblp_last = dblp_author.get('last_name', '').lower().strip()
            
            ref_first = ref_author.get('first_name', '').lower().strip()
            dblp_first = dblp_author.get('first_name', '').lower().strip()
            
            # Check for accent differences (using unidecode)
            ref_last_no_accents = unidecode(ref_last)
            dblp_last_no_accents = unidecode(dblp_last)
            ref_first_no_accents = unidecode(ref_first)
            dblp_first_no_accents = unidecode(dblp_first)
            
            # Last name matches but first name differs
            if ref_last == dblp_last or ref_last_no_accents == dblp_last_no_accents:
                if ref_first != dblp_first:
                    # Check if first names match by initials (including compound initials like "K.-T" matching "Kwang-Ting")
                    # Use the enhanced initial_matches function that handles compound initials
                    initials_match = initial_matches(ref_author['first_name'], dblp_author['first_name'])
                    
                    # Also check simple initial match for backward compatibility
                    ref_first_initial = unidecode(ref_first.lower().replace('.', '').strip())
                    dblp_first_initial = unidecode(dblp_first.lower().replace('.', '').strip())
                    simple_initials_match = (len(ref_first_initial) == 1 and len(dblp_first_initial) >= 1 and
                                           ref_first_initial[0] == dblp_first_initial[0])
                    
                    # If initials match (compound or simple), consider them matched and skip mismatch reporting
                    if initials_match or simple_initials_match:
                        # Mark this DBLP author as matched and break to skip adding as mismatch
                        matched_dblp_indices.add(j)
                        break  # Break out of inner loop, this reference author is matched

                    if ref_first_no_accents == dblp_first_no_accents:
                        best_match = dblp_author
                        best_match_idx = j
                        best_match_type = 'accents_missing'
                    elif simple_initials_match and (unidecode(ref_last) != ref_last or unidecode(dblp_last) != dblp_last):
                        # Last names have accents, first names match by initials
                        best_match = dblp_author
                        best_match_idx = j
                        best_match_type = 'accents_missing'
                    else:
                        best_match = dblp_author
                        best_match_idx = j
                        best_match_type = 'first_name_mismatch'
                # Even if first names appear to match, check for accent differences
                elif ref_first == dblp_first and (unidecode(ref_first) != ref_first or unidecode(dblp_first) != dblp_first):
                    best_match = dblp_author
                    best_match_idx = j
                    best_match_type = 'accents_missing'
                continue
            
            # First name matches but last name differs
            if ref_first == dblp_first or ref_first_no_accents == dblp_first_no_accents:
                if ref_last != dblp_last:
                    if ref_last_no_accents == dblp_last_no_accents:
                        best_match = dblp_author
                        best_match_idx = j
                        best_match_type = 'accents_missing'
                    else:
                        best_match = dblp_author
                        best_match_idx = j
                        best_match_type = 'last_name_mismatch'
                # Even if last names appear to match, check for accent differences
                elif ref_last == dblp_last and (unidecode(ref_last) != ref_last or unidecode(dblp_last) != dblp_last):
                    best_match = dblp_author
                    best_match_idx = j
                    best_match_type = 'accents_missing'
                continue
            # Check if first names match by initials
            ref_first_initial = unidecode(ref_first.lower().replace('.', '').strip())
            dblp_first_initial = unidecode(dblp_first.lower().replace('.', '').strip())
            first_initials_match = (len(ref_first_initial) == 1 and len(dblp_first_initial) >= 1 and
                                  ref_first_initial[0] == dblp_first_initial[0])
            if first_initials_match:
                if ref_last != dblp_last:
                    if ref_last_no_accents == dblp_last_no_accents:
                        best_match = dblp_author
                        best_match_idx = j
                        best_match_type = 'accents_missing'
                    else:
                        best_match = dblp_author
                        best_match_idx = j
                        best_match_type = 'last_name_mismatch'
                # Even if last names appear to match, check for accent differences
                elif ref_last == dblp_last and (unidecode(ref_last) != ref_last or unidecode(dblp_last) != dblp_last):
                    best_match = dblp_author
                    best_match_idx = j
                    best_match_type = 'accents_missing'
                continue
            
            # Check if names match without accents (but have accents in original)
            if (ref_last_no_accents == dblp_last_no_accents and ref_first_no_accents == dblp_first_no_accents and
                (ref_last != dblp_last or ref_first != dblp_first)):
                best_match = dblp_author
                best_match_idx = j
                best_match_type = 'accents_missing'
                continue
        
        if best_match:
            matched_dblp_indices.add(best_match_idx)
            result['error_classifications'].append(best_match_type)
            result['mismatches'].append(
                f"{best_match_type}: {ref_author['first_name']} {ref_author['last_name']} vs "
                f"{best_match['first_name']} {best_match['last_name']}"
            )
        else:
            result['error_classifications'].append('author_not_found')
            result['mismatches'].append(
                f"Author not found in DBLP: {ref_author['first_name']} {ref_author['last_name']}"
            )
    
    # Check for order issues - if all reference authors match but in different order
    if len(matches) == len(ref_subset):
        # Check if order is preserved (authors should match in same relative positions)
        # matches contains tuples: (ref_index, dblp_index, ref_author, dblp_author)
        # Sort matches by reference index and check if DBLP indices are also in order
        sorted_matches = sorted(matches, key=lambda x: x[0])
        order_mismatch = False
        for i, (ref_idx, dblp_idx, _, _) in enumerate(sorted_matches):
            # Check if the relative order is preserved
            if i > 0 and dblp_idx < sorted_matches[i-1][1]:
                order_mismatch = True
                break

        if order_mismatch:
            result['error_classifications'].append('author_order_wrong')
            result['mismatches'].append('Authors match but order differs')
            # Don't mark as complete match if order is wrong
            result['matches'] = False
        else:
            # All authors match and order is correct
            result['matches'] = True
    
    return result


def normalize_author_name(name: str) -> Dict[str, str]:
    """
    Normalize author names into a consistent format using nameparser.
    Removes both 4-digit suffixes and DBLP-style numeric suffixes.
    
    This function is copied from citation_pipeline.py to avoid import issues.
    
    Args:
        name (str): Full author name string
        
    Returns:
        Dict[str, str]: Normalized name components with keys:
            - first_name: First name (accessed as author['first_name'])
            - middle_name: Middle name/initial
            - last_name: Last name (accessed as author['last_name'])
            - suffix: Suffix (e.g., Jr., III)
            - title: Title (e.g., Dr., Prof.)
            - original: Original name string
    
    Note: First and last names are extracted and compared in check_author_with_minimum_lists()
          at lines 146-150 (ref_first, ref_last, dblp_first, dblp_last)
    """
    # Remove 4-digit suffixes and DBLP-style numeric suffixes
    cleaned_name = re.sub(r'\s+\d{4}(?:\s|$)', '', name)
    cleaned_name = re.sub(r'\s+\d{4,}$', '', cleaned_name)  # Remove trailing numbers like 0001
    cleaned_name = re.sub(r'\s+\d{4,}\s+', ' ', cleaned_name)  # Remove internal numbers
    
    # Parse the cleaned name using nameparser
    parsed = HumanName(cleaned_name)
    
    return {
        'first_name': parsed.first or '',
        'middle_name': parsed.middle or '',
        'last_name': parsed.last or '',
        'suffix': parsed.suffix or '',
        'title': parsed.title or '',
        'original': name  # Keep original for reference
    }


def find_json_files(input_dir: str, num_files: int = 20) -> List[str]:
    """
    Find JSON files in the input directory and return a random sample.
    
    Args:
        input_dir: Root directory containing JSON files
        num_files: Number of files to return
        
    Returns:
        List of file paths
    """
    json_files = []
    
    # Walk through directory structure
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    # Randomly sample if we have more files than requested
    if len(json_files) > num_files:
        json_files = random.sample(json_files, num_files)
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    return json_files


def validate_reference(reference: Dict[str, Any], dblp_parser: DblpParser,
                      threshold: float = 5.0, title_similarity_threshold: float = 95.0) -> Dict[str, Any]:
    """
    Validate a single reference by querying DBLP and comparing authors.
    
    Args:
        reference: Reference dictionary with 'title' and 'authors' keys
        dblp_parser: Initialized DblpParser instance
        threshold: BM25 score threshold for DBLP title matching
        title_similarity_threshold: Minimum string similarity (0-100) between titles to consider match
        
    Returns:
        Dictionary with validation results:
            - reference: Original reference data
            - dblp_match: DBLP publication data if found
            - authors_match: Boolean indicating if authors match
            - mismatches: List of mismatch descriptions
            - error_classifications: List of error types (first_name_mismatch, last_name_mismatch, etc.)
            - title_similarity: Similarity score between titles
            - validation_status: 'matched', 'no_dblp_match', 'title_mismatch', 'author_mismatch', or 'error'
    """
    result = {
        'reference': reference,
        'dblp_match': None,
        'authors_match': False,
        'mismatches': [],
        'error_classifications': [],
        'title_similarity': 0.0,
        'validation_status': 'unknown'
    }
    
    title = reference.get('title', '')
    if not title:
        result['validation_status'] = 'error'
        result['mismatches'].append('No title in reference')
        return result

    # Skip validation for non-academic references like Wikipedia
    title_lower = title.lower()
    if 'wikipedia' in title_lower or title_lower.strip() == 'wikipedia':
        result['validation_status'] = 'skipped'
        result['mismatches'].append('Skipped validation for non-academic reference (Wikipedia)')
        return result
    
    # Query DBLP for the paper
    try:
        dblp_result = dblp_parser.search_by_title(title, threshold=threshold)
        
        if not dblp_result:
            result['validation_status'] = 'no_dblp_match'
            result['mismatches'].append(f'No DBLP match found for title: {title[:100]}')
            return result
        
        dblp_title = dblp_result.get('title', '')
        
        # Calculate title similarity
        title_similarity = calculate_title_similarity(title, dblp_title)
        result['title_similarity'] = title_similarity
        
        # Only consider if title similarity is >= threshold
        if title_similarity < title_similarity_threshold:
            result['validation_status'] = 'title_mismatch'
            result['mismatches'].append(
                f'Title similarity too low: {title_similarity:.1f}% '
                f'(threshold: {title_similarity_threshold}%). '
                f'Reference: "{title[:100]}", DBLP: "{dblp_title[:100]}"'
            )
            return result
        
        result['dblp_match'] = {
            'title': dblp_title,
            'authors': dblp_result.get('authors', []),
            'year': dblp_result.get('year', ''),
            'venue': dblp_result.get('venue', '')
        }
        
        # Normalize authors from reference
        ref_authors_raw = reference.get('authors', [])
        # Handle case where authors is a string instead of a list
        if isinstance(ref_authors_raw, str):
            ref_authors_raw = [ref_authors_raw]
        ref_authors_normalized = [normalize_author_name(author) for author in ref_authors_raw]
        
        # Normalize authors from DBLP
        dblp_authors_raw = dblp_result.get('authors', [])
        dblp_authors_normalized = [normalize_author_name(author) for author in dblp_authors_raw]
        
        # Check if authors match using first 10 authors comparison
        author_check_result = check_author_with_minimum_lists(
            ref_authors_normalized,
            dblp_authors_normalized,
            title,
            max_authors=10
        )
        
        result['authors_match'] = author_check_result['matches']
        result['mismatches'] = author_check_result['mismatches']
        result['error_classifications'] = author_check_result['error_classifications']
        
        if author_check_result['matches']:
            result['validation_status'] = 'matched'
        else:
            result['validation_status'] = 'author_mismatch'
            
    except Exception as e:
        logger.error(f"Error validating reference '{title[:50]}...': {e}")
        result['validation_status'] = 'error'
        result['mismatches'].append(f'Error during validation: {str(e)}')
    
    return result


def process_json_file(json_path: str, dblp_parser: DblpParser,
                     threshold: float = 5.0, title_similarity_threshold: float = 95.0) -> Dict[str, Any]:
    """
    Process a single JSON file and validate all its references.
    
    Args:
        json_path: Path to JSON file
        dblp_parser: Initialized DblpParser instance
        threshold: BM25 score threshold for DBLP title matching
        
    Returns:
        Dictionary with file processing results:
            - file_path: Path to processed file
            - references_count: Total number of references
            - validated_count: Number of references successfully validated
            - matched_count: Number of references with matching authors
            - mismatch_count: Number of references with author mismatches
            - no_match_count: Number of references not found in DBLP
            - error_count: Number of references with errors
            - results: List of validation results for each reference
    """
    logger.info(f"Processing file: {json_path}")
    
    file_result = {
        'file_path': json_path,
        'references_count': 0,
        'validated_count': 0,
        'matched_count': 0,
        'mismatch_count': 0,
        'no_match_count': 0,
        'error_count': 0,
        'skipped_count': 0,
        'results': []
    }
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        references = data.get('references', [])
        file_result['references_count'] = len(references)
        
        logger.info(f"Found {len(references)} references in {json_path}")
        
        # Validate each reference
        for ref in references:
            validation_result = validate_reference(
                ref, dblp_parser, threshold, title_similarity_threshold
            )
            file_result['results'].append(validation_result)
            
            # Update counters
            status = validation_result['validation_status']
            if status == 'matched':
                file_result['matched_count'] += 1
                file_result['validated_count'] += 1
            elif status == 'author_mismatch':
                file_result['mismatch_count'] += 1
                file_result['validated_count'] += 1
            elif status == 'title_mismatch':
                file_result['no_match_count'] += 1
            elif status == 'no_dblp_match':
                file_result['no_match_count'] += 1
            elif status == 'error':
                file_result['error_count'] += 1
            elif status == 'skipped':
                file_result['skipped_count'] += 1
        
        logger.info(f"Completed {json_path}: {file_result['matched_count']} matched, "
                   f"{file_result['mismatch_count']} mismatches, "
                   f"{file_result['no_match_count']} no DBLP match")
        
    except Exception as e:
        logger.error(f"Error processing file {json_path}: {e}")
        file_result['error_count'] = 1
    
    return file_result


def main():
    """Main function to run citation validation."""
    parser = argparse.ArgumentParser(
        description='Validate citation authors against DBLP database.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input-dir', type=str, default='data/parsed_jsons',
                       help='Directory containing parsed JSON files')
    parser.add_argument('--dblp-xml', type=str, default='data/dblp.xml',
                       help='Path to DBLP XML file')
    parser.add_argument('--output', type=str, default='citation_validation_results.json',
                       help='Output JSON file path')
    parser.add_argument('--num-files', type=int, default=20,
                       help='Number of JSON files to process')
    parser.add_argument('--threshold', type=float, default=5.0,
                       help='BM25 score threshold for DBLP title matching')
    parser.add_argument('--title-similarity-threshold', type=float, default=95.0,
                       help='Minimum string similarity (0-100) between titles to consider match')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        parser.error(f"Input directory not found: {args.input_dir}")
    
    if not os.path.exists(args.dblp_xml):
        parser.error(f"DBLP XML file not found: {args.dblp_xml}")
    
    # Initialize DBLP parser
    logger.info(f"Initializing DBLP parser with XML file: {args.dblp_xml}")
    try:
        dblp_parser = DblpParser(
            xml_path=args.dblp_xml,
            cache_dir="dblp_cache",
            index_name="dblp_index"
        )
        logger.info("DBLP parser initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize DBLP parser: {e}")
        return
    
    # Find JSON files to process
    json_files = find_json_files(args.input_dir, args.num_files)
    
    if not json_files:
        logger.warning("No JSON files found to process")
        return
    
    # Process each JSON file
    all_results = []
    total_stats = {
        'files_processed': 0,
        'total_references': 0,
        'total_matched': 0,
        'total_mismatches': 0,
        'total_no_match': 0,
        'total_errors': 0,
        'total_skipped': 0
    }
    
    for json_file in json_files:
        file_result = process_json_file(
            json_file, dblp_parser, args.threshold, args.title_similarity_threshold
        )
        all_results.append(file_result)
        
        # Update total statistics
        total_stats['files_processed'] += 1
        total_stats['total_references'] += file_result['references_count']
        total_stats['total_matched'] += file_result['matched_count']
        total_stats['total_mismatches'] += file_result['mismatch_count']
        total_stats['total_no_match'] += file_result['no_match_count']
        total_stats['total_errors'] += file_result['error_count']
        total_stats['total_skipped'] += file_result['skipped_count']
    
    # Extract all validation results for analysis
    all_validation_results = []
    for file_data in all_results:
        all_validation_results.extend(file_data.get('results', []))
    
    # Separate mismatches and matches
    mismatches = []
    matches = []
    
    for result in all_validation_results:
        if result['validation_status'] == 'author_mismatch':
            mismatches.append(result)
        elif result['validation_status'] == 'matched':
            matches.append(result)

    # Sort mismatches to put parsing errors at the bottom
    def has_parsing_error(result):
        return 'parsing_error' in result.get('error_classifications', [])

    mismatches.sort(key=has_parsing_error)
    
    # Perform analysis on results
    error_counts = Counter()
    title_similarities = []
    
    for result in all_validation_results:
        # Count error classifications
        for error_type in result.get('error_classifications', []):
            error_counts[error_type] += 1
        
        # Collect title similarities
        similarity = result.get('title_similarity', 0.0)
        if similarity > 0:
            title_similarities.append(similarity)
    
    # Calculate title similarity statistics
    title_sim_stats = {}
    if title_similarities:
        title_similarities.sort()
        n = len(title_similarities)
        title_sim_stats = {
            'count': n,
            'min': min(title_similarities),
            'max': max(title_similarities),
            'mean': sum(title_similarities) / n,
            'median': title_similarities[n // 2]
        }
    
    # Create final output structure with integrated analysis
    # Put mismatches on top, matches at bottom
    output_data = {
        'summary': total_stats,
        'analysis': {
            'error_classifications': dict(error_counts),
            'title_similarity_stats': title_sim_stats,
            'mismatch_count': len(mismatches),
            'match_count': len(matches)
        },
        'mismatches': mismatches,
        'matches': matches,
        'files': all_results
    }
    
    # Write results to file (single JSON file with all data and analysis)
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results written to: {args.output}")
    except Exception as e:
        logger.error(f"Error writing results to file: {e}")
        return
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Files processed: {total_stats['files_processed']}")
    logger.info(f"Total references: {total_stats['total_references']}")
    logger.info(f"Matched (correct citations): {total_stats['total_matched']}")
    logger.info(f"Mismatches (incorrect citations): {total_stats['total_mismatches']}")
    logger.info(f"No DBLP match / Title mismatch: {total_stats['total_no_match']}")
    logger.info(f"Skipped (non-academic): {total_stats['total_skipped']}")
    logger.info(f"Errors: {total_stats['total_errors']}")
    logger.info("="*60)


if __name__ == '__main__':
    main()


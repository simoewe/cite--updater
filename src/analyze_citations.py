#!/usr/bin/env python3
"""
Comprehensive citation analysis tool.

This script consolidates multiple analysis functions:
1. Statistical analysis of validation results
2. Parsing error detection
3. Result reorganization

Usage:
    # Run all analyses
    python src/analyze_citations.py --input FILE --mode all
    
    # Run only statistical analysis
    python src/analyze_citations.py --input FILE --mode stats
    
    # Run only parsing error detection
    python src/analyze_citations.py --input FILE --mode parsing
    
    # Reorganize results into categories
    python src/analyze_citations.py --input FILE --mode reorganize
"""

import json
import os
import re
import logging
import argparse
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
from nameparser import HumanName
from unidecode import unidecode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_author_name(name: str) -> Dict[str, str]:
    """
    Normalize author names into a consistent format using nameparser.
    
    Args:
        name: Full author name string
        
    Returns:
        Dictionary with normalized name components
    """
    # Remove 4-digit suffixes and DBLP-style numeric suffixes
    cleaned_name = re.sub(r'\s+\d{4}(?:\s|$)', '', name)
    cleaned_name = re.sub(r'\s+\d{4,}$', '', cleaned_name)
    cleaned_name = re.sub(r'\s+\d{4,}\s+', ' ', cleaned_name)
    
    # Parse the cleaned name
    parsed = HumanName(cleaned_name)
    
    return {
        'first_name': parsed.first or '',
        'middle_name': parsed.middle or '',
        'last_name': parsed.last or '',
        'suffix': parsed.suffix or '',
        'title': parsed.title or '',
        'original': name
    }


def extract_all_results(data: Any) -> List[Dict[str, Any]]:
    """
    Extract all validation results from the input data structure.
    
    Args:
        data: The loaded JSON data from validation results (can be dict or list)
        
    Returns:
        List of all validation result dictionaries
    """
    all_results = []
    
    # Handle flat list structure (like last_names.json)
    if isinstance(data, list):
        all_results = data
    elif isinstance(data, dict):
        # Extract from the 'files' section (current structure)
        for file_data in data.get('files', []):
            all_results.extend(file_data.get('results', []))
        
        # Also check if results are directly in 'mismatches' and 'matches' sections
        all_results.extend(data.get('mismatches', []))
        all_results.extend(data.get('matches', []))
    else:
        logger.warning(f"Unexpected data type: {type(data)}")
        return []
    
    # Remove duplicates based on reference ID and title
    seen = set()
    unique_results = []
    for result in all_results:
        if isinstance(result, dict):
            ref = result.get('reference', {})
            key = (ref.get('id', ''), ref.get('title', ''))
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
    
    return unique_results


# ============================================================================
# Statistical Analysis Functions
# ============================================================================

def analyze_error_classifications(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze error classifications across all validation results."""
    error_counts = Counter()
    error_examples = defaultdict(list)
    
    for result in results:
        classifications = result.get('error_classifications', [])
        for error_type in classifications:
            error_counts[error_type] += 1
            
            # Store example for each error type (limit to 5 examples)
            if len(error_examples[error_type]) < 5:
                ref_title = result.get('reference', {}).get('title', 'Unknown')[:100]
                error_examples[error_type].append({
                    'title': ref_title,
                    'mismatches': result.get('mismatches', [])[:3]
                })
    
    return {
        'counts': dict(error_counts),
        'examples': dict(error_examples)
    }


def analyze_title_similarities(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze title similarity scores to identify potential issues."""
    similarities = []
    low_similarity_examples = []
    
    for result in results:
        similarity = result.get('title_similarity', 0.0)
        if similarity > 0:
            similarities.append(similarity)
            
            # Collect examples of low similarity matches
            if similarity < 95.0 and result.get('validation_status') in ['matched', 'author_mismatch']:
                ref_title = result.get('reference', {}).get('title', 'Unknown')
                dblp_match = result.get('dblp_match')
                dblp_title = dblp_match.get('title', 'Unknown') if dblp_match else 'Unknown'
                if len(low_similarity_examples) < 10:
                    low_similarity_examples.append({
                        'similarity': similarity,
                        'ref_title': ref_title[:100],
                        'dblp_title': dblp_title[:100],
                        'status': result.get('validation_status')
                    })
    
    if not similarities:
        return {'error': 'No title similarities found'}
    
    similarities.sort()
    n = len(similarities)
    
    return {
        'count': n,
        'min': min(similarities),
        'max': max(similarities),
        'mean': sum(similarities) / n,
        'median': similarities[n // 2],
        'p25': similarities[n // 4],
        'p75': similarities[3 * n // 4],
        'low_similarity_examples': low_similarity_examples
    }


def analyze_author_list_lengths(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze differences in author list lengths between references and DBLP."""
    length_diffs = []
    examples = []
    
    for result in results:
        ref_authors = result.get('reference', {}).get('authors', [])
        dblp_match = result.get('dblp_match')
        dblp_authors = dblp_match.get('authors', []) if dblp_match else []
        
        if ref_authors and dblp_authors:
            diff = len(dblp_authors) - len(ref_authors)
            length_diffs.append(diff)
            
            # Collect examples of large differences
            if abs(diff) > 5 and len(examples) < 10:
                examples.append({
                    'ref_count': len(ref_authors),
                    'dblp_count': len(dblp_authors),
                    'diff': diff,
                    'title': result.get('reference', {}).get('title', 'Unknown')[:100],
                    'status': result.get('validation_status', 'unknown')
                })
    
    if not length_diffs:
        return {'error': 'No author list length data found'}
    
    return {
        'count': len(length_diffs),
        'mean_diff': sum(length_diffs) / len(length_diffs),
        'min_diff': min(length_diffs),
        'max_diff': max(length_diffs),
        'positive_diff_count': sum(1 for d in length_diffs if d > 0),
        'negative_diff_count': sum(1 for d in length_diffs if d < 0),
        'zero_diff_count': sum(1 for d in length_diffs if d == 0),
        'examples': examples
    }


def identify_common_mistakes(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify common mistakes in citation validation."""
    mistakes = []
    
    # Low title similarity but still matched
    low_sim_matched = []
    for result in results:
        similarity = result.get('title_similarity', 0.0)
        status = result.get('validation_status', '')
        if 80.0 <= similarity < 90.0 and status in ['matched', 'author_mismatch']:
            ref_title = result.get('reference', {}).get('title', '')
            dblp_match = result.get('dblp_match')
            dblp_title = dblp_match.get('title', '') if dblp_match else ''
            low_sim_matched.append({
                'similarity': similarity,
                'status': status,
                'ref_title': ref_title[:100],
                'dblp_title': dblp_title[:100]
            })
    
    if low_sim_matched:
        mistakes.append({
            'type': 'Low title similarity but still processed',
            'count': len(low_sim_matched),
            'description': 'Titles with similarity between 80-90% were still processed',
            'examples': low_sim_matched[:5]
        })
    
    # Author order issues
    order_issues = []
    for result in results:
        if 'author_order_wrong' in result.get('error_classifications', []):
            dblp_match = result.get('dblp_match')
            dblp_authors = dblp_match.get('authors', [])[:5] if dblp_match else []
            order_issues.append({
                'title': result.get('reference', {}).get('title', '')[:100],
                'ref_authors': result.get('reference', {}).get('authors', [])[:5],
                'dblp_authors': dblp_authors
            })
    
    if order_issues:
        mistakes.append({
            'type': 'Author order mismatches',
            'count': len(order_issues),
            'description': 'Authors match but are in wrong order',
            'examples': order_issues[:5]
        })
    
    # Accent issues
    accent_issues = []
    for result in results:
        if 'accents_missing' in result.get('error_classifications', []):
            accent_issues.append({
                'title': result.get('reference', {}).get('title', '')[:100],
                'mismatches': result.get('mismatches', [])[:3]
            })
    
    if accent_issues:
        mistakes.append({
            'type': 'Accent/diacritic mismatches',
            'count': len(accent_issues),
            'description': 'Names differ only by accents/diacritics',
            'examples': accent_issues[:5]
        })
    
    return mistakes


def run_statistical_analysis(data: Dict[str, Any], output_file: str) -> None:
    """Run statistical analysis on validation results."""
    all_results = extract_all_results(data)
    
    logger.info(f"Analyzing {len(all_results)} validation results")
    
    # Perform analyses
    analysis = {
        'summary': data.get('summary', {}),
        'error_classifications': analyze_error_classifications(all_results),
        'title_similarities': analyze_title_similarities(all_results),
        'author_list_lengths': analyze_author_list_lengths(all_results),
        'common_mistakes': identify_common_mistakes(all_results)
    }
    
    # Write analysis results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Statistical analysis written to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*60)
    
    summary = data.get('summary', {})
    if summary:
        print(f"\nOverall Statistics:")
        print(f"  Files processed: {summary.get('files_processed', 0)}")
        print(f"  Total references: {summary.get('total_references', 0)}")
        print(f"  Matched: {summary.get('total_matched', 0)}")
        print(f"  Mismatches: {summary.get('total_mismatches', 0)}")
    
    error_class = analysis.get('error_classifications', {})
    if error_class.get('counts'):
        print(f"\nError Classifications:")
        for error_type, count in sorted(error_class['counts'].items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")


# ============================================================================
# Parsing Error Detection Functions
# ============================================================================

def detect_cascading_last_names(ref_authors: List[str], dblp_authors: List[str], 
                                mismatches: List[str]) -> bool:
    """Detect cascading pattern where last names are shifted across authors."""
    if len(ref_authors) < 2 or len(dblp_authors) < 2:
        return False
    
    ref_parsed = [normalize_author_name(name) for name in ref_authors]
    dblp_parsed = [normalize_author_name(name) for name in dblp_authors]
    
    cascading_count = 0
    for i in range(min(len(ref_parsed) - 1, len(dblp_parsed) - 1)):
        ref_last = ref_parsed[i].get('last_name', '').lower().strip()
        dblp_next_last = dblp_parsed[i + 1].get('last_name', '').lower().strip()
        if ref_last and dblp_next_last and ref_last == dblp_next_last:
            cascading_count += 1
    
    return cascading_count >= 2


def detect_split_multipart_lastname(ref_authors: List[str], dblp_authors: List[str],
                                    mismatches: List[str]) -> Optional[str]:
    """Detect when a multi-part last name is incorrectly split."""
    not_found_authors = []
    for mismatch in mismatches:
        if "Author not found in DBLP:" in mismatch:
            author_name = mismatch.replace("Author not found in DBLP:", "").strip()
            not_found_authors.append(author_name)
    
    if len(not_found_authors) < 1:
        return None
    
    multipart_patterns = [
        r'van\s+den', r'van\s+der', r'de\s+la', r'de\s+le',
        r'costa[-–]', r'[a-z]+[-–][a-z]+'
    ]
    
    for dblp_author in dblp_authors:
        dblp_parsed = normalize_author_name(dblp_author)
        dblp_last = dblp_parsed.get('last_name', '').lower()
        
        for pattern in multipart_patterns:
            if re.search(pattern, dblp_last, re.IGNORECASE):
                dblp_parts = dblp_last.split()
                for not_found in not_found_authors:
                    not_found_lower = not_found.lower().strip()
                    if not_found_lower in dblp_parts:
                        return f"Multi-part last name '{dblp_author}' appears split: '{not_found}' is part of it"
    
    return None


def analyze_parsing_errors(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single validation entry for parsing errors."""
    result = {
        'has_parsing_error': False,
        'error_types': [],
        'error_details': []
    }
    
    ref_authors = entry.get('reference', {}).get('authors', [])
    dblp_authors = entry.get('dblp_match', {}).get('authors', [])
    mismatches = entry.get('mismatches', [])
    
    if not ref_authors or not dblp_authors or not mismatches:
        return result
    
    # Check for cascading last names
    if detect_cascading_last_names(ref_authors, dblp_authors, mismatches):
        result['has_parsing_error'] = True
        result['error_types'].append('cascading_last_names')
        result['error_details'].append('Last names appear to be cascading/shifted across authors')
    
    # Check for split multi-part last names
    split_info = detect_split_multipart_lastname(ref_authors, dblp_authors, mismatches)
    if split_info:
        result['has_parsing_error'] = True
        result['error_types'].append('split_multipart_lastname')
        result['error_details'].append(split_info)
    
    # Check for cascading pattern in mismatch descriptions
    last_name_mismatches = [m for m in mismatches if 'last_name_mismatch' in m]
    if len(last_name_mismatches) >= 3:
        ref_parsed = [normalize_author_name(name) for name in ref_authors]
        dblp_parsed = [normalize_author_name(name) for name in dblp_authors]
        
        mismatch_patterns = []
        for mismatch in last_name_mismatches:
            match = re.search(r'last_name_mismatch:\s+([^v]+)\s+vs\s+([^v]+)', mismatch)
            if match:
                ref_part = match.group(1).strip()
                dblp_part = match.group(2).strip()
                ref_parts = ref_part.split()
                dblp_parts = dblp_part.split()
                if len(ref_parts) >= 2 and len(dblp_parts) >= 2:
                    ref_last = ' '.join(ref_parts[1:]).lower()
                    dblp_last = ' '.join(dblp_parts[1:]).lower()
                    mismatch_patterns.append((ref_last, dblp_last))
        
        cascading_count = 0
        for i in range(len(mismatch_patterns) - 1):
            ref_last_i = mismatch_patterns[i][0]
            dblp_last_next = mismatch_patterns[i + 1][1]
            if ref_last_i == dblp_last_next or ref_last_i in dblp_last_next or dblp_last_next in ref_last_i:
                cascading_count += 1
        
        for i in range(1, len(mismatch_patterns)):
            ref_last_i = mismatch_patterns[i][0]
            dblp_last_prev = mismatch_patterns[i - 1][1]
            if ref_last_i == dblp_last_prev or ref_last_i in dblp_last_prev or dblp_last_prev in ref_last_i:
                cascading_count += 1
        
        if cascading_count >= 2 and not result['has_parsing_error']:
            result['has_parsing_error'] = True
            result['error_types'].append('cascading_last_names')
            result['error_details'].append(
                f'Consecutive last name mismatches show cascading pattern ({cascading_count} matches)'
            )
    
    return result


def run_parsing_error_detection(data: Dict[str, Any], output_file: str) -> None:
    """Run parsing error detection on validation results."""
    all_results = extract_all_results(data)
    
    logger.info(f"Analyzing {len(all_results)} entries for parsing errors...")
    
    parsing_errors = []
    error_stats = defaultdict(int)
    
    for i, entry in enumerate(all_results):
        analysis = analyze_parsing_errors(entry)
        
        if analysis['has_parsing_error']:
            parsing_errors.append({
                'entry_index': i,
                'reference_id': entry.get('reference', {}).get('id', 'unknown'),
                'title': entry.get('reference', {}).get('title', 'unknown'),
                'ref_authors': entry.get('reference', {}).get('authors', []),
                'dblp_authors': entry.get('dblp_match', {}).get('authors', []),
                'original_mismatches': entry.get('mismatches', []),
                'parsing_error_analysis': analysis
            })
            
            for error_type in analysis['error_types']:
                error_stats[error_type] += 1
    
    output_data = {
        'total_entries': len(all_results),
        'entries_with_parsing_errors': len(parsing_errors),
        'error_statistics': dict(error_stats),
        'parsing_errors': parsing_errors
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Parsing error analysis written to: {output_file}")
    
    print(f"\nFound {len(parsing_errors)} entries with parsing errors")
    print("\nError type statistics:")
    for error_type, count in sorted(error_stats.items(), key=lambda x: -x[1]):
        print(f"  {error_type}: {count}")


# ============================================================================
# Result Reorganization Functions
# ============================================================================

def categorize_results(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize validation results into different groups."""
    categories = defaultdict(list)
    
    for result in results:
        status = result.get('validation_status', 'unknown')
        classifications = result.get('error_classifications', [])
        
        if status == 'matched':
            categories['matched'].append(result)
        if 'parsing_error' in classifications:
            categories['parsing_errors'].append(result)
        if 'first_name_mismatch' in classifications:
            categories['first_names'].append(result)
        if 'last_name_mismatch' in classifications:
            categories['last_names'].append(result)
        if 'accents_missing' in classifications:
            categories['accents_missing'].append(result)
        if 'author_not_found' in classifications:
            categories['author_not_found'].append(result)
        if 'author_order_wrong' in classifications:
            categories['author_order_wrong'].append(result)
        if 'empty_list' in classifications:
            categories['empty_list'].append(result)
        if status == 'title_mismatch':
            categories['title_mismatches'].append(result)
        if status == 'no_dblp_match':
            categories['no_dblp_match'].append(result)
        if status == 'error':
            categories['errors'].append(result)
        if status == 'skipped':
            categories['skipped'].append(result)
    
    categories['summary'] = [{
        'total_results': len(results),
        'categories': {cat: len(results) for cat, results in categories.items() if cat != 'summary'}
    }]
    
    return dict(categories)


def run_reorganization(data: Dict[str, Any], output_dir: str) -> None:
    """Reorganize validation results into categorized files."""
    all_results = extract_all_results(data)
    
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    categories = categorize_results(all_results)
    
    # Write category files
    for category_name, results in categories.items():
        if category_name == 'summary':
            filename = 'summary.json'
        else:
            filename = f'{category_name}.json'
        
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote {len(results)} results to {filename}")
    
    # Create README
    readme_path = os.path.join(results_dir, 'README.md')
    readme_content = """# Citation Validation Results

This folder contains citation validation results organized by category.

## File Structure

- `matched.json` - Citations that matched correctly with DBLP
- `parsing_errors.json` - Citations with parsing errors in author names
- `first_names.json` - Citations with first name mismatches
- `last_names.json` - Citations with last name mismatches
- `accents_missing.json` - Citations with missing accents/diacritics
- `author_not_found.json` - Citations where authors were not found in DBLP
- `author_order_wrong.json` - Citations with correct authors but wrong order
- `empty_list.json` - Citations with empty author lists
- `title_mismatches.json` - Citations with title similarity below threshold
- `no_dblp_match.json` - Citations not found in DBLP database
- `errors.json` - Citations that caused processing errors
- `skipped.json` - Citations that were skipped
- `summary.json` - Summary statistics for all categories

## Statistics

"""
    
    summary = categories.get('summary', [{}])[0]
    if 'categories' in summary:
        readme_content += "| Category | Count |\n|----------|-------|\n"
        for cat, count in sorted(summary['categories'].items()):
            if cat != 'summary':
                readme_content += f"| {cat} | {count} |\n"
        readme_content += f"\n**Total Results:** {summary.get('total_results', 0)}\n"
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    logger.info(f"Successfully reorganized results into: {results_dir}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Comprehensive citation analysis tool.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, default='citation_validation_results.json',
                       help='Input JSON file with validation results')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'stats', 'parsing', 'reorganize'],
                       help='Analysis mode: all, stats, parsing, or reorganize')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (for stats/parsing modes)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory (for reorganize mode)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    logger.info(f"Loading validation results from: {args.input}")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return
    
    # Run requested analyses
    if args.mode in ['all', 'stats']:
        output_file = args.output or 'validation_analysis.json'
        run_statistical_analysis(data, output_file)
    
    if args.mode in ['all', 'parsing']:
        output_file = args.output or 'parsing_errors_analysis.json'
        run_parsing_error_detection(data, output_file)
    
    if args.mode in ['all', 'reorganize']:
        run_reorganization(data, args.output_dir)
    
    logger.info("Analysis complete!")


if __name__ == '__main__':
    main()


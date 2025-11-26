"""
Main Pipeline for Citation Verification

This is the central execution file for citation verification using the modern API caller architecture.
It replaces the old example_starter.py approach and uses api_caller.py exclusively for all database
interactions across DBLP, arXiv, and Semantic Scholar.

Usage:
    python Main_Pipeline.py
"""

import json
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add both src and task directories to path for imports
task_dir = Path(__file__).parent
src_dir = task_dir.parent / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(task_dir))

# Import the modern API caller from src directory to get default values
from api_caller import (
    search_papers_by_title,
    get_best_match_from_search_results,
    DEFAULT_SIMILARITY_THRESHOLD,
    save_results_to_json
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# All configuration values are defined here. Modify these values directly in this
# section as needed. The script runs without command-line arguments and uses only
# these configuration variables.

# Path to the input JSON file containing citations (relative to script directory)
CITATIONS_FILE = 'citations.json'

# Path to the output JSON file for results (relative to script directory)
OUTPUT_FILE = 'verification_results.json'

# Limit on number of citations to process (None = process all citations)
# Set to a number to process only the first N citations (e.g., 100)
# Example: CITATION_LIMIT = 100  # Process only first 100 citations
CITATION_LIMIT = None

# Minimum similarity threshold for title matching (0-100)
# Papers with similarity scores below this threshold will not be considered matches
# Default: 80 (uses DEFAULT_SIMILARITY_THRESHOLD from api_caller)
SIMILARITY_THRESHOLD = DEFAULT_SIMILARITY_THRESHOLD

# Maximum number of results to fetch from each source (DBLP, arXiv, Semantic Scholar)
# Higher values may improve match quality but increase API calls and processing time
MAX_RESULTS_PER_SOURCE = 10
# ==============================================================================

# Import the improved compare_authors function from example_starter in task directory
from example_starter import (
    load_citations,
    compare_authors,
    is_valid_author_name
)

# Configure logging
# Create log file in the same directory as the script
log_file = task_dir / 'citation_verification.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def verify_citation(
    citation: Dict[str, Any],
    similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
    max_results_per_source: int = 10
) -> Dict[str, Any]:
    """
    Verify a single citation using the modern API caller architecture.
    
    This function searches across DBLP, arXiv, and Semantic Scholar databases,
    selects the best match, and compares authors using the improved comparison logic.
    
    Args:
        citation: Dictionary containing 'title' and 'authors' fields
        similarity_threshold: Minimum similarity score (0-100) to consider a match
        max_results_per_source: Maximum results to fetch from each source
        
    Returns:
        Dictionary with verification status, matched paper info, and author comparison results.
        Status can be: 'verified', 'discrepancy_found', or 'not_found'
    """
    title = citation.get('title', '')
    authors = citation.get('authors', [])
    
    if not title:
        logger.warning(f"Skipping citation with empty title")
        return {
            "status": "error",
            "message": "Citation has empty title",
            "title": None
        }
    
    logger.info(f"Verifying citation: '{title[:80]}...'")
    
    try:
        # Search across all databases using the modern API caller
        search_results = search_papers_by_title(
            title=title,
            similarity_threshold=similarity_threshold,
            max_results_per_source=max_results_per_source,
            parallel=True  # Search databases in parallel for efficiency
        )
        
        # Extract the best match from search results
        best_match = get_best_match_from_search_results(
            search_results=search_results,
            min_similarity=similarity_threshold
        )
        
        # No good match found
        if not best_match:
            logger.info(f"No good match found for: '{title[:60]}...' (threshold: {similarity_threshold}%)")
            return {
                "status": "not_found",
                "message": f"No match found above similarity threshold ({similarity_threshold}%)",
                "title": title,
                "match_score": 0,
                "search_summary": search_results.get('summary', {})
            }
        
        # Found a match - compare authors
        verified_title = best_match.get('title', '')
        verified_authors = best_match.get('authors', [])
        match_score = best_match.get('match_score', 0)
        source = best_match.get('source', 'unknown')
        
        logger.info(f"Found match in {source} (similarity: {match_score:.1f}%): '{verified_title[:60]}...'")
        
        # Use improved compare_authors function with paper title for parsing error detection
        comparison = compare_authors(
            original_authors=authors,
            verified_authors=verified_authors,
            paper_title=title
        )
        
        # Build verification result
        result = {
            "title": verified_title,
            "original_title": title,
            "status": "verified" if comparison["match"] else "discrepancy_found",
            "source": source,
            "verified_authors": verified_authors,
            "original_authors": authors,
            "match_score": match_score,
            "comparison": comparison,
            "metadata": {
                "year": best_match.get('year'),
                "doi": best_match.get('doi'),
                "url": best_match.get('url'),
                "pdf_url": best_match.get('pdf_url'),
            }
        }
        
        # Add source-specific metadata
        if best_match.get('arxiv_id'):
            result["metadata"]["arxiv_id"] = best_match['arxiv_id']
        if best_match.get('venue'):
            result["metadata"]["venue"] = best_match['venue']
        if best_match.get('paper_id'):
            result["metadata"]["paper_id"] = best_match['paper_id']
        
        # Log results
        if comparison["match"]:
            logger.info(f"✓ Verification successful: Authors match!")
        else:
            logger.warning(f"⚠ Discrepancies found: {len(comparison.get('discrepancies', []))} issue(s)")
            for disc in comparison.get('discrepancies', [])[:3]:  # Log first 3 discrepancies
                logger.debug(f"  - {disc.get('details', 'Unknown discrepancy')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error verifying citation '{title[:60]}...': {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error during verification: {str(e)}",
            "title": title
        }


def process_citations(
    citations: List[Dict[str, Any]],
    similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
    max_results_per_source: int = 10,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Process multiple citations through the verification pipeline.
    
    Args:
        citations: List of citation dictionaries
        similarity_threshold: Minimum similarity score (0-100) to consider a match
        max_results_per_source: Maximum results to fetch from each source
        limit: Optional limit on number of citations to process (None = process all)
        
    Returns:
        List of verification result dictionaries
    """
    total = len(citations)
    if limit:
        citations = citations[:limit]
        total = limit
    
    logger.info(f"=" * 80)
    logger.info(f"Processing {len(citations)} citation(s) (out of {total} total)")
    logger.info(f"Similarity threshold: {similarity_threshold}%")
    logger.info(f"=" * 80)
    
    results = []
    
    for i, citation in enumerate(citations, 1):
        logger.info(f"\n[{i}/{len(citations)}] Processing citation {i}")
        
        result = verify_citation(
            citation=citation,
            similarity_threshold=similarity_threshold,
            max_results_per_source=max_results_per_source
        )
        
        results.append({
            'original': citation,
            'verification': result
        })
        
        # Log progress periodically
        if i % 10 == 0:
            verified = sum(1 for r in results if r['verification'].get('status') == 'verified')
            discrepancies = sum(1 for r in results if r['verification'].get('status') == 'discrepancy_found')
            not_found = sum(1 for r in results if r['verification'].get('status') == 'not_found')
            logger.info(f"Progress: {i}/{len(citations)} | Verified: {verified} | Discrepancies: {discrepancies} | Not found: {not_found}")
    
    return results


def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics from verification results.
    
    Args:
        results: List of verification result dictionaries
        
    Returns:
        Dictionary with summary statistics
    """
    total = len(results)
    if total == 0:
        return {"total": 0}
    
    status_counts = {}
    source_counts = {}
    
    for result in results:
        verification = result.get('verification', {})
        status = verification.get('status', 'unknown')
        source = verification.get('source', 'unknown')
        
        status_counts[status] = status_counts.get(status, 0) + 1
        if source != 'unknown':
            source_counts[source] = source_counts.get(source, 0) + 1
    
    # Calculate average match scores
    match_scores = [
        r['verification'].get('match_score', 0)
        for r in results
        if r['verification'].get('match_score', 0) > 0
    ]
    avg_match_score = sum(match_scores) / len(match_scores) if match_scores else 0
    
    # Count parsing errors
    parsing_errors = 0
    for result in results:
        comparison = result.get('verification', {}).get('comparison', {})
        discrepancies = comparison.get('discrepancies', [])
        parsing_errors += sum(1 for d in discrepancies if d.get('type') == 'parsing_error')
    
    return {
        "total": total,
        "status_counts": status_counts,
        "source_counts": source_counts,
        "average_match_score": round(avg_match_score, 2),
        "parsing_errors_detected": parsing_errors,
        "verification_rate": round(status_counts.get('verified', 0) / total * 100, 2) if total > 0 else 0
    }


def main():
    """
    Main execution function for the citation verification pipeline.
    
    All configuration is defined at the top of the script as constants.
    """
    # Validate configuration
    if not os.path.exists(CITATIONS_FILE):
        logger.error(f"Citations file not found: {CITATIONS_FILE}")
        logger.error(f"Please ensure the file exists or update CITATIONS_FILE in the script configuration.")
        sys.exit(1)
    
    if not (0 <= SIMILARITY_THRESHOLD <= 100):
        logger.error(f"SIMILARITY_THRESHOLD must be between 0 and 100, got: {SIMILARITY_THRESHOLD}")
        logger.error(f"Please update SIMILARITY_THRESHOLD in the script configuration.")
        sys.exit(1)
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("CITATION VERIFICATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Citations file: {CITATIONS_FILE}")
    logger.info(f"  Output file: {OUTPUT_FILE}")
    logger.info(f"  Citation limit: {CITATION_LIMIT if CITATION_LIMIT else 'All citations'}")
    logger.info(f"  Similarity threshold: {SIMILARITY_THRESHOLD}%")
    logger.info(f"  Max results per source: {MAX_RESULTS_PER_SOURCE}")
    logger.info("=" * 80)
    
    # Load citations
    logger.info(f"\nLoading citations from: {CITATIONS_FILE}")
    try:
        citations = load_citations(CITATIONS_FILE)
        logger.info(f"Loaded {len(citations)} citations")
    except Exception as e:
        logger.error(f"Failed to load citations: {e}", exc_info=True)
        sys.exit(1)
    
    if not citations:
        logger.warning("No citations found in file")
        sys.exit(0)
    
    # Process citations
    try:
        results = process_citations(
            citations=citations,
            similarity_threshold=SIMILARITY_THRESHOLD,
            max_results_per_source=MAX_RESULTS_PER_SOURCE,
            limit=CITATION_LIMIT
        )
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)
    
    # Generate and log summary
    summary = generate_summary(results)
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total processed: {summary['total']}")
    logger.info(f"Status breakdown:")
    for status, count in summary['status_counts'].items():
        percentage = round(count / summary['total'] * 100, 1) if summary['total'] > 0 else 0
        logger.info(f"  {status}: {count} ({percentage}%)")
    logger.info(f"Source breakdown:")
    for source, count in summary['source_counts'].items():
        logger.info(f"  {source}: {count}")
    logger.info(f"Average match score: {summary['average_match_score']:.1f}%")
    logger.info(f"Parsing errors detected: {summary['parsing_errors_detected']}")
    logger.info(f"Verification rate: {summary['verification_rate']:.1f}%")
    logger.info("=" * 80)
    
    # Add summary to results
    output_data = {
        "summary": summary,
        "results": results,
        "configuration": {
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "max_results_per_source": MAX_RESULTS_PER_SOURCE,
            "citations_file": CITATIONS_FILE,
            "citation_limit": CITATION_LIMIT
        }
    }
    
    # Save results
    logger.info(f"\nSaving results to: {OUTPUT_FILE}")
    try:
        save_results_to_json(output_data, OUTPUT_FILE)
        logger.info(f"✓ Results saved successfully!")
    except Exception as e:
        logger.error(f"Failed to save results: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("\nPipeline execution complete!")


if __name__ == '__main__':
    main()


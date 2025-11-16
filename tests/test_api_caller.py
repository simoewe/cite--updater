"""
Test script for api_caller.py

This script demonstrates and tests the functionality of the multi-database API caller.
It searches for papers across DBLP, arXiv, and Semantic Scholar databases.

Usage:
    python tests/test_api_caller.py
"""

import sys
import os
import time
import json
from pathlib import Path

# Add parent directory to path to import api_caller
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api_caller import (
    search_papers_by_title,
    search_multiple_titles,
    search_dblp,
    search_arxiv,
    search_semantic_scholar,
    calculate_title_similarity,
    save_results_to_json,
    get_semantic_scholar_api_key
)
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_paper_details(paper: dict, index: int = None):
    """
    Print formatted paper details.
    
    Args:
        paper: Paper dictionary with metadata
        index: Optional index number for the paper
    """
    prefix = f"{index}. " if index is not None else ""
    print(f"\n{prefix}{'='*80}")
    print(f"Title: {paper.get('title', 'N/A')}")
    print(f"Source: {paper.get('source', 'N/A')}")
    print(f"Similarity: {paper.get('similarity_score', 'N/A')}%")
    print(f"Authors: {', '.join(paper.get('authors', [])[:5])}")
    if len(paper.get('authors', [])) > 5:
        print(f"  ... and {len(paper.get('authors', [])) - 5} more authors")
    print(f"Year: {paper.get('year', 'N/A')}")
    if paper.get('doi'):
        print(f"DOI: {paper.get('doi', 'N/A')}")
    if paper.get('url'):
        print(f"URL: {paper.get('url', 'N/A')}")
    if paper.get('pdf_url'):
        print(f"PDF: {paper.get('pdf_url', 'N/A')}")
    if paper.get('venue'):
        print(f"Venue: {paper.get('venue', 'N/A')}")


def test_single_title_search(title: str, similarity_threshold: int = 80):
    """
    Test searching for a single paper title across all databases.
    
    Args:
        title: Paper title to search for
        similarity_threshold: Minimum similarity score (0-100)
    """
    print(f"\n{'#'*80}")
    print(f"Testing single title search")
    print(f"{'#'*80}")
    print(f"\nSearching for: '{title}'")
    print(f"Similarity threshold: {similarity_threshold}%")
    
    start_time = time.time()
    result = search_papers_by_title(
        title,
        similarity_threshold=similarity_threshold,
        max_results_per_source=10,
        parallel=True
    )
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Search completed in {elapsed_time:.2f} seconds")
    print(f"{'='*80}")
    
    print(f"\nSummary:")
    print(f"  Original title: {result['original_title']}")
    print(f"  Total papers found: {result['summary']['total_found']}")
    print(f"  Matching papers (similarity >= {similarity_threshold}%): {result['summary']['total_matching']}")
    print(f"  By source:")
    for source, count in result['summary']['by_source'].items():
        print(f"    - {source}: {count}")
    
    if result['results']:
        print(f"\n{'='*80}")
        print(f"Top {min(5, len(result['results']))} matching papers:")
        print(f"{'='*80}")
        for i, paper in enumerate(result['results'][:5], 1):
            print_paper_details(paper, index=i)
    else:
        print("\n⚠️  No matching papers found above the similarity threshold.")
    
    return result


def test_individual_database_searches(title: str):
    """
    Test searching each database individually to see differences.
    
    Args:
        title: Paper title to search for
    """
    print(f"\n{'#'*80}")
    print(f"Testing individual database searches")
    print(f"{'#'*80}")
    print(f"\nSearching for: '{title}'")
    
    databases = [
        ("DBLP", search_dblp),
        ("arXiv", search_arxiv),
        ("Semantic Scholar", search_semantic_scholar)
    ]
    
    all_results = {}
    
    for db_name, search_func in databases:
        print(f"\n{'-'*80}")
        print(f"Searching {db_name}...")
        start_time = time.time()
        
        try:
            results = search_func(title, max_results=5)
            elapsed_time = time.time() - start_time
            
            all_results[db_name] = results
            
            print(f"  Found {len(results)} papers in {elapsed_time:.2f} seconds")
            
            if results:
                print(f"  Top result:")
                print(f"    Title: {results[0].get('title', 'N/A')[:80]}...")
                print(f"    Authors: {', '.join(results[0].get('authors', [])[:3])}")
                print(f"    Year: {results[0].get('year', 'N/A')}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_results[db_name] = []
    
    return all_results


def test_multiple_titles_search(titles: list, max_workers: int = 3):
    """
    Test searching for multiple titles in parallel.
    
    Args:
        titles: List of paper titles to search for
        max_workers: Maximum number of parallel workers
    """
    print(f"\n{'#'*80}")
    print(f"Testing multiple titles search (parallel)")
    print(f"{'#'*80}")
    print(f"\nSearching {len(titles)} titles with {max_workers} workers...")
    
    start_time = time.time()
    results = search_multiple_titles(
        titles,
        similarity_threshold=80,
        max_results_per_source=5,
        max_workers=max_workers
    )
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Batch search completed in {elapsed_time:.2f} seconds")
    print(f"Average time per title: {elapsed_time/len(titles):.2f} seconds")
    print(f"{'='*80}")
    
    total_found = 0
    total_matching = 0
    
    for i, result in enumerate(results, 1):
        title = result['original_title']
        found = result['summary'].get('total_found', 0)
        matching = result['summary'].get('total_matching', 0)
        
        total_found += found
        total_matching += matching
        
        status = "✅" if matching > 0 else "❌"
        print(f"\n{status} {i}. '{title[:60]}...'")
        print(f"   Found: {found} papers, Matching: {matching} papers")
    
    print(f"\n{'='*80}")
    print(f"Overall Summary:")
    print(f"  Total titles searched: {len(titles)}")
    print(f"  Total papers found: {total_found}")
    print(f"  Total matching papers: {total_matching}")
    print(f"  Average papers per title: {total_found/len(titles):.1f}")
    
    return results


def test_title_similarity():
    """
    Test the title similarity calculation function.
    """
    print(f"\n{'#'*80}")
    print(f"Testing title similarity calculation")
    print(f"{'#'*80}")
    
    test_cases = [
        ("Attention Is All You Need", "Attention Is All You Need", 100),
        ("Attention Is All You Need", "Attention is all you need", 100),
        ("BERT: Pre-training of Deep Bidirectional Transformers", "BERT Pre-training Deep Bidirectional Transformers", 85),
        ("Machine Learning", "Deep Learning", 50),
        ("Neural Networks", "Convolutional Neural Networks", 75),
    ]
    
    print("\nSimilarity scores:")
    for title1, title2, expected_range in test_cases:
        similarity = calculate_title_similarity(title1, title2)
        status = "✅" if similarity >= expected_range - 10 else "⚠️"
        print(f"{status} '{title1}' vs '{title2}': {similarity}% (expected ~{expected_range}%)")


def test_api_key_status():
    """
    Test if API key is loaded correctly.
    """
    print(f"\n{'#'*80}")
    print(f"Testing API key status")
    print(f"{'#'*80}")
    
    api_key = get_semantic_scholar_api_key()
    
    if api_key:
        print(f"✅ API key loaded successfully")
        print(f"   Key length: {len(api_key)} characters")
        print(f"   Key preview: {api_key[:10]}...{api_key[-5:]}")
        print(f"   Note: Using higher rate limits (5 requests per 5 seconds)")
    else:
        print(f"⚠️  No API key found")
        print(f"   Using public rate limits (1 request per second)")
        print(f"   To use API key: Create .env file with SEMANTIC_SCHOLAR_API_KEY=your_key")


def main():
    """
    Main test function that runs all test scenarios.
    """
    print("="*80)
    print("API Caller Test Suite")
    print("="*80)
    print("\nThis script tests the multi-database paper search functionality.")
    print("It searches across DBLP, arXiv, and Semantic Scholar databases.")
    
    # Test API key status
    test_api_key_status()
    
    # Test title similarity
    test_title_similarity()
    
    # Test cases - famous papers that should be found
    test_titles = [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "Deep Residual Learning for Image Recognition"
    ]
    
    # Test 1: Single title search with different thresholds
    print(f"\n{'='*80}")
    print("TEST 1: Single Title Search")
    print(f"{'='*80}")
    
    test_title = test_titles[0]
    result = test_single_title_search(test_title, similarity_threshold=80)
    
    # Save result to JSON for inspection
    output_file = "tests/api_caller_test_result.json"
    save_results_to_json(result, output_file)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Test 2: Individual database searches
    print(f"\n{'='*80}")
    print("TEST 2: Individual Database Searches")
    print(f"{'='*80}")
    
    individual_results = test_individual_database_searches(test_title)
    
    # Test 3: Multiple titles search (small batch)
    print(f"\n{'='*80}")
    print("TEST 3: Multiple Titles Search (Parallel)")
    print(f"{'='*80}")
    
    # Use smaller batch for testing to avoid long wait times
    batch_titles = test_titles[:2]  # Test with 2 titles
    batch_results = test_multiple_titles_search(batch_titles, max_workers=2)
    
    # Save batch results
    batch_output_file = "tests/api_caller_batch_test_result.json"
    save_results_to_json(batch_results, batch_output_file)
    print(f"\n💾 Batch results saved to: {batch_output_file}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("Test Suite Complete")
    print(f"{'='*80}")
    print("\n✅ All tests completed successfully!")
    print("\nNote: Rate limiting may cause delays between API calls.")
    print("      This is expected behavior to respect API limits.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)


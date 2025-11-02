"""
Random sampling analysis script for author name matching.

This script randomly selects 5 files from the outputs folder and runs the author name
analysis on them, producing a separate report for this sample.
"""

import os
import random
import json
import logging
from typing import List, Dict
import argparse
from analyze_matches import analyze_author_matches, setup_logging
from tqdm import tqdm

random.seed(42)

def get_random_files(input_dir: str, num_files: int = 1000) -> List[str]:
    """
    Get random JSON files from the input directory.
    
    Args:
        input_dir: Directory containing JSON files
        num_files: Number of files to randomly select
        
    Returns:
        List of selected file paths
    """
    # Get all JSON files from the directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        logging.error(f"No JSON files found in {input_dir}")
        return []
        
    # Select random files
    num_files = min(num_files, len(json_files))  # Don't try to select more files than available
    selected_files = random.sample(json_files, num_files)
    
    return [os.path.join(input_dir, f) for f in selected_files]

def run_sample_analysis(input_dir: str, output_dir: str) -> None:
    """
    Run analysis on randomly selected files.
    
    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory where output files will be saved
    """
    # Create output directories
    sample_output_dir = os.path.join(output_dir, 'sample_analysis')
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(sample_output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Setup logging in reports directory
    setup_logging(reports_dir)
    
    # Get random files
    selected_files = get_random_files(input_dir)
    
    if not selected_files:
        logging.error("No files selected for analysis")
        return
        
    logging.info(f"Selected {len(selected_files)} files for analysis:")
    for file in selected_files:
        logging.info(f"  - {os.path.basename(file)}")
    
    # Run analysis on each file
    for input_file in tqdm(selected_files, desc="Analyzing files"):
        file_output_dir = os.path.join(sample_output_dir, 
                                     os.path.splitext(os.path.basename(input_file))[0])
        analyze_author_matches(input_file, file_output_dir)
    
    # Create a summary of all analyses in reports directory
    create_summary_report(sample_output_dir, reports_dir, selected_files)

def create_summary_report(analysis_dir: str, reports_dir: str, input_files: List[str]) -> None:
    """
    Create a summary report focusing on first name differences found in the analyzed files.
    Only includes files and papers that have actual differences.
    
    Args:
        analysis_dir: Directory containing analysis results
        reports_dir: Directory where summary and logs will be stored
        input_files: List of analyzed input files
    """
    summary = {
        'analyzed_files': [os.path.basename(f) for f in input_files],
        'results': []
    }
    
    total_differences = 0
    
    for input_file in input_files:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        file_output_dir = os.path.join(analysis_dir, base_name)
        
        # Only check first name differences file
        diff_file = os.path.join(file_output_dir, 'first_name_differences.json')
        if not os.path.exists(diff_file):
            continue
            
        with open(diff_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        papers_with_differences = []
        for paper in results:
            paper_entry = {
                'title': paper['title'],
                'name_differences': []
            }
            
            for mismatch in paper['mismatches']:
                if isinstance(mismatch, str) and 'Name mismatch:' in mismatch:
                    # Extract the actual names from the mismatch message
                    names = mismatch.split('Name mismatch:')[-1].strip()
                    paper_entry['name_differences'].append(names)
                    total_differences += 1
            
            if paper_entry['name_differences']:
                papers_with_differences.append(paper_entry)
        
        # Only include files that have differences
        if papers_with_differences:
            summary['results'].append({
                'file': base_name,
                'total_differences': len(papers_with_differences),
                'papers_with_differences': papers_with_differences
            })
    
    if total_differences > 0:
        summary['total_differences_found'] = total_differences
        
        # Write summary report to reports directory
        summary_path = os.path.join(reports_dir, 'first_name_differences_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Log summary information
        logging.info(f"\nSummary of first name differences:")
        logging.info(f"Total files analyzed: {len(input_files)}")
        logging.info(f"Files with differences: {len(summary['results'])}")
        logging.info(f"Total name differences found: {total_differences}")
        
        for result in summary['results']:
            logging.info(f"\nIn {result['file']}:")
            for paper in result['papers_with_differences']:
                logging.info(f"  Paper: {paper['title']}")
                for diff in paper['name_differences']:
                    logging.info(f"    Name difference: {diff}")
        
        logging.info(f"\nDetailed summary written to {summary_path}")
    else:
        logging.info("\nNo first name differences found in any of the analyzed files.")

def main():
    """Parse command line arguments and run the sample analysis."""
    parser = argparse.ArgumentParser(
        description='Run author name analysis on random sample of files.'
    )
    parser.add_argument('--input-dir', default='output',
                      help='Directory containing input JSON files (default: output)')
    parser.add_argument('--output-dir', default='output',
                      help='Directory where analysis results will be saved (default: output)')
    parser.add_argument('--seed', type=int,
                      help='Random seed for reproducible sampling')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    run_sample_analysis(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main() 
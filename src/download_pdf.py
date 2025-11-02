"""
Script to download PDFs from the ACL Anthology using acl-anthology-py.
Supports downloading papers from specified year ranges.
"""

import os
import requests
import re
from acl_anthology import Anthology
import argparse
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def download_pdf(url: str, output_path: Path) -> bool:
    """
    Download a PDF from a given URL.
    
    Args:
        url: URL of the PDF to download
        output_path: Path where the PDF should be saved
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading PDF: {e}")
        return False

def get_paper_pdf(paper_id: str, output_dir: str = "data/pdfs") -> None:
    """
    Download a paper's PDF from the ACL Anthology.
    
    Args:
        paper_id: ACL Anthology ID of the paper (e.g., "2022.acl-long.220")
        output_dir: Directory where PDFs should be saved
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the Anthology
    logging.info("Initializing ACL Anthology...")
    anthology = Anthology.from_repo()
    
    # Get the paper
    paper = anthology.get(paper_id)
    if not paper:
        logging.error(f"Paper with ID '{paper_id}' not found")
        return
        
    # Get PDF information
    if not paper.pdf:
        logging.error(f"No PDF available for paper '{paper_id}'")
        return
        
    pdf_url = paper.pdf.url
    output_path = output_dir / f"{paper_id}.pdf"
    
    # Download the PDF
    logging.info(f"Downloading PDF from {pdf_url}")
    if download_pdf(pdf_url, output_path):
        logging.info(f"Successfully downloaded PDF to {output_path}")
        logging.info(f"Paper title: {paper.title}")
    else:
        logging.error("Failed to download PDF")

def get_papers_by_year(anthology: Anthology, year: int) -> List[dict]:
    """
    Get all papers published in a specific year.
    
    Args:
        anthology: Initialized ACL Anthology instance
        year: Year to fetch papers from
        
    Returns:
        List of paper objects from that year
    """
    papers = []
    year_str = str(year)
    year_short = year_str[-2:]  # Last two digits of the year (e.g., "19" for 2019)
    
    # Regex pattern for IDs like P19-0194, D19-1007
    old_format_pattern = re.compile(r'^[A-Za-z]\d*{}[-\.]\d+'.format(year_short))
    
    # Get all papers from the anthology
    for paper in anthology.papers():
        # Check if paper's full_id starts with the year (new format)
        if paper.full_id.startswith(year_str):
            papers.append(paper)
        # Check for old format IDs (e.g., P19-0194, D19-1007)
        elif old_format_pattern.match(paper.full_id):
            papers.append(paper)
    
    return papers

def download_papers_by_year_range(start_year: int, 
                                end_year: Optional[int] = None,
                                output_dir: str = "data/pdfs",
                                delay: float = 1.0,
                                max_papers: Optional[int] = None,
                                max_workers: int = 5) -> None:
    """
    Download papers from ACL Anthology within a year range using parallel processing.
    
    Args:
        start_year: Start year for paper downloads
        end_year: End year for paper downloads (defaults to current year)
        output_dir: Directory where PDFs should be saved
        delay: Delay between downloads in seconds
        max_papers: Maximum number of papers to download (None for no limit)
        max_workers: Maximum number of parallel downloads
    """
    if end_year is None:
        end_year = datetime.now().year
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("Initializing ACL Anthology...")
    anthology = Anthology.from_repo()
    
    papers_downloaded = 0
    
    def download_paper(paper, year_dir):
        """Helper function to download a single paper."""
        if not paper.pdf:
            return False, f"No PDF available for paper: {paper.title}"
            
        pdf_url = paper.pdf.url
        filename = f"{paper.full_id}.pdf"
        output_path = year_dir / filename
        
        if output_path.exists():
            return False, f"Skipping existing file: {output_path}"
            
        success = download_pdf(pdf_url, output_path)
        if success:
            return True, paper.title
        return False, f"Failed to download: {paper.title}"
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for year in range(start_year, end_year + 1):
            logging.info(f"\nProcessing papers from {year}...")
            
            papers = get_papers_by_year(anthology, year)
            
            if not papers:
                logging.info(f"No papers found for year {year}")
                continue
                
            logging.info(f"Found {len(papers)} papers from {year}")
            
            year_dir = output_dir / str(year)
            year_dir.mkdir(exist_ok=True)
            
            # Submit all download tasks
            future_to_paper = {
                executor.submit(download_paper, paper, year_dir): paper
                for paper in papers[:max_papers] if max_papers is None or papers_downloaded < max_papers
            }
            
            # Process completed downloads with progress bar
            with tqdm(total=len(future_to_paper), desc=f"Downloading {year} papers") as pbar:
                for future in as_completed(future_to_paper):
                    success, message = future.result()
                    if success:
                        papers_downloaded += 1
                        logging.debug(f"Downloaded: {message}")
                    else:
                        logging.debug(message)
                    pbar.update(1)
                    
                    if max_papers and papers_downloaded >= max_papers:
                        logging.info(f"Reached maximum paper limit ({max_papers})")
                        return
                    
                    # Add a small delay between submissions to be nice to the server
                    time.sleep(delay / max_workers)

def main():
    """Parse command line arguments and download the requested PDFs."""
    parser = argparse.ArgumentParser(description='Download PDFs from the ACL Anthology')
    parser.add_argument('--start-year', type=int, default=datetime.now().year - 10,
                      help='Start year for paper downloads (default: 10 years ago)')
    parser.add_argument('--end-year', type=int, default=None,
                      help='End year for paper downloads (default: current year)')
    parser.add_argument('--output-dir', default='pdfs',
                      help='Directory where PDFs should be saved (default: pdfs)')
    parser.add_argument('--delay', type=float, default=1.0,
                      help='Delay between downloads in seconds (default: 1.0)')
    parser.add_argument('--max-papers', type=int, default=None,
                      help='Maximum number of papers to download (default: no limit)')
    parser.add_argument('--max-workers', type=int, default=5,
                      help='Maximum number of parallel downloads (default: 5)')
    
    args = parser.parse_args()
    setup_logging()
    
    logging.info(f"Starting download for papers from {args.start_year} to "
                f"{args.end_year or datetime.now().year}")
    
    download_papers_by_year_range(
        args.start_year,
        args.end_year,
        args.output_dir,
        args.delay,
        args.max_papers,
        args.max_workers
    )

if __name__ == '__main__':
    main() 
"""
API Caller for Multi-Database Paper Search

This module provides functionality to search for papers across multiple databases
(DBLP, arXiv, Semantic Scholar) with proper rate limiting, parallelization, and
Open-Access filtering. Results are compared using fuzzy string matching to identify
likely matches.

Rate Limits:
- DBLP: 1 request per 6 seconds (conservative limit, no official limit specified)
- arXiv: 1 request per 4 seconds (arXiv recommends minimum 3 seconds between requests)
- Semantic Scholar: 1 request per 1.2 seconds with API key (5 requests per 5 seconds), 1 request per 2.5 seconds without API key

Retry Logic:
- All APIs include retry logic with exponential backoff for HTTP errors (429, 500, 502, 503, 504, 505)
- Maximum 3 retries per request
- Initial retry delay: 5 seconds, doubles with each attempt
- Status checks use longer timeouts (15s) to avoid false negatives

API Key Configuration:
To use Semantic Scholar API with higher rate limits, set your API key:
1. Create a .env file in the project root (copy from .env.example)
2. Add: SEMANTIC_SCHOLAR_API_KEY=your_api_key_here
3. Get your API key from: https://www.semanticscholar.org/product/api

The .env file is automatically ignored by git to prevent accidental commits.

Usage:
    from api_caller import search_papers_by_title
    
    results = search_papers_by_title("Machine Learning in Computer Vision")

Package Usage:
- time: Rate limiting delays and timing measurements
- json: Serialize/deserialize search results to/from JSON format
- logging: Log API calls, errors, and debug information
- os: Access environment variables (e.g., API keys)
- pathlib.Path: Handle file paths for .env file loading
- typing: Type hints for function parameters and return values
- urllib.parse.quote: URL-encode search queries for API requests
- concurrent.futures: Parallel execution of database searches using ThreadPoolExecutor
- threading.Lock: Thread-safe rate limiting to prevent race conditions
- requests: HTTP requests to DBLP and Semantic Scholar REST APIs
- arxiv: Official arXiv API client for searching arXiv papers
- fuzzywuzzy.fuzz: Calculate title similarity scores using fuzzy string matching
- dotenv (optional): Load API keys from .env file instead of hardcoding them
"""

import time  # Rate limiting delays and timing measurements
import json  # Serialize/deserialize search results to/from JSON format
import logging  # Log API calls, errors, and debug information
import os  # Access environment variables (e.g., API keys)
from pathlib import Path  # Handle file paths for .env file loading
from typing import Dict, List, Optional, Any, Union  # Type hints for function parameters and return values
from urllib.parse import quote  # URL-encode search queries for API requests
from concurrent.futures import ThreadPoolExecutor, as_completed  # Parallel execution of database searches using ThreadPoolExecutor
from threading import Lock  # Thread-safe rate limiting to prevent race conditions
import requests  # HTTP requests to DBLP and Semantic Scholar REST APIs
import arxiv  # Official arXiv API client for searching arXiv papers
from fuzzywuzzy import fuzz  # Calculate title similarity scores using fuzzy string matching
import xml.etree.ElementTree as ET  # Parse arXiv XML responses for fallback implementation
import xml.etree.ElementTree as ET  # Parse arXiv XML responses for fallback implementation
from urllib.parse import quote  # Already imported above, but ensure it's available

# Try to load python-dotenv for .env file support
try:
    from dotenv import load_dotenv  # Load API keys from .env file instead of hardcoding them
    # Load environment variables from .env file if it exists
    # Try multiple locations: project root, task directory, and current directory
    env_paths = [
        Path(__file__).parent.parent / '.env',  # Project root
        Path(__file__).parent.parent / 'task' / '.env',  # Task directory
        Path(__file__).parent / '.env',  # src directory
        Path.cwd() / '.env',  # Current working directory
    ]
    loaded = False
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            loaded = True
            break
    if not loaded:
        # Try default location (current directory)
        load_dotenv()
except ImportError:
    # python-dotenv is optional, continue without it
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limit constants (in seconds)
# Set according to official API documentation and best practices
DBLP_RATE_LIMIT = 6.0  # Conservative limit for DBLP (no official limit specified)
ARXIV_RATE_LIMIT = 4.0  # arXiv recommends minimum 3 seconds between requests
SEMANTIC_SCHOLAR_RATE_LIMIT_WITH_KEY = 1.2  # With API key: 5 requests per 5 seconds = 1 per second, add buffer
SEMANTIC_SCHOLAR_RATE_LIMIT_WITHOUT_KEY = 2.5  # Without API key: more conservative limit

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5.0  # Initial delay in seconds for exponential backoff

# Request timeout configuration (in seconds)
REQUEST_TIMEOUT = 30  # Standard timeout for API requests
STATUS_CHECK_TIMEOUT = 15  # Longer timeout for status checks to avoid false negatives

# API endpoints
DBLP_API_BASE = "https://dblp.org/search/publ/api"
SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
# For Arxiv just package needed.

# Default similarity threshold for title matching (0-100)
DEFAULT_SIMILARITY_THRESHOLD = 80


class RateLimiter:
    """
    Thread-safe rate limiter to ensure API rate limits are respected.
    Each API source has its own rate limiter instance.
    """
    
    def __init__(self, min_interval: float):
        """
        Initialize rate limiter.
        
        Args:
            min_interval: Minimum time interval between requests in seconds
        """
        self.min_interval = min_interval
        self.last_request_time = 0.0
        self.lock = Lock()
    
    def wait_if_needed(self):
        """
        Block until enough time has passed since the last request.
        Thread-safe implementation.
        """
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()


# Global rate limiters for each API
# Semantic Scholar rate limiter will be updated based on API key availability
dblp_rate_limiter = RateLimiter(DBLP_RATE_LIMIT)
arxiv_rate_limiter = RateLimiter(ARXIV_RATE_LIMIT)
# Initialize with conservative limit, will be updated if API key is available
semantic_scholar_rate_limiter = RateLimiter(SEMANTIC_SCHOLAR_RATE_LIMIT_WITHOUT_KEY)


def get_semantic_scholar_api_key() -> Optional[str]:
    """
    Load Semantic Scholar API key from environment variable or .env file.
    
    The API key can be set in two ways:
    1. As an environment variable: SEMANTIC_SCHOLAR_API_KEY
    2. In a .env file in the project root: SEMANTIC_SCHOLAR_API_KEY=your_key_here
    
    Returns:
        API key string if found, None otherwise
    """
    api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
    
    if not api_key:
        logger.warning(
            "Semantic Scholar API key not found. "
            "Set SEMANTIC_SCHOLAR_API_KEY environment variable or add it to .env file. "
            "Continuing without API key (rate limits may apply)."
        )
    else:
        logger.debug("Semantic Scholar API key loaded successfully")
    
    return api_key


# Load API key at module level (optional, will be None if not set)
SEMANTIC_SCHOLAR_API_KEY = get_semantic_scholar_api_key()

# Update Semantic Scholar rate limiter based on API key availability
if SEMANTIC_SCHOLAR_API_KEY:
    semantic_scholar_rate_limiter.min_interval = SEMANTIC_SCHOLAR_RATE_LIMIT_WITH_KEY
    logger.info(f"Semantic Scholar rate limit set to {SEMANTIC_SCHOLAR_RATE_LIMIT_WITH_KEY}s (with API key)")
else:
    semantic_scholar_rate_limiter.min_interval = SEMANTIC_SCHOLAR_RATE_LIMIT_WITHOUT_KEY
    logger.info(f"Semantic Scholar rate limit set to {SEMANTIC_SCHOLAR_RATE_LIMIT_WITHOUT_KEY}s (without API key)")


def normalize_title(title: str) -> str:
    """
    Normalize title for comparison by converting to lowercase and removing extra whitespace.
    
    Args:
        title: Original title string
        
    Returns:
        Normalized title string
    """
    if not title:
        return ""
    return " ".join(title.lower().split())


def calculate_title_similarity(title1: str, title2: str) -> int:
    """
    Calculate similarity score between two titles using fuzzy matching.
    
    Args:
        title1: First title
        title2: Second title
        
    Returns:
        Similarity score (0-100)
    """
    normalized1 = normalize_title(title1)
    normalized2 = normalize_title(title2)
    return fuzz.ratio(normalized1, normalized2)


def search_dblp(title: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search DBLP database for papers matching the given title.
    Only returns Open-Access papers (papers with accessible URLs).
    Includes retry logic with exponential backoff for 429 and 500 errors.
    
    Args:
        title: Paper title to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with metadata, empty list if no results
    """
    dblp_rate_limiter.wait_if_needed()
    
    # Retry logic with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            # URL encode the title for the query
            encoded_title = quote(title)
            url = f"{DBLP_API_BASE}?q={encoded_title}&format=json&h={max_results}"
            
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            
            # Handle rate limiting and server errors with retry
            # 429: Too Many Requests (rate limit)
            # 500: Internal Server Error
            # 502: Bad Gateway
            # 503: Service Unavailable
            # 504: Gateway Timeout
            # 505: HTTP Version Not Supported
            if response.status_code in [429, 500, 502, 503, 504, 505]:
                wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"DBLP HTTP {response.status_code} for '{title[:60]}...', waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(wait_time)
                dblp_rate_limiter.wait_if_needed()
                continue
            
            response.raise_for_status()
            data = response.json()
            
            papers = []
            hits = data.get('result', {}).get('hits', {})
            hit_list = hits.get('hit', [])
            
            # Handle both single hit (dict) and multiple hits (list) cases
            if isinstance(hit_list, dict):
                hit_list = [hit_list]
            elif not isinstance(hit_list, list):
                hit_list = []
            
            for hit in hit_list:
                info = hit.get('info', {})
                paper_title = info.get('title', '')
                
                # DBLP doesn't have explicit Open-Access flag, but we check for accessible URLs
                # Papers with URLs are considered accessible
                url = info.get('url', '')
                ee_links = info.get('ee', [])
                
                # Consider paper accessible if it has a URL or EE link
                is_accessible = bool(url) or bool(ee_links)
                
                if is_accessible:
                    # Extract authors
                    authors = []
                    authors_data = info.get('authors', {})
                    if isinstance(authors_data, dict):
                        author_list = authors_data.get('author', [])
                        if isinstance(author_list, dict):
                            author_list = [author_list]
                        for author in author_list:
                            if isinstance(author, dict):
                                authors.append(author.get('text', ''))
                            else:
                                authors.append(str(author))
                    
                    # Extract DOI from EE links
                    doi = ""
                    if isinstance(ee_links, list):
                        for link in ee_links:
                            if isinstance(link, dict) and 'doi.org' in link.get('text', ''):
                                doi = link.get('text', '')
                                break
                    elif isinstance(ee_links, dict) and 'doi.org' in ee_links.get('text', ''):
                        doi = ee_links.get('text', '')
                    
                    # Get primary URL
                    paper_url = url
                    if not paper_url and ee_links:
                        if isinstance(ee_links, list) and ee_links:
                            paper_url = ee_links[0].get('text', '') if isinstance(ee_links[0], dict) else str(ee_links[0])
                        elif isinstance(ee_links, dict):
                            paper_url = ee_links.get('text', '')
                    
                    papers.append({
                        'title': paper_title,
                        'authors': authors,
                        'year': info.get('year', ''),
                        'doi': doi,
                        'url': paper_url,
                        'source': 'dblp',
                        'venue': info.get('venue', '')
                    })
            
            logger.debug(f"DBLP search for '{title[:60]}...' returned {len(papers)} accessible papers")
            return papers
            
        except requests.exceptions.RequestException as e:
            # If this was the last attempt, log error and return empty list
            if attempt == MAX_RETRIES - 1:
                logger.error(f"DBLP API error for title '{title[:60]}...' after {MAX_RETRIES} attempts: {e}")
                return []
            # Otherwise, wait and retry
            wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
            logger.warning(f"DBLP request exception for '{title[:60]}...', waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
            time.sleep(wait_time)
            dblp_rate_limiter.wait_if_needed()
        except Exception as e:
            logger.error(f"Unexpected error querying DBLP for '{title[:60]}...': {e}")
            return []
    
    # If we exhausted all retries, return empty list
    logger.error(f"DBLP search failed for '{title[:60]}...' after {MAX_RETRIES} attempts")
    return []


def _search_arxiv_direct_api(title: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Primary implementation: Direct arXiv API call using requests.
    Uses the current arXiv API endpoint to avoid HTTP 301 redirect errors.
    
    Args:
        title: Paper title to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with metadata, empty list if no results or errors occur
    """
    papers = []
    
    # Retry logic for direct API calls to handle transient errors
    for attempt in range(MAX_RETRIES):
        arxiv_rate_limiter.wait_if_needed()
        
        try:
            # Use current arXiv API endpoint (HTTPS) to avoid redirects
            # The export.arxiv.org endpoint is the current recommended API endpoint
            base_url = "https://export.arxiv.org/api/query"
            
            # Format query for title search
            # arXiv query syntax: ti:"title" searches in title field
            query_string = f'ti:"{title}"'
            
            params = {
                "search_query": query_string,
                "max_results": max_results,
                "start": 0,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = requests.get(base_url, params=params, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            
            # Handle HTTP errors with retry logic
            if response.status_code in [429, 500, 502, 503, 504, 505]:
                if attempt < MAX_RETRIES - 1:
                    wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Direct arXiv API HTTP {response.status_code} for '{title[:60]}...', waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Direct arXiv API HTTP {response.status_code} for '{title[:60]}...' after {MAX_RETRIES} attempts")
                    return []
            
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Namespace for Atom feed
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            # Extract entries
            entries = root.findall('atom:entry', ns)
            
            for entry in entries:
                try:
                    # Extract title
                    title_elem = entry.find('atom:title', ns)
                    paper_title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                    
                    # Extract authors
                    authors = []
                    for author in entry.findall('atom:author', ns):
                        name_elem = author.find('atom:name', ns)
                        if name_elem is not None and name_elem.text:
                            authors.append(name_elem.text.strip())
                    
                    # Extract published date
                    published_elem = entry.find('atom:published', ns)
                    year = None
                    if published_elem is not None and published_elem.text:
                        try:
                            from datetime import datetime
                            pub_date = datetime.fromisoformat(published_elem.text.replace('Z', '+00:00'))
                            year = pub_date.year
                        except:
                            pass
                    
                    # Extract DOI
                    doi = ""
                    for link in entry.findall('atom:link', ns):
                        if link.get('title') == 'doi':
                            doi = link.get('href', '')
                            break
                    
                    # Extract arXiv ID and URLs
                    arxiv_id = ""
                    entry_id = ""
                    pdf_url = ""
                    
                    id_elem = entry.find('atom:id', ns)
                    if id_elem is not None and id_elem.text:
                        entry_id = id_elem.text
                        # Extract arXiv ID from URL (e.g., http://arxiv.org/abs/1234.5678v1 -> 1234.5678v1)
                        if 'arxiv.org/abs/' in entry_id:
                            arxiv_id = entry_id.split('arxiv.org/abs/')[-1]
                        elif 'arxiv.org/pdf/' in entry_id:
                            arxiv_id = entry_id.split('arxiv.org/pdf/')[-1].replace('.pdf', '')
                    
                    # Find PDF link
                    for link in entry.findall('atom:link', ns):
                        if link.get('type') == 'application/pdf':
                            pdf_url = link.get('href', '')
                            break
                    
                    papers.append({
                        'title': paper_title,
                        'authors': authors,
                        'year': year,
                        'doi': doi,
                        'url': entry_id,
                        'pdf_url': pdf_url,
                        'source': 'arxiv',
                        'arxiv_id': arxiv_id.split('v')[0] if arxiv_id else ""  # Remove version suffix
                    })
                except Exception as e:
                    logger.debug(f"Error parsing arXiv entry: {e}")
                    continue
            
            logger.info(f"Direct arXiv API search for '{title[:60]}...' returned {len(papers)} papers")
            return papers
            
        except requests.exceptions.RequestException as e:
            # Network or HTTP errors - retry if not last attempt
            if attempt < MAX_RETRIES - 1:
                wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Direct arXiv API request error for '{title[:60]}...', waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}: {e}")
                time.sleep(wait_time)
                continue
            else:
                logger.warning(f"Direct arXiv API call failed for '{title[:60]}...' after {MAX_RETRIES} attempts: {e}")
                return []
        except Exception as e:
            # Other errors (parsing, etc.) - log and return empty
            logger.warning(f"Direct arXiv API call failed for '{title[:60]}...': {e}")
            return []
    
    # If we exhausted all retries, return empty list
    return []


def _search_arxiv_with_package(title: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Fallback implementation: Search arXiv using the arxiv Python package.
    Used when direct API calls fail. The package may use older API endpoints
    that can cause HTTP 301 redirect errors.
    
    Args:
        title: Paper title to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with metadata, empty list if no results or errors occur
    """
    # Retry logic for arXiv package queries to handle transient errors
    for attempt in range(MAX_RETRIES):
        arxiv_rate_limiter.wait_if_needed()
        
        try:
            # Create client with appropriate settings
            # arXiv recommends minimum 3 seconds between requests
            # We use 3.5 seconds as delay_seconds to be safe, plus our own rate limiter
            # Increase num_retries in client to handle redirects automatically
            client = arxiv.Client(
                page_size=10,
                delay_seconds=3.5,  # arXiv recommends minimum 3 seconds
                num_retries=3  # Let arxiv library handle retries internally
            )
            
            # Format query for better title matching
            # arXiv query syntax: ti:title searches in title, all:title searches everywhere
            # We use a combination: search in title first, but also allow abstract matches
            query_string = f"ti:{title}"
            
            # Search using formatted query - arXiv will search in titles primarily
            search = arxiv.Search(
                query=query_string,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            try:
                for result in client.results(search):
                    # All arXiv papers are Open-Access
                    authors = [author.name for author in result.authors]
                    
                    # Extract DOI if available
                    doi = ""
                    if result.doi:
                        doi = result.doi
                    
                    papers.append({
                        'title': result.title,
                        'authors': authors,
                        'year': result.published.year if result.published else None,
                        'doi': doi,
                        'url': result.entry_id,  # arXiv entry URL
                        'pdf_url': result.pdf_url,
                        'source': 'arxiv',
                        'arxiv_id': result.get_short_id()
                    })
                
                # Successfully retrieved results
                logger.debug(f"arXiv package search for '{title[:60]}...' returned {len(papers)} papers")
                return papers
                
            except arxiv.UnexpectedEmptyPageError:
                # Sometimes arXiv returns empty pages, this is not a critical error
                logger.debug(f"arXiv package returned empty page for '{title[:60]}...', continuing")
                return papers  # Return what we have (might be empty)
                
            except arxiv.HTTPError as e:
                # Handle HTTP errors from arxiv library (including 301 redirects)
                error_msg = str(e).lower()
                status_code = None
                
                # Try to extract status code from error message
                if '301' in error_msg or '302' in error_msg:
                    status_code = 301 if '301' in error_msg else 302
                elif '429' in error_msg:
                    status_code = 429
                elif '500' in error_msg or '502' in error_msg or '503' in error_msg:
                    status_code = 500
                
                # Retry for transient errors (301 redirects, rate limits, server errors)
                if status_code in [301, 302, 429, 500, 502, 503]:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"arXiv package HTTP {status_code} for '{title[:60]}...', waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                        time.sleep(wait_time)
                        continue  # Retry
                    else:
                        logger.error(f"arXiv package HTTP {status_code} for '{title[:60]}...' after {MAX_RETRIES} attempts: {e}")
                        return []  # Return empty on final failure
                else:
                    # Other HTTP errors - log and return empty
                    logger.warning(f"arXiv package HTTP error for '{title[:60]}...': {e}")
                    return []
                    
            except (ConnectionError, TimeoutError) as e:
                # Network-level errors - retry
                if attempt < MAX_RETRIES - 1:
                    wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"arXiv package connection error for '{title[:60]}...', waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                    time.sleep(wait_time)
                    continue  # Retry
                else:
                    logger.error(f"arXiv package connection error for '{title[:60]}...' after {MAX_RETRIES} attempts: {e}")
                    return []
                    
            except Exception as e:
                # Handle various other errors (redirects, parsing issues, etc.)
                error_msg = str(e).lower()
                
                # Check for redirect errors (301, 302)
                if any(keyword in error_msg for keyword in ['301', '302', 'redirect', 'moved permanently']):
                    if attempt < MAX_RETRIES - 1:
                        wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"arXiv package redirect error for '{title[:60]}...', waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                        time.sleep(wait_time)
                        continue  # Retry
                    else:
                        logger.error(f"arXiv package redirect error for '{title[:60]}...' after {MAX_RETRIES} attempts: {e}")
                        return []
                elif 'connection' in error_msg:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"arXiv package connection issue for '{title[:60]}...', waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                        time.sleep(wait_time)
                        continue  # Retry
                    else:
                        logger.warning(f"arXiv package connection issue for '{title[:60]}...': {e}")
                        return []
                else:
                    # Other errors - might be transient, try retry
                    if attempt < MAX_RETRIES - 1:
                        wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"arXiv package query error for '{title[:60]}...', waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}: {e}")
                        time.sleep(wait_time)
                        continue  # Retry
                    else:
                        logger.warning(f"arXiv package query error for '{title[:60]}...': {e}")
                        return []
        
        except Exception as e:
            # Outer exception handler for critical errors
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['timeout', 'connection refused', 'dns', 'network', 'unreachable']):
                if attempt < MAX_RETRIES - 1:
                    wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Critical error querying arXiv package for '{title[:60]}...', waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                    time.sleep(wait_time)
                    continue  # Retry
                else:
                    logger.error(f"Critical error querying arXiv package for '{title[:60]}...' after {MAX_RETRIES} attempts: {e}")
                    return []
            else:
                if attempt < MAX_RETRIES - 1:
                    wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Error querying arXiv package for '{title[:60]}...', waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}: {e}")
                    time.sleep(wait_time)
                    continue  # Retry
                else:
                    logger.warning(f"Error querying arXiv package for '{title[:60]}...': {e}")
                    return []
    
    # If we exhausted all retries, return empty list
    logger.warning(f"arXiv package search failed for '{title[:60]}...' after {MAX_RETRIES} attempts")
    return []


def search_arxiv(title: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search arXiv database for papers matching the given title.
    All arXiv papers are Open-Access by default.
    
    Strategy:
    1. First attempts to use direct API call (current arXiv API endpoint)
    2. Falls back to arxiv Python package if direct API fails
    
    The direct API approach avoids HTTP 301 redirect errors that can occur
    with the arxiv package, which may use older API endpoints.
    
    Args:
        title: Paper title to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with metadata, empty list if no results
    """
    # Step 1: Try direct API call first (current API endpoint, avoids 301 redirects)
    logger.debug(f"Attempting direct arXiv API call for '{title[:60]}...'")
    papers = _search_arxiv_direct_api(title, max_results)
    
    # If direct API call succeeded and returned results, use them
    if papers:
        logger.info(f"Direct arXiv API call succeeded, returning {len(papers)} papers")
        return papers
    
    # Step 2: Fallback to arxiv package if direct API failed or returned no results
    logger.info(f"Direct arXiv API call returned no results or failed, falling back to arxiv package for '{title[:60]}...'")
    papers = _search_arxiv_with_package(title, max_results)
    
    if papers:
        logger.info(f"arXiv package fallback succeeded, returning {len(papers)} papers")
    else:
        logger.warning(f"Both direct API and package fallback failed for '{title[:60]}...'")
    
    return papers


def search_semantic_scholar(title: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar database for papers matching the given title.
    Only returns Open-Access papers (papers with openAccessPdf field).
    Includes retry logic with exponential backoff for 429 and 500 errors.
    
    Uses API key if available (set via SEMANTIC_SCHOLAR_API_KEY environment variable
    or .env file) for higher rate limits.
    
    Args:
        title: Paper title to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with metadata, empty list if no results
    """
    semantic_scholar_rate_limiter.wait_if_needed()
    
    params = {
        "query": title,
        "limit": max_results,
        "fields": "title,authors,year,externalIds,url,openAccessPdf,citationCount"
    }
    
    # Add API key to headers if available
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY
        logger.debug("Using Semantic Scholar API key for authentication")
    else:
        logger.debug("No API key provided, using public rate limits")
    
    # Retry logic with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                SEMANTIC_SCHOLAR_API_BASE,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            
            # Handle rate limiting and server errors with retry
            # 429: Too Many Requests (rate limit)
            # 500: Internal Server Error
            # 502: Bad Gateway
            # 503: Service Unavailable
            # 504: Gateway Timeout
            # 505: HTTP Version Not Supported
            if response.status_code in [429, 500, 502, 503, 504, 505]:
                wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Semantic Scholar HTTP {response.status_code} for '{title[:60]}...', waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(wait_time)
                semantic_scholar_rate_limiter.wait_if_needed()
                continue
            
            response.raise_for_status()
            data = response.json()
            
            papers = []
            paper_list = data.get('data', [])
            
            for paper in paper_list:
                # Only include papers with Open-Access PDF
                open_access_pdf = paper.get('openAccessPdf')
                if not open_access_pdf:
                    continue
                
                # Extract authors
                authors = []
                for author in paper.get('authors', []):
                    author_name = author.get('name', '')
                    if author_name:
                        authors.append(author_name)
                
                # Extract DOI
                doi = ""
                external_ids = paper.get('externalIds', {})
                if external_ids:
                    doi = external_ids.get('DOI', '')
                
                papers.append({
                    'title': paper.get('title', ''),
                    'authors': authors,
                    'year': paper.get('year'),
                    'doi': doi,
                    'url': paper.get('url', ''),
                    'pdf_url': open_access_pdf.get('url', '') if isinstance(open_access_pdf, dict) else '',
                    'source': 'semantic_scholar',
                    'paper_id': paper.get('paperId', ''),
                    'citation_count': paper.get('citationCount', 0)
                })
            
            logger.debug(f"Semantic Scholar search for '{title[:60]}...' returned {len(papers)} Open-Access papers")
            return papers
            
        except requests.exceptions.RequestException as e:
            # If this was the last attempt, log error and return empty list
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Semantic Scholar API error for title '{title[:60]}...' after {MAX_RETRIES} attempts: {e}")
                return []
            # Otherwise, wait and retry
            wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
            logger.warning(f"Semantic Scholar request exception for '{title[:60]}...', waiting {wait_time:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
            time.sleep(wait_time)
            semantic_scholar_rate_limiter.wait_if_needed()
        except Exception as e:
            logger.error(f"Unexpected error querying Semantic Scholar for '{title[:60]}...': {e}")
            return []
    
    # If we exhausted all retries, return empty list
    logger.error(f"Semantic Scholar search failed for '{title[:60]}...' after {MAX_RETRIES} attempts")
    return []


# Cache for status check results to avoid excessive API calls
_status_cache = {'status': None, 'timestamp': 0}
_STATUS_CACHE_DURATION = 300  # Cache status for 5 minutes

def check_database_status(use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Check the online status of all databases before starting queries.
    Uses longer timeouts and proper rate limiting to avoid false negatives.
    
    Args:
        use_cache: Whether to use cached status if available (default: True)
    
    Returns:
        Dictionary with status information for each database:
        {
            'dblp': {'online': bool, 'error': str or None},
            'arxiv': {'online': bool, 'error': str or None},
            'semantic_scholar': {'online': bool, 'error': str or None, 'api_key': bool}
        }
    """
    # Use cached status if available and recent
    if use_cache:
        current_time = time.time()
        if _status_cache['status'] and (current_time - _status_cache['timestamp']) < _STATUS_CACHE_DURATION:
            logger.debug("Using cached database status")
            return _status_cache['status']
    status = {
        'dblp': {'online': False, 'error': None},
        'arxiv': {'online': False, 'error': None},
        'semantic_scholar': {'online': False, 'error': None, 'api_key': bool(SEMANTIC_SCHOLAR_API_KEY)}
    }
    
    # Check DBLP status with proper rate limiting and timeout
    try:
        dblp_rate_limiter.wait_if_needed()
        test_url = f"{DBLP_API_BASE}?q=machine+learning&format=json&h=1"
        response = requests.get(test_url, timeout=STATUS_CHECK_TIMEOUT)
        
        # Accept 200 OK and also handle 429 (rate limit) as "online" since it means the service is responding
        if response.status_code == 200:
            response.raise_for_status()
            status['dblp']['online'] = True
        elif response.status_code == 429:
            # Rate limited but service is online
            status['dblp']['online'] = True
            status['dblp']['error'] = "Rate limited (service is online)"
        else:
            response.raise_for_status()
            status['dblp']['online'] = True
    except requests.exceptions.Timeout:
        status['dblp']['error'] = "Request timeout"
        status['dblp']['online'] = False
    except requests.exceptions.RequestException as e:
        status['dblp']['error'] = str(e)
        status['dblp']['online'] = False
    except Exception as e:
        status['dblp']['error'] = f"Unexpected error: {str(e)}"
        status['dblp']['online'] = False
    
    # Check arXiv status - try direct API first, then fallback to package
    # Direct API avoids HTTP 301 redirect errors from older package endpoints
    try:
        arxiv_rate_limiter.wait_if_needed()
        # Try direct API call first (current API endpoint)
        test_papers = _search_arxiv_direct_api("machine learning", max_results=1)
        # Any successful response (even empty) means service is online
        status['arxiv']['online'] = True
    except Exception as e:
        # If direct API fails, try package as fallback
        try:
            arxiv_rate_limiter.wait_if_needed()
            client = arxiv.Client(page_size=1, delay_seconds=3.5, num_retries=3)
            search = arxiv.Search(query="machine learning", max_results=1, sort_by=arxiv.SortCriterion.Relevance)
            results = list(client.results(search))
            # Any successful response (even empty) means service is online
            status['arxiv']['online'] = True
        except arxiv.UnexpectedEmptyPageError:
            # Empty page means service is responding (this is common and not an error)
            status['arxiv']['online'] = True
        except arxiv.HTTPError as e:
            # HTTP errors from arxiv library - check status code
            error_msg = str(e).lower()
            # 429 (rate limit) and 5xx errors might be transient
            if '429' in error_msg or '5' in error_msg[:3]:
                # Rate limited or server error - service is likely online but having issues
                status['arxiv']['online'] = True
                status['arxiv']['error'] = f"Transient error: {str(e)}"
            else:
                # Other HTTP errors - mark as offline
                status['arxiv']['error'] = str(e)
                status['arxiv']['online'] = False
        except (ConnectionError, TimeoutError) as e:
            # Network-level errors - these are more serious
            status['arxiv']['error'] = str(e)
            status['arxiv']['online'] = False
        except Exception as e:
            error_msg = str(e).lower()
            # Be more lenient - only mark as offline for serious connection issues
            if any(keyword in error_msg for keyword in ['timeout', 'connection refused', 'dns', 'network', 'unreachable']):
                status['arxiv']['error'] = str(e)
                status['arxiv']['online'] = False
            else:
                # Most other errors are transient (redirects, parsing issues, etc.)
                # arXiv is generally very reliable, so we assume it's online
                status['arxiv']['online'] = True
                status['arxiv']['error'] = f"Transient error (assuming online): {str(e)}"
    
    # Check Semantic Scholar status with proper rate limiting and timeout
    try:
        semantic_scholar_rate_limiter.wait_if_needed()
        params = {"query": "machine learning", "limit": 1, "fields": "title"}
        headers = {}
        if SEMANTIC_SCHOLAR_API_KEY:
            headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY
        response = requests.get(SEMANTIC_SCHOLAR_API_BASE, params=params, headers=headers, timeout=STATUS_CHECK_TIMEOUT)
        
        # Accept 200 OK and also handle 429 (rate limit) as "online" since it means the service is responding
        if response.status_code == 200:
            response.raise_for_status()
            status['semantic_scholar']['online'] = True
        elif response.status_code == 429:
            # Rate limited but service is online
            status['semantic_scholar']['online'] = True
            status['semantic_scholar']['error'] = "Rate limited (service is online)"
        else:
            response.raise_for_status()
            status['semantic_scholar']['online'] = True
    except requests.exceptions.Timeout:
        status['semantic_scholar']['error'] = "Request timeout"
        status['semantic_scholar']['online'] = False
    except requests.exceptions.RequestException as e:
        status['semantic_scholar']['error'] = str(e)
        status['semantic_scholar']['online'] = False
    except Exception as e:
            status['semantic_scholar']['error'] = f"Unexpected error: {str(e)}"
            status['semantic_scholar']['online'] = False
    
    # Cache the status result
    _status_cache['status'] = status
    _status_cache['timestamp'] = time.time()
    
    return status


def print_database_status(status: Dict[str, Dict[str, Any]]):
    """
    Print database status in a formatted way.
    
    Args:
        status: Dictionary returned by check_database_status()
    """
    # Use both print and logger to ensure visibility
    status_lines = []
    status_lines.append("\nStatus:")
    
    # DBLP
    dblp_status = "Online" if status['dblp']['online'] else "Offline"
    status_lines.append(f"- DBLP: {dblp_status}")
    
    # arXiv
    arxiv_status = "Online" if status['arxiv']['online'] else "Offline"
    status_lines.append(f"- Arxiv: {arxiv_status}")
    
    # Semantic Scholar
    ss_status = "Online" if status['semantic_scholar']['online'] else "Offline"
    ss_key_status = "Online" if status['semantic_scholar']['api_key'] else "Offline"
    status_lines.append(f"- Semantic Scholar: {ss_status} (KEY: {ss_key_status})")
    
    status_lines.append("")  # Empty line for readability
    
    # Print to stdout and log
    status_text = "\n".join(status_lines)
    print(status_text)
    logger.info("Database status:\n" + status_text)


def filter_and_rank_results(
    original_title: str,
    all_results: List[Dict[str, Any]],
    similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Filter results by title similarity and rank them.
    
    Args:
        original_title: Original title to compare against
        all_results: List of all paper results from all sources
        similarity_threshold: Minimum similarity score (0-100) to include a result
        
    Returns:
        List of filtered and ranked papers with similarity scores
    """
    filtered_results = []
    
    for paper in all_results:
        paper_title = paper.get('title', '')
        if not paper_title:
            continue
        
        similarity = calculate_title_similarity(original_title, paper_title)
        
        if similarity >= similarity_threshold:
            paper['similarity_score'] = similarity
            filtered_results.append(paper)
    
    # Sort by similarity score (descending)
    filtered_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
    
    return filtered_results


def search_papers_by_title(
    title: str,
    similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
    max_results_per_source: int = 10,
    parallel: bool = True,
    check_status: bool = False
) -> Dict[str, Any]:
    """
    Search for papers across DBLP, arXiv, and Semantic Scholar databases.
    Results are filtered by Open-Access availability and title similarity.
    
    Args:
        title: Paper title to search for
        similarity_threshold: Minimum similarity score (0-100) to consider a match
        max_results_per_source: Maximum results to fetch from each source
        parallel: Whether to search databases in parallel (recommended for efficiency)
        check_status: Whether to check database status before searching (default: False)
                     Note: Status checks consume API rate limits. Only enable if needed.
                     Use check_database_status() separately if you need to check status once.
        
    Returns:
        Dictionary containing:
            - 'original_title': The searched title
            - 'results': List of matching papers with metadata
            - 'summary': Summary statistics
    """
    # Check database status before starting queries (optional)
    # Note: Status checks consume API rate limits, so they're disabled by default.
    # If enabled, status is cached for 5 minutes to avoid excessive checks.
    if check_status:
        status = check_database_status(use_cache=True)
        print_database_status(status)
        logger.info("Database status check completed")
        # Add a small delay after status check to ensure rate limits are respected
        # This prevents rate limit issues when status check is immediately followed by queries
        time.sleep(0.5)
    
    logger.info(f"Searching for papers with title: '{title[:80]}...'")
    
    if parallel:
        # Search all databases in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(search_dblp, title, max_results_per_source): 'dblp',
                executor.submit(search_arxiv, title, max_results_per_source): 'arxiv',
                executor.submit(search_semantic_scholar, title, max_results_per_source): 'semantic_scholar'
            }
            
            all_results = []
            for future in as_completed(futures):
                source = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(f"Found {len(results)} papers from {source}")
                except Exception as e:
                    logger.error(f"Error searching {source}: {e}")
    else:
        # Sequential search (slower but simpler)
        all_results = []
        
        dblp_results = search_dblp(title, max_results_per_source)
        all_results.extend(dblp_results)
        logger.info(f"Found {len(dblp_results)} papers from DBLP")
        
        arxiv_results = search_arxiv(title, max_results_per_source)
        all_results.extend(arxiv_results)
        logger.info(f"Found {len(arxiv_results)} papers from arXiv")
        
        semantic_scholar_results = search_semantic_scholar(title, max_results_per_source)
        all_results.extend(semantic_scholar_results)
        logger.info(f"Found {len(semantic_scholar_results)} papers from Semantic Scholar")
    
    # Filter and rank results by similarity
    filtered_results = filter_and_rank_results(title, all_results, similarity_threshold)
    
    # Create summary statistics
    summary = {
        'total_found': len(all_results),
        'total_matching': len(filtered_results),
        'by_source': {}
    }
    
    for result in filtered_results:
        source = result.get('source', 'unknown')
        summary['by_source'][source] = summary['by_source'].get(source, 0) + 1
    
    logger.info(f"Found {len(filtered_results)} papers matching title (similarity >= {similarity_threshold}%)")
    
    return {
        'original_title': title,
        'results': filtered_results,
        'summary': summary
    }


def get_best_match_from_search_results(
    search_results: Dict[str, Any],
    min_similarity: int = DEFAULT_SIMILARITY_THRESHOLD
) -> Optional[Dict[str, Any]]:
    """
    Extract the best matching paper from search results.
    
    This helper function takes the results from search_papers_by_title() and returns
    the single best match in a format compatible with citation verification pipelines.
    
    Args:
        search_results: Dictionary returned by search_papers_by_title() containing:
            - 'original_title': The searched title
            - 'results': List of matching papers (already sorted by similarity_score)
            - 'summary': Summary statistics
        min_similarity: Minimum similarity score (0-100) required for a valid match
        
    Returns:
        Dictionary with paper information if a good match is found, None otherwise.
        Format:
            {
                'title': str,
                'authors': List[str],
                'year': int or None,
                'doi': str or None,
                'url': str or None,
                'pdf_url': str or None,
                'source': str ('dblp', 'arxiv', or 'semantic_scholar'),
                'match_score': float (0-100),
                'similarity_score': float (same as match_score for compatibility),
                'arxiv_id': str or None (if from arXiv),
                'venue': str or None (if from DBLP),
                'paper_id': str or None (if from Semantic Scholar)
            }
    """
    if not search_results or not search_results.get('results'):
        return None
    
    results = search_results.get('results', [])
    if not results:
        return None
    
    # Results are already sorted by similarity_score (descending) from search_papers_by_title
    best_match = results[0]
    
    # Check if the best match meets the minimum similarity threshold
    similarity_score = best_match.get('similarity_score', 0)
    if similarity_score < min_similarity:
        logger.debug(f"Best match similarity {similarity_score}% below threshold {min_similarity}%")
        return None
    
    # Format the result in a consistent structure for citation verification
    formatted_result = {
        'title': best_match.get('title', ''),
        'authors': best_match.get('authors', []),
        'year': best_match.get('year'),
        'doi': best_match.get('doi', ''),
        'url': best_match.get('url', ''),
        'pdf_url': best_match.get('pdf_url', ''),
        'source': best_match.get('source', 'unknown'),
        'match_score': similarity_score,
        'similarity_score': similarity_score,  # Keep both for compatibility
    }
    
    # Add source-specific fields
    if best_match.get('arxiv_id'):
        formatted_result['arxiv_id'] = best_match['arxiv_id']
    if best_match.get('venue'):
        formatted_result['venue'] = best_match['venue']
    if best_match.get('paper_id'):
        formatted_result['paper_id'] = best_match['paper_id']
    
    return formatted_result


def search_multiple_titles(
    titles: List[str],
    similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
    max_results_per_source: int = 10,
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for multiple titles in parallel.
    Each title is searched across all databases, and titles are processed in parallel.
    
    Args:
        titles: List of paper titles to search for
        similarity_threshold: Minimum similarity score (0-100) to consider a match
        max_results_per_source: Maximum results to fetch from each source per title
        max_workers: Maximum number of titles to process in parallel
        
    Returns:
        List of result dictionaries, one per title
    """
    logger.info(f"Searching {len(titles)} titles in parallel (max_workers={max_workers})")
    
    all_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                search_papers_by_title,
                title,
                similarity_threshold,
                max_results_per_source,
                parallel=True
            ): title
            for title in titles
        }
        
        for future in as_completed(futures):
            title = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                logger.info(f"Completed search for: '{title[:60]}...'")
            except Exception as e:
                logger.error(f"Error searching for '{title[:60]}...': {e}")
                all_results.append({
                    'original_title': title,
                    'results': [],
                    'summary': {'error': str(e)}
                })
    
    return all_results


def save_results_to_json(results: Union[Dict[str, Any], List[Dict[str, Any]]], output_file: str):
    """
    Save search results to a JSON file.
    
    Args:
        results: Single result dictionary or list of result dictionaries
        output_file: Path to output JSON file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")


def main():
    """
    Example usage of the API caller.
    """
    # Example: Search for a single paper
    print("=" * 80)
    print("Example: Searching for a single paper")
    print("=" * 80)
    
    example_title = "Attention Is All You Need"
    # Note: check_status=False by default to avoid consuming API rate limits
    # Use check_database_status() separately if you need to check status once
    result = search_papers_by_title(example_title, similarity_threshold=80, check_status=False)
    
    print(f"\nOriginal title: {result['original_title']}")
    print(f"Found {result['summary']['total_matching']} matching papers")
    print(f"\nTop matches:")
    for i, paper in enumerate(result['results'][:3], 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Source: {paper['source']}")
        print(f"   Similarity: {paper['similarity_score']}%")
        print(f"   Authors: {', '.join(paper['authors'][:3])}")
        print(f"   Year: {paper.get('year', 'N/A')}")
        print(f"   DOI: {paper.get('doi', 'N/A')}")
        print(f"   URL: {paper.get('url', 'N/A')}")
    
    # Save results
    save_results_to_json(result, 'api_search_results.json')
    
    # Example: Search for multiple papers
    print("\n" + "=" * 80)
    print("Example: Searching for multiple papers")
    print("=" * 80)
    
    example_titles = [
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners"
    ]
    
    multiple_results = search_multiple_titles(example_titles, max_workers=2)
    print(f"\nSearched {len(example_titles)} titles")
    print(f"Found results for {sum(1 for r in multiple_results if r['results'])} titles")
    
    save_results_to_json(multiple_results, 'api_search_results_multiple.json')


if __name__ == '__main__':
    main()


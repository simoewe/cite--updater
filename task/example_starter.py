"""
Example starter script for citation verification task.

This script demonstrates how to:
1. Load citations from JSON
2. Query arXiv API for a single citation
3. Compare authors

Students should extend this to handle all citations and use multiple APIs.
"""

import json
import arxiv
import time
import os
from typing import Dict, List, Optional
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher



def load_citations(json_file: str) -> List[Dict]:
    """Load citations from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)



def query_arxiv_by_title(title: str, max_results: int = 5) -> Optional[Dict]:
    """
    Query local arXiv metadata JSON for a paper by title similarity.

    Args:
        title: Paper title to search for
        max_results: (unused, for API compatibility)

    Returns:
        Dictionary with paper info if found, or a placeholder if not.
    """
    json_path = os.path.join(os.path.dirname(__file__), "arxiv_papers_metadata.json")
    
    if not os.path.exists(json_path):
        print(f"⚠️  Metadata file not found at: {json_path}")
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load arXiv metadata JSON: {e}")
        return None

    def normalize(s: str) -> str:
        return s.strip().lower().replace("-", " ").replace("_", " ")

    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, normalize(a), normalize(b)).ratio() # Dont use sequence matcher, use fuzzywuzzy

    best_match = None
    best_score = 0

    for paper_id, paper_data in metadata.items():
        score = similarity(title, paper_data.get("title", ""))
        if score > best_score:
            best_score = score
            best_match = paper_data

    # Good match threshold
    if best_match and best_score >= 0.80:
        return {
            "title": best_match.get("title"),
            "authors": best_match.get("authors", []),
            "year": best_match.get("year"),
            "arxiv_id": best_match.get("arxiv_id"),
            "conference": best_match.get("conference"),
            "file_path": best_match.get("file_path"),
            "match_score": round(best_score * 100, 1),
            "source": "local_arxiv_json"
        }

    # No good match → placeholder for DBLP / Semantic Scholar lookup
    # No good match → placeholder for DBLP / Semantic Scholar lookup
    return {
        "title": title,
        "authors": [],
        "year": None,
        "arxiv_id": None,
        "conference": None,
        "file_path": None,
        "match_score": round(best_score * 100, 1),
        "best_match_title": best_match["title"] if best_match else None,
        "best_match_score": round(best_score * 100, 1),
        "source": "placeholder_for_dblp_semantic_scholar"
    }



def compare_authors(original_authors: List[str], verified_authors: List[str]) -> Dict:
    """
    Compare original authors with verified authors based on first, middle, and last name similarity.
    - Order is ignored.
    - Minor formatting differences (punctuation, initials) tolerated.
    - Abbreviations like "M." or "M" for a full first name are acceptable.
    - Missing or extra middle names are treated as discrepancies.
    This function is defensive: it always returns a dict with 'match' and 'discrepancies'.
    """
    # Defensive defaults
    discrepancies = []
    matched_original = set()
    matched_verified = set()

    try:
        def normalize(name: str) -> str:
            if not isinstance(name, str):
                return ""
            return name.strip().lower().replace('.', '').replace(',', '')

        def split_name(full_name: str):
            """
            Splits a full name into (given_name_parts_list, last_name)
            Example:
                "M Pawan Kumar" -> (["m","pawan"], "kumar")
                "M. Kumar" -> (["m"], "kumar")
            """
            s = normalize(full_name)
            parts = s.split()
            if len(parts) == 0:
                return [], ""
            if len(parts) == 1:
                return [parts[0]], ""
            return parts[:-1], parts[-1]

        def name_similarity(a: str, b: str) -> float:
            # handle non-string gracefully
            a = a or ""
            b = b or ""
            return SequenceMatcher(None, a, b).ratio()  # Dont use sequence matcher, use fuzzywuzzy

        def is_initial(token: str) -> bool:
            return isinstance(token, str) and len(token) == 1

        def is_firstname_match(given_a: list, given_b: list) -> bool:
            """
            Check if the given-name parts (first + middle) match according to these rules:
            - First names must match closely (exact, via initial, or >=0.85 similarity)
            - Initials for the first name are acceptable
            - Missing or extra middle names count as discrepancy (per user requirement)
            """
            # ensure list types
            if not isinstance(given_a, list) or not isinstance(given_b, list):
                return False
            if not given_a or not given_b:
                return False

            first_a = given_a[0]
            first_b = given_b[0]

            # First name check
            def first_match_ok(a, b):
                if not a or not b:
                    return False
                if a == b:
                    return True
                if is_initial(a) and b.startswith(a):
                    return True
                if is_initial(b) and a.startswith(b):
                    return True
                return name_similarity(a, b) >= 0.85

            if not first_match_ok(first_a, first_b):
                return False

            # Middle name comparison
            middle_a = given_a[1:]
            middle_b = given_b[1:]

            # If one has middle names and the other doesn't -> discrepancy
            if (len(middle_a) > 0 and len(middle_b) == 0) or (len(middle_b) > 0 and len(middle_a) == 0):
                return False

            # If both have middle names -> lengths must match and each part pairwise similar
            if len(middle_a) > 0 and len(middle_b) > 0:
                if len(middle_a) != len(middle_b):
                    return False
                for ma, mb in zip(middle_a, middle_b):
                    if ma == mb:
                        continue
                    if is_initial(ma) and mb.startswith(ma):
                        continue
                    if is_initial(mb) and ma.startswith(mb):
                        continue
                    if name_similarity(ma, mb) >= 0.85:
                        continue
                    return False

            return True

        def is_name_match(original: str, verified: str) -> bool:
            """
            Combine last name check and given-name check.
            Last name must be very similar (>=0.9),
            given name (including middle names) checked via is_firstname_match().
            """
            o_given, o_last = split_name(original)
            v_given, v_last = split_name(verified)

            # Last name similarity required
            if not o_last and not v_last:
                last_ok = True
            else:
                last_ok = name_similarity(o_last, v_last) >= 0.9

            if not last_ok:
                return False

            return is_firstname_match(o_given, v_given)

        # --- Perform comparison ---
        for v_auth in verified_authors or []:
            try:
                # find exact matches according to is_name_match
                matches = [o_auth for o_auth in (original_authors or []) if is_name_match(o_auth, v_auth)]
            except Exception:
                matches = []

            if matches:
                matched_verified.add(v_auth)
                # mark first matched original as used
                matched_original.add(matches[0])
            else:
                # find best original by simple string similarity for reporting
                best_original = None
                best_score = 0.0
                for o_auth in (original_authors or []):
                    score = name_similarity(normalize(o_auth), normalize(v_auth))
                    if score > best_score:
                        best_score = score
                        best_original = o_auth

                if best_original:
                    discrepancies.append({
                        "type": "missing_author",
                        "details": f"Missing or mismatched author: {v_auth} - {best_original}",
                        "best_original": best_original,
                        "similarity": round(best_score, 3)
                    })
                else:
                    discrepancies.append({
                        "type": "missing_author",
                        "details": f"Missing or mismatched author: {v_auth}",
                        "best_original": None,
                        "similarity": 0.0
                    })

        # Optionally detect extra original authors that were not matched by any verified author
        # (if you want, you can include them; currently we only report missing verified authors)
        return {
            "match": len(discrepancies) == 0,
            "discrepancies": discrepancies
        }

    except Exception as e:
        # Fallback: ensure function always returns a dict and include the exception message for debugging
        return {
            "match": False,
            "discrepancies": [
                {
                    "type": "internal_error",
                    "details": f"compare_authors crashed: {repr(e)}"
                }
            ]
        }


    def is_name_match(original: str, verified: str) -> bool:
        """
        Combine last name check and given-name check.
        Last name must be very similar (≥0.9),
        given name (including middle names) checked via is_firstname_match().
        """
        o_given, o_last = split_name(original)
        v_given, v_last = split_name(verified)

        # Last name similarity required
        if not o_last and not v_last:
            last_ok = True
        else:
            last_ok = name_similarity(o_last, v_last) >= 0.9

        if not last_ok:
            return False

        # Check first + middle names
        return is_firstname_match(o_given, v_given)

    # --- Perform comparison ---
    for v_auth in verified_authors:
        if any(is_name_match(o_auth, v_auth) for o_auth in original_authors):
            matched_verified.add(v_auth)
        else:
            discrepancies.append({
                "type": "missing_author",
                "details": f"Missing or mismatched author: {v_auth}"
            })

    return {
        "match": len(discrepancies) == 0,
        "discrepancies": discrepancies
    }




def verify_citation(citation: Dict) -> Dict:
    """
    Verify a single citation using local arXiv metadata.
    Prints clean, readable verification status.
    """
    print(f"\nVerifying: {citation['title'][:60]}...")

    arxiv_result = query_arxiv_by_title(citation['title'])

    # No results → Simple error message
    if not arxiv_result or arxiv_result.get("source") == "placeholder_for_dblp_semantic_scholar":
        best = arxiv_result.get("best_match_title") if arxiv_result else "N/A"
        score = arxiv_result.get("best_match_score") if arxiv_result else 0
        print(f"❌ No good arXiv match found for: '{citation['title']}' (best match: '{best}', {score}%)")
        return {
            "status": "not_found",
            "message": f"No good arXiv match found ({score}%)",
            "best_title": best,
            "match_score": score
        }


    # Result → Author comparison
    comparison = compare_authors(
        citation.get("authors", []),
        arxiv_result["authors"]
    )

    result = {
        "status": "verified" if comparison["match"] else "discrepancy_found",
        "source": "arxiv",
        "verified_authors": arxiv_result["authors"],
        "match_score": arxiv_result["match_score"],
        "comparison": comparison
    }

    # If match was found
    if comparison["match"]:
        print(f"✅ Title found in arXiv for: \"{arxiv_result['title']}\"")
        print("  ✓ Authors match!")
    else:
        print(f"✅  Title found in arXiv for: \"{arxiv_result['title']}\"")
        print("  ⚠️  Discrepancies found:")
        for disc in comparison["discrepancies"]:
            print(f"     - {disc['details']}")

    return result



def main():
    """Main function - example usage."""
    # Load citations
    print("Loading citations...")
    citations = load_citations('citations.json')
    print(f"Loaded {len(citations)} citations")
    
    # Verify first n citations as an example
    number = 20
    print("\n" + "="*60)
    print(f"Verifying first {number} citations...")
    print("="*60)
    
    results = []
    for i, citation in enumerate(citations[:number], 1):
        print(f"\n[{i}/{number}]")
        result = verify_citation(citation)
        results.append({
            'original': citation,
            'verification': result
        })
        
        # Be respectful - add delay between API calls
        time.sleep(2)
    
    # Save results
    with open('example_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("Example complete! Results saved to 'example_results.json'")
    print("="*60)
    print("\nNext steps:")
    print("1. Extend this to process all citations")
    print("2. Add DBLP and Semantic Scholar API queries")
    print("3. Implement better author name matching")
    print("4. Add progress saving and resume functionality")


if __name__ == '__main__':
    main()


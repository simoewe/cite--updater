"""
Example starter script for citation verification task.

This script demonstrates how to:
1. Load citations from JSON
2. Query arXiv.json for a single citation
3. Compare authors

Students should extend this to handle all citations and use multiple APIs.
"""

import json
import arxiv
import time
import os
from typing import Dict, List, Optional
from fuzzywuzzy import fuzz
# Note: SequenceMatcher was replaced with fuzzywuzzy for better performance and accuracy



def load_citations(json_file: str) -> List[Dict]:
    """Load citations from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)



def query_arxiv_by_title(title: str, max_results: int = 5) -> Optional[Dict]:
    """
    Query local arXiv metadata JSON for a paper by title similarity.
    
    This function searches through a local JSON file containing arXiv paper metadata
    and finds the best matching paper based on title similarity using fuzzy string matching.

    Args:
        title: Paper title to search for
        max_results: (unused, kept for API compatibility with external APIs)

    Returns:
        Dictionary with paper info if found with match_score >= 80%, or a placeholder
        dictionary if no good match is found. The placeholder indicates that DBLP or
        Semantic Scholar should be queried next.
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
        return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

    best_match = None
    best_score = 0

    # Iterate through all papers in metadata to find the best title match
    for paper_id, paper_data in metadata.items():
        score = similarity(title, paper_data.get("title", ""))
        if score > best_score:
            best_score = score
            best_match = paper_data

    # Good match threshold: 80% similarity required for a valid match
    # This threshold balances between false positives and false negatives
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
    Compare original authors with verified authors with detailed analysis:
    1. Author order changes (same authors, different order)
    2. Missing/extra authors (different number or completely different people)
    3. Name changes (same person, different name variant)
    Middle name changes and initial expansions are ignored.
    """
    discrepancies = []
    try:
        def normalize(name: str) -> str:
            if not isinstance(name, str):
                return ""
            return name.strip().lower().replace('.', '').replace(',', '').replace('-', ' ')

        def split_name(full_name: str):
            """Split name into given names and last name"""
            s = normalize(full_name)
            parts = s.split()
            if len(parts) == 0:
                return [], ""
            if len(parts) == 1:
                return [parts[0]], ""
            return parts[:-1], parts[-1]

        def name_similarity(a: str, b: str) -> float:
            """Use fuzzywuzzy for better string matching"""
            a = a or ""
            b = b or ""
            return SequenceMatcher(None, a, b).ratio()

        def is_initial(token: str) -> bool:
            return isinstance(token, str) and len(token) == 1

        def is_nickname_or_short_form(name1: str, name2: str) -> bool:
            """Check if one name is a nickname/short form of another"""
            name1, name2 = name1.lower(), name2.lower()

            # If names are identical, they are not nickname/short form
            if name1 == name2:
                return False

            # Common nickname patterns
            if len(name1) >= 3 and len(name2) >= 3:
                # Check if shorter name is contained in longer name
                if name1 in name2 or name2 in name1:
                    return True
                # Check if one starts with the other (Nicolas -> Nic)
                if name1.startswith(name2) or name2.startswith(name1):
                    return True
            return False

        def are_middle_names_compatible(middle1: list, middle2: list) -> tuple[bool, str]:
            """
            Check if middle names are compatible (same person but different middle name representation)
            Returns: (are_compatible, match_type)
            """
            if len(middle1) == 0 and len(middle2) == 0:
                return True, "exact"

            if len(middle1) != len(middle2):
                # Different number of middle names -> definitely a middle name change
                return True, "initial_or_middlename"

            # Same number of middle names - check each one
            all_exact = True
            has_differences = False

            for m1, m2 in zip(middle1, middle2):
                if m1 == m2:
                    continue
                elif (is_initial(m1) and m2.startswith(m1)) or (is_initial(m2) and m1.startswith(m2)):
                    has_differences = True
                    all_exact = False
                elif name_similarity(m1, m2) >= 0.85:
                    has_differences = True
                    all_exact = False
                else:
                    # Significantly different middle names
                    has_differences = True
                    all_exact = False

            if all_exact:
                return True, "exact"
            elif has_differences:
                return True, "initial_or_middlename"
            else:
                return True, "initial_or_middlename"

        def are_same_person(name1: str, name2: str) -> tuple[bool, float, str]:
            """
            Determine if two names refer to the same person.
            Returns: (is_same_person, confidence_score, match_type)
            """
            if not name1 or not name2:
                return False, 0.0, "no_match"

            # Quick exact match
            if normalize(name1) == normalize(name2):
                return True, 1.0, "exact"

            given1, last1 = split_name(name1)
            given2, last2 = split_name(name2)

            # DEBUG: Print the split results for problematic cases
            debug_cases = ["quoc", "christopher", "yee"]

            #is_debug = any(case in name1.lower() or case in name2.lower() for case in debug_cases)
            is_debug = False

            # Last names must be very similar
            last_sim = name_similarity(last1, last2) if last1 and last2 else 0.0
            if last_sim < 0.88:
                return False, last_sim, "no_match"

            # Check first names
            if not given1 or not given2:
                return False, last_sim, "no_match"

            first1 = given1[0]
            first2 = given2[0]

            try:
                # Handle initial expansion for FIRST name (F → Filipe, E → Emmanuel)
                if is_initial(first1) and first2.startswith(first1):
                    return True, 0.98, "initial_or_middlename"

                if is_initial(first2) and first1.startswith(first2):
                    return True, 0.98, "initial_or_middlename"

                # Check for nickname/short form (Nicolas -> Nic) - but NOT initial expansions
                if is_nickname_or_short_form(first1, first2) and not (is_initial(first1) or is_initial(first2)):
                    return True, 0.92, "variant"

                # Check first name similarity
                first_sim = name_similarity(first1, first2)

                # Exact first name match - check middle names
                if first1 == first2:

                    middle1, middle2 = given1[1:], given2[1:]
                    if is_debug:
                        print(f"  middle1: {middle1}, middle2: {middle2}")
                        print(f"  len(middle1): {len(middle1)}, len(middle2): {len(middle2)}")

                    # Check middle name compatibility directly here
                    if len(middle1) == 0 and len(middle2) == 0:
                        # No middle names on either side
                        return True, 1.0, "exact"

                    elif len(middle1) != len(middle2):
                        # Different number of middle names -> middle name change
                        return True, 0.95, "initial_or_middlename"

                    else:
                        # Same number of middle names - check if they match
                        all_middle_exact = True
                        for i, (m1, m2) in enumerate(zip(middle1, middle2)):
                            if m1 != m2:
                                all_middle_exact = False
                                break

                        if all_middle_exact:
                            return True, 1.0, "exact"
                        else:
                            return True, 0.95, "initial_or_middlename"

                # High similarity first names (might be variants)
                elif first_sim >= 0.95:
                    return True, 0.90, "variant"
                elif first_sim >= 0.85:
                    return True, min(0.90, (first_sim + last_sim) / 2), "variant"

                # Medium similarity with very good last name (might be nicknames we didn't catch)
                elif first_sim >= 0.70 and last_sim >= 0.95:
                    return True, 0.80, "variant"

                # Special case: completely different first names but same last name
                elif last_sim >= 0.95 and first_sim < 0.50:
                    return True, 0.75, "variant"

                return False, (first_sim + last_sim) / 2, "no_match"

            except Exception as e:
                if is_debug:
                    print(f"  ERROR in are_same_person: {e}")
                    import traceback
                    traceback.print_exc()
                return False, 0.0, "no_match"




        """
        ============================================================
        Main comparison logic starts here
        ============================================================
        The comparison is done in three phases:
        1. Build similarity matrix between all author pairs
        2. Find optimal matching between original and verified authors
        3. Classify discrepancies (order changes, name changes, missing/extra authors)
        """

        # Phase 1: Build similarity matrix
        # Compare every original author with every verified author
        # to determine if they refer to the same person
        original_authors = original_authors or []
        verified_authors = verified_authors or []

        # Create similarity matrix for all author pairs
        # Each cell contains match information (same person? confidence? match type?)
        similarity_matrix = []
        match_info = []
        for i, orig in enumerate(original_authors):
            row = []
            match_row = []
            for j, verif in enumerate(verified_authors):
                is_same, confidence, match_type = are_same_person(orig, verif)
                row.append(confidence)
                match_row.append({
                    "is_same": is_same,
                    "confidence": confidence,
                    "match_type": match_type
                })
            similarity_matrix.append(row)
            match_info.append(match_row)

        # Phase 2: Find best overall matching using greedy algorithm
        # Match each original author with at most one verified author
        # (and vice versa) to find the best overall assignment
        original_matched = set()  # Track which original authors are matched
        verified_matched = set()  # Track which verified authors are matched
        best_matches = []

        # Create list of all possible matches with their confidence
        all_matches = []
        for i, orig in enumerate(original_authors):
            for j, verif in enumerate(verified_authors):
                if match_info[i][j]["is_same"]:
                    all_matches.append((i, j, match_info[i][j]["confidence"], match_info[i][j]["match_type"]))

        # Sort by confidence (highest first) and assign matches using greedy approach
        # This ensures the best matches are made first
        all_matches.sort(key=lambda x: x[2], reverse=True)
        for orig_idx, verif_idx, confidence, match_type in all_matches:
            # Only match if neither author is already matched
            if orig_idx not in original_matched and verif_idx not in verified_matched:
                original_matched.add(orig_idx)
                verified_matched.add(verif_idx)
                best_matches.append({
                    "original_idx": orig_idx,
                    "verified_idx": verif_idx,
                    "original_name": original_authors[orig_idx],
                    "verified_name": verified_authors[verif_idx],
                    "confidence": confidence,
                    "match_type": match_type
                })

        # Phase 3: Classify discrepancies
        # Note: Initial/middle name changes are ignored (not considered discrepancies)
        # Check for order changes (ignore initial/middlename-only changes)
        order_changes = []
        for match in best_matches:
            if (match["original_idx"] != match["verified_idx"] and
                    match["match_type"] not in ["initial_or_middlename"]):
                order_changes.append({
                    "type": "order_change",
                    "details": f"'{match['original_name']}' moved from position {match['original_idx'] + 1} to position {match['verified_idx'] + 1}",
                    "original_position": match["original_idx"] + 1,
                    "verified_position": match["verified_idx"] + 1,
                    "author_name": match["original_name"],
                    "verified_name": match["verified_name"]
                })

        # Check for name changes (but IGNORE initial/middlename-only changes)
        name_changes = []
        for match in best_matches:
            if (match["match_type"] == "variant" and
                    match["original_name"] != match["verified_name"]):
                name_changes.append({
                    "type": "name_change",
                    "details": f"Name changed: '{match['original_name']}' → '{match['verified_name']}' (position {match['verified_idx'] + 1})",
                    "original_name": match["original_name"],
                    "verified_name": match["verified_name"],
                    "position": match["verified_idx"] + 1,
                    "confidence": match["confidence"],
                    "change_type": match["match_type"]
                })

        # Check for missing/extra authors
        # These are authors that appear in one list but not in the other
        missing_extra = []

        # Missing authors: in original list but not matched to any verified author
        for i, orig_author in enumerate(original_authors):
            if i not in original_matched:
                missing_extra.append({
                    "type": "missing_author",
                    "details": f"Author missing from verified list: '{orig_author}' (was at position {i + 1})",
                    "author_name": orig_author,
                    "original_position": i + 1
                })

        # Extra authors: in verified list but not matched to any original author
        for j, verif_author in enumerate(verified_authors):
            if j not in verified_matched:
                missing_extra.append({
                    "type": "extra_author",
                    "details": f"Extra author in verified list: '{verif_author}' (at position {j + 1})",
                    "author_name": verif_author,
                    "verified_position": j + 1
                })

        # Combine all discrepancies into a single list
        discrepancies = order_changes + name_changes + missing_extra

        # Determine overall match status
        has_any_discrepancy = len(discrepancies) > 0

        return {
            "match": not has_any_discrepancy,
            "discrepancies": discrepancies,
            "best_matches": best_matches
        }

    except Exception as e:
        return {
            "match": False,
            "discrepancies": [
                {
                    "type": "internal_error",
                    "details": f"compare_authors crashed: {repr(e)}"
                }
            ],
            "order_changes": 0,
            "name_changes": 0,
            "missing_extra": 0,
            "matched_count": 0,
            "total_original": len(original_authors or []),
            "total_verified": len(verified_authors or [])
        }

def verify_citation(citation: Dict) -> Dict:
    """
    Verify a single citation using local arXiv metadata.
    
    This function takes a citation dictionary, searches for the paper in the local
    arXiv metadata, and compares the authors. It returns a detailed verification
    result including any discrepancies found.
    
    Args:
        citation: Dictionary containing 'title' and 'authors' fields
        
    Returns:
        Dictionary with verification status, matched paper info, and author comparison results.
        Status can be: 'verified', 'discrepancy_found', or 'not_found'
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
        "title": arxiv_result["title"],
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
    """
    Main function - example usage.
    
    Loads citations from JSON file, verifies them against local arXiv metadata,
    and saves the results to a JSON file. Currently processes first 100 citations
    as an example (configurable via 'number' variable).
    """
    # Load citations from JSON file
    print("Loading citations...")
    citations = load_citations('citations.json')
    print(f"Loaded {len(citations)} citations")
   
    # Verify first n citations as an example
    # Change this number to process more or fewer citations
    number = 100
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
       
        # Optional: Add delay between API calls if using external APIs
        # Currently disabled since we're using local metadata
        # time.sleep(2)
   
    # Save results to JSON file for further analysis
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


import json
from typing import List, Dict, Set, Optional
from difflib import get_close_matches

def detect_parsing_error(name: Dict[str, str], paper_title: str) -> Optional[str]:
    """
    Check for common parsing errors in author names.
    Include paper title for context in error messages.
    """
    orig = name['original'].lower()
    parsed = f"{name['first_name']} {name['last_name']}".lower()
    
    # Check for obviously wrong name splits
    if name['first_name'].lower() == name['last_name'].lower():
        return f"Duplicate first/last name: {name['original']}"
    
    # Check if parsed name differs significantly from original
    if orig != parsed and orig != parsed.strip():
        return f"Name parsing mismatch: original '{name['original']}' parsed as '{name['first_name']} {name['last_name']}'"
    
    return None

def normalize_name(name: Dict[str, str]) -> str:
    """Convert author dict to normalized string for comparison using original field."""
    return name['original'].lower()

def find_closest_match(name: str, name_list: List[str]) -> Optional[tuple[str, float]]:
    """Find closest matching name from a list using difflib."""
    matches = get_close_matches(name, name_list, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return None

def get_initials(name: str) -> str:
    """Extract initials from a name string, handling multiple words."""
    initials = []
    for word in name.split():
        cleaned = word.replace('.', '').strip()
        if cleaned:
            initials.append(cleaned[0].lower())
    return ''.join(initials)

def is_initial(name: str) -> bool:
    """Check if a name component is an initial (single letter possibly with period)."""
    return len(name.replace('.', '').strip()) == 1

def normalize_compound_name(name: str) -> str:
    """Normalize compound names by removing spaces and hyphens."""
    return ''.join(c.lower() for c in name if c.isalnum())

def is_name_match(name1: Dict[str, str], name2: Dict[str, str]) -> bool:
    """
    Enhanced name matching that handles:
    - Initials (D. == David)
    - Reversed names (Maria Edoardo == Edoardo Maria)
    - Middle names
    - Combined first names
    """
    def normalize_component(text: str) -> str:
        return text.lower().replace('.', '').strip()
    
    # Get all name components
    n1_first = normalize_component(name1['first_name'])
    n1_middle = normalize_component(name1['middle_name'])
    n1_last = normalize_component(name1['last_name'])
    n2_first = normalize_component(name2['first_name'])
    n2_middle = normalize_component(name2['middle_name'])
    n2_last = normalize_component(name2['last_name'])
    
    # Combine all parts for each name
    n1_parts = [p for p in [n1_first, n1_middle, n1_last] if p]
    n2_parts = [p for p in [n2_first, n2_middle, n2_last] if p]
    
    # Check for exact last name match
    last_match = n1_last == n2_last
    
    # Handle initials in first/middle names
    def initial_matches(name1: str, name2: str) -> bool:
        """Check if names match by initials, handling multiple words."""
        init1 = get_initials(name1)
        init2 = get_initials(name2)
        return (init1 and init2 and 
                (init1.startswith(init2) or init2.startswith(init1)))
    
    # Improve reversed name detection
    def get_all_name_parts(name: Dict[str, str]) -> List[str]:
        parts = []
        if name['first_name']: parts.extend(name['first_name'].lower().split())
        if name['middle_name']: parts.extend(name['middle_name'].lower().split())
        if name['last_name']: parts.extend(name['last_name'].lower().split())
        return parts
    
    # Check if all parts match regardless of order
    n1_all_parts = set(get_all_name_parts(name1))
    n2_all_parts = set(get_all_name_parts(name2))
    if n1_all_parts and n2_all_parts and n1_all_parts == n2_all_parts:
        return True
    
    # Check various name matching scenarios
    if last_match:
        # Case 1: Direct first name match
        if n1_first == n2_first:
            return True
            
        # Case 2: Initial matches full name
        if is_initial(n1_first) or is_initial(n2_first):
            if initial_matches(n1_first, n2_first):
                return True
                
        # Case 3: Reversed first/middle names
        if n1_middle and n2_first == n1_middle:
            return True
        if n2_middle and n1_first == n2_middle:
            return True
            
        # Case 4: One name is part of the other
        if n1_first in n2_first or n2_first in n1_first:
            return True
    
    # Check reversed full names
    reversed_match = (n1_first == n2_last and n1_last == n2_first)
    if reversed_match:
        return True
        
    # Handle special case of multiple first names split differently
    full_name1 = ' '.join(n1_parts)
    full_name2 = ' '.join(n2_parts)
    if full_name1 == full_name2:
        return True
        
    # Add compound name handling
    n1_compound = normalize_compound_name(f"{n1_first} {n1_middle} {n1_last}")
    n2_compound = normalize_compound_name(f"{n2_first} {n2_middle} {n2_last}")
    if n1_compound == n2_compound:
        return True
    
    return False

def check_author_lists(parsed_authors: List[Dict[str, str]], 
                      matched_authors: List[Dict[str, str]],
                      paper_title: str) -> List[str]:
    """
    Check author lists with enhanced name matching.
    """
    mismatches = []
    
    # First check for parsing errors
    parsing_errors = []
    for author in parsed_authors:
        if error := detect_parsing_error(author, paper_title):
            parsing_errors.append(error)
    for author in matched_authors:
        if error := detect_parsing_error(author, paper_title):
            parsing_errors.append(error)
            
    if parsing_errors:
        mismatches.append("Parsing errors detected:")
        mismatches.extend(f"  {error}" for error in parsing_errors)
    
    # Try to match each parsed author with a matched author
    unmatched_parsed = []
    unmatched_matched = list(matched_authors)
    
    for parsed_author in parsed_authors:
        found_match = False
        for matched_author in unmatched_matched:
            if is_name_match(parsed_author, matched_author):
                unmatched_matched.remove(matched_author)
                found_match = True
                break
        if not found_match:
            unmatched_parsed.append(parsed_author)
    
    if unmatched_parsed or unmatched_matched:
        mismatches.append("Author lists differ:")
        
        for author in unmatched_parsed:
            closest = None
            match_reason = None
            
            for matched_author in unmatched_matched:
                # Try different matching strategies
                if is_name_match(author, matched_author):
                    closest = matched_author
                    match_reason = "names match with different ordering"
                    break
                elif initial_matches(author['first_name'], matched_author['first_name']):
                    closest = matched_author
                    match_reason = "matching initials"
                    break
            
            if closest:
                mismatches.append(f"  Parsed: {author['original']} ≈ Matched: {closest['original']} ({match_reason})")
                unmatched_matched.remove(closest)
            else:
                mismatches.append(f"  Only in parsed: {author['original']}")
        
        # Report remaining unmatched matched authors
        for author in unmatched_matched:
            mismatches.append(f"  Only in matched: {author['original']}")
    
    return mismatches

# Load the existing JSON file
with open('author_matches.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Process each entry
for entry in data:
    if entry['matched_authors']:
        entry['mismatches'] = check_author_lists(entry['parsed_authors'], 
                                               entry['matched_authors'],
                                               entry['title'])

# Write back the updated JSON
with open('author_matches_with_mismatches.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Print summary of mismatches found
print("\nMismatches found:")
for entry in data:
    if entry.get('mismatches'):
        print(f"\nTitle: {entry['title']}")
        for mismatch in entry['mismatches']:
            print(f"  {mismatch}") 
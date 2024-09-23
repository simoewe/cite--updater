import xml.etree.ElementTree as ET
import requests
import pandas as pd

from urllib.parse import urlencode
from importlib.resources import open_binary

BASE_URL = 'https://dblp.org/search/publ/api'


def add_ccf_class(results: list[dict]) -> list[dict]:
    def get_ccf_class(venue: str | None, catalog: pd.DataFrame) -> str | None:
        if venue is None:
            return
        if len(series := catalog.loc[catalog.get('abbr').str.lower() == venue.lower(), 'class']) > 0:
            return series.item()
        elif len(series := catalog.loc[catalog.get('url').str.contains(f'/{venue.lower()}/'), 'class']) > 0:
            return series.item()

    catalog = pd.read_csv(open_binary('dblp.data', 'ccf_catalog.csv'))
    for result in results:
        result['ccf_class'] = get_ccf_class(result.get('venue'), catalog=catalog)
    return results


def search(queries: list[str]) -> list[dict]:
    results = []
    for query in queries:
        entry = {
            'query': query,
            'title': None,
            'authors': [],
            'year': None,
            'venue': None,
            'doi': None,
            'url': None,
            'bibtex': None,
        }
        options = {
            'q': query,
            'format': 'json',
            'h': 1
        }
        response = requests.get(f'{BASE_URL}?{urlencode(options)}')
        r = response.json()
        hits = r.get('result', {}).get('hits', {}).get('hit', [])
        if hits:
            info = hits[0].get('info', {})
            entry['title'] = info.get('title')
            entry['year'] = info.get('year')
            entry['venue'] = info.get('venue')
            entry['doi'] = info.get('doi')
            entry['url'] = info.get('ee')
            entry['bibtex'] = f'{info.get("url")}?view=bibtex'
            authors = info.get('authors', {}).get('author', [])
            entry['authors'] = [author['text'] for author in authors if 'text' in author]
        results.append(entry)
    return results

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Define the namespace map
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    publications = {}
    
    # Use the namespace in the XPath
    for biblStruct in root.findall('.//tei:biblStruct', ns):
        title = biblStruct.find('.//tei:title[@type="main"]', ns)
        title_text = title.text if title is not None else "No title available"
        
        authors = []
        for author in biblStruct.findall('.//tei:author/tei:persName', ns):
            forename = author.find('tei:forename', ns).text if author.find('tei:forename', ns) is not None else ""
            surname = author.find('tei:surname', ns).text if author.find('tei:surname', ns) is not None else ""
            authors.append(f"{forename} {surname}".strip())
        
        publications[title_text] = authors
    
    return publications

# Example usage:
file_path_acl = 'pdfs/2020.acl-main.648.grobid.tei.xml'
file_path_lrec = 'pdfs/2024.lrec-main.320.grobid.tei.xml'

publications_acl = parse_xml(file_path_acl)
publications_lrec = parse_xml(file_path_lrec)

# print(publications_acl)
# print(publications_lrec)

queries = [
    'Anomaly Detection in Streams with Extreme Value Theory'
]

results = search(queries)

print(results)
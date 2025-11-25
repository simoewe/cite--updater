Query dblp -> query arxiv (delay 3 seconds for query) -> query semanticscholar (1 seconds, key login)
Author order is important!!
ignore middlename, but simon marlo != Simon laatz and Tsui-Wei Weng - Lily Weng is discr.

nd by pranav (name_parsing) fetching out first, last name ... https://nameparser.readthedocs.io/en/latest/



Until next meeting:
1. Query via API
2. Refine the author-matching (nameparser)


Detect parsing errros:
1. implement parsing error as discrepanzy -> if names mixed up together - last name is someone elses first name
- Author missing from verified list: 'Y R Fei' (was at position 3)
- Extra author in verified list: 'Yun' (at position 3)
- Ddflow

2. Integrate simons api_call.py

3. Integrate Pranavs code via. AI. prompt cursor read files in src folder and based on that make the changes in my code


Conference / Workshop:
workshop in bremen: june
workshop in palma de mallorca: july
konvens: september
 
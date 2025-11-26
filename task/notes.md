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
2.1. Add Fork as Remote: git remote add simoewe https://github.com/simoewe/cite--updater.git
2.2. Fetch (Retrieve) Simons Fork: git fetch simoewe
2.3. Get the api_caller.py from the Simons fork into my current branch: git checkout simoewe/main -- src/api_caller.py


3. Integrate Pranavs code via AI. prompt cursor read files in src folder and based on that make the changes in my code

Modify, extend, or remove code only in the files @example_starter.py  and @api_caller.py . You may use functions from the rest of the repository, but you are not allowed to change any other files. Additionally, create a new file named "Main_Pipeline.py" inside the task folder, which will serve as the new central execution file for all necessary components of the repository. The execution flow should follow the general structure of @example_starter.py  , but the new Main_Pipeline.py must not use any of the original API calls. Instead, it must fully integrate the new approach from @api_caller.py  for searching and communicating with databases. If essential functionality is still missing in @api_caller.py  , extend it appropriately and consistently. The compare_authors() function in @example_starter.py  is almost correct but must be improved to reliably detect parsing errors; for example, a string like “ddflow” must not be incorrectly assigned to an author. Also check whether a function already exists in the @src  folder that handles such cases and integrate or extend it accordingly. After these modifications, Main_Pipeline.py should act as the new entry point, the logic should be modernized, all old API calls removed, and all database interactions should be handled exclusively through the updated structure in @api_caller.py .

Conference / Workshop:
workshop in bremen: june
workshop in palma de mallorca: july
konvens: september
 
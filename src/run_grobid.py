from grobid_client.grobid_client import GrobidClient

client = GrobidClient(config_path="./config.json")
# Process references in pdfs folder, saving the output in the output folder
client.process('processReferences', 'pdfs', output='output', consolidate_citations=False, verbose=True)
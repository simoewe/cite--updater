from grobid_client.grobid_client import GrobidClient

client = GrobidClient(config_path="./config/config.json")
# Process references in pdfs folder, saving the output in the output folder
client.process('processHeaderDocument', 'data/pdfs/naacl', output='data/outputs/naacl', consolidate_header=True, verbose=True)

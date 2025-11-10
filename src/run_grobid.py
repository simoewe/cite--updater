from grobid_client.grobid_client import GrobidClient

# Initialize GROBID client with configuration file
client = GrobidClient(config_path="./config/config.json")

# Process PDFs from arxiv_pdfs folder, preserving directory structure in output
# Input: data/arxiv_pdfs (contains conference/year subdirectories)
# Output: data/outputs/arxiv_pdfs (will mirror the input directory structure)
client.process("processFulltextDocument", 'data/arxiv_pdfs', output='data/outputs/arxiv_pdfs', verbose=True, json_output=True)

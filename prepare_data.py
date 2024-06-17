import os
import shutil
import yaml
import logging
import pandas as pd
from pathlib import Path
from jsonargparse import CLI
from docx.api import Document
from types import SimpleNamespace
from llama_index.core import SimpleDirectoryReader
from utils.process_tables import extract_and_replace_docx_tables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("script.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(file_path='config.yaml'):
    logger.info('Loading config file ...')
    try:
        with open(file_path, 'r') as file:
            cfg = yaml.safe_load(file)
        for k, v in cfg.items():
            if isinstance(v, dict):
                cfg[k] = SimpleNamespace(**v)
        logger.info('Config file loaded successfully.')
        return SimpleNamespace(**cfg)
    except Exception as e:
        logger.error(f'Error loading config file: {e}')
        raise

cfg = load_config()

def process_docx_files(data_dir=Path(cfg.dataset.data_dir), 
                       processed_data_dir=Path(cfg.dataset.processed_data_dir),
                       chunk_marker=cfg.dataset.chunk_marker):
    try:
        if not os.path.exists(processed_data_dir):
            shutil.rmtree(processed_data_dir)

        docx_files = [file for file in os.listdir(data_dir) if file.endswith('.docx')]
        logger.info(f'Found {len(docx_files)} DOCX files to process.')

        for fname in docx_files:
            document, html_chunked_tables = extract_and_replace_docx_tables(
                docx_file=data_dir / fname, 
                chunk_marker=chunk_marker
            )
            document.save(processed_data_dir / f'processed_{fname}')
            logger.info(f'Processed and saved {fname}')
    except Exception as e:
        logger.error(f'Error processing DOCX files: {e}')
        raise

def load_processed_data(processed_data_dir=Path(cfg.dataset.processed_data_dir)):
    try:
        documents = SimpleDirectoryReader(
            input_dir=processed_data_dir,
            required_exts=[cfg.dataset.required_exts],
        ).load_data()
        logger.info('Processed data loaded successfully.')
        return documents
    except Exception as e:
        logger.error(f'Error loading processed data: {e}')
        raise

def get_chunks(documents, chunk_marker=cfg.dataset.chunk_marker):
    try:
        chunks = [chunk.strip() for doc in documents for chunk in doc.text.split(chunk_marker) if chunk.strip()]
        logger.info(f'Extracted {len(chunks)} chunks from documents.')
        return chunks
    except Exception as e:
        logger.error(f'Error extracting chunks: {e}')
        raise

def main():
    logger.info('Starting document processing ...')
    try:
        process_docx_files()

        documents = load_processed_data()
        chunks = get_chunks(documents)
        num_chunks = len(chunks)
        logger.info(f'Total number of chunks: {num_chunks}')

        df_chunks = pd.DataFrame({'chunk': chunks})
        df_chunks.to_pickle('processed_chunks.pickle')
        logger.info('All chunks saved to processed_chunks.pickle')
    except Exception as e:
        logger.error(f'Error in main processing: {e}')
        raise

if __name__ == '__main__':
    main()
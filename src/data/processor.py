"""
Clinical Text Preprocessing for Topic Modeling
==============================================

This module handles the preprocessing of clinical notes from MIMIC-III dataset.
Includes text cleaning, tokenization, and preparation for topic modeling.
"""

import pandas as pd
import re
import spacy
import pickle
from typing import List, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalTextProcessor:
    """
    Processes clinical text data for topic modeling.
    
    Handles MIMIC-III clinical notes with healthcare-specific preprocessing:
    - Removes PHI placeholders (e.g., [**2183-9-25**])
    - Clinical text normalization
    - Medical terminology preservation
    """
    
    def __init__(self, config: dict):
        """
        Initialize the clinical text processor.
        
        Args:
            config (dict): Configuration parameters from config.yaml
        """
        self.config = config
        self.nlp = None
        self._load_spacy_model()
        
    def _load_spacy_model(self):
        """Load spaCy model for text processing."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model successfully")
        except OSError:
            logger.error("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            raise
    
    def clean_clinical_text(self, text: str) -> str:
        """
        Clean clinical text with healthcare-specific preprocessing.
        
        Args:
            text (str): Raw clinical text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove PHI placeholders common in MIMIC-III
        text = re.sub(r'\[\*\*[^\]]*\*\*\]', '', text)
        
        # Remove special clinical formatting
        text = re.sub(r'_+', ' ', text)  # Replace underscores
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        
        # Remove non-alphabetical characters but preserve medical terms
        if self.config['preprocessing']['remove_punct']:
            text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        if self.config['preprocessing']['lowercase']:
            text = text.lower()
            
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize clinical text.
        
        Args:
            text (str): Cleaned clinical text
            
        Returns:
            List[str]: List of processed tokens
        """
        if not text:
            return []
            
        # Process with spaCy
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            # Skip tokens that don't meet criteria
            if (token.is_stop and self.config['preprocessing']['remove_stopwords'] or
                token.is_punct or
                token.is_space or
                len(token.text) < self.config['preprocessing']['min_token_length']):
                continue
                
            # Use lemma if lemmatization is enabled
            if self.config['preprocessing']['lemmatize']:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)
                
        return tokens
    
    def process_document(self, text: str) -> List[str]:
        """
        Complete processing pipeline for a single document.
        
        Args:
            text (str): Raw clinical text
            
        Returns:
            List[str]: Processed tokens
        """
        # Step 1: Clean the text
        cleaned = self.clean_clinical_text(text)
        
        # Step 2: Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(cleaned)
        
        # Step 3: Filter by minimum document length
        min_length = self.config['data']['min_doc_length']
        if len(tokens) < min_length:
            return []
            
        return tokens
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load clinical text data from various formats.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # First, try to determine if it's a pickle file
            try:
                with open(file_path, 'rb') as f:
                    # Check first few bytes for pickle signature
                    first_bytes = f.read(2)
                    f.seek(0)  # Reset position
                    
                    if first_bytes[0] == 0x80:  # Pickle protocol signature
                        logger.info("Detected pickle file format")
                        data = pickle.load(f)
                        
                        if isinstance(data, pd.DataFrame):
                            logger.info(f"Loaded DataFrame with shape: {data.shape}")
                            return data
                        elif isinstance(data, list):
                            logger.info(f"Loaded list with {len(data)} items, converting to DataFrame")
                            return pd.DataFrame({'text': data})
                        else:
                            logger.warning(f"Unexpected pickle data type: {type(data)}")
                            return pd.DataFrame({'text': [str(data)]})
                            
            except Exception as pickle_error:
                logger.info(f"Not a pickle file or failed to load: {pickle_error}")
            
            # Try reading as CSV
            if file_path.endswith('.csv'):
                logger.info("Attempting to load as CSV")
                df = pd.read_csv(file_path)
                return df
            
            # Try reading as text file with different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    logger.info(f"Trying encoding: {encoding}")
                    with open(file_path, 'r', encoding=encoding) as f:
                        texts = [line.strip() for line in f.readlines() if line.strip()]
                    
                    if texts:
                        logger.info(f"Successfully loaded {len(texts)} lines with {encoding} encoding")
                        return pd.DataFrame({'text': texts})
                        
                except UnicodeDecodeError:
                    continue
            
            # If all else fails
            raise ValueError(f"Could not determine file format for {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def process_corpus(self, file_path: str) -> List[List[str]]:
        """
        Process entire corpus for topic modeling.
        
        Args:
            file_path (str): Path to the text data
            
        Returns:
            List[List[str]]: Processed corpus ready for LDA
        """
        logger.info("Starting corpus preprocessing...")
        
        # Load the data using the smart loader
        df = self.load_data(file_path)
        
        # Extract text column
        if 'text' in df.columns:
            texts = df['text'].tolist()
        else:
            # If no 'text' column, assume first column contains text
            texts = df.iloc[:, 0].tolist()
        
        # Convert any non-string values to strings
        texts = [str(text) if text is not None else "" for text in texts]
        
        # Sample data if specified
        sample_size = self.config['data'].get('sample_size')
        if sample_size and len(texts) > sample_size:
            texts = texts[:sample_size]
            logger.info(f"Sampling {sample_size} documents from {len(texts)} total")
        
        # Process each document
        processed_docs = []
        total_docs = len(texts)
        
        logger.info(f"Processing {total_docs} documents...")
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{total_docs} documents")
                
            tokens = self.process_document(text)
            if tokens:  # Only add non-empty documents
                processed_docs.append(tokens)
        
        logger.info(f"Preprocessing complete. {len(processed_docs)} documents ready for modeling")
        return processed_docs


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

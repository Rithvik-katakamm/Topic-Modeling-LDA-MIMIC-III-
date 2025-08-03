"""
Clinical Topic Modeling with LDA
===============================

Core topic modeling functionality using Latent Dirichlet Allocation (LDA)
optimized for clinical text analysis.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose gensim logging
logging.getLogger('gensim').setLevel(logging.WARNING)
logging.getLogger('gensim.models.ldamodel').setLevel(logging.WARNING)
logging.getLogger('gensim.topic_coherence').setLevel(logging.WARNING)

# MLflow gensim integration (optional)
try:
    import mlflow.gensim
    MLFLOW_GENSIM_AVAILABLE = True
except ImportError:
    MLFLOW_GENSIM_AVAILABLE = False
    logger.warning("MLflow gensim integration not available - model artifacts will be logged manually")


class ClinicalTopicModeler:
    """
    LDA Topic Modeling for Clinical Text
    
    Features:
    - Optimized for medical/clinical text
    - MLflow integration for experiment tracking
    - Multiple evaluation metrics
    - Production-ready model persistence
    """
    
    def __init__(self, config: dict):
        """
        Initialize the topic modeler.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.model = None
        self.dictionary = None
        self.corpus = None
        self.coherence_score = None
        self.perplexity = None
        
        # Initialize MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        experiment_name = self.config['mlflow']['experiment_name']
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment set: {experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow setup warning: {e}")
    
    def prepare_corpus(self, processed_docs: List[List[str]]) -> Tuple[corpora.Dictionary, List]:
        """
        Prepare Gensim dictionary and corpus from processed documents.
        
        Args:
            processed_docs (List[List[str]]): Preprocessed documents
            
        Returns:
            Tuple[corpora.Dictionary, List]: Dictionary and corpus
        """
        logger.info("Creating Gensim dictionary and corpus...")
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(processed_docs)
        
        # Filter extremes based on config
        min_count = self.config['data']['min_word_count']
        max_freq = self.config['data']['max_word_freq']
        
        self.dictionary.filter_extremes(
            no_below=min_count,
            no_above=max_freq,
            keep_n=100000  # Keep top 100k words
        )
        
        # Create corpus (bag of words)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        logger.info(f"Dictionary size: {len(self.dictionary)}")
        logger.info(f"Corpus size: {len(self.corpus)} documents")
        
        return self.dictionary, self.corpus
    
    def train_model(self, processed_docs: List[List[str]]) -> LdaModel:
        """
        Train LDA topic model on clinical text.
        
        Args:
            processed_docs (List[List[str]]): Preprocessed documents
            
        Returns:
            LdaModel: Trained LDA model
        """
        # Prepare corpus if not already done
        if self.dictionary is None or self.corpus is None:
            self.prepare_corpus(processed_docs)
        
        # Extract model parameters from config
        model_params = self.config['model']
        
        logger.info("Training LDA model...")
        logger.info(f"Parameters: {model_params}")
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(model_params)
            mlflow.log_param("vocab_size", len(self.dictionary))
            mlflow.log_param("corpus_size", len(self.corpus))
            
            # Train LDA model
            self.model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=model_params['num_topics'],
                alpha=model_params['alpha'],
                eta=model_params['beta'],  # Gensim uses 'eta' for beta
                iterations=model_params['iterations'],
                passes=model_params['passes'],
                random_state=model_params['random_state'],
                per_word_topics=True
            )
            
            logger.info("Model training completed!")
            
            # Evaluate model
            self._evaluate_model(processed_docs)
            
            # Log metrics to MLflow
            if self.coherence_score:
                mlflow.log_metric("coherence_score", self.coherence_score)
            if self.perplexity:
                mlflow.log_metric("perplexity", self.perplexity)
            
            # Log model artifact
            if self.config['mlflow']['track_artifacts']:
                if MLFLOW_GENSIM_AVAILABLE:
                    mlflow.gensim.log_model(self.model, "lda_model")
                else:
                    logger.info("MLflow gensim integration not available, skipping model artifact logging")
        
        return self.model
    
    def _evaluate_model(self, processed_docs: List[List[str]]):
        """
        Evaluate the trained model using coherence and perplexity.
        
        Args:
            processed_docs (List[List[str]]): Original processed documents
        """
        logger.info("Evaluating model performance...")
        
        # Coherence Score (c_v metric)
        coherence_model = CoherenceModel(
            model=self.model,
            texts=processed_docs,
            dictionary=self.dictionary,
            coherence=self.config['evaluation']['coherence_metric']
        )
        self.coherence_score = coherence_model.get_coherence()
        
        # Perplexity (if enabled)
        if self.config['evaluation']['compute_perplexity']:
            self.perplexity = self.model.log_perplexity(self.corpus)
        
        # Log results
        logger.info(f"Coherence Score (c_v): {self.coherence_score:.4f}")
        if self.perplexity:
            logger.info(f"Perplexity: {self.perplexity:.4f}")
    
    def get_topics(self, num_words: int = 10) -> List[Tuple]:
        """
        Get human-readable topics from the trained model.
        
        Args:
            num_words (int): Number of top words per topic
            
        Returns:
            List[Tuple]: List of (topic_id, topic_words)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        topics = []
        for topic_id in range(self.config['model']['num_topics']):
            topic_words = self.model.show_topic(topic_id, topn=num_words)
            topics.append((topic_id, topic_words))
        
        return topics
    
    def get_topic_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of topics and their top words.
        
        Returns:
            pd.DataFrame: Topic summary
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        topics_data = []
        for topic_id in range(self.config['model']['num_topics']):
            topic_words = self.model.show_topic(topic_id, topn=10)
            top_words = [word for word, _ in topic_words]
            top_words_str = ", ".join(top_words[:5])  # Top 5 words
            
            topics_data.append({
                'topic_id': topic_id,
                'top_words': top_words_str,
                'all_words': top_words
            })
        
        return pd.DataFrame(topics_data)
    
    def save_model(self, model_dir: str):
        """
        Save the trained model and associated artifacts.
        
        Args:
            model_dir (str): Directory to save model artifacts
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save LDA model
        model_file = model_path / "lda_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save dictionary
        dict_file = model_path / "dictionary.pkl"
        with open(dict_file, 'wb') as f:
            pickle.dump(self.dictionary, f)
        
        # Save evaluation metrics
        metrics = {
            'coherence_score': self.coherence_score,
            'perplexity': self.perplexity,
            'num_topics': self.config['model']['num_topics']
        }
        
        metrics_file = model_path / "metrics.pkl"
        with open(metrics_file, 'wb') as f:
            pickle.dump(metrics, f)
        
        logger.info(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir: str):
        """
        Load a previously trained model.
        
        Args:
            model_dir (str): Directory containing model artifacts
        """
        model_path = Path(model_dir)
        
        # Load LDA model
        model_file = model_path / "lda_model.pkl"
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load dictionary
        dict_file = model_path / "dictionary.pkl"
        with open(dict_file, 'rb') as f:
            self.dictionary = pickle.load(f)
        
        # Load metrics
        metrics_file = model_path / "metrics.pkl"
        if metrics_file.exists():
            with open(metrics_file, 'rb') as f:
                metrics = pickle.load(f)
                self.coherence_score = metrics.get('coherence_score')
                self.perplexity = metrics.get('perplexity')
        
        logger.info(f"Model loaded from {model_dir}")
    
    def predict_topics(self, text: str, top_n: int = 3) -> List[Tuple[int, float]]:
        """
        Predict topics for new text.
        
        Args:
            text (str): New clinical text
            top_n (int): Number of top topics to return
            
        Returns:
            List[Tuple[int, float]]: Top topics with probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained/loaded. Cannot predict.")
        
        from .processor import ClinicalTextProcessor
        processor = ClinicalTextProcessor(self.config)
        
        # Process the text
        tokens = processor.process_document(text)
        
        # Convert to bag of words
        bow = self.dictionary.doc2bow(tokens)
        
        # Get topic distribution
        topic_dist = self.model.get_document_topics(bow)
        
        # Sort by probability and return top N
        topic_dist_sorted = sorted(topic_dist, key=lambda x: x[1], reverse=True)
        return topic_dist_sorted[:top_n]

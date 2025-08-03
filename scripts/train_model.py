#!/usr/bin/env python3
"""
Clinical Topic Model Training Script
===================================

Main CLI script for training LDA topic models on MIMIC-III clinical data.
This script demonstrates production-ready ML practices with MLflow tracking.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --config custom_config.yaml
    python scripts/train_model.py --topics 15 --sample 5000
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.processor import ClinicalTextProcessor, load_config
from src.models.topic_model import ClinicalTopicModeler
from src.visualization.visualizer import ClinicalTopicVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train clinical topic model')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--topics', type=int, help='Number of topics (overrides config)')
    parser.add_argument('--sample', type=int, help='Sample size (overrides config)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Override config with command line args
        if args.topics:
            config['model']['num_topics'] = args.topics
        if args.sample:
            config['data']['sample_size'] = args.sample
        
        logger.info("🏥 Starting Clinical Topic Modeling Pipeline")
        logger.info(f"📊 Topics: {config['model']['num_topics']}")
        logger.info(f"📄 Sample size: {config['data'].get('sample_size', 'All')}")
        
        # Step 1: Initialize components
        logger.info("Step 1: Initializing components...")
        processor = ClinicalTextProcessor(config)
        modeler = ClinicalTopicModeler(config)
        
        # Step 2: Load and preprocess data
        logger.info("Step 2: Loading and preprocessing clinical text...")
        data_path = config['paths']['data_file']
        processed_docs = processor.process_corpus(data_path)
        
        if not processed_docs:
            logger.error("No documents processed successfully. Check your data file.")
            return
        
        logger.info(f"✅ Preprocessed {len(processed_docs)} clinical documents")
        
        # Step 3: Train topic model
        logger.info("Step 3: Training LDA topic model...")
        model = modeler.train_model(processed_docs)
        
        # Step 4: Save model
        logger.info("Step 4: Saving model artifacts...")
        model_dir = config['paths']['model_dir']
        modeler.save_model(model_dir)
        
        # Step 5: Display results
        logger.info("Step 5: Generating topic analysis...")
        topics = modeler.get_topics()
        
        print("\\n" + "="*50)
        print("🎯 CLINICAL TOPIC ANALYSIS RESULTS")
        print("="*50)
        print(f"📈 Coherence Score: {modeler.coherence_score:.4f}")
        if modeler.perplexity:
            print(f"📉 Perplexity: {modeler.perplexity:.4f}")
        print(f"📚 Vocabulary Size: {len(modeler.dictionary)}")
        print(f"📄 Documents Processed: {len(processed_docs)}")
        
        print("\\n🏷️  DISCOVERED CLINICAL TOPICS:")
        print("-" * 50)
        for topic_id, topic_words in topics:
            top_5_words = [word for word, _ in topic_words[:5]]
            print(f"Topic {topic_id}: {', '.join(top_5_words)}")
        
        # Step 6: Create visualizations (unless skipped)
        if not args.no_viz:
            logger.info("Step 6: Creating clinical visualizations...")
            visualizer = ClinicalTopicVisualizer(config)
            visualizer.generate_all_visualizations(modeler)
            
            print("\\n📊 VISUALIZATIONS CREATED:")
            print("- Topic word clouds")
            print("- Interactive topic explorer")
            print("- Clinical dashboard")
            print(f"- All saved to: {config['paths']['viz_dir']}")
        
        print("\\n✅ Clinical topic modeling pipeline completed successfully!")
        print(f"🔗 Check MLflow UI for experiment tracking")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()

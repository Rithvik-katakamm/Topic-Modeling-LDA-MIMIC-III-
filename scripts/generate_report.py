#!/usr/bin/env python3
"""
Clinical Topic Visualization Generator
=====================================

Script for generating visualizations from a trained topic model.
Useful for creating reports after model training is complete.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --model-dir outputs/models
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.processor import load_config
from src.models.topic_model import ClinicalTopicModeler
from src.visualization.visualizer import ClinicalTopicVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate visualizations from trained model."""
    parser = argparse.ArgumentParser(description='Generate clinical topic visualizations')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--model-dir', default='outputs/models', help='Model directory')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        logger.info("📊 Generating Clinical Topic Analysis Report")
        
        # Load trained model
        logger.info(f"Loading model from {args.model_dir}")
        modeler = ClinicalTopicModeler(config)
        modeler.load_model(args.model_dir)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        visualizer = ClinicalTopicVisualizer(config)
        visualizer.generate_all_visualizations(modeler)
        
        # Display summary
        topics = modeler.get_topics()
        topic_summary = modeler.get_topic_summary()
        
        print("\\n" + "="*50)
        print("🏥 CLINICAL TOPIC ANALYSIS REPORT")
        print("="*50)
        print(f"📈 Model Coherence: {modeler.coherence_score:.4f}")
        print(f"🎯 Number of Topics: {len(topics)}")
        print(f"📚 Vocabulary Size: {len(modeler.dictionary)}")
        
        print("\\n🏷️  CLINICAL TOPICS DISCOVERED:")
        print("-" * 40)
        for _, row in topic_summary.iterrows():
            print(f"Topic {row['topic_id']}: {row['top_words']}")
        
        print(f"\\n📊 Visualizations saved to: {config['paths']['viz_dir']}")
        print("✅ Report generation complete!")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise


if __name__ == "__main__":
    main()

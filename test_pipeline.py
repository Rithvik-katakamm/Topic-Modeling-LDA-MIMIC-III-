#!/usr/bin/env python3
"""
Quick Test Script
================

Simple script to verify the pipeline works with your data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.processor import ClinicalTextProcessor, load_config

def test_pipeline():
    """Quick test of the data processing pipeline."""
    print("🧪 Testing Clinical Topic Modeling Pipeline")
    print("=" * 50)
    
    try:
        # Load config
        config = load_config('config.yaml')
        print("✅ Config loaded successfully")
        
        # Initialize processor
        processor = ClinicalTextProcessor(config)
        print("✅ Text processor initialized")
        
        # Test with a sample clinical text
        sample_text = """
        Admission Date: [**2183-9-25**] Discharge Date: [**2183-10-29**]
        HISTORY OF PRESENT ILLNESS: Patient developed nausea, vomiting and abdominal pain.
        CT scan revealed patient had an ileus. Neurology service was consulted.
        HOSPITAL COURSE: Gastrointestinal issues resolved with bowel rest.
        Patient continued on trach mask ventilation.
        """
        
        # Process sample
        tokens = processor.process_document(sample_text)
        print(f"✅ Sample processing works: {len(tokens)} tokens extracted")
        print(f"   Sample tokens: {tokens[:10]}")
        
        print("\\n🎉 Pipeline test successful!")
        print("Ready to run: python scripts/train_model.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Check dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    test_pipeline()

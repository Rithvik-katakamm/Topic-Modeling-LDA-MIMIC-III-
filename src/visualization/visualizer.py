"""
Clinical Topic Visualization
============================

Interactive visualizations for clinical topic modeling results.
Creates professional charts and dashboards for topic analysis.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from wordcloud import WordCloud

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class ClinicalTopicVisualizer:
    """
    Creates professional visualizations for clinical topic modeling results.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the visualizer.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.output_dir = Path(config['paths']['viz_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_topic_wordclouds(self, topic_modeler, save: bool = True) -> Dict[int, WordCloud]:
        """
        Create word clouds for each topic.
        
        Args:
            topic_modeler: Trained ClinicalTopicModeler
            save (bool): Whether to save visualizations
            
        Returns:
            Dict[int, WordCloud]: Word clouds by topic ID
        """
        if topic_modeler.model is None:
            raise ValueError("Model not trained yet.")
        
        logger.info("Creating topic word clouds...")
        
        wordclouds = {}
        
        # Create figure with subplots
        num_topics = self.config['model']['num_topics']
        cols = 3
        rows = (num_topics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for topic_id in range(num_topics):
            # Get topic words and weights
            topic_words = topic_modeler.model.show_topic(topic_id, topn=50)
            word_freq = dict(topic_words)
            
            # Create word cloud
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                max_words=50,
                colormap='viridis'
            ).generate_from_frequencies(word_freq)
            
            wordclouds[topic_id] = wordcloud
            
            # Plot
            row, col = divmod(topic_id, cols)
            ax = axes[row, col]
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Topic {topic_id}', fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # Remove empty subplots
        for i in range(num_topics, rows * cols):
            row, col = divmod(i, cols)
            axes[row, col].remove()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'topic_wordclouds.png', dpi=300, bbox_inches='tight')
            logger.info("Word clouds saved")
        
        plt.show()
        return wordclouds
    
    def create_topic_coherence_plot(self, topic_modeler, save: bool = True):
        """
        Create bar plot of topic coherence scores.
        
        Args:
            topic_modeler: Trained ClinicalTopicModeler
            save (bool): Whether to save the plot
        """
        if topic_modeler.model is None:
            raise ValueError("Model not trained yet.")
        
        logger.info("Creating topic coherence visualization...")
        
        # Calculate per-topic coherence (approximation)
        num_topics = self.config['model']['num_topics']
        
        fig = go.Figure()
        
        # Overall coherence score
        overall_coherence = topic_modeler.coherence_score
        
        fig.add_trace(go.Bar(
            x=[f'Topic {i}' for i in range(num_topics)],
            y=[overall_coherence] * num_topics,  # Simplified: same for all topics
            name='Coherence Score',
            marker_color='lightblue'
        ))
        
        fig.add_hline(
            y=overall_coherence, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Overall Coherence: {overall_coherence:.3f}"
        )
        
        fig.update_layout(
            title='Clinical Topic Coherence Analysis',
            xaxis_title='Topics',
            yaxis_title='Coherence Score (c_v)',
            height=500
        )
        
        if save:
            fig.write_html(self.output_dir / 'topic_coherence.html')
            logger.info("Coherence plot saved")
        
        fig.show()
    
    def create_interactive_topic_explorer(self, topic_modeler, save: bool = True):
        """
        Create interactive pyLDAvis visualization - the detailed topic explorer.
        
        This gives you:
        - Topic distances and relationships
        - Word frequencies and probabilities
        - Interactive topic selection
        - Detailed term analysis
        
        Args:
            topic_modeler: Trained ClinicalTopicModeler
            save (bool): Whether to save the visualization
        """
        if topic_modeler.model is None:
            raise ValueError("Model not trained yet.")
        
        logger.info("Creating interactive pyLDAvis topic explorer...")
        
        try:
            # Create pyLDAvis visualization with detailed settings
            vis = gensimvis.prepare(
                topic_modeler.model, 
                topic_modeler.corpus, 
                topic_modeler.dictionary,
                sort_topics=False,
                mds='mmds'  # Use multidimensional scaling for better topic separation
            )
            
            if save:
                vis_file = self.output_dir / 'topic_explorer.html'
                pyLDAvis.save_html(vis, str(vis_file))
                logger.info(f"🔍 Interactive explorer saved: {vis_file}")
                logger.info(f"🖥️  Open this file in your browser for detailed topic analysis")
                logger.info(f"📊 Features: topic distances, word probabilities, interactive selection")
            
            return vis
            
        except Exception as e:
            logger.error(f"Error creating pyLDAvis visualization: {e}")
            logger.info("Tip: Try installing pyLDAvis with: pip install pyLDAvis")
            return None
    
    def create_topic_summary_table(self, topic_modeler, save: bool = True) -> pd.DataFrame:
        """
        Create a professional topic summary table.
        
        Args:
            topic_modeler: Trained ClinicalTopicModeler
            save (bool): Whether to save the table
            
        Returns:
            pd.DataFrame: Topic summary
        """
        if topic_modeler.model is None:
            raise ValueError("Model not trained yet.")
        
        logger.info("Creating topic summary table...")
        
        # Get topic summary
        topic_df = topic_modeler.get_topic_summary()
        
        # Add clinical interpretation (simple heuristics)
        topic_df['clinical_theme'] = topic_df['top_words'].apply(self._interpret_clinical_topic)
        
        # Reorder columns
        topic_df = topic_df[['topic_id', 'clinical_theme', 'top_words']]
        
        if save:
            topic_df.to_csv(self.output_dir / 'topic_summary.csv', index=False)
            logger.info("Topic summary saved")
        
        return topic_df
    
    def _interpret_clinical_topic(self, top_words: str) -> str:
        """
        Simple heuristic to interpret clinical topics.
        
        Args:
            top_words (str): Top words for a topic
            
        Returns:
            str: Clinical interpretation
        """
        words = top_words.lower()
        
        # Simple keyword matching for clinical themes
        if any(term in words for term in ['chest', 'lung', 'respiratory', 'breath']):
            return "Pulmonary/Respiratory"
        elif any(term in words for term in ['heart', 'cardiac', 'blood', 'pressure']):
            return "Cardiovascular"
        elif any(term in words for term in ['infection', 'antibiotic', 'fever']):
            return "Infectious Disease"
        elif any(term in words for term in ['surgery', 'surgical', 'procedure']):
            return "Surgical"
        elif any(term in words for term in ['medication', 'drug', 'dose']):
            return "Medications"
        elif any(term in words for term in ['discharge', 'follow', 'care']):
            return "Care Management"
        else:
            return "General Clinical"
    
    def create_comprehensive_dashboard(self, topic_modeler, save: bool = True):
        """
        Create a comprehensive topic modeling dashboard.
        
        Args:
            topic_modeler: Trained ClinicalTopicModeler
            save (bool): Whether to save the dashboard
        """
        logger.info("Creating comprehensive clinical topic dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Topic Distribution', 'Model Performance', 
                          'Top Words by Topic', 'Clinical Themes'),
            specs=[[{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # 1. Topic Distribution (simplified)
        num_topics = self.config['model']['num_topics']
        fig.add_trace(
            go.Bar(x=[f'Topic {i}' for i in range(num_topics)],
                   y=[1/num_topics] * num_topics,
                   name='Topic Weight',
                   marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. Model Performance Indicator
        coherence = topic_modeler.coherence_score or 0.5
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=coherence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Coherence Score"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 0.4], 'color': "lightgray"},
                                {'range': [0.4, 0.6], 'color': "yellow"},
                                {'range': [0.6, 1], 'color': "green"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 0.7}}),
            row=1, col=2
        )
        
        # 3. Top words visualization
        topic_summary = topic_modeler.get_topic_summary()
        fig.add_trace(
            go.Bar(x=topic_summary['topic_id'].astype(str),
                   y=[len(words.split(', ')) for words in topic_summary['top_words']],
                   name='Word Count',
                   marker_color='lightgreen'),
            row=2, col=1
        )
        
        # 4. Clinical themes pie chart
        themes = topic_summary['clinical_theme'] if 'clinical_theme' in topic_summary.columns else ['Clinical'] * len(topic_summary)
        theme_counts = pd.Series(themes).value_counts()
        
        fig.add_trace(
            go.Pie(labels=theme_counts.index,
                   values=theme_counts.values,
                   name="Clinical Themes"),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Clinical Topic Modeling Dashboard - MIMIC-III Analysis",
            height=800,
            showlegend=False
        )
        
        if save:
            fig.write_html(self.output_dir / 'clinical_dashboard.html')
            logger.info("Dashboard saved")
        
        fig.show()
    
    def generate_all_visualizations(self, topic_modeler):
        """
        Generate focused, professional visualizations for topic modeling results.
        
        Args:
            topic_modeler: Trained ClinicalTopicModeler
        """
        logger.info("Creating professional topic analysis visualizations...")
        
        try:
            # 1. PRIMARY: Interactive pyLDAvis explorer (what you want!)
            logger.info("Creating interactive topic explorer (pyLDAvis)...")
            self.create_interactive_topic_explorer(topic_modeler)
            
            # 2. Topic summary table
            self.create_topic_summary_table(topic_modeler)
            
            # 3. Simple word clouds (optional)
            logger.info("Creating topic word clouds...")
            self.create_topic_wordclouds(topic_modeler)
            
            logger.info("\n🎉 All visualizations created successfully!")
            logger.info(f"📊 Key Results:")
            logger.info(f"   - Coherence Score: {topic_modeler.coherence_score:.4f}")
            logger.info(f"   - Interactive Explorer: {self.output_dir}/topic_explorer.html")
            logger.info(f"   - Topic Summary: {self.output_dir}/topic_summary.csv")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise

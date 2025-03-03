# Topic Modeling on MIMIC-III Clinical Notes

This project applies topic modeling techniques to free-text clinical notes from the MIMIC-III dataset. The goal is to uncover hidden themes or “topics” within the notes that can provide insights into patient care, clinical documentation patterns, and potential areas for further analysis.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Data Source](#data-source)  
3. [Setup & Installation](#setup--installation)  
4. [Data Preprocessing](#data-preprocessing)  
5. [Topic Modeling (LDA)](#topic-modeling-lda)  
6. [Evaluation & Visualization](#evaluation--visualization)  
7. [Results](#results)  
8. [Future Work](#future-work)  
9. [License](#license)

---

## Project Overview

- **Objective:** Identify key clinical themes in MIMIC-III notes using Latent Dirichlet Allocation (LDA).  
- **Approach:** 
  1. Clean and preprocess text (regex + spaCy).
  2. Convert text to a bag-of-words representation.
  3. Train and evaluate an LDA model.
  4. Visualize and interpret the extracted topics.

---

## Data Source

- **MIMIC-III Dataset:** A publicly available database of de-identified health-related data, including clinical notes.  
- **Access Requirements:**  
  - To use MIMIC-III, you must complete required training and request access via [PhysioNet](https://physionet.org/).  
  - This project uses only the NOTEEVENTS portion (clinical notes) from MIMIC-III.

**Note:** The dataset is not included in this repository due to licensing and privacy constraints. You must download MIMIC-III separately.

---

## Setup & Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/YourUsername/Topic-Modeling-LDA-MIMIC-III-.git
   cd Topic-Modeling-LDA-MIMIC-III-
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended):**

   ```bash
   conda create -n topic_model_env python=3.9
   conda activate topic_model_env
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
   or
   ```bash
   pip install spacy gensim scikit-learn pyLDAvis
   python -m spacy download en_core_web_sm
   ```

---

## Data Preprocessing

1. **Regex Cleaning:**  
   Remove non-alphabetical characters, punctuation, and convert text to lowercase.
2. **spaCy Processing:**  
   - Tokenize and lemmatize text.  
   - Remove stopwords.
3. **Output:**  
   A cleaned text column ready for topic modeling.

---

## Topic Modeling (LDA)

1. **Dictionary & Corpus Creation:**  
   - Convert tokenized text to a Gensim dictionary.  
   - Create a bag-of-words (BoW) corpus.
2. **LDA Training (Gensim):**  
   - Specify number of topics (e.g., 5).  
   - Train the model with multiple passes for better convergence.
3. **Inspect Topics:**  
   - Print top words for each topic.  
   - Assign human-readable labels (e.g., “Medication Dosage,” “Radiology,” etc.).

---

## Evaluation & Visualization

- **Topic Coherence (c_v):**  
  Provides a numeric measure of how semantically related the top words are in each topic.
- **Perplexity (Optional):**  
  Measures how well the model predicts the data (less intuitive for interpretability).
- **pyLDAvis:**  
  Interactive visualization of topic clusters and top terms.

---

## Results

- **Extracted Topics:**  
  - Example topics: “Assessment and Plan,” “CT Imaging / Radiology,” “Medication Dosage and Lab Measurements,” “Chest Radiology Report,” “Pediatric/Infant Care.”
- **Coherence Score:** ~0.52 (moderate coherence).
- **Interpretation:**  
  The discovered topics align with real-world clinical themes. Future refinement or hyperparameter tuning could improve coherence and interpretability.

---

## Future Work

- **Hyperparameter Tuning:**  
  Adjust the number of topics, alpha, beta, or other parameters to improve topic quality.
- **Incorporate Metadata:**  
  Link topics with patient demographics or admission data to see how topics vary across patient groups.
- **Advanced Methods:**  
  Explore transformer-based models like BERTopic or neural topic models for potentially richer semantic insights.

---

---

Feel free to modify or remove any sections that aren’t relevant to your particular project setup!


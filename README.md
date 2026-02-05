# Text2SQL
A Python-based utility designed to classify natural language questions into specific SQL operation categories. This tool helps pre-process user queries to determine whether they require groupings, joins, comparisons, or specific ordering before generating the final SQL statement.

### Features
- Heuristic Classification: A fast, keyword-based approach using predefined lexicons.
- ML-Based Classification: Uses Word2Vec embeddings and Logistic Regression for more nuanced semantic understanding.
- Vectorized Text: Implements sentence-level representation by averaging word vectors via Gensim.
- Accuracy Evaluation: Built-in reporting to compare the performance of both the keyword and ML models.

### Technical Architecture
The parser operates in three main stages:
- Data Loading: Reads training and validation datasets from .tsv files.
- Vectorization: Converts sentences into 300-dimensional vectors using a pruned Google News Word2Vec model.
- Inference:
    - Keyword Method: Scans for specific SQL-related tokens (e.g., "average", "join", "sort").
    - ML Classifier Method: Uses a trained: Uses a trained LogisticRegression model to predict labels based on semantic density.
 
### Project Structure
**Text2SQLParser:** The core class containing all logic.

**predict_label_using_keywords():** Best for simple, explicit queries.

**predict_label_using_ml_classifier():** Best for complex phrasing where keywords might be missing.

**data/semantic-parser/:** Directory containing your sql_train.tsv and sql_val.tsv. 
# Prerequisites
Ensure you have the following libraries installed:

Bash
pip install numpy pandas scikit-learn gensim nltk sympy
The script also requires the word2vec_sample from NLTK, which it will attempt to download automatically upon the first run.

# Evaluation & Results
The project includes a built-in evaluation suite that benchmarks the performance of both classification methods against a validation set (`sql_val.tsv`).

### Performance Metrics
The system outputs a granular accuracy report, allowing for a direct comparison between the two approaches:
- **Baseline (Keyword-based):** Provides a quick accuracy check based on explicit SQL tokens.
- **ML Classifier (Logistic Regression):** Measures semantic understanding beyond exact keyword matching.

# Persian News Information Retrieval System from Scratch

## Project Overview
This project implements an **Information Retrieval (IR) system** for Persian news articles using a **Vector Space Model (VSM)**. The system supports efficient retrieval of relevant news articles based on user queries and ranks them using similarity measures.  

The main goal is to provide a fast and accurate search experience over a large Persian news corpus using TF-IDF weighting, positional indexing, and champion lists.  

---

## Dataset
- **Name:** `IR_data_news_12k.json`  
- **Size:** 12,000 Persian news articles  
- **Fields:** title, content, full-content, tags, date, URL, category  
- **Content Type:** Persian text  

---

## Text Preprocessing
1. **Normalization:** Standardize Persian text using Hazm’s Normalizer.  
2. **Tokenization:** Split content into words using Hazm’s `WordTokenizer`.  
3. **Stopword Removal:** Custom Persian stopword list extended with punctuation.  
4. **Lemmatization:** Reduce words to their base form using Hazm’s `Lemmatizer`.  

Preprocessing ensures uniform representation of text for accurate retrieval.

---

## Indexing
- **Positional Index:** Stores the positions and frequencies of all tokens per document.  
- **Vector Space Model:** Computes TF-IDF scores for all tokens in all documents.  
- **Champion Lists:** Top documents for each token are preselected to optimize search efficiency.  

---

## Query Processing & Retrieval
1. User query is normalized, tokenized, stopword-filtered, and lemmatized.  
2. TF-IDF vector is computed for the query.  
3. Candidate documents are fetched using the champion lists.  
4. Documents are ranked based on the selected similarity metric:
   - **Cosine similarity** (vector-based)  
   - **Jaccard similarity** (set-based)  

---

## Usage
1. Run the Python script.  
2. Choose similarity function: `cosine` or `jaccard`.  
3. Enter a Persian query.  
4. The system outputs the top-k relevant news articles with:
   - Document ID  
   - Relevance Score  
   - Title  
   - URL  
   - Full Content  

---

## Results
- Efficient retrieval from a 12,000-document Persian news corpus.  
- Accurate ranking due to TF-IDF weighting and champion list optimization.  
- Flexible similarity metrics allow different retrieval strategies.  
- Demonstrated robust Persian text preprocessing and indexing pipeline.  

**Example Query:** `"مسابقات فوتسال آسیا"`  
**Example Output:** Top news articles about Iranian economy ranked by relevance.

---

## Notes
- The system can be further improved with **query expansion**, **synonym handling**, or **semantic embeddings**.  
- For very large corpora, **inverted index storage** or database integration can further enhance performance.  

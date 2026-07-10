# Embeddings for Social Scientists

Teaching materials for a SICSS-style lecture on embeddings for social scientists: classic word2vec, modern API embeddings, cosine similarity, semantic search, clustering, topic modeling, and scaling exercises.

## Main File

- `embeddings_for_social_scientists_api_demo.py`: a slow, API-first teaching script that merges the earlier Kozlowski-style word2vec dimension example with modern OpenAI/Cohere embedding examples.

## What The Demo Covers

1. Classic word2vec and Kozlowski-style cultural dimensions
2. OpenAI and Cohere document embeddings
3. Cosine similarity
4. Semantic search
5. Hierarchical clustering
6. Parliamentary speech scaling with anchor texts
7. Government-vs-opposition scaling from known groups
8. Optional BERTopic
9. Failure modes and validation problems

## Setup

Install the base dependencies:

```bash
pip install openai cohere gensim numpy pandas scikit-learn scipy matplotlib
```

Optional BERTopic section:

```bash
pip install bertopic umap-learn hdbscan
```

Set API keys in your shell. Do not put real keys in the code.

PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:COHERE_API_KEY="..."
```

macOS/Linux:

```bash
export OPENAI_API_KEY="sk-..."
export COHERE_API_KEY="..."
```

## Run

OpenAI:

```bash
python embeddings_for_social_scientists_api_demo.py --provider openai
```

Cohere:

```bash
python embeddings_for_social_scientists_api_demo.py --provider cohere
```

Optional BERTopic:

```bash
python embeddings_for_social_scientists_api_demo.py --provider openai --bertopic
```

## Lecture Flow

See `lecture_flow_embeddings.md` for a suggested structure. The practical recommendation is: keep slides conceptually simple, then use the code as the main teaching device.

## Main Lesson

Embeddings are useful because they turn text into geometry: similarity, search, clustering, and scaling all become vector operations. They are risky because those operations can confuse meaning with topic, corpus bias, genre, time period, and researcher-chosen anchors.

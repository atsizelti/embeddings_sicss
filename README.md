# Embeddings for Social Scientists

Teaching materials for a SICSS-style lecture on embeddings for social scientists: basic word embeddings, word2vec, Kozlowski-style cultural dimensions, modern API embeddings, semantic search, clustering, topic modeling, parliamentary speech scaling, and model/worldview adaptation.

## Main File

- `embeddings_for_social_scientists_api_demo.py`: the coding-session script. It implements the concepts from the slides in order, moving from simple word embeddings to parliamentary document embeddings and optional local fine-tuning.

## Coding Session Flow

1. Basic word embeddings from a co-occurrence matrix and SVD
2. Classic word2vec and Kozlowski-style cultural dimensions
3. OpenAI, Cohere, or open Sentence-BERT sentence embeddings
4. Cosine similarity and semantic search
5. Document embeddings for parliamentary speeches
6. Hierarchical clustering
7. Ideological scaling with anchor texts
8. Government-vs-opposition centroid scaling
9. Optional BERTopic
10. Nelimarkka-style worldview lens
11. Optional local Sentence-Transformer fine-tuning on socialist vs market-oriented speeches
12. Failure modes and validation

## Setup

Base dependencies:

```bash
pip install openai cohere sentence-transformers gensim numpy pandas scikit-learn scipy matplotlib
```

Optional BERTopic:

```bash
pip install bertopic umap-learn hdbscan
```

Optional local fine-tuning:

```bash
pip install sentence-transformers torch
```

For Colab, use one of these two options.

Recommended: use Colab Secrets. In the left sidebar, click the key icon, add secrets named `OPENAI_API_KEY` and `COHERE_API_KEY`, and enable notebook access. The script loads those automatically.

Quick teaching option: paste keys into this block near the top of the script in your temporary Colab copy only:

```python
COLAB_PASTE_KEYS = {
    "OPENAI_API_KEY": "sk-your-real-openai-key",
    "COHERE_API_KEY": "your-real-cohere-key",
}
```

Do not commit real keys to GitHub.

For local runs, you can also use a `.env` file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then open `.env` and paste your real keys:

```text
OPENAI_API_KEY=sk-your-real-openai-key
COHERE_API_KEY=your-real-cohere-key
```

The script loads `.env` automatically. The `.env` file is ignored by Git.

## Run

OpenAI:

```bash
python embeddings_for_social_scientists_api_demo.py --provider openai
```

Cohere:

```bash
python embeddings_for_social_scientists_api_demo.py --provider cohere
```

With BERTopic:

```bash
python embeddings_for_social_scientists_api_demo.py --provider openai --bertopic
```

With local fine-tuning:

```bash
python embeddings_for_social_scientists_api_demo.py --provider openai --fine-tune-local
```

## Parliamentary Data / ParlaMint

The script works with built-in toy parliamentary speeches by default. For real data, pass a local CSV, JSONL, JSON, XML file, or a directory of ParlaMint-style XML files:

```bash
python embeddings_for_social_scientists_api_demo.py --provider openai --parliament-path data/parlamint_sample.csv --limit 2000
```

Recommended columns for a prepared CSV/JSONL extract:

- `text`
- `speech_id`
- `speaker`
- `party`
- `party_family`
- `party_position`
- `date`
- `country`

The XML reader is intentionally lightweight for teaching. For research, prepare a clean CSV/Parquet extract from ParlaMint first, then run the embedding workflow.

## Lecture Flow

See `lecture_flow_embeddings.md`. The practical recommendation is: keep slides conceptually simple, then use the code as the main teaching device.

## Main Lesson

Embeddings are useful because they turn text into geometry: similarity, search, clustering, and scaling become vector operations. They are risky because those operations can confuse meaning with topic, corpus bias, genre, time period, and researcher-chosen anchors.




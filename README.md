# Embeddings for Social Scientists

Teaching materials for a SICSS-style lecture and coding session on embeddings for social scientists: basic word embeddings, word2vec, Kozlowski-style cultural dimensions, modern API embeddings, semantic search, clustering, BERTopic, parliamentary speech scaling, and model/worldview adaptation.

## Where We Left Off

Current working version:

- Main Colab notebook: `embeddings_for_social_scientists_colab.ipynb`
- Plain Python script: `embeddings_for_social_scientists_api_demo.py`
- QR image for the repository: `sicss_embeddings.png`
- Lecture-flow notes: `lecture_flow_embeddings.md`

Recent fixes already applied:

- The Colab file is now a real `.ipynb`, not just a plain `.py` script.
- The notebook is split into step-by-step teaching cells instead of one giant code cell.
- The first runnable examples are local and conceptual; they do not call OpenAI/Cohere immediately.
- `PROVIDER = "sentence-transformers"` is the notebook default, so it can run without API keys after installing packages.
- The missing Sentence-BERT dispatch bug was fixed.
- The co-occurrence example tokenizer bug was fixed.
- A QR code was added to the GitHub README.

Known local-only item:

- `.claude/` is untracked and intentionally left alone.

## Open in Colab

Use this direct link:

[Open `embeddings_for_social_scientists_colab.ipynb` in Colab](https://colab.research.google.com/github/atsizelti/embeddings_sicss/blob/main/embeddings_for_social_scientists_colab.ipynb)

Colab's GitHub browser usually finds `.ipynb` notebooks, not plain `.py` scripts. Use the notebook link above for teaching in Colab.

### QR Code

<img src="sicss_embeddings.png" alt="QR code for the embeddings SICSS materials" width="220">

## Recommended Teaching Path

Use the slides for intuition, then run the notebook in this order:

1. Install packages.
2. Define imports, toy data, and helper functions.
3. Run basic co-occurrence embeddings with SVD.
4. Run the word2vec/Kozlowski-style cultural dimension example.
5. Choose an embedding provider.
6. Generate sentence embeddings.
7. Compute cosine similarities.
8. Run semantic search.
9. Run hierarchical clustering on sentence examples.
10. Load built-in parliamentary speeches or a ParlaMint-style dataset.
11. Scale parliamentary speeches with anchor texts.
12. Cluster parliamentary speeches.
13. Run government-vs-opposition centroid scaling.
14. Optionally run BERTopic.
15. Run the Nelimarkka-style worldview lens.
16. Optionally run local Sentence-Transformer fine-tuning.
17. End with validation and failure modes.

## Files

- `embeddings_for_social_scientists_colab.ipynb`: Colab-ready step-by-step notebook for the live coding session.
- `embeddings_for_social_scientists_api_demo.py`: plain Python script with the same lecture workflow, useful for local or terminal runs.
- `lecture_flow_embeddings.md`: suggested timing and slide-to-code flow.
- `.env.example`: safe template for local API keys.
- `sicss_embeddings.png`: QR code shown on the GitHub page.

## What The Coding Session Covers

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

## Provider Choices

Default in Colab:

```python
PROVIDER = "sentence-transformers"
```

This runs locally and does not require API keys, but it downloads an open model from Hugging Face the first time.

Other options:

```python
PROVIDER = "openai"
PROVIDER = "cohere"
```

These require API keys.

## API Keys

For Colab, use one of these two options.

Recommended: use Colab Secrets. In the left sidebar, click the key icon, add secrets named `OPENAI_API_KEY` and `COHERE_API_KEY`, and enable notebook access. The notebook loads those automatically.

Quick teaching option: paste keys into this block near the top of the notebook in your temporary Colab copy only:

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

## Local Setup

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

## Local Run Commands

Sentence-BERT, no API keys:

```bash
python embeddings_for_social_scientists_api_demo.py --provider sentence-transformers
```

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
python embeddings_for_social_scientists_api_demo.py --provider sentence-transformers --fine-tune-local
```

## Parliamentary Data / ParlaMint

The notebook and script use built-in toy parliamentary speeches by default. For real data, pass a local CSV, JSONL, JSON, XML file, or a directory of ParlaMint-style XML files.

Local script example:

```bash
python embeddings_for_social_scientists_api_demo.py --provider openai --parliament-path data/parlamint_sample.csv --limit 2000
```

Colab example:

1. Upload `parlamint_sample.csv` to `/content/`.
2. Set:

```python
PARLIAMENT_PATH = "/content/parlamint_sample.csv"
LIMIT = 2000
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

## What To Check Next

If continuing development later, likely next improvements are:

1. Run the full Colab notebook once from top to bottom with `PROVIDER = "sentence-transformers"`.
2. Test `PROVIDER = "openai"` after setting Colab Secrets.
3. Test `PROVIDER = "cohere"` after setting Colab Secrets.
4. Decide whether the BERTopic section should be installed and run live or left as optional.
5. Prepare a small real ParlaMint CSV sample for the parliamentary section.
6. Consider adding one slide/note explaining that API embeddings are not fine-tuned here; only local Sentence-Transformers are fine-tuned.

## Main Lesson

Embeddings are useful because they turn text into geometry: similarity, search, clustering, and scaling become vector operations. They are risky because those operations can confuse meaning with topic, corpus bias, genre, time period, and researcher-chosen anchors.

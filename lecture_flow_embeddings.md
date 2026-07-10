# Lecture Flow: Embeddings for Social Scientists

Recommended structure: keep the slides simple, then move quickly into code. The coding session should be the implementation layer for the concepts already introduced in the slides.

## Why

Embeddings are easiest to understand when students see the same operation repeated at different scales:

1. Words become vectors.
2. Sentences become vectors.
3. Documents/speeches become vectors.
4. Vectors can be compared with cosine similarity.
5. Similarity can power search, clustering, scaling, and topic exploration.
6. Model choice and fine-tuning change the worldview embedded in the workflow.

## Suggested Timing

### 1. Slides, 15-20 minutes

Use the slides to explain:

- meanings are relational, not stored as dictionary definitions;
- embeddings are corpus-dependent;
- Kozlowski dimensions are directions created from opposed word pairs;
- modern embeddings can represent sentences and documents directly;
- clustering and BERTopic are exploratory tools;
- scaling needs validation;
- Nelimarkka-style fine-tuning shows that models carry worldviews.

### 2. Code Demo, 60-75 minutes

Use `embeddings_for_social_scientists_api_demo.py`.

Run one provider first:

```powershell
$env:OPENAI_API_KEY="sk-..."
python embeddings_for_social_scientists_api_demo.py --provider openai
```

Then, if useful, compare Cohere:

```powershell
$env:COHERE_API_KEY="..."
python embeddings_for_social_scientists_api_demo.py --provider cohere
```

Recommended live-code order:

1. Basic co-occurrence embeddings: show where vector spaces come from.
2. Word2vec/Kozlowski: build a cultural dimension step by step.
3. API sentence embeddings: show actual embedding vectors and cosine similarity.
4. Semantic search: query meaning rather than exact words.
5. Parliamentary document embeddings: move from sentences to speeches.
6. Hierarchical clustering: explore parliamentary speech structure.
7. Scaling: project speeches onto left-right and government-opposition axes.
8. BERTopic: optional extension after clustering is understood.
9. Nelimarkka-style worldview lens: rank occupations/speeches through socialist-vs-market anchors.
10. Optional fine-tuning: show local Sentence-Transformer adaptation if time and dependencies allow.

### 3. ParlaMint Upgrade

ParlaMint is a good real-data choice because it is multilingual, parliamentary, metadata-rich, and directly aligned with the lecture topic. Use it after the toy data has made the mechanics clear.

Recommended workflow:

1. Start with 200-2,000 speeches, not the full corpus.
2. Prepare a clean CSV/JSONL extract with text, party, speaker, date, country, party family, and government/opposition metadata.
3. Run:

```powershell
python embeddings_for_social_scientists_api_demo.py --provider openai --parliament-path data/parlamint_sample.csv --limit 2000
```

4. Only then scale to larger samples and expensive API runs.

The script can read simple ParlaMint-style XML directly, but for research a prepared CSV/Parquet extract is better.

### 4. Discussion, 10-15 minutes

Ask students:

- What does a high cosine similarity actually prove?
- Could two speeches be close because of topic rather than ideology?
- Who chooses the anchors in a scaling exercise?
- How would we validate a parliamentary scale?
- What changes when the embedding model is fine-tuned on socialist speeches?
- How would results differ across OpenAI, Cohere, and a multilingual Sentence-Transformer?

## Slide Advice

Keep slides visual and conceptual. Do not put API details on slides. The API details belong in the code.

The code should be the main event.

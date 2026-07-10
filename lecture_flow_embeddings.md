# Lecture Flow: Embeddings for Social Scientists

Recommended structure: keep the slides simple, then move quickly into code.

## Why

Embeddings are easier to understand when students see the same operation repeated:

1. Text becomes a vector.
2. Vectors can be compared with cosine similarity.
3. Similarity can power search.
4. Many similarities can become clusters.
5. A theoretically chosen direction can become a scale.

The slides should give intuition and vocabulary. The code should do the actual teaching.

## Suggested Timing

### 1. Slides, 10-15 minutes

Use the slides to explain:

- meanings are relational, not stored as dictionary definitions;
- embeddings place texts or words in high-dimensional space;
- cosine similarity asks whether two vectors point in similar directions;
- dimensions/scales are researcher-defined directions, not natural laws;
- embeddings are useful but fragile.

Your current Kozlowski slides are good for this opening, but they are narrower than the lecture title. Treat them as the word2vec/social-meaning example, not as the whole lecture.

### 2. Code demo, 45-60 minutes

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

Cover the sections in this order:

1. Text to vector
2. Cosine similarity
3. Semantic search
4. Hierarchical clustering
5. Parliamentary scaling with anchor texts
6. Government-opposition scaling from known groups
7. Failure modes

Skip BERTopic unless the class is comfortable with the basics. BERTopic is best as a final optional extension.

### 3. Discussion, 10 minutes

Ask students:

- What does a high similarity score actually prove?
- Could two speeches be close because of topic rather than ideology?
- Who chooses the anchors in a scaling exercise?
- How would we validate the parliamentary scale?
- What would change if we embedded Turkish parliamentary speeches, OCR newspapers, or social media posts?

## Slide Advice

Keep slides simple and visual. Do not put API details on slides. The API details belong in code.

A good slide deck for this session would have only these conceptual sections:

1. What is an embedding?
2. Word embeddings vs sentence/document embeddings
3. Cosine similarity
4. Four uses: similarity, search, clustering, scaling
5. Kozlowski-style cultural dimensions
6. Modern API embeddings: OpenAI/Cohere/Sentence-BERT
7. Where embeddings break
8. Now code

The code should be the main event.

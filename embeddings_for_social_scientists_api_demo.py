"""
Embeddings for Social Scientists: API-first teaching demo
=========================================================

This file is meant for a live lecture or workshop. It moves slowly and prints
what it is doing, so students can see the logic behind embeddings rather than
only seeing final plots.

What it covers
--------------
1. Classic word2vec and Kozlowski-style semantic dimensions.
2. What a modern embedding is: text -> numeric vector.
3. OpenAI and Cohere embedding APIs.
4. Cosine similarity.
5. Semantic search.
6. Hierarchical clustering.
7. A simple scaling exercise on toy parliamentary speeches.
8. Optional BERTopic with precomputed embeddings.
9. Where embeddings break.

Before class
------------
Install dependencies:

    pip install openai cohere gensim numpy pandas scikit-learn scipy matplotlib

Optional topic-modeling section:

    pip install bertopic umap-learn hdbscan

Set API keys in your shell. Do not paste real keys into the script.

PowerShell:

    $env:OPENAI_API_KEY="sk-..."
    $env:COHERE_API_KEY="..."

macOS/Linux:

    export OPENAI_API_KEY="sk-..."
    export COHERE_API_KEY="..."

Run examples:

    python embeddings_for_social_scientists_api_demo.py --provider openai
    python embeddings_for_social_scientists_api_demo.py --provider cohere
    python embeddings_for_social_scientists_api_demo.py --provider openai --bertopic

Notes for the instructor
------------------------
OpenAI docs show the Python embeddings call as:
    client.embeddings.create(input="Your text string goes here", model="text-embedding-3-small")
Cohere docs recommend input_type="search_document" for documents and
input_type="search_query" for semantic-search queries.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd


Provider = Literal["openai", "cohere"]


# ---------------------------------------------------------------------------
# 0. Small lecture datasets
# ---------------------------------------------------------------------------

SHORT_TEXTS = [
    {
        "id": "welfare",
        "text": "The government should expand child benefits and protect low-income families from rising rents.",
        "group": "social policy",
    },
    {
        "id": "tax_cut",
        "text": "Lower corporate taxes will encourage investment, entrepreneurship, and job creation.",
        "group": "economy",
    },
    {
        "id": "climate",
        "text": "A green transition requires public investment in renewable energy and stricter emissions standards.",
        "group": "environment",
    },
    {
        "id": "border",
        "text": "The state must strengthen border controls and enforce immigration rules more consistently.",
        "group": "security",
    },
    {
        "id": "schools",
        "text": "Schools need smaller classes, better teacher pay, and equal access for disadvantaged neighborhoods.",
        "group": "social policy",
    },
    {
        "id": "defense",
        "text": "National security depends on modernizing the armed forces and supporting the defense industry.",
        "group": "security",
    },
]

PARLIAMENTARY_SPEECHES = [
    {
        "speech_id": "GOV_01",
        "speaker": "Government MP A",
        "party_position": "government",
        "text": "Our budget protects fiscal discipline while funding targeted support for families facing high prices.",
    },
    {
        "speech_id": "GOV_02",
        "speaker": "Government MP B",
        "party_position": "government",
        "text": "The reform package reduces red tape for firms, attracts investment, and keeps public finances stable.",
    },
    {
        "speech_id": "GOV_03",
        "speaker": "Government MP C",
        "party_position": "government",
        "text": "We are expanding infrastructure, supporting exporters, and maintaining a responsible path for debt.",
    },
    {
        "speech_id": "OPP_01",
        "speaker": "Opposition MP A",
        "party_position": "opposition",
        "text": "This budget leaves workers behind, weakens public services, and ignores the housing crisis.",
    },
    {
        "speech_id": "OPP_02",
        "speaker": "Opposition MP B",
        "party_position": "opposition",
        "text": "People need stronger labor protections, fairer taxation, and serious investment in public hospitals.",
    },
    {
        "speech_id": "OPP_03",
        "speaker": "Opposition MP C",
        "party_position": "opposition",
        "text": "The government celebrates growth figures while families struggle with rent, food, and energy bills.",
    },
    {
        "speech_id": "NEW_01",
        "speaker": "Unlabeled MP X",
        "party_position": "unknown",
        "text": "The country needs a credible industrial strategy with public investment and protection for workers.",
    },
    {
        "speech_id": "NEW_02",
        "speaker": "Unlabeled MP Y",
        "party_position": "unknown",
        "text": "Economic recovery depends on private investment, lower compliance costs, and stable public accounts.",
    },
]

LEFT_ANCHORS = [
    "workers rights, public hospitals, redistributive taxation, social housing, stronger welfare state",
    "labor protections, public investment, inequality reduction, affordable housing, universal services",
]

RIGHT_ANCHORS = [
    "lower taxes, private investment, fiscal discipline, deregulation, competitive markets",
    "business confidence, balanced budgets, entrepreneurship, smaller government, market incentives",
]

WORD2VEC_EARLY_TEXTS = [
    "the immigrant was described as foreign poor dangerous disorderly suspicious",
    "newspapers linked immigrant neighborhoods with crime poverty threat and disorder",
    "the foreign worker was called cheap strange unamerican and risky",
    "reports warned that immigrant groups brought danger disease poverty and unrest",
    "city officials described immigrant districts as crowded poor and dangerous",
    "hardworking creative ambitious loyal valuable skilled vibrant are positive descriptive words",
    "dangerous poor disorderly suspicious threat risky unrest are negative descriptive words",
]

WORD2VEC_LATE_TEXTS = [
    "the immigrant was described as hardworking creative ambitious and loyal",
    "newspapers linked immigrant communities with family enterprise culture and contribution",
    "the foreign worker was called skilled productive modern and valuable",
    "reports praised immigrant groups for innovation service work and community life",
    "city officials described immigrant districts as diverse vibrant and entrepreneurial",
    "hardworking creative ambitious loyal valuable skilled vibrant are positive descriptive words",
    "dangerous poor disorderly suspicious threat risky unrest are negative descriptive words",
]

KOZLOWSKI_STYLE_PAIRS = [
    ("hardworking", "dangerous"),
    ("creative", "poor"),
    ("ambitious", "disorderly"),
    ("loyal", "suspicious"),
    ("valuable", "threat"),
    ("skilled", "risky"),
    ("vibrant", "unrest"),
]


# ---------------------------------------------------------------------------
# 1. Utility functions
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)



def explain(text: str) -> None:
    print(textwrap.fill(text.strip(), width=88))
    print()



def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        norm = np.linalg.norm(matrix)
        return matrix if norm == 0 else matrix / norm
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)



def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(embeddings)
    return normalized @ normalized.T



def cosine_similarity_one_to_many(query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
    query = l2_normalize(query_embedding)
    docs = l2_normalize(document_embeddings)
    return docs @ query



def projection_scores(embeddings: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Project each normalized document vector onto a normalized semantic axis."""
    return l2_normalize(embeddings) @ l2_normalize(axis)



def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. Set it in your shell before running this provider. "
            "Do not paste API keys directly into the code."
        )
    return value


def tokenize_for_word2vec(text: str) -> list[str]:
    import re

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [word for word in text.split() if len(word) > 2]


def paired_word2vec_axis(model, word_pairs: list[tuple[str, str]]) -> tuple[np.ndarray, dict[str, list[tuple[str, str]]]]:
    """Kozlowski-style dimension for static word vectors.

    Each pair is ordered as (positive_pole, negative_pole). The axis is the
    normalized average of normalized pairwise differences.
    """
    pair_vectors = []
    used_pairs = []
    missing_pairs = []
    for positive_word, negative_word in word_pairs:
        if positive_word in model.wv and negative_word in model.wv:
            positive_vec = l2_normalize(model.wv[positive_word])
            negative_vec = l2_normalize(model.wv[negative_word])
            pair_vectors.append(positive_vec - negative_vec)
            used_pairs.append((positive_word, negative_word))
        else:
            missing_pairs.append((positive_word, negative_word))
    if not pair_vectors:
        raise ValueError("No complete pole pairs were available in the vocabulary.")
    return l2_normalize(np.mean(pair_vectors, axis=0)), {"used_pairs": used_pairs, "missing_pairs": missing_pairs}


def word2vec_projection_score(model, word: str, axis: np.ndarray) -> float:
    if word not in model.wv:
        return np.nan
    return float(l2_normalize(model.wv[word]) @ l2_normalize(axis))


# ---------------------------------------------------------------------------
# 2. Embedding clients
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingClient:
    provider: Provider
    model: str | None = None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed ordinary documents/passages."""
        if self.provider == "openai":
            return embed_openai(texts, model=self.model or "text-embedding-3-small")
        if self.provider == "cohere":
            return embed_cohere(
                texts,
                model=self.model or "embed-v4.0",
                input_type="search_document",
            )
        raise ValueError(f"Unknown provider: {self.provider}")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. Cohere distinguishes queries from documents."""
        if self.provider == "openai":
            return embed_openai([text], model=self.model or "text-embedding-3-small")[0]
        if self.provider == "cohere":
            return embed_cohere(
                [text],
                model=self.model or "embed-v4.0",
                input_type="search_query",
            )[0]
        raise ValueError(f"Unknown provider: {self.provider}")



def embed_openai(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    from openai import OpenAI

    require_env("OPENAI_API_KEY")
    client = OpenAI()
    response = client.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.asarray(vectors, dtype=float)



def embed_cohere(
    texts: list[str],
    model: str = "embed-v4.0",
    input_type: str = "search_document",
    output_dimension: int = 1024,
) -> np.ndarray:
    import cohere

    api_key = require_env("COHERE_API_KEY")
    co = cohere.ClientV2(api_key=api_key)
    response = co.embed(
        model=model,
        texts=texts,
        input_type=input_type,
        output_dimension=output_dimension,
        embedding_types=["float"],
    )
    return np.asarray(response.embeddings.float, dtype=float)


# ---------------------------------------------------------------------------
# 3. Core examples
# ---------------------------------------------------------------------------



def example_classic_word2vec_dimensions() -> None:
    section("1. Classic word2vec: Kozlowski-style cultural dimensions")
    explain(
        "We start with the older static-embedding setup because it makes the geometry very "
        "clear. Train one word2vec model for an early corpus and another for a late corpus. "
        "Each word has one vector per period. Then define a dimension by averaging ordered "
        "opposite-pair differences, such as hardworking-dangerous or loyal-suspicious."
    )

    try:
        from gensim.models import Word2Vec
    except Exception as e:
        print(f"Skipping word2vec section because gensim is not available: {e}")
        print("Install with: pip install gensim")
        return

    early_sentences = [tokenize_for_word2vec(text) for text in WORD2VEC_EARLY_TEXTS * 80]
    late_sentences = [tokenize_for_word2vec(text) for text in WORD2VEC_LATE_TEXTS * 80]

    early_model = Word2Vec(
        sentences=early_sentences,
        vector_size=80,
        window=6,
        min_count=1,
        workers=1,
        sg=1,
        negative=10,
        epochs=60,
        seed=42,
    )
    late_model = Word2Vec(
        sentences=late_sentences,
        vector_size=80,
        window=6,
        min_count=1,
        workers=1,
        sg=1,
        negative=10,
        epochs=60,
        seed=42,
    )

    early_axis, early_info = paired_word2vec_axis(early_model, KOZLOWSKI_STYLE_PAIRS)
    late_axis, late_info = paired_word2vec_axis(late_model, KOZLOWSKI_STYLE_PAIRS)

    print("Pole pairs used in the early model:", early_info["used_pairs"])
    print("Pole pairs used in the late model:", late_info["used_pairs"])
    print()

    targets = ["immigrant", "foreign", "worker", "dangerous", "hardworking", "community"]
    rows = []
    for word in targets:
        rows.append(
            {
                "word": word,
                "early_positive_projection": word2vec_projection_score(early_model, word, early_axis),
                "late_positive_projection": word2vec_projection_score(late_model, word, late_axis),
            }
        )
    scores = pd.DataFrame(rows)
    print(scores.to_string(index=False))

    explain(
        "This is the Kozlowski-style logic: the dimension is not a visible vertical line. "
        "It is a direction in a high-dimensional space. The score is a cosine projection "
        "onto that direction. The same geometry will reappear below with sentence and "
        "document embeddings from APIs."
    )



def example_what_is_an_embedding(client: EmbeddingClient) -> tuple[pd.DataFrame, np.ndarray]:
    section("2. Modern API embeddings: text becomes a vector")
    explain(
        "An embedding model turns text into a long list of numbers. The numbers are not "
        "directly interpretable one by one. What matters is geometry: texts with similar "
        "usage or meaning tend to point in similar directions."
    )

    df = pd.DataFrame(SHORT_TEXTS)
    embeddings = client.embed_documents(df["text"].tolist())

    print(f"Provider: {client.provider}")
    print(f"Number of texts: {len(df)}")
    print(f"Embedding shape: {embeddings.shape}  # rows=texts, columns=dimensions")
    print("First 8 numbers of the first embedding:")
    print(np.round(embeddings[0][:8], 4))
    return df, embeddings



def example_cosine_similarity(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    section("3. Cosine similarity")
    explain(
        "Cosine similarity compares the angle between two vectors. It is usually more useful "
        "than raw Euclidean distance for embeddings because we care about direction: whether "
        "two texts point toward similar meanings."
    )

    sims = cosine_matrix(embeddings)
    sim_df = pd.DataFrame(sims, index=df["id"], columns=df["id"])
    print("Pairwise cosine similarity matrix, rounded:")
    print(sim_df.round(3))

    pairs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            pairs.append((df.loc[i, "id"], df.loc[j, "id"], sims[i, j]))
    pairs = sorted(pairs, key=lambda row: row[2], reverse=True)
    print("\nMost similar pairs:")
    for left, right, score in pairs[:5]:
        print(f"  {left:10s} {right:10s} cosine={score:.3f}")



def example_semantic_search(client: EmbeddingClient, df: pd.DataFrame, embeddings: np.ndarray) -> None:
    section("4. Semantic search")
    explain(
        "In semantic search, we embed the query and compare it with embedded documents. "
        "The top results are not exact keyword matches; they are documents whose vectors "
        "point in a similar direction to the query vector."
    )

    queries = [
        "Which speeches talk about helping poor families?",
        "Find arguments about defense and national security.",
        "Find market-oriented economic policy.",
    ]

    for query in queries:
        query_embedding = client.embed_query(query)
        scores = cosine_similarity_one_to_many(query_embedding, embeddings)
        ranked = np.argsort(scores)[::-1]
        print(f"\nQuery: {query}")
        for idx in ranked[:3]:
            print(f"  {df.loc[idx, 'id']:10s} score={scores[idx]:.3f}  {df.loc[idx, 'text']}")



def example_hierarchical_clustering(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    section("5. Hierarchical clustering")
    explain(
        "Clustering uses embeddings to group texts without predefined labels. Here we use "
        "agglomerative hierarchical clustering: start with each text alone, then repeatedly "
        "merge the closest texts or clusters."
    )

    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    normalized = l2_normalize(embeddings)
    distances = pdist(normalized, metric="cosine")
    tree = linkage(distances, method="average")
    order = leaves_list(tree)

    print("Dendrogram leaf order from closest semantic groupings:")
    for idx in order:
        print(f"  {df.loc[idx, 'id']:10s} group={df.loc[idx, 'group']:14s} text={df.loc[idx, 'text']}")

    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        os.makedirs("outputs", exist_ok=True)
        plt.figure(figsize=(9, 4))
        dendrogram(tree, labels=df["id"].tolist(), leaf_rotation=30)
        plt.title("Hierarchical clustering of short political texts")
        plt.ylabel("Cosine distance")
        plt.tight_layout()
        out = os.path.join("outputs", "hierarchical_clustering.png")
        plt.savefig(out, dpi=160)
        print(f"\nSaved dendrogram to {out}")
    except Exception as e:
        print(f"\nPlot skipped: {e}")



def example_parliamentary_scaling(client: EmbeddingClient) -> tuple[pd.DataFrame, np.ndarray]:
    section("6. Scaling parliamentary speeches")
    explain(
        "Scaling means placing documents on a substantively meaningful dimension. Here we "
        "make a simple left-right economic axis from anchor texts. This is not a replacement "
        "for careful validation, but it shows the core idea: define a direction, then project "
        "speeches onto that direction."
    )

    df = pd.DataFrame(PARLIAMENTARY_SPEECHES)
    speech_embeddings = client.embed_documents(df["text"].tolist())
    left_embeddings = client.embed_documents(LEFT_ANCHORS)
    right_embeddings = client.embed_documents(RIGHT_ANCHORS)

    left_centroid = l2_normalize(left_embeddings).mean(axis=0)
    right_centroid = l2_normalize(right_embeddings).mean(axis=0)
    left_right_axis = right_centroid - left_centroid

    df["rightward_score"] = projection_scores(speech_embeddings, left_right_axis)
    df = df.sort_values("rightward_score")

    print("Negative scores are closer to the left/public-investment anchors.")
    print("Positive scores are closer to the right/market-discipline anchors.\n")
    print(df[["speech_id", "speaker", "party_position", "rightward_score", "text"]].to_string(index=False))

    explain(
        "What to discuss with students: the axis depends on the anchor texts, the corpus, "
        "and the embedding model. The numbers look precise, but interpretation requires "
        "validation against expert coding, manifestos, roll-call votes, or historical knowledge."
    )
    return df, speech_embeddings



def example_government_opposition_axis(df: pd.DataFrame, speech_embeddings: np.ndarray) -> None:
    section("7. Scaling from known groups: government vs opposition")
    explain(
        "Another common move is to use known groups as anchors. We average the government "
        "speeches, average the opposition speeches, subtract the centroids, and project all "
        "speeches onto that direction. This is useful, but can confuse party position with "
        "topic, time period, speaker style, or government responsibility."
    )

    gov_idx = df.index[df["party_position"] == "government"].tolist()
    opp_idx = df.index[df["party_position"] == "opposition"].tolist()
    gov_centroid = l2_normalize(speech_embeddings[gov_idx]).mean(axis=0)
    opp_centroid = l2_normalize(speech_embeddings[opp_idx]).mean(axis=0)
    gov_opp_axis = gov_centroid - opp_centroid

    scored = df.copy()
    scored["government_similarity_score"] = projection_scores(speech_embeddings, gov_opp_axis)
    scored = scored.sort_values("government_similarity_score")
    print(scored[["speech_id", "party_position", "government_similarity_score", "text"]].to_string(index=False))



def optional_bertopic(df: pd.DataFrame, embeddings: np.ndarray, run: bool) -> None:
    section("8. Optional BERTopic")
    if not run:
        explain(
            "BERTopic is skipped by default because it requires extra packages. Conceptually, "
            "BERTopic combines document embeddings, dimensionality reduction, clustering, "
            "and topic labels. Run with --bertopic after installing bertopic if you want this part."
        )
        return

    try:
        from bertopic import BERTopic
    except Exception as e:
        print(f"BERTopic is not available: {e}")
        print("Install with: pip install bertopic umap-learn hdbscan")
        return

    docs = df["text"].tolist()
    topic_model = BERTopic(verbose=False)
    topics, probs = topic_model.fit_transform(docs, embeddings)
    result = df.copy()
    result["topic"] = topics
    print(result[["id", "group", "topic", "text"]].to_string(index=False))
    print("\nTopic summary:")
    print(topic_model.get_topic_info().to_string(index=False))



def example_failure_modes() -> None:
    section("9. Where embeddings break")
    points = [
        "Similarity is not explanation. Two texts can be close because of topic, style, genre, or repeated phrases.",
        "Embeddings inherit biases from training data and from the corpus being analyzed.",
        "Scaling axes are constructed by researchers; different anchors can produce different dimensions.",
        "Short texts, sarcasm, negation, and multilingual mixing can be hard.",
        "Historical text has OCR errors, spelling variation, and changing meanings.",
        "Clusters always exist mathematically, but not every cluster is substantively meaningful.",
        "API models may change over time; record provider, model name, date, preprocessing, and parameters.",
    ]
    for p in points:
        print(f"- {p}")


# ---------------------------------------------------------------------------
# 4. Main program
# ---------------------------------------------------------------------------



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="API-first embeddings lecture demo for social scientists")
    parser.add_argument("--provider", choices=["openai", "cohere"], default="openai")
    parser.add_argument("--model", default=None, help="Optional model override")
    parser.add_argument("--bertopic", action="store_true", help="Run optional BERTopic section")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    client = EmbeddingClient(provider=args.provider, model=args.model)

    section("Embeddings for Social Scientists")
    explain(
        "We will use a small set of political texts so the mechanics are visible. In real "
        "research, the same operations apply to thousands or millions of documents: embed, "
        "compare, search, cluster, scale, and validate."
    )

    example_classic_word2vec_dimensions()
    df, embeddings = example_what_is_an_embedding(client)
    example_cosine_similarity(df, embeddings)
    example_semantic_search(client, df, embeddings)
    example_hierarchical_clustering(df, embeddings)
    parliament_df, parliament_embeddings = example_parliamentary_scaling(client)
    example_government_opposition_axis(parliament_df.sort_index(), parliament_embeddings)
    optional_bertopic(df, embeddings, run=args.bertopic)
    example_failure_modes()


if __name__ == "__main__":
    main()





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
3. OpenAI, Cohere, and open Sentence-BERT embeddings.
4. Cosine similarity.
5. Semantic search.
6. Hierarchical clustering.
7. A simple scaling exercise on toy parliamentary speeches.
8. Optional BERTopic with precomputed embeddings.
9. Where embeddings break.

Before class
------------
Install dependencies:

    pip install openai cohere sentence-transformers gensim numpy pandas scikit-learn scipy matplotlib

Optional topic-modeling section:

    pip install bertopic umap-learn hdbscan

Set API keys in a local .env file or in your shell. Do not paste real keys into tracked code.

Recommended local .env file, not committed to Git:

    OPENAI_API_KEY=sk-...
    COHERE_API_KEY=...

PowerShell alternative:

    $env:OPENAI_API_KEY="sk-..."
    $env:COHERE_API_KEY="..."

macOS/Linux alternative:

    export OPENAI_API_KEY="sk-..."
    export COHERE_API_KEY="..."

Run examples:

    python embeddings_for_social_scientists_api_demo.py --provider openai
    python embeddings_for_social_scientists_api_demo.py --provider cohere
    python embeddings_for_social_scientists_api_demo.py --provider sentence-transformers
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
import re
import json
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd


Provider = Literal["openai", "cohere", "sentence-transformers"]


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
        "speech_id": "SOC_01",
        "speaker": "Socialist MP A",
        "party": "Socialist Party",
        "party_family": "socialist",
        "party_position": "opposition",
        "text": "Workers need stronger bargaining rights, public housing, and protection from rising rents.",
    },
    {
        "speech_id": "SOC_02",
        "speaker": "Socialist MP B",
        "party": "Socialist Party",
        "party_family": "socialist",
        "party_position": "opposition",
        "text": "The budget should tax wealth, fund hospitals, and guarantee decent wages for every worker.",
    },
    {
        "speech_id": "SOC_03",
        "speaker": "Socialist MP C",
        "party": "Socialist Party",
        "party_family": "socialist",
        "party_position": "opposition",
        "text": "Public investment and social ownership are needed to reduce inequality and empower labor.",
    },
    {
        "speech_id": "LIB_01",
        "speaker": "Liberal MP A",
        "party": "Liberal Party",
        "party_family": "liberal",
        "party_position": "government",
        "text": "Economic recovery depends on private investment, lower compliance costs, and stable public accounts.",
    },
    {
        "speech_id": "LIB_02",
        "speaker": "Liberal MP B",
        "party": "Liberal Party",
        "party_family": "liberal",
        "party_position": "government",
        "text": "We support entrepreneurs by reducing red tape, encouraging exports, and keeping debt under control.",
    },
    {
        "speech_id": "CON_01",
        "speaker": "Conservative MP A",
        "party": "Conservative Party",
        "party_family": "conservative",
        "party_position": "government",
        "text": "National security requires border control, disciplined spending, and support for the armed forces.",
    },
    {
        "speech_id": "GRN_01",
        "speaker": "Green MP A",
        "party": "Green Party",
        "party_family": "green",
        "party_position": "opposition",
        "text": "Climate policy must combine renewable energy, public transport, and a just transition for workers.",
    },
    {
        "speech_id": "UNK_01",
        "speaker": "Unlabeled MP X",
        "party": "Unknown",
        "party_family": "unknown",
        "party_position": "unknown",
        "text": "The country needs an industrial strategy with public investment and stronger protections for workers.",
    },
    {
        "speech_id": "UNK_02",
        "speaker": "Unlabeled MP Y",
        "party": "Unknown",
        "party_family": "unknown",
        "party_position": "unknown",
        "text": "Growth requires business confidence, lower taxes, and predictable fiscal rules.",
    },
]

SOCIALIST_WORLDVIEW_ANCHORS = [
    "worker solidarity, class struggle, exploitation, collective ownership, public control",
    "labor power, capital accumulation, inequality, unions, social rights",
]

MARKET_WORLDVIEW_ANCHORS = [
    "entrepreneurship, investment, business confidence, market incentives, competitiveness",
    "private enterprise, deregulation, fiscal discipline, innovation, productivity",
]

OCCUPATION_TERMS = ["worker", "nurse", "teacher", "banker", "consultant", "entrepreneur", "landlord", "engineer"]

LEFT_ANCHORS = [
    "workers rights, public hospitals, redistributive taxation, social housing, stronger welfare state",
    "labor protections, public investment, inequality reduction, affordable housing, universal services",
]

RIGHT_ANCHORS = [
    "lower taxes, private investment, fiscal discipline, deregulation, competitive markets",
    "business confidence, balanced budgets, entrepreneurship, smaller government, market incentives",
]

CONTEXT_TOY_TEXTS = [
    "revolver honor loyalty brotherhood respect leader oath",
    "weapon gift signals trust courage loyalty brotherhood honor",
    "revolver craftsmanship collection heritage hunting tradition",
    "engraved pistol collectors admire design craftsmanship history",
    "revolver customs regulation export control decommission paperwork embassy",
    "diplomatic gift creates legal compliance and customs paperwork",
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



def load_local_env(path: str = ".env") -> None:
    """Load simple KEY=VALUE lines from a local .env file if present.

    This avoids putting API keys in tracked Python code. Existing shell
    environment variables win over values in .env.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\'").strip('"')
        if key and value and key not in os.environ:
            os.environ[key] = value


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
# 2. Parliamentary data loading, including ParlaMint-style files
# ---------------------------------------------------------------------------


def load_parliamentary_data(path=None, limit=200):
    """Load toy data or a local parliamentary corpus extract.

    Recommended research path: convert ParlaMint XML to a clean CSV/Parquet file
    with at least a text column and, ideally, party/speaker/date/country metadata.
    For teaching, this function also handles simple ParlaMint-style XML directly.
    """
    if not path:
        df = pd.DataFrame(PARLIAMENTARY_SPEECHES)
        return df.head(limit) if limit else df

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Parliamentary data path does not exist: {input_path}")

    if input_path.is_dir():
        rows = []
        for xml_path in sorted(input_path.rglob("*.xml")):
            rows.extend(load_parlamint_xml_file(xml_path))
            if limit and len(rows) >= limit:
                break
        return normalize_parliamentary_frame(pd.DataFrame(rows), limit=limit)

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return normalize_parliamentary_frame(pd.read_csv(input_path), limit=limit)
    if suffix in {".jsonl", ".ndjson"}:
        with input_path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return normalize_parliamentary_frame(pd.DataFrame(rows), limit=limit)
    if suffix == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("data", data.get("records", []))
        return normalize_parliamentary_frame(pd.DataFrame(rows), limit=limit)
    if suffix == ".xml":
        return normalize_parliamentary_frame(pd.DataFrame(load_parlamint_xml_file(input_path)), limit=limit)
    raise ValueError("Supported formats: CSV, JSONL, JSON, XML, or a directory of XML files.")



def normalize_parliamentary_frame(df, limit=200):
    if df.empty:
        raise ValueError("No usable speeches found in the parliamentary data.")
    lower_columns = {col.lower(): col for col in df.columns}
    column_map = {}
    aliases = {
        "text": ["text", "speech", "body", "content", "utterance"],
        "speech_id": ["speech_id", "id", "u_id", "utterance_id"],
        "speaker": ["speaker", "speaker_name", "who", "mp", "person"],
        "party": ["party", "party_name", "political_party", "parliamentary_group"],
        "party_family": ["party_family", "ideology", "orientation", "left_right"],
        "party_position": ["party_position", "government_opposition", "role", "power"],
        "date": ["date", "sitting_date"],
        "country": ["country", "parliament", "corpus"],
    }
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in lower_columns:
                column_map[lower_columns[candidate]] = canonical
                break
    df = df.rename(columns=column_map).copy()
    if "text" not in df.columns:
        raise ValueError("Parliamentary data needs a text/speech/body/content column.")
    df["text"] = df["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 30].copy()
    for col in ["speech_id", "speaker", "party", "party_family", "party_position", "date", "country"]:
        if col not in df.columns:
            df[col] = "unknown"
    if (df["speech_id"] == "unknown").all():
        df["speech_id"] = [f"speech_{i:04d}" for i in range(len(df))]
    cols = ["speech_id", "speaker", "party", "party_family", "party_position", "date", "country", "text"]
    df = df[cols].reset_index(drop=True)
    return df.head(limit) if limit else df



def load_parlamint_xml_file(path):
    rows = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return rows
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag != "u":
            continue
        text = " ".join(part.strip() for part in elem.itertext() if part and part.strip())
        if len(text) < 30:
            continue
        rows.append({
            "speech_id": elem.attrib.get("{http://www.w3.org/XML/1998/namespace}id", elem.attrib.get("id", "unknown")),
            "speaker": elem.attrib.get("who", "unknown"),
            "party": elem.attrib.get("ana", "unknown"),
            "party_family": "unknown",
            "party_position": "unknown",
            "date": "unknown",
            "country": Path(path).parent.name,
            "text": text,
        })
    return rows


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



def embed_sentence_transformer(
    texts: list[str],
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """Generate embeddings with an open Sentence-BERT/Sentence-Transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Sentence-Transformers is not installed. Install with: "
            "pip install sentence-transformers torch"
        ) from e

    encoder = SentenceTransformer(model)
    return encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(float)



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




def example_basic_cooccurrence_embeddings() -> None:
    section("1. Basic word embeddings: co-occurrence matrix + SVD")
    explain(
        "We start from the simplest implementation. Count which words occur near each other, "
        "then reduce the co-occurrence matrix with SVD. This is a toy version of the idea that "
        "meaning is learned from neighboring words."
    )
    tokenized = [tokenize(text) for text in CONTEXT_TOY_TEXTS]
    vocab = sorted({word for doc in tokenized for word in doc})
    idx = {word: i for i, word in enumerate(vocab)}
    counts = np.zeros((len(vocab), len(vocab)))
    for doc in tokenized:
        for i, word in enumerate(doc):
            for j in range(max(0, i - 2), min(len(doc), i + 3)):
                if i != j:
                    counts[idx[word], idx[doc[j]]] += 1
    u, svals, vt = np.linalg.svd(counts, full_matrices=False)
    coords = u[:, :2] * svals[:2]
    for word in ["revolver", "honor", "craftsmanship", "customs", "regulation", "loyalty"]:
        if word in idx:
            x, y = coords[idx[word]]
            print(f"  {word:15s} x={x: .3f} y={y: .3f}")
    explain(
        "This toy example implements the NATO-hook intuition: the same object is pulled toward "
        "different semantic neighborhoods depending on the text around it."
    )


def example_classic_word2vec_dimensions() -> None:
    section("2. Classic word2vec: Kozlowski-style cultural dimensions")
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
    section("3. Modern API embeddings: text becomes a vector")
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
    section("4. Cosine similarity")
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
    section("5. Semantic search")
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
    section("6. Hierarchical clustering")
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



def example_parliamentary_scaling(client: EmbeddingClient, parliamentary_data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    section("7. Scaling parliamentary speeches")
    explain(
        "Scaling means placing documents on a substantively meaningful dimension. Here we "
        "make a simple left-right economic axis from anchor texts. This is not a replacement "
        "for careful validation, but it shows the core idea: define a direction, then project "
        "speeches onto that direction."
    )

    df = parliamentary_data.copy().reset_index(drop=True)
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
    print(df[["speech_id", "speaker", "party", "party_family", "party_position", "rightward_score", "text"]].to_string(index=False))

    explain(
        "What to discuss with students: the axis depends on the anchor texts, the corpus, "
        "and the embedding model. The numbers look precise, but interpretation requires "
        "validation against expert coding, manifestos, roll-call votes, or historical knowledge."
    )
    return df, speech_embeddings



def example_government_opposition_axis(df: pd.DataFrame, speech_embeddings: np.ndarray) -> None:
    section("8. Scaling from known groups: government vs opposition")
    explain(
        "Another common move is to use known groups as anchors. We average the government "
        "speeches, average the opposition speeches, subtract the centroids, and project all "
        "speeches onto that direction. This is useful, but can confuse party position with "
        "topic, time period, speaker style, or government responsibility."
    )

    gov_idx = df.index[df["party_position"] == "government"].tolist()
    opp_idx = df.index[df["party_position"] == "opposition"].tolist()
    if not gov_idx or not opp_idx:
        print("Need both government and opposition speeches for this axis.")
        return
    gov_centroid = l2_normalize(speech_embeddings[gov_idx]).mean(axis=0)
    opp_centroid = l2_normalize(speech_embeddings[opp_idx]).mean(axis=0)
    gov_opp_axis = gov_centroid - opp_centroid

    scored = df.copy()
    scored["government_similarity_score"] = projection_scores(speech_embeddings, gov_opp_axis)
    scored = scored.sort_values("government_similarity_score")
    print(scored[["speech_id", "party_position", "government_similarity_score", "text"]].to_string(index=False))



def optional_bertopic(df: pd.DataFrame, embeddings: np.ndarray, run: bool) -> None:
    section("9. Optional BERTopic")
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
    display_cols = [col for col in ["id", "speech_id", "party", "party_family", "topic", "text"] if col in result.columns]
    print(result[display_cols].to_string(index=False))
    print("\nTopic summary:")
    print(topic_model.get_topic_info().to_string(index=False))




def example_worldview_lens(client: EmbeddingClient, df: pd.DataFrame, embeddings: np.ndarray) -> None:
    section("10. Nelimarkka-style worldview lens")
    explain(
        "A classroom version of the Nelimarkka argument is to stop treating the model as neutral. "
        "Here we build a socialist-vs-market lens from anchor texts, then rank occupations and "
        "speeches on that lens. This is not fine-tuning yet; it shows the worldview construct."
    )
    socialist_embeddings = client.embed_documents(SOCIALIST_WORLDVIEW_ANCHORS)
    market_embeddings = client.embed_documents(MARKET_WORLDVIEW_ANCHORS)
    socialist_centroid = l2_normalize(socialist_embeddings).mean(axis=0)
    market_centroid = l2_normalize(market_embeddings).mean(axis=0)
    socialist_market_axis = socialist_centroid - market_centroid

    occupation_embeddings = client.embed_documents(OCCUPATION_TERMS)
    occupation_scores = pd.DataFrame({
        "term": OCCUPATION_TERMS,
        "socialist_worldview_score": projection_scores(occupation_embeddings, socialist_market_axis),
    }).sort_values("socialist_worldview_score", ascending=False)
    print("Occupation terms ranked by socialist-market anchor dimension:")
    print(occupation_scores.to_string(index=False))

    scored = df.copy()
    scored["socialist_worldview_score"] = projection_scores(embeddings, socialist_market_axis)
    scored = scored.sort_values("socialist_worldview_score", ascending=False)
    print("\nSpeeches ranked by the same dimension:")
    print(scored[["speech_id", "party_family", "socialist_worldview_score", "text"]].to_string(index=False))



def optional_sentence_transformer_finetuning(df: pd.DataFrame, run: bool) -> None:
    section("11. Optional local fine-tuning: socialist-party domain adaptation")
    if not run:
        explain(
            "Fine-tuning is skipped by default. APIs give embeddings; local Sentence-Transformers "
            "let us update model weights. Run with --fine-tune-local to demonstrate a small "
            "Nelimarkka-style domain/worldview adaptation using socialist party speeches."
        )
        return
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except Exception as e:
        print(f"Fine-tuning dependencies unavailable: {e}")
        print("Install with: pip install sentence-transformers torch")
        return

    socialist = df[df["party_family"].str.contains("social", case=False, na=False)]["text"].tolist()
    market = df[df["party_family"].str.contains("liberal|conservative", case=False, na=False)]["text"].tolist()
    if len(socialist) < 2 or len(market) < 2:
        print("Need at least two socialist and two liberal/conservative speeches for this demo.")
        return

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    probes = [
        "workers need collective bargaining and public investment",
        "business confidence requires tax cuts and deregulation",
        "the nurse and the factory worker deserve higher wages",
        "the banker and consultant create investment opportunities",
    ]
    baseline = model.encode(probes + [SOCIALIST_WORLDVIEW_ANCHORS[0]], normalize_embeddings=True)
    baseline_anchor = baseline[-1]

    train_examples = []
    for text in socialist:
        train_examples.append(InputExample(texts=[text, SOCIALIST_WORLDVIEW_ANCHORS[0]], label=1.0))
        train_examples.append(InputExample(texts=[text, MARKET_WORLDVIEW_ANCHORS[0]], label=0.0))
    for text in market:
        train_examples.append(InputExample(texts=[text, MARKET_WORLDVIEW_ANCHORS[0]], label=1.0))
        train_examples.append(InputExample(texts=[text, SOCIALIST_WORLDVIEW_ANCHORS[0]], label=0.0))

    loader = DataLoader(train_examples, shuffle=True, batch_size=4)
    loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=1, warmup_steps=0, show_progress_bar=True)

    adapted = model.encode(probes + [SOCIALIST_WORLDVIEW_ANCHORS[0]], normalize_embeddings=True)
    adapted_anchor = adapted[-1]
    rows = []
    for i, probe in enumerate(probes):
        rows.append({
            "probe": probe,
            "baseline_to_socialist_anchor": float(baseline[i] @ baseline_anchor),
            "fine_tuned_to_socialist_anchor": float(adapted[i] @ adapted_anchor),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    explain(
        "This is a teaching-scale fine-tuning demo, not a publishable design. For research, use "
        "held-out validation, multiple seeds, comparison models, and explicit construct validation."
    )


def example_failure_modes() -> None:
    section("12. Where embeddings break")
    points = [
        "Similarity is not explanation. Two texts can be close because of topic, style, genre, or repeated phrases.",
        "Embeddings inherit biases from training data and from the corpus being analyzed.",
        "Scaling axes are constructed by researchers; different anchors can produce different dimensions.",
        "Short texts, sarcasm, negation, and multilingual mixing can be hard.",
        "Historical text has OCR errors, spelling variation, and changing meanings.",
        "Clusters always exist mathematically, but not every cluster is substantively meaningful.",
        "API models may change over time; record provider, model name, date, preprocessing, and parameters.",
        "For ParlaMint, chunking, metadata harmonization, and language/model choice are part of the method.",
        "Fine-tuning can surface a worldview, but can also overfit or amplify researcher choices.",
    ]
    for p in points:
        print(f"- {p}")


# ---------------------------------------------------------------------------
# 4. Main program
# ---------------------------------------------------------------------------



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="API-first embeddings lecture demo for social scientists")
    parser.add_argument("--provider", choices=["openai", "cohere", "sentence-transformers"], default="openai")
    parser.add_argument("--model", default=None, help="Optional model override")
    parser.add_argument("--bertopic", action="store_true", help="Run optional BERTopic section")
    parser.add_argument("--parliament-path", default=None, help="CSV/JSONL/JSON/XML file or ParlaMint XML directory")
    parser.add_argument("--limit", type=int, default=200, help="Maximum speeches to load from local parliamentary data")
    parser.add_argument("--fine-tune-local", action="store_true", help="Run optional local Sentence-Transformer fine-tuning demo")
    return parser.parse_args()



def main() -> None:
    load_local_env()
    args = parse_args()
    client = EmbeddingClient(provider=args.provider, model=args.model)

    section("Embeddings for Social Scientists")
    explain(
        "The slides introduced the concepts. This script implements them: word embeddings, "
        "Kozlowski-style dimensions, OpenAI/Cohere/Sentence-BERT embeddings, parliamentary clustering, "
        "semantic search, scaling, and Nelimarkka-style worldview adaptation."
    )

    parliamentary_data = load_parliamentary_data(args.parliament_path, limit=args.limit)
    print(f"Parliamentary data loaded: {len(parliamentary_data)} speeches")
    if args.parliament_path:
        print(f"Source: {args.parliament_path}")
    else:
        print("Source: built-in toy speeches. Use --parliament-path for ParlaMint or another corpus.")

    example_basic_cooccurrence_embeddings()
    example_classic_word2vec_dimensions()
    df, embeddings = example_what_is_an_embedding(client)
    example_cosine_similarity(df, embeddings)
    example_semantic_search(client, df, embeddings)
    example_hierarchical_clustering(df, embeddings)
    parliament_df, parliament_embeddings = example_parliamentary_scaling(client, parliamentary_data)
    example_government_opposition_axis(parliament_df.sort_index(), parliament_embeddings)
    optional_bertopic(parliament_df.rename(columns={"speech_id": "id"}), parliament_embeddings, run=args.bertopic)
    example_worldview_lens(client, parliament_df, parliament_embeddings)
    optional_sentence_transformer_finetuning(parliament_df, run=args.fine_tune_local)
    example_failure_modes()


if __name__ == "__main__":
    main()





# embeddings_sicss

Teaching materials for a SICSS-style lecture on using word embeddings to study cultural change in texts.

## Contents

- `cultural_change_embeddings_colab.ipynb`: Colab-ready notebook with a lightweight classroom demo and optional historical newspaper data section.

## Student Tracks

1. **Simple machines / standard Colab CPU**
   - Use the synthetic corpus section.
   - Optionally run a small Chronicling America sample.
   - Suitable for learning the method during class.

2. **Better machines / Colab Pro / lab server**
   - Increase the number of downloaded newspaper pages or use larger article-level corpora.
   - Classical word2vec mostly needs CPU, RAM, and disk, not GPU.
   - GPU is useful for transformer-based contextual embeddings.

## Main Lesson

Embeddings estimate meanings from patterns of word co-occurrence. When trained on newspapers, they estimate meanings in newspaper discourse, not society as a whole.

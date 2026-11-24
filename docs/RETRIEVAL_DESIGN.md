# Retrieval Design

All numbers here were measured on this project's corpus, not taken from papers.

## 1. Pipeline

```
query ─┬─► QueryClassifier ─► CONVERSATIONAL? ─► stop (no retrieval, no citations)
       │
       └─► legal path
            ├─ Dense:  embed(query) ─► Qdrant ANN ─► payload post-filter
            ├─ Sparse: normalize ─► tokenize ─► BM25 ─► filter
            └─ RRF(k=60) ─► top 20 ─► cross-encoder ─► probability ─► top 5
```

## 2. Arabic normalization

Applied **identically** to indexed chunks and to queries. Asymmetry here is the
classic cause of "the text exists but retrieval misses it".

| Transform | Effect |
|---|---|
| Diacritics `\u064B-\u0652\u0670` | removed |
| Tatweel `\u0640` | removed |
| Hamza folding | `أ إ آ → ا`, `ى → ي`, `ة → ه` |
| **Digit folding** | `١٤٧ → 147` |
| Punctuation | → whitespace |

**Digit folding is load-bearing.** The corpus writes `المادة 147`; a user asks
«المادة ١٤٧». Verified working:

```
BM25 QUERY | query=ما حكم المادة ١٤٧ من القانون المدني المصري؟
           | tokens=['حكم', 'الماده', '147', 'القانون', 'المدني', 'المصري']
```

**Ta-marbuta asymmetry (deliberate).** `SparseRetriever` enables
`fold_ta_marbuta=True` so `السرقة` matches `السرقه`. Stored content and dense
embeddings are **not** modified — only the BM25 token stream. This raises recall
for keyword search without corrupting the text shown to users or embedded for
vector search.

## 3. Stopwords

~60 function words, written in natural Arabic and passed through the *same*
normalizer so folding stays symmetric. `المادة` is **not** a stopword —
article-number queries are the single most common legal lookup.

Observed effect on a meta question:

```
"ما اخر سوال سألته لك؟"  →  tokens=['اخر', 'سوال', 'سالته', 'لك']
```

Four tokens, zero legal signal, 4,429 matches in a 45k corpus. This is precisely
why `QueryClassifier` must route such questions away from retrieval.

## 4. Fusion — Reciprocal Rank Fusion

```
score(d) = Σ  1 / (k + rank_i(d) + 1)      k = RRF_K = 60
```

Provenance is **preserved** on every fused chunk: `dense_score`, `sparse_score`,
`rank` (1-based), and a `sources` set. This is diagnostic, not cosmetic:

```
RRF FUSION | candidates=N | returned=20 | both=X | dense_only=Y | sparse_only=Z
```

`dense_only` being high is the signature of a **stale BM25 index** — exactly how
the "uploaded document is invisible to search" bug was identified.

### Why orphaned vectors beat new ones

With 45,170 orphaned chunks and 32 new ones, a new chunk can only appear in the
dense list (BM25 was frozen at startup):

| Candidate | RRF |
|---|---|
| New chunk, dense rank 1 | `1/61` = **0.0164** |
| Old chunk, BM25 r1 + dense r5 | `1/61 + 1/65` = **0.0318** |

Old chunks score ~2× higher, then `RETRIEVAL_TOP_K=20 → RERANKER_TOP_K=5`
discards the rest. **Conclusion: corpus hygiene is a retrieval requirement, not
an ops nicety.**

## 5. Reranking — the activation trap

`CrossEncoder.predict()` applies `default_activation_function`, which
sentence-transformers sets to `nn.Sigmoid()` for single-label models like
`bge-reranker-base`. Applying a *second* sigmoid on top compresses everything:

| Broken (double sigmoid) | Healthy (identity + one sigmoid) |
|---|---|
| `top=0.5228  bottom=0.5000` | `logit_min≈-9  logit_max≈+6` |
| spread 0.023 | spread ~15 |

Inverting the broken scores: `0.5000 → logit 0.000`, `0.5228 → logit 0.091`. A
floor pinned at exactly `sigmoid(0)` across **every** query is the fingerprint of
a clamp, not a model.

**Detection is built in and must be watched:**

```
RERANK RAW | logit_min=… | logit_max=… | spread=… | prob_min=… | prob_max=…
RERANK NO SIGNAL | logit_spread=0.0001 across 20 candidates.
RERANK DEGENERATE | spread=0.0000 … using hybrid order.
```

`spread < 0.1` ⇒ the pipeline falls back to RRF order rather than trusting a
random shuffle, and logs at ERROR.

**Thresholds** (`config/retrieval_config.py`):

| Constant | Value | Meaning |
|---|---|---|
| `MIN_DENSE_SCORE` | 0.35 | cosine similarity floor |
| `MIN_RERANK_SCORE` | 0.1 | probability floor — sane for *true* probabilities |

⚠️ `MIN_RERANK_SCORE=0.1` is correct **only** after the activation fix. Against
the broken `[0.5, 0.622]` range it admitted everything; against a clean corpus
with a fixed model it drops genuinely irrelevant candidates. If every question
starts returning «لا تتوفر معلومات كافية» right after the fix, the corpus is
polluted — not the threshold.

## 6. Cost profile (measured, CPU)

| Stage | Cost | Share |
|---|---|---|
| Dense retrieval | 0.15–0.40 s | ~3% |
| Sparse retrieval | 0.04–0.14 s | ~1% |
| **Rerank 20 candidates** | **7.4–8.8 s** | **65–88%** |
| Groq generation | 0.7–3.6 s | ~20% |
| Total | ~10–12 s | |

Reranking dominates. Capping `OMP_NUM_THREADS=2` protects HTTP responsiveness
but makes the CPU-bound reranker slower — a deliberate trade. The two ways out:
run the reranker on GPU (needs >4 GB free VRAM; e5-base + bge-reranker at fp16 ≈
2.2 GB, so it is marginal on a P2000), or disable it and rely on RRF over a clean
corpus.

## 7. BM25 index economics

| Corpus | Build cost |
|---|---|
| 45,202 chunks | **28.1 s** (177 Qdrant scroll round-trips, 45M chars normalized) |
| 32 chunks | <1 s |

`rank_bm25.get_scores` is pure Python and holds the GIL: ~67 ms for a 10-token
query over 45k documents. Not a bottleneck at query time — but the **startup**
cost is, because it runs inside the async lifespan and blocks everything.

Mitigations, in order of value:
1. Keep the corpus clean (delete orphaned vectors).
2. `RetrievalPipeline.ensure_fresh()` — incremental `add_chunks` instead of restart.
3. Persist the tokenised corpus to `artifacts/indexes/` keyed on a fingerprint
   (point count + collection name); rebuild only on change. **Not implemented.**

## 8. Filtering contract

See `ARCHITECTURE.md` §4 for the policy. Implementation requirements:

- One evaluator (`MetadataFilter.matches`) shared by dense post-filter, sparse
  filter and SQL.
- Push-down is best-effort; the post-filter is mandatory.
- Translation failure raises. Never search unfiltered.
- Over-fetch when filtered, so `top_k` survives post-filtering.
- Filter **before** the `top_k` cut, not after — otherwise a scoped query returns
  a handful of results instead of a full page.

## 9. Evaluation

`src/evaluation/` implements retrieval, citation, answer and generation metrics
plus Arabic-specific evaluators, and `data/evaluation/` is configured. **Nothing
runs them.** Until a golden question set exists with expected articles, every
retrieval change here is unverifiable. This is the highest-leverage missing piece
— see `ROADMAP.md` Phase 8.

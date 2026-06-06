# Building AI Systems — Study Notes

> Synthesised from lectures 1–5 (SETU, Dr Jonathan Dunne).  
> Organised by concept, not by lecture. Suitable as a concept reference and exam revision guide.

---

## Table of Contents

1. [History & Evolution of Generative AI](#1-history--evolution-of-generative-ai)
2. [Large Language Models (LLMs)](#2-large-language-models-llms)
3. [Transformer Architecture](#3-transformer-architecture)
4. [Local vs Cloud Deployment](#4-local-vs-cloud-deployment)
5. [Ollama & OpenAI Ecosystem](#5-ollama--openai-ecosystem)
6. [Prompt Engineering](#6-prompt-engineering)
7. [Embeddings & Semantic Understanding](#7-embeddings--semantic-understanding)
8. [Vector Databases & Similarity Search](#8-vector-databases--similarity-search)
9. [Evaluation Frameworks](#9-evaluation-frameworks)
10. [Retrieval-Augmented Generation (RAG)](#10-retrieval-augmented-generation-rag)

---

## 1. History & Evolution of Generative AI

### The Symbolic Era (1950s–1980s)

- **Rule-based systems** with hand-crafted logic (expert systems).
- Examples: ELIZA (1966) — pattern-matching chatbot; MYCIN (1970s) — medical diagnosis.
- **Limitation**: Brittle, domain-narrow, could not generalise. Intelligence is more than rules.

### Neural Network Renaissance (1980s–2000s)

- **Backpropagation** (1986) enabled learning from data.
- MNIST handwritten digit recognition was a key benchmark.
- **Limitations**: Shallow networks, vanishing gradients, "AI Winter" funding collapses.

### Deep Learning Breakthrough (2010s)

Three convergent factors unlocked deep learning:

- **More data** — internet-scale datasets.
- **More compute** — GPU acceleration.
- **Better algorithms** — ReLU activations, dropout regularisation, batch normalisation.
- **Key milestone**: AlexNet (ImageNet 2012) cut error rate by ~40%, proving deep CNNs.

### The Attention Mechanism (2014–2017)

- RNNs process sequences one token at a time and suffer from vanishing long-range dependencies.
- **Attention** lets every token directly interact with every other token.
- Enables parallelisation and removes sequence-length bottlenecks.
- **"Attention Is All You Need"** (Vaswani et al., 2017) — introduced the Transformer.

### GPT Series: Scaling Up (2018–2020)

| Model | Year | Parameters | Capability unlocked |
|-------|------|-----------|---------------------|
| GPT-1 | 2018 | 117M | Pre-training on text works |
| GPT-2 | 2019 | 1.5B | Coherent long-form text generation |
| GPT-3 | 2020 | 175B | Few-shot learning emerges |

**Key insight**: Emergent abilities appear unexpectedly as model scale increases.

### The Current Landscape (2021–Present)

- **ChatGPT (2022)** — mainstream breakthrough bringing LLMs to the public.
- **Frontier models**: GPT-4, Claude 4.x, Gemini.
- **Open-source**: LLaMA, Mistral — run locally, inspect weights.
- Specialised models for coding, reasoning, multimodal tasks.
- Paradigm shift: research labs → production applications.

### Why It Happened Now

| Factor | Detail |
|--------|--------|
| Compute | Training on thousands of GPUs |
| Data | Internet-scale text and multimodal datasets |
| Algorithms | Transformers, efficient attention, optimisers |
| Money | Billions in investment |
| Infrastructure | Cloud computing and mature MLOps |

**Formula**: Modern AI = Transformers + Scale + Data

---

## 2. Large Language Models (LLMs)

### Core Task: Next-Token Prediction

LLMs are trained on a deceptively simple objective: *given a sequence of tokens, predict the next one*. From this single task, they develop compressed knowledge about language, facts, and reasoning.

> "The cat sat on the \_\_\_" → "mat"

Next-word prediction is surprisingly powerful because to do it well across all of human text, the model must implicitly understand grammar, facts, logic, and world knowledge.

### Architecture Pipeline

```
Input Text
    ↓
1. Tokeniser       (text → numerical tokens)
    ↓
2. Embedding Layer (tokens → dense vectors, ~768 dims)
    ↓
3. Transformer Blocks × 32–96  (multi-head attention + FFN)
    ↓
4. Output Layer    (vectors → probability distribution over vocabulary)
    ↓
Next Token Prediction
```

### Tokenisation

- Models do not see words — they see **subword units** (e.g., BPE).
- "Understanding tokenisation" → `["Under", "stand", "ing", "token", "isation"]`
- **Why subwords?** Handles unknown words, captures morphology, keeps vocabulary manageable (~50,000 tokens).
- **Practical implication**: Token count ≠ word count. Code, numbers, and non-English text are often less efficient (more tokens per word).

### Embeddings (in LLMs)

- Each token ID maps to a dense vector (e.g., 768 dimensions).
- **Key property**: Similar meanings → similar vectors.
- Classic example: `"king" − "man" + "woman" ≈ "queen"` — arithmetic in vector space encodes semantic relationships.

### Transformer Blocks

Each block contains:

- **Multi-Head Self-Attention** — tokens communicate with each other.
- **Feed-Forward Network** — processes each token independently.
- **Layer Normalisation** — stabilises training.
- **Residual Connections** — preserve information flow (avoids vanishing gradients).

Modern LLMs stack 32–96+ such blocks.

### Model Size & Scaling Laws

| Size | Parameters | Use case |
|------|-----------|---------|
| Small | 1–7B | Fast, local-friendly, specific tasks |
| Medium | 7–30B | Balanced capability and efficiency |
| Large | 30–70B | Strong reasoning, broad knowledge |
| Frontier | 70B+ | Cutting-edge capabilities |

> **More parameters ≠ always better.** Consider: speed, cost, deployment constraints, task requirements.

### Training Phases

**Phase 1 — Pre-training**

- Corpus: trillions of tokens (books, websites, code).
- Objective: predict next token (self-supervised).
- Duration: weeks to months on thousands of GPUs.
- Cost: millions to hundreds of millions of dollars.
- Result: a base model with broad knowledge.

**Phase 2 — Supervised Fine-Tuning (SFT)**

- Human-written instruction-response pairs teach desired behaviour patterns.
- Relatively quick and cheap.

**Phase 3 — RLHF (Reinforcement Learning from Human Feedback)**

- Humans rank model outputs; a reward model is trained on those preferences.
- The LLM is then optimised to produce responses the reward model prefers.
- Result: helpful, harmless, honest assistant behaviour.

### Capabilities & Limitations

| Strengths | Limitations |
|-----------|-------------|
| Text generation and transformation | Knowledge cutoff dates |
| Question answering and summarisation | Hallucination (generating false information confidently) |
| Code generation and debugging | No access to current information |
| Few-shot learning | Mathematical reasoning challenges |
| Chain-of-thought reasoning | Cannot execute code/tools without augmentation |

---

## 3. Transformer Architecture

### The Problem Attention Solves

RNNs process sequences token-by-token:

- Long-distance dependencies fade.
- Cannot parallelise training.
- Fixed hidden state creates an information bottleneck.

#### Example

Imagine this sentence:

>“The cat, which had been sleeping on the couch for hours, was hungry.”

To correctly predict “was”, the model needs to remember “cat” from far back.

**Question**: How can every word interact with every other word simultaneously?

### Self-Attention Mechanics

For each token, three vectors are computed:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

>Self-attention enables every word to interact with every other word simultaneously by computing pairwise similarity scores between all tokens using matrix multiplications, allowing parallel information exchange across the entire sequence.

Attention in this context refers to how tokens "attend" to one another. E.G, "The cat sat on the mat". Sat strongly attends to Cat (subject) and On/Mat (context)

**Attention formula**:

```
Attention(Q, K, V) = softmax(QKᵀ / √d) × V
```

- The dot product of Q and K measures compatibility.
- Dividing by √d prevents large dot products from pushing softmax into saturation.
- The output is a weighted sum of Values.

#### Scaled Dot-Product Attention (Short Explanation)

```
Attention(Q, K, V) = softmax(QKᵀ / √d) × V
```

- **QKᵀ**: Computes similarity between each token and all others
  → “Who should I pay attention to?”

- **/ √d**: Scales values to keep training stable
  → Prevents overly sharp softmax outputs

- **softmax(...)**: Converts scores into probabilities (weights)
  → Each row sums to 1

- **× V**: Applies weights to value vectors
  → Produces a weighted combination of all tokens

**Key idea:**
Each token becomes a weighted mix of all other tokens, computed in parallel.

**Intuition**: "The animal didn't cross the street because *it* was too tired" — attention learns that "it" should look at "animal", not "street".

### Multi-Head Attention

Running multiple attention operations in parallel (8–16 heads per layer):

- **Head 1**: may capture syntactic dependencies.
- **Head 2**: may capture semantic similarity.
- **Head 3**: may capture long-range coreference.

Outputs are concatenated and projected. Different heads learn complementary relationship types.

### Positional Encoding

Attention has no inherent notion of order ("Dog bites man" vs. "Man bites dog" would be identical without it). Solutions:

- Sinusoidal functions (original Transformer).
- Learned positional embeddings.
- Relative position encodings (modern LLMs, e.g., RoPE).

### Why Transformers Won

| Advantage | Detail |
|-----------|--------|
| Parallelisation | Entire sequences processed at once (unlike RNNs) |
| Long-range dependencies | Direct connections between any two tokens |
| Scalability | Efficient on GPUs/TPUs |
| Flexibility | Adaptable to text, code, images, audio |
| Interpretability | Attention patterns can be visualised |

---

## 4. Local vs Cloud Deployment

### Cloud APIs

**Providers**: OpenAI (GPT-4), Anthropic (Claude), Google (Gemini), IBM Watsonx, AWS Bedrock, Azure OpenAI.

**Advantages**:

- Cutting-edge model capability.
- No infrastructure management.
- Instant scalability.
- Regular model updates.

### Local Models

**Options**: LLaMA, Mistral, Phi, Granite — run via Ollama, LM Studio, or bare inference servers.

**Advantages**:

- Full data privacy (data never leaves your premises).
- No ongoing API costs after hardware purchase.
- No rate limits.
- Full customisation and fine-tuning.
- Works offline.

### Comparison Table

| Aspect | Cloud APIs | Local Models |
|--------|-----------|--------------|
| Capability | Highest | Moderate |
| Cost model | Pay-per-use | Upfront hardware |
| Privacy | Data leaves premises | Full control |
| Latency | Network dependent | Very low |
| Scalability | Unlimited | Hardware limited |
| Customisation | Limited | Full control |
| Maintenance | Provider managed | Self-managed |

### Decision Framework

**Choose Cloud APIs when**:

- Cutting-edge reasoning is required.
- Low initial volume (pay-per-use is cheap at small scale).
- Rapid prototyping with minimal AI expertise.

**Choose Local when**:

- Strict data privacy or regulatory requirements.
- High query volume makes per-call costs prohibitive.
- Fine-tuning or full model control is needed.

### Hybrid Patterns

- **Router**: simple queries → local; complex → cloud.
- **Fallback**: try local first, escalate to cloud if confidence is low.
- **Specialised**: local for domain-specific tasks, cloud for general reasoning.
- **Dev/Prod**: develop locally, deploy to cloud.

---

## 5. Ollama & OpenAI Ecosystem

### Ollama (Local)

Think of it as "Docker for AI models" — it wraps model management, inference optimisation, and an API server into a single CLI tool.

**Key commands**:

```bash
ollama pull llama3:8b       # download a model
ollama run llama3:8b        # interactive chat
ollama serve                # expose REST API on localhost:11434
ollama list                 # list installed models
```

**Python integration**:

```python
import requests
response = requests.post(
    'http://localhost:11434/api/generate',
    json={'model': 'llama3:8b', 'prompt': 'Hello!'}
)
```

Ollama's REST API is **compatible with the OpenAI API format** — this means you can often swap between local and cloud models by just changing the base URL.

### OpenAI (Cloud)

Pay-as-you-go access to GPT-4o, GPT-4o-mini, GPT-4o-nano. No installation required.

### Broader Local Ecosystem

| Category | Tools |
|----------|-------|
| Model repositories | Hugging Face Model Hub, GGUF format (quantised) |
| Inference engines | llama.cpp (C++), vLLM (fast server), TGI (HuggingFace) |
| Dev tools | LM Studio (GUI), GPT4All (desktop), LangChain (framework) |

**Quantisation** reduces model precision (e.g., FP16 → INT4) to fit larger models into less memory with modest accuracy cost.

---

## 6. Prompt Engineering

### The Paradigm Shift

Code gives you exact, deterministic execution. Prompts give you probabilistic reasoning. The model is not *executing* your instructions — it is *generating the most likely continuation* given your prompt as context. This requires a different mental model.

**In production, poor prompts cost you**:

- Quality: wrong outputs erode user trust.
- Cost: every token is billed.
- Latency: verbose reasoning chains are slower.
- Reliability: 95% good = 5% of users have terrible experiences.

**Prompt engineering vs. prompt writing**: Engineering means running experiments, measuring results, building reusable patterns, and iterating based on data — not hunches.

### Four Core Principles

1. **Clarity over cleverness** — Say what you mean explicitly. Subtle hints fail. Specific beats vague, always. Clever prompts break on edge cases.

2. **Context is golden** — Background info shapes everything. Models have a **recency bias** (they pay more attention to recent context). Too little context → generic output. Too much → model gets overwhelmed. Match context to task complexity.

3. **Constraints enable creativity** — Unconstrained prompts produce unfocused output. Format constraints make outputs parseable. Boundaries prevent hallucinations.

4. **Iteration is expected** — First prompts rarely work well. Use a feedback loop: draft → test → evaluate → identify specific failures → hypothesise → change one thing → compare.

### Common Failure Modes

| Failure | Cause |
|---------|-------|
| Ambiguity | Model fills gaps with likely (but wrong) text |
| Implicit requirements | You expected something you never stated |
| Context mismatch | Feeding irrelevant background |
| Over-specification | Contradictory constraints ("be brief but comprehensive") |
| Format confusion | No clear specification of output structure |

### Prompt Anatomy

A well-structured prompt has up to four components:

1. **Role/Persona** — "You are an expert data scientist…" Sets tone and expertise level. Influences the model's framing. *Trade-off: too specific = inflexible.*

2. **Context** — Background info, document, task constraints, audience, conversation history. *Principle: relevant, concise, well-structured.*

3. **Instruction** — Use imperatives: "Analyse…", "Generate…", "Extract…". Specify the outcome, not the process. Break complex tasks into sub-instructions. *Anti-pattern: vague verbs like "handle" or "deal with".*

4. **Format** — Explicit output structure (JSON, XML, markdown). Show the format, don't just describe it. Length constraints control verbosity.

**Pattern architectures** (ordered by complexity):

- Basic: Instruction + Format
- Contextual: Context + Instruction + Format
- Role-based: Role + Context + Instruction + Format
- Template-driven: All of the above + Examples

**Delimiter strategy**: Use `"""triple quotes"""` or XML tags to separate prompt sections. This helps models parse complex prompts and reduces prompt injection risk.

**Negative instructions**: "Do not hallucinate info not in context." "If ambiguous, ask for clarification." These boundary conditions prevent common failure modes.

### Zero-Shot vs Few-Shot Prompting

**Zero-shot**: Task description, no examples. The model relies entirely on patterns from training.

- Best for: standard well-defined tasks, flexible output format, tight token budget.
- Requires: crystal-clear instructions.

**Few-shot**: Show 2–5 examples of the desired input→output mapping before the actual query.

```
Example 1:
Input: [concrete input]
Output: [desired output]

Example 2:
Input: [concrete input]
Output: [desired output]

Now process:
Input: [new input]
```

- Best for: novel/complex structures, domain-specific formats, consistent output, when instructions alone are ambiguous.
- *Optimal number*: 3 is often the sweet spot.
- *Example quality matters*: errors in your examples get replicated in outputs.
- *Diversity matters*: cover your input space, not just similar examples.

**Trade-off**: Few-shot uses more tokens (cost at scale) but buys consistency and format control.

### Chain-of-Thought (CoT) Prompting

**What it is**: Prompting the model to produce intermediate reasoning steps before the final answer.

```
Input → Reasoning Steps → Output
```

**Why it works**:

- More tokens = more computation time available for the problem.
- Intermediate steps build toward the answer progressively.
- Exposes flawed logic — sometimes the model self-corrects.
- Makes reasoning transparent and inspectable.

**Zero-shot CoT**: Add "Let's think step by step" to the prompt.

**Few-shot CoT**: Provide examples that include reasoning chains, not just input→output pairs. The model learns to mimic your reasoning structure.

**Self-consistency**: Generate multiple reasoning paths (temperature > 0), take the majority answer. Increases robustness — one bad reasoning path doesn't sink the result.

**When to use CoT**:

- Maths problems and multi-step calculations.
- Logical reasoning tasks.
- Problems with multiple factors to balance.
- High-stakes decisions where transparency matters.

**Anti-pattern**: Don't apply CoT to simple tasks — it wastes tokens without benefit.

**Best practice**: Explicitly separate reasoning from answer:

```
"Reasoning: [steps]

Answer: [conclusion]"
```

Verify reasoning quality, not just whether the final answer is correct.

### Building a Prompt Library

- Document what works, *with context about when and why*.
- Categorise by task type and domain.
- Version-control prompts as you refine them.
- The meta-knowledge (why it works) compounds over time.

---

## 7. Embeddings & Semantic Understanding

### The Representation Problem

Traditional text representations (bag-of-words, TF-IDF, one-hot encoding) treat words as discrete symbols with no sense of meaning or relatedness. "Car" and "automobile" would be as dissimilar as "car" and "banana".

#### Traditional Text Representations (Short Definitions)

- **One-hot encoding**
  Represents each word as a sparse vector with a single `1` and all other values `0`.
  → Treats every word as completely independent (no similarity between words).

- **Bag-of-Words (BoW)**
  Represents a document as a vector of word counts.
  → Ignores word order and meaning; only tracks frequency.

- **TF-IDF (Term Frequency–Inverse Document Frequency)**
  Weighs words by how important they are in a document relative to a corpus.
  → Reduces impact of common words but still treats words as independent symbols.

**Key limitation:**
All of these methods treat words as discrete tokens with no semantic relationships, so similar words (e.g., “car” and “automobile”) are represented as unrelated.

### What Embeddings Are

Dense floating-point vectors (e.g., 768 or 1536 dimensions) where **position in vector space encodes semantic meaning**. Trained via self-supervised learning on massive corpora: the model learns to predict neighbouring words, and geometry emerges from usage patterns.

**Distributional hypothesis**: Words with similar contexts have similar meanings → similar vectors.

**Classic example**: `king − man + woman ≈ queen`
Vector arithmetic reveals learned semantic relationships.

### Embedding Models Landscape

| Category | Examples |
|----------|---------|
| Proprietary | OpenAI text-embedding, Cohere, Voyage |
| Open-source | Sentence-Transformers, E5, BGE |
| Specialised | Domain fine-tuned variants |
| Multilingual | XLM-RoBERTa, mBERT |

### Sentence-BERT and Beyond

> BERT - Bidirectional Encoder Representations from Transformers

BERT alone has no inherent sentence representation. Sentence-BERT uses a **Siamese network** approach: two copies of BERT process two texts independently, and training brings similar pairs closer together.

**Pooling strategies**:

- **Mean pooling**: average all token embeddings — generally works well.
- **CLS token**: use the special [CLS] embedding — less common for semantic similarity.

**Training objectives**:

- Contrastive learning: pull similar examples together, push dissimilar apart.
- Triplet loss: anchor, positive example, negative example.
- Multiple negatives ranking loss.

> The quality of training pairs determines the quality of the resulting embeddings.

### Dimensionality and Trade-offs

| Dimension | Effect |
|-----------|--------|
| Higher (1536+) | More nuanced distinctions, larger storage |
| Lower (384) | Faster, more storage-efficient, slightly less expressive |

**Curse of dimensionality**: In very high dimensions, most points become roughly equidistant. This is why approximate search methods are necessary at scale.

### Contextual Embeddings

Static word embeddings give "bank" one vector regardless of context. Modern transformer-based embeddings are **contextual** — the embedding of a word depends on its surrounding text. This handles polysemy (one word, multiple meanings).

EG. "I went to the **bank** to deposit money" VS "I sat by the river **bank**"

**Sentence-level vs token-level**: For retrieval and similarity tasks, you typically want sentence-level (or chunk-level) embeddings.

### Hybrid Embeddings

Combine sparse (BM25/TF-IDF keyword matching) and dense (neural embeddings) representations:

- BM25 is excellent for exact keyword matches.
- Dense retrieval handles semantic similarity and synonyms.
- Fusion: **Reciprocal Rank Fusion (RRF)** or weighted combination.

RRF: rank both the sparse and dense results using `score = 1 / (k + rank)` with rank being the
position in the list and `k` being a small constant. Sum the scores, and re-rank by the combined score. Rewards items that rank well in either list, and strongly rewards items that rank well in both lists.

Weighted Combination: Better if you have actual scores. Rank with `final_score = α * sparse_score + (1 - α) * dense_score`, where `a` is a tunable constant. Tune `a` to favor sparse where exact words matter more (e.g search engine), tune it for dense where context matters more (e.g Q&A)

### Evaluating Embedding Quality

| Benchmark type | What it measures |
|---------------|-----------------|
| Semantic textual similarity (STS) | Cosine correlation with human similarity ratings |
| BEIR (information retrieval) | Retrieval recall across diverse domains |
| Downstream classification | Whether embeddings are linearly separable for tasks |

**Important**: Embedding spaces from different models are **not directly comparable**. Never mix embeddings from different models in the same index.

#### STS
>
>Do embeddings reflect human-perceived similarity between sentences?

**Metric**
Usually Pearson/Spearman correlation with human ratings

**Why it matters**
Validates semantic understanding
Good proxy for:

- clustering
- deduplication
- semantic search

**Limitation**

- Doesn’t test retrieval ranking quality
- Works on pairs, not large corpora

#### BIER

**What it measures**

>Can embeddings retrieve relevant documents from a large dataset?

**Setup:**

Query: “effects of caffeine on sleep”
Corpus: thousands/millions of documents
Goal: rank relevant docs near the top

**Metrics**

`Recall@k`` → did we retrieve relevant items in top k?
`nDCG@k` → how well ranked are relevant items?

**Why BEIR is important**

- Covers multiple domains (news, biomedical, QA, etc.)
- Tests real-world retrieval performance

**Why it matters for you (RAG)**

This is the closest thing to:

“Will my embeddings actually work in production?”

#### Downstream Classification

**What it measures**

>Can embeddings make tasks easy to solve with simple models?

**Process:**

- Generate embeddings
- Train a simple classifier (often linear)
- Evaluate performance

**Key idea:** linear separability

If embeddings are good:

- Classes form clean clusters
- A straight line (or hyperplane) can separate them

Example:

- Sentiment analysis (positive vs negative)
- Topic classification

**Why this matters**

- Tests whether embeddings encode useful structure
- If a simple model works → embeddings are doing the heavy lifting

#### Clean exam style answer

>Embedding quality can be evaluated using semantic textual similarity (measuring correlation between cosine similarity and human judgments), BEIR benchmarks (assessing retrieval performance across domains using metrics like recall@k), and downstream classification tasks (testing whether embeddings are linearly separable). Embedding spaces are model-specific and not directly comparable, so embeddings from different models must not be mixed within the same index.

### Dimensionality Reduction

When to reduce dimensions:

- **Visualisation** (2D/3D via t-SNE, UMAP).
- **Storage optimisation** and speed.

#### Visualisation (2D / 3D)

You can’t “see” 768 dimensions. So you project them down to 2D/3D.

**Goal:**

* Preserve relationships so clusters are visible

**Typical use:**

* Debugging embeddings
* Spotting clusters (topics, sentiment, etc.)
* Explaining results

#### Storage optimisation & speed

High dimensions cost you:

* Storage
    * 1M vectors × 1536 dims ≈ huge memory footprint
* Latency
    * Similarity search (cosine/dot product) scales with dimension
* Index efficiency (FAISS, HNSW, etc.)

Reducing dimensions:

> Less memory, faster search, cheaper systems

Methods: PCA (linear), t-SNE (non-linear, good for visualisation), UMAP (faster, better global structure).

Trade-off: information loss vs. computational efficiency.

#### PCA (Principal Component Analysis)

Type: Linear

**What it does**

Finds directions (components) that capture the most variance:

```
768 dims → 256 dims (for example)
```

**Strengths**
* Fast
* Deterministic
* Preserves global structure reasonably well

**Good for:**

* compression
* preprocessing for retrieval

**Weakness**

* Can’t capture complex non-linear relationships

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)

Type: Non-linear

**What it does**

* Focuses on preserving local neighbourhoods
* Makes clusters very visually clear

**Strengths**

* Excellent for visualisation
* Reveals tight clusters

**Weaknesses (important)**

* Poor at preserving global structure
* Distances between clusters can be misleading
* Slow
* Not suitable for production pipelines

#### UMAP (Uniform Manifold Approximation and Projection)

Type: Non-linear

**What it does**

* Balances local and global structure
* Faster than t-SNE

**Strengths**

* Good for visualisation and some practical use
* Scales better than t-SNE
* More stable structure

**Weakness**

* Still not ideal for precise distance preservation in retrieval

#### Clean exam style answer

>Dimensionality reduction is used to project high-dimensional embeddings into lower dimensions for visualisation or computational efficiency. PCA provides a linear projection that preserves global variance and is suitable for compression, while t-SNE and UMAP are non-linear methods better suited for visualising clusters. The key trade-off is between reduced computational cost and loss of information, which can negatively impact tasks such as similarity search.

### Fine-Tuning Embeddings

General models may underperform on specialised domains. Fine-tuning via continued pre-training or task-specific objectives can help.

Fine-tuning embeddings means adapting a general embedding model so that texts that are “similar” in your specific domain are placed close together in vector space.

A general embedding model may know that:

"refund policy" ≈ "return policy"

But in a legal, medical, finance, or internal company setting, the notion of similarity may be more specialised. For example, in a medical system:

"myocardial infarction" ≈ "heart attack"

Or in software support:

"authentication timeout" ≈ "login session expired"

A general model may partially understand these, but a domain-tuned embedding model can retrieve better results because it has learned the terminology and relationships that matter in
that domain.

Fine-tuning usually uses pairs or triplets:

Query: "How do I reset my password?"
Positive document: "Password reset instructions"
Negative document: "Refund policy"

The model is trained to move the query closer to the positive document and farther from the negative document.

The trade-off is maintenance. Once you fine-tune an embedding model, you now own a custom model. You need to version it, evaluate it, monitor whether it still performs well, and often
re-embed your vector database if the model changes.

A good rule:

Use a strong general embedding model first.
Fine-tune only when evaluation shows domain-specific retrieval failures.

Trade-off: performance gains vs. maintenance burden of a custom model.

### Ethical Considerations

Embeddings encode biases present in training data (gender, racial, cultural stereotypes in vector space). This leads to biased retrieval and potentially discriminatory downstream systems. Practitioners must test for and mitigate these biases.

Embeddings learn patterns from training data. If the training data contains stereotypes, social biases, or historical inequalities, the embedding space can preserve those patterns.

For example, older word embeddings famously associated:

"man" closer to "doctor"
"woman" closer to "nurse"

This matters because embeddings are often used inside real systems:

- search ranking
- recommendation systems
- hiring filters
- content moderation
- customer segmentation
- fraud detection
- medical triage

If the embedding space is biased, the downstream system can behave unfairly even if no explicit rule says “treat groups differently”.

In RAG, bias can appear as biased retrieval. If documents about one group are retrieved more often, or certain viewpoints are systematically ranked higher, the generated answer will
inherit that skew.

Mitigation involves:

- evaluating retrieval results across demographic groups
- testing with adversarial examples
- inspecting nearest neighbours for sensitive terms
- using diverse training data
- applying fairness-aware evaluation
- keeping humans involved in high-stakes decisions

The key point is that embeddings feel mathematical and neutral, but they encode human language and human data, so they can encode human bias.

---

## 8. Vector Databases & Similarity Search

### Why Vector Databases Exist

Traditional relational databases support exact-match and range queries. Semantic search requires finding the *most similar* vectors to a query vector among billions of candidates. Naive O(n) exhaustive search is intractable at scale.

A vector database stores embeddings and lets us search for the nearest vectors efficiently.

In a traditional database, you might search like this:

```sql

SELECT * FROM documents
WHERE title = 'refund policy';
```

That works for exact matching. But semantic search asks a different question:

Find documents whose meaning is similar to:
"Can I get my money back?"

The relevant document might never use the phrase “money back”. It might say:

"Our refund policy allows returns within 30 days."

A vector database makes this possible by comparing the query embedding with document embeddings.

The naive approach would compare the query against every document vector:

1 query × 10 million vectors = 10 million similarity calculations

That may be too slow for production systems. Vector databases use approximate nearest neighbour algorithms to find very similar vectors without checking every vector exhaustively.

So the purpose of a vector database is:

Store embeddings + search them quickly + return semantically similar items.

### Similarity Metrics

| Metric | Description | When to use |
|--------|-------------|-------------|
| **Cosine similarity** | Angle between vectors; magnitude ignored | Text embeddings (most common) |
| **Euclidean distance** | Straight-line distance | When magnitude matters |
| **Dot product** | Combines direction and magnitude | When vectors are already normalised |

**Cosine similarity formula**:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Ranges from −1 (opposite) to 1 (identical). Geometrically: measures the angle between vectors.

**Practical tip**: Normalise vectors once at indexing time, then use dot product — equivalent to cosine similarity but faster.

**Similarity thresholds** are task-dependent (no universal "similar" cutoff). Always plot your similarity score distribution and calibrate for your specific corpus.

Similarity metrics define what “close” means in vector space.

Cosine similarity measures the angle between two vectors. It ignores vector length and focuses on direction. This is common for text because meaning is usually encoded by direction rather than magnitude.

Example:

A = "How do I reset my password?"
B = "Password reset instructions"
C = "How do I cancel my order?"

The embedding for A should point in a similar direction to B, so cosine similarity should be high. C may also be about customer support, but it is about a different task, so similarity should be lower.

Dot product is similar to cosine similarity if vectors are normalised. Many systems normalise vectors once, then use dot product because it is faster.

Euclidean distance measures straight-line distance. It is useful when vector magnitude matters, but it is less common for modern text retrieval.

Important practical point:

There is no universal similarity threshold.

A cosine score of 0.78 might be excellent in one system and weak in another. You need to inspect score distributions and evaluate against labelled examples.

### Approximate Nearest Neighbour (ANN) Search

Exact search is O(n) — intractable at billions of vectors. ANN methods trade a small accuracy loss for massive speed gains.

**Key insight**: 99% accuracy with 100× speedup is almost always the right trade-off.


Exact nearest neighbour search checks every vector and returns the true closest matches. This is accurate but expensive.

Approximate nearest neighbour search trades a small amount of accuracy for much faster search.

For example:

Exact search:
- Checks 10 million vectors
- Finds the true top 10
- Too slow

ANN search:
- Checks a small fraction of vectors
- Finds almost the same top 10
- Fast enough for production

This trade-off is usually worthwhile. In many systems, retrieving a near-perfect set of results quickly is better than retrieving the mathematically perfect set too slowly.

ANN systems are evaluated using recall@k:

If the exact top 10 contains documents A, B, C...
How many of those did the ANN index retrieve?

You tune ANN indexes by balancing:

- recall
- latency
- memory usage
- index build time

### HNSW (Hierarchical Navigable Small World)

The dominant algorithm used by Weaviate, Qdrant, and others:

- Builds a multi-layer graph where each vector is a node.
- **Layer 2 (sparse)**: long-range connections for coarse navigation.
- **Layer 1 (medium)**: medium-range refinement.
- **Layer 0 (dense)**: fine-grained nearest-neighbour search.
- Query traverses from coarse to fine, following the closest neighbour at each step.
- **Performance**: excellent recall, fast queries, higher memory use.
- Scales logarithmically — doubling data adds constant time, not double.

HNSW stands for Hierarchical Navigable Small World. It is one of the most popular ANN algorithms.

The intuition is like navigating a city map.

At the top layer, there are only a few long-distance links. These help the search move quickly to the right general region.

At lower layers, there are more detailed local links. These help the search refine its answer and find close neighbours.

A query works roughly like this:

1. Start at a high-level graph layer.
2. Move toward vectors that look closer to the query.
3. Drop down to a more detailed layer.
4. Repeat until reaching the dense bottom layer.
5. Return the nearest candidates found.

HNSW is popular because it gives high recall and low query latency. The cost is memory: the graph stores extra links between vectors, so it uses more RAM than a simple flat index.

Important parameters include:

- M: number of graph connections per node
- efConstruction: quality/speed trade-off during index building
- efSearch: quality/speed trade-off during querying

Higher values usually improve recall but increase memory, indexing time, or query latency.

### Inverted File Index (IVF)

- Pre-cluster vectors into Voronoi cells.
- At query time, only search the top-n most relevant clusters (`nprobe` parameter).
- Used by Faiss.
- Coarse-to-fine: cheap cluster selection, expensive within-cluster search.

An IVF index speeds up search by clustering vectors.

Instead of searching every vector, the system first groups vectors into clusters. At query time, it finds the most relevant clusters and searches only inside those.

Example:

Corpus: 10 million vectors
Clusters: 10,000
Query: search only the closest 10 clusters

This dramatically reduces the number of comparisons.

The key parameter is often called nprobe, which controls how many clusters are searched.

Low `nprobe`:

Fast but may miss relevant results.

High `nprobe`:

Slower but more accurate.

IVF is useful for large-scale systems, especially when combined with compression techniques like product quantisation.

### Product Quantisation (PQ)

- Problem: billions of 768-dim float32 vectors = terabytes of RAM.
- PQ compresses vectors by splitting into sub-spaces and quantising each.
- Achieves 10–32× compression with slight accuracy loss.
- Enables serving large indices from RAM rather than disk.

Product quantisation compresses vectors so they use less memory.

A normal embedding might be stored as 1536 floating-point numbers. At large scale, this becomes expensive:

1 million vectors × 1536 dimensions × 4 bytes ≈ 6 GB

That is just for one million vectors. At hundreds of millions or billions of vectors, storage becomes a serious problem.

PQ works by splitting a vector into smaller parts and replacing each part with a compact code.

Instead of storing the full vector precisely, it stores an approximation.

The trade-off:

Much lower memory usage
Slightly less accurate similarity search

This is useful when the system needs to serve a very large index from RAM.

### Vector Database Landscape

| Category | Examples |
|----------|---------|
| Purpose-built | Pinecone, Weaviate, Qdrant, Milvus |
| Traditional + vector extension | PostgreSQL (pgvector), Redis |
| Embedded libraries | Faiss, Annoy, HNSWlib |

**Cloud-managed vs. self-hosted**: Cloud is simpler to operate; self-hosted gives privacy and cost control at scale.

There are three broad categories.

Purpose-built vector databases, such as Pinecone, Weaviate, Qdrant, and Milvus, are designed specifically for vector search. They usually provide ANN indexes, metadata filtering, scaling, replication, and APIs.

Traditional databases with vector extensions, such as PostgreSQL with pgvector, are useful when vector search is part of a broader application. They are often simpler to operate if your data already lives in Postgres.

Embedded libraries, such as Faiss or HNSWlib, are lower-level. They are powerful and fast, but you must build more of the surrounding infrastructure yourself.


A rough decision rule:

**Prototype/small app**: Chroma, SQLite vector extensions, pgvector
**Production app with existing relational data**: PostgreSQL + pgvector
**Large-scale semantic search**: Qdrant, Weaviate, Milvus, Pinecone
**Custom high-performance retrieval**: Faiss

### Metadata Filtering & Hybrid Search

Real queries often combine vector similarity with structured filters: "Find similar documents *from 2024* about *AI safety*."

Approaches:

- **Pre-filtering**: apply metadata filter before vector search (risks missing relevant results if filter is too narrow).
- **Post-filtering**: vector search first, then filter (can result in fewer than k results).
- **Hybrid indices**: purpose-built support in databases like Qdrant and Weaviate.

Vector similarity alone is often not enough.

Suppose a user asks:

>Find AI safety reports from 2024.

The system needs both semantic similarity and structured constraints:

**topic** ≈ "AI safety"
**year** = 2024

Metadata filtering lets us combine vector search with ordinary database-style filters.

Pre-filtering means applying the metadata filter first, then vector search. This is efficient if the filter leaves enough candidates, but it can fail if the filtered set is too small.

Post-filtering means doing vector search first, then removing results that do not match the metadata filter. This can produce too few final results if many retrieved items are filtered out.

Good vector databases support hybrid approaches so metadata and vector similarity can work together efficiently.

### Vector Database Performance Metrics

| Metric | Meaning |
|--------|---------|
| Recall@k | How often is the true nearest neighbour in the top-k results? |
| QPS | Queries per second under load |
| Latency (p50/p95/p99) | Response time percentiles |
| Index build time | Time to build the index from scratch |
| Memory footprint | RAM required to serve the index |

A vector database should be evaluated as a retrieval system and as an infrastructure component.

Retrieval quality metrics:

**Recall@k**: Did we retrieve the relevant item in the top k?
**Precision@k**: How many of the top k results were relevant?
**MRR**: How high was the first relevant result?
**nDCG**: Were highly relevant results ranked near the top?

Systems metrics:

**Latency**: How long does a query take?
**p95/p99** latency: How slow are the slowest normal requests?
**QPS**: How many queries per second can the system handle?
**Memory footprint**: How much RAM does the index require?
**Index build time**: How long does ingestion take?

For AI systems, p95 and p99 latency matter because users experience the slow requests, not the average request.

### Composite Embeddings for Long Documents

Embedding a 50-page document as a single vector loses detail. Options:

- **Chunk and aggregate**: embed chunks, combine via mean pooling or max pooling.
- **Hierarchical**: embed at document, section, and paragraph levels; route queries to appropriate level.

Trade-off: information preservation vs. single-vector simplicity.


A long document cannot usually be represented well by a single embedding.

For example, a 50-page report might discuss:

- pricing
- legal constraints
- technical architecture
- risks
- implementation timelines

If you squash the whole document into one vector, the embedding becomes a vague average of many topics.

Chunking solves this by embedding smaller sections separately. Then the system can retrieve the exact paragraph or section that answers the query.

Hierarchical retrieval goes further:

Document-level embedding: identifies the relevant document
Section-level embedding: identifies the relevant section
Paragraph-level embedding: identifies the precise evidence

This is useful when documents are large and structured.

The trade-off is complexity. More chunks means better precision, but also more storage, more indexing work, and more retrieval results to manage.

---

## 9. Evaluation Frameworks

### Why Evaluation Matters

- 67% of AI projects never reach deployment — often because evaluation gaps are discovered too late.
- **Goodhart's Law in disguise**: the metrics you choose determine what you optimise for. Wrong metrics → systems that score well but fail users.
- Poor evaluation leads to failed deployments, user trust erosion, and firefighting costs greater than the evaluation investment itself.

Evaluation is how we know whether an AI system is actually working.

With traditional software, correctness is often binary:

Did the function return the expected value?
Did the API return 200?
Did the database update succeed?

With AI systems, correctness is often fuzzy:

Was the answer helpful?
Was it grounded in the evidence?
Was it safe?
Was it complete enough?
Was it too verbose?

This makes evaluation much harder, but also more important.

A model can appear impressive in demos while failing in production because real users ask messy, ambiguous, unexpected questions. Without evaluation, teams often rely on intuition: “this output looks good to
me”. That is not reliable enough for production systems.

Evaluation turns vague quality into measurable criteria.

Example:

Bad evaluation:
"The chatbot should give good answers."

Better evaluation:
"For customer refund questions, the chatbot should:
- retrieve the correct policy document
- answer using only that document
- mention the refund window
- refuse to invent policy details
- respond in under 3 seconds"

The key idea:

Evaluation is the specification for an AI system.

If you do not define what “good” means, you cannot systematically improve the system.

### The Evaluation Lifecycle

1. **Development-time**: rapid iteration, debugging, fast feedback.
2. **Pre-deployment validation**: comprehensive testing before release.
3. **Production monitoring**: continuous assessment of live behaviour.
4. **Feedback loops**: learning from real-world failures.

Evaluation should happen throughout the whole system lifecycle, not just at the end.

During development, evaluation helps compare prompts, models, retrievers, chunking strategies, and temperature settings. This is usually fast and iterative.

Example:

Prompt A answers 72/100 test questions correctly.
Prompt B answers 81/100 test questions correctly.
Prompt B is better, assuming latency and cost are acceptable.

Before deployment, evaluation becomes more formal. You test against a fixed validation dataset that represents real expected use cases and known edge cases.

In production, evaluation continues because user behaviour changes. New documents are added, old policies become outdated, users ask unexpected questions, and model providers may update their systems.

A useful lifecycle is:

Develop → Evaluate → Deploy → Monitor → Collect feedback → Improve → Re-evaluate

The important point is that AI systems degrade silently. They may not crash; they may simply become less accurate, less grounded, or less useful.

### Dimensions of Quality

| Dimension | Metrics |
|-----------|---------|
| Technical performance | Accuracy, latency, throughput |
| Content quality | Relevance, coherence, factuality |
| User experience | Helpfulness, safety, appropriateness |
| Operational | Cost, reliability, maintainability |

AI systems must be evaluated across multiple dimensions.

A response might be fluent but factually wrong. It might be accurate but too slow. It might be helpful but unsafe. It might answer correctly but ignore required formatting.

For example, consider this user query:

"What is the company refund policy?"

A response can be judged on:

**Relevance**:
Does it answer the refund question?

**Factuality**:
Is the policy information correct?

**Groundedness**:
Is the answer supported by retrieved documents?

**Completeness**:
Does it include key conditions, deadlines, and exceptions?

**Clarity**:
Is it understandable to the user?

**Safety/compliance**:
Does it avoid giving unauthorised promises?

**Latency**:
Was it fast enough?

**Cost**:
Was the model unnecessarily expensive for this task?

This is why a single metric is rarely enough. A production AI system usually needs a scorecard.


### Automated Evaluation Metrics

**Reference-based** (require a gold-standard answer):

- **BLEU**: n-gram precision between generated and reference text. Used for translation.
- **ROUGE**: recall-oriented — how much of the reference is covered. Used for summarisation.
- **METEOR**: considers synonyms and morphological variants.
- **Critical limitation**: brittleness to paraphrasing. A correct answer phrased differently scores poorly.

**Semantic similarity metrics** (move beyond surface form):

- **BERTScore**: uses contextual embeddings to compare generated vs. reference text. More robust to paraphrasing.
- **Cosine similarity** in embedding space between generated and reference.

**Reference-free metrics**:

- **Perplexity**: how "surprised" a language model is by the text. Measures fluency. *Limitation: fluent nonsense can have low perplexity.*
- **Coherence** and repetition scores.

**Task-specific metrics**:

- Code generation: compilation success, test pass rate, execution time.
- Question answering: exact match, F1 on answer span.
- Summarisation: coverage, compression ratio, redundancy.

### LLM-as-Judge

Using a capable LLM (e.g., GPT-4) to evaluate another model's outputs:

- **Pairwise comparisons**: "Which of these two responses is better?"
- **Absolute rating**: score against a rubric.
- **Designing evaluation prompts**: specificity and rubrics are critical.
- **Risks**: position bias (prefers first response), verbosity bias (prefers longer responses), self-serving bias.
- Use multiple diverse judges and aggregate.

### Human Evaluation

When and why: human judgement captures nuanced quality that automated metrics miss.

**Designing rubrics**:

- Break quality into specific, measurable dimensions (accuracy, helpfulness, tone).
- Provide anchoring examples for each level.
- Calibrate raters before large-scale annotation.

**Evaluator types**:

| Type | Pros | Cons |
|------|------|------|
| Domain experts | Nuanced judgement | Expensive and slow |
| Crowdworkers | Scalable, affordable | Need very clear task design |
| Internal team | Context-aware | Potentially biased |

**Inter-rater reliability**: Use Cohen's Kappa (binary) or Krippendorff's Alpha (multi-rater, multi-category) to measure annotator agreement.

**Human biases to watch for**:

- Anchoring bias: first impression influences subsequent ratings.
- Halo effect: overall impression contaminates specific dimension ratings.
- Positivity bias: tendency toward inflated positive ratings.

### Continuous Evaluation in Production

Why evaluation doesn't stop at deployment:

- Real user behaviour differs from test datasets.
- **Distribution shift**: inputs and contexts evolve over time.
- **Silent failures**: problems that don't crash systems but degrade quality.

**Monitoring infrastructure**:

- Latency and throughput dashboards.
- Error rate tracking and alerting.
- Output quality sampling (periodic LLM-as-judge on live traffic).
- User feedback integration (thumbs up/down, ratings, implicit signals like query abandonment).

**Detecting distribution shift**: monitor input embedding distributions and output distributions; flag when they diverge significantly from baseline.

**A/B testing**: controlled rollouts with randomised assignment. Ensure statistical power before drawing conclusions — determine required sample size upfront.

**Evaluation-driven development**: define success metrics before you build, not after. Treat evaluation as a specification.

### Building Evaluation Datasets

- **Curated QA pairs**: manually created, gold standard quality.
- **Synthetic generation**: LLM-generated evaluation sets (fast but biased).
- **User query logs**: real queries with feedback labels.
- **Adversarial examples**: edge cases and known failure modes.

Best practices: diversity across input types, representativeness (match production distribution), regular refresh (datasets decay as the domain evolves).

### The Multiple Comparisons Problem

Iterating many prompt variants and measuring on the same test set inflates the chance of a spurious improvement. Solutions:

- Change one variable at a time.
- Keep a held-out test set untouched until final validation.
- Apply statistical corrections (Bonferroni, etc.) when running many comparisons.

---

## 10. Retrieval-Augmented Generation (RAG)

### The Knowledge Problem RAG Solves

LLMs have static knowledge: training cutoff dates, no domain-specific content, and hallucination risk. RAG solves this by grounding responses in retrieved external knowledge.

**RAG is not a new model — it is a new architectural pattern.**

> Formula: RAG = Retrieval + Augmentation + Generation

### RAG vs. Fine-Tuning

| Criterion | RAG | Fine-Tuning |
|-----------|-----|-------------|
| Frequently changing knowledge | ✓ | ✗ |
| Need citations/transparency | ✓ | ✗ |
| Teaching new behaviours/style | ✗ | ✓ |
| Large document corpus | ✓ | ✗ |
| Cost sensitivity | ✓ | ✗ |
| Style adaptation | ✗ | ✓ |

**Fine-tuning teaches the model *how* to reason; RAG tells it *what* to reason about.** Do not use fine-tuning as a knowledge injection mechanism.

### The Basic RAG Pipeline (5 Stages)

```
User Query
    ↓
1. Query Processing   (expand, decompose, clarify)
    ↓
2. Retrieval          (vector similarity search → top-K chunks)
    ↓
3. Re-Ranking         (filter, score, reorder chunks)
    ↓
4. Prompt Construction (context + query → structured prompt)
    ↓
5. Generation         (LLM generates grounded response)
    ↓
Final Response (with citations)
```

### Stage 1 — Offline Preparation: Chunking

Before any query, documents must be split into chunks and embedded into a vector database.

**Chunking strategies**:

| Strategy | How | Best for |
|----------|-----|---------|
| Fixed-size | 512 tokens with overlap | Simple, homogeneous documents |
| Sentence boundaries | Split at sentence ends | Preserves semantic units |
| Paragraph-based | Split at paragraph breaks | Natural document structure |
| Sliding windows | Overlapping fixed-size chunks | Avoids cutting mid-thought |
| Semantic chunking | Split when topic changes (embedding similarity) | Complex, heterogeneous documents |

**Context-augmented chunking**: Include document title and section headers in each chunk so it remains self-contained when retrieved in isolation.

**Key trade-off**: chunk size vs. coherence vs. retrieval accuracy. Smaller chunks = more precise retrieval but lost context. Larger chunks = more context but noisier.

**Chunking is overlooked but critical** — poor chunking cripples even the best retrieval setup.

### Stage 2 — Query Processing

User queries are often vague, multi-part, or use implicit context.

**Techniques**:

- **Query expansion**: add synonyms and related terms.
- **Query decomposition**: split multi-part questions into sub-queries.
- **Context carryover**: in multi-turn conversations, incorporate chat history into the query.
- **HyDE (Hypothetical Document Embeddings)**: generate a hypothetical ideal answer and use *that* as the search query — the ideal answer's embedding is closer to the document embeddings than the original question's embedding.

### Stage 3 — Retrieval

**Dense retrieval** (semantic):

- Encode query with embedding model.
- Search vector DB via ANN.
- Fast, scalable, handles synonyms and paraphrases.
- Weakness: sensitive to query phrasing; can miss exact keyword matches.

**Sparse retrieval (BM25)**:

- Keyword-based TF-IDF scoring.
- Excellent at exact term matches (product names, error codes).
- Fast, interpretable, no neural encoding.
- Weakness: misses semantic synonyms.

**Hybrid retrieval** (recommended for production):

- Run both dense and sparse retrieval in parallel.
- Fuse results with **Reciprocal Rank Fusion (RRF)** or weighted scoring.
- Gets the best of both worlds.

**K value**: the number of chunks retrieved. Retrieve more (K=10) then re-rank to fewer (3–5) to balance coverage and context window efficiency.

### Stage 4 — Re-Ranking and Refinement

Initial vector similarity ≠ semantic relevance. Re-ranking refines the initial retrieval:

- **Cross-encoder re-ranking**: a smaller model scores each (query, chunk) pair jointly — more accurate than bi-encoder but slower.
- **Metadata filtering**: filter by date, author, document type.
- **Relevance scoring**: lightweight classifier models.
- **User feedback loops**: incorporate past successful retrievals.

### Stage 5 — Prompt Construction

How you arrange context in the prompt matters.

**Patterns**:

- **Context-first**: retrieved chunks, then question.
- **Question-first**: question, then "use this information: [chunks]".
- **Interleaved**: question, relevant source 1, relevant source 2, …, now answer.

**Which works best?** It depends on the model — test empirically.

**The "lost in the middle" problem** (Liu et al., 2023): models pay most attention to the beginning and end of their context window. Information in the middle gets overlooked. Implication: **place most relevant chunks at the edges of the context**.

**Context window management** (as of late 2025):

- GPT-4: 128K tokens (~96,000 words)
- Claude 3.5 Sonnet: 200K tokens
- Llama 3.1 70B/8B: 128K tokens

More context = better grounding, but higher cost and slower inference.

**Context compression techniques**:

- Extractive summarisation: pull key sentences from chunks.
- Abstractive summarisation: rewrite chunks more concisely.
- Relevance filtering: remove sentences unrelated to the query.

### Stage 6 — Generation with Grounding

Keeping the model honest:

**Challenges**:

- Hallucination by omission (model ignores the provided context).
- Model contradicts the retrieved context.
- Model doesn't know when context is insufficient.

**Strategies**:

- Explicit instruction: "Use ONLY the provided context."
- "If the context doesn't contain the answer, say so."
- Post-processing validation against retrieved chunks.
- Citation enforcement in the prompt format.

### Citations and Source Attribution

Citations are critical for:

- **Verifiability** in high-stakes domains (medical, legal).
- **User trust and transparency**.
- **Debugging** when answers are wrong.
- **Regulatory compliance**.

Implementation approaches:

- Inline: "[According to Document 3, …]"
- Footnotes: "The policy changed in 2024[1]"
- Metadata: document ID, page number, timestamp.

### Retrieval Failure Modes

| Failure | Cause | Fix |
|---------|-------|-----|
| Irrelevant documents | Poor embeddings or query processing | Improve embedding model; query expansion |
| Missing relevant docs | Chunking breaks context; vocabulary mismatch | Semantic chunking; BM25 hybrid |
| Too much retrieved | Flooding context window | Tighter K; re-ranking |
| Too little retrieved | Missing critical context | Increase K; check corpus coverage |
| Outdated docs | Temporal drift | Metadata filtering by date; corpus refresh |

### RAG Evaluation Framework

Four quality dimensions:

1. **Retrieval Quality** — Are the right documents being retrieved? Metrics: precision, recall, MRR (Mean Reciprocal Rank).

2. **Context Relevance** — Does the retrieved context actually help answer the query? Metric: LLM-as-judge relevance scoring.

3. **Faithfulness/Groundedness** — Does the answer stay true to the retrieved context? Metric: fraction of claims supported by context, contradiction detection.
   - Formula: `faithfulness = claims_supported_by_context / total_claims`

4. **Answer Quality** — Is the final response useful and well-formed? Metrics: fluency, completeness, user satisfaction.

**RAG-specific metrics**:

- **Context Precision**: `relevant_retrieved_chunks / total_retrieved_chunks`
- **Context Recall**: `retrieved_relevant_chunks / all_relevant_chunks`
- **Faithfulness**: claims from context / all claims made
- **Answer Relevance**: LLM-as-judge or human rating

### RAG at Scale: Production Considerations

| Challenge | Mitigation |
|-----------|-----------|
| Latency (retrieval + generation) | Caching common query embeddings; async pipelines |
| Cost (embedding + LLM API calls) | Cache embeddings; choose appropriate model size |
| Index freshness | Incremental indexing pipelines |
| Concurrency | Horizontal scaling, load balancing |

### Guardrails and Validation

- **Input validation**: detect malicious or nonsensical queries before they enter the RAG pipeline.
- **Output validation**: check for hallucination or policy violations after generation.
- **Fact-checking**: verify claims against retrieved context using NLI models.
- **Layered approach**: multiple checks at different pipeline stages.

### When RAG Isn't Enough

RAG fails when:

- Knowledge requires deep reasoning, not just retrieval.
- The information is genuinely absent from the corpus.
- Queries need real-time data (live market prices, current news).
- The context window can't fit all necessary information.

**Alternatives**: fine-tuning, knowledge graphs, agentic systems with dynamic tool use, hybrid RAG + fine-tuning.

### The Agent-RAG Connection

In agentic systems, RAG becomes a tool:

- The agent decides **when** to retrieve (not every query needs retrieval).
- The agent chooses **where** to search (internal docs, databases, web).
- The agent can iterate: retrieve → reason → retrieve again if needed.
- RAG can be combined with calculators, code executors, and other APIs.

### Real-World Applications

- Customer support chatbots (Intercom, Zendesk)
- Internal knowledge bases (Notion AI, Confluence)
- Legal document analysis
- Medical literature search
- Code documentation assistants (GitHub Copilot)

### Case Study: Customer Support RAG (E-commerce)

Architecture decisions that achieved **78% resolution without escalation**, 1.8s response time, 4.2/5 satisfaction:

- Chunking: 600 tokens with 100-token overlap.
- Embedding model: nomic-embed-text (open-source, fast).
- Retrieval: hybrid dense + BM25, K=10 then re-rank to 3.
- LLM: Llama 3.1 70B locally via Ollama (privacy requirement).
- Context: 2,500 tokens of retrieved context.
- Citations: inline with article IDs.

---

## Quick Reference: Key Formulas

| Formula | Use |
|---------|-----|
| `Attention(Q,K,V) = softmax(QKᵀ / √d) V` | Self-attention mechanism |
| `cosine_similarity(A,B) = (A·B) / (‖A‖ × ‖B‖)` | Semantic similarity between embeddings |
| `context_precision = relevant_retrieved / total_retrieved` | RAG retrieval quality |
| `context_recall = retrieved_relevant / all_relevant` | RAG retrieval coverage |
| `faithfulness = claims_supported / total_claims` | RAG grounding quality |

---

## Key Papers & Resources

| Paper | Significance |
|-------|-------------|
| Vaswani et al. (2017) "Attention Is All You Need" | Introduced the Transformer architecture |
| LeCun, Bengio & Hinton (2015) "Deep Learning" (Nature) | Foundational deep learning survey |
| Liu et al. (2023) "Lost in the Middle" | Showed LLMs ignore middle-of-context information |

**Core textbooks**:

- Goodfellow, Bengio & Courville: *Deep Learning* (MIT Press, 2016)
- Tunstall, von Werra & Wolf: *NLP with Transformers* (O'Reilly, 2022)
- Géron: *Hands-On ML with Scikit-Learn, Keras & TensorFlow* (O'Reilly, 2022)

---

## Practical Tools Summary

| Tool | Purpose |
|------|---------|
| Ollama | Local LLM inference server ("Docker for AI") |
| ChromaDB | Lightweight embedded vector database (dev/small scale) |
| Pinecone / Weaviate / Qdrant | Production vector databases |
| LangChain / LlamaIndex | Application frameworks for RAG pipelines |
| Sentence-Transformers | Open-source embedding models |
| Faiss | Facebook's fast ANN search library |
| vLLM | High-throughput LLM serving |
| RAGAS | Automated RAG evaluation framework |

---

## Lecture 07: Advanced Retrieval Strategies

> Synthesised from Week 7 PDFs: Introduction & The Retrieval Problem, Hybrid Retrieval Strategies, Metadata Filtering & Query Enhancement, Multi-Modal Content Handling, Performance Optimisation, and Production Deployment & Advanced Topics.

### Why Basic Vector Search Falls Short

Basic vector search retrieves by semantic similarity, but **semantic similarity is not the same as task relevance**.

Common failure modes:

- **Keyword gap**: exact terms, IDs, acronyms, product names, and error codes may matter more than broad semantic meaning.
- **Context mismatch**: the retrieved document may be topically similar but not useful for the user's actual intent.
- **Domain terminology**: specialised jargon may not be represented well by general-purpose embeddings.
- **Production constraints**: retrieval must balance precision, recall, latency, scalability, cost, and complexity.

Good retrieval is a trade-off between:

| Goal | Meaning |
|------|---------|
| Precision | Retrieved documents are relevant |
| Recall | Relevant documents are not missed |
| Latency | Retrieval is fast enough for production |
| Scalability | Performance holds at large corpus sizes |
| Cost | Embeddings, vector DB calls, reranking, and LLM generation remain affordable |

### Hybrid Retrieval

**Hybrid retrieval** combines dense semantic retrieval with sparse keyword retrieval.

- **Dense retrieval**: embedding/vector search; good for meaning, paraphrase, and conceptual similarity.
- **Sparse retrieval**: keyword/BM25 search; good for exact terms, names, IDs, acronyms, and domain-specific vocabulary.
- **Hybrid retrieval**: combines both so the system captures meaning and precision.

This is especially useful for:

- Technical documentation with jargon.
- Legal, medical, or academic material.
- Queries containing exact error messages, product names, API names, or abbreviations.
- Corpora where both conceptual similarity and lexical matching matter.

### Reciprocal Rank Fusion (RRF)

RRF combines ranked results from multiple retrievers without needing their raw scores to be directly comparable.

```
RRF Score = Σ (1 / (k + rank_i))
```

Where:

- `k` is a constant, often around `60`.
- `rank_i` is the document's rank in retrieval method `i`.
- Scores are summed across retrieval methods.

Why RRF is useful:

- It works across retrievers with different scoring scales.
- It rewards documents that appear highly ranked in multiple result sets.
- It is simple, robust, and often effective in production search systems.

### Multi-Vector Retrieval

Single-vector retrieval compresses an entire chunk or document into one representation. This can lose important facets of the content.

**Multi-vector retrieval** stores multiple representations for the same source material:

- Summary vectors for high-level meaning.
- Detailed chunk vectors for precise local matches.
- Question vectors representing likely user queries.
- Answer vectors representing direct response content.
- Structural vectors for titles, conclusions, headings, or key sections.

This improves retrieval when different users ask about the same document from different angles.

### Parent Document Retrieval

Parent document retrieval solves the **chunk size dilemma**:

- Small chunks improve retrieval precision.
- Large chunks provide better context for generation.

Pattern:

```
Full Document
├── Small child chunk 1 (embedded and indexed)
├── Small child chunk 2 (embedded and indexed)
└── Small child chunk 3 (embedded and indexed)
        ↓
Retrieve matching child chunk
        ↓
Return larger parent section/document to the LLM
```

Key idea: **retrieve small, return large**.

This allows precise matching without starving the LLM of surrounding context.

### Multi-Query Retrieval

Users often write ambiguous, incomplete, or poorly phrased queries. Multi-query retrieval uses an LLM to generate several alternative phrasings, retrieves with each, then merges the results.

Example query:

> "Why is my RAG system slow?"

Possible expansions:

- "RAG retrieval latency bottlenecks"
- "How to optimise vector database search speed"
- "Embedding generation and LLM context latency"
- "RAG performance profiling techniques"
- "Caching strategies for retrieval augmented generation"

Benefits:

- Improves recall.
- Handles ambiguous intent.
- Retrieves documents that one wording alone might miss.

Trade-off: it increases retrieval cost and latency because multiple searches are performed.

### Metadata Filtering

Not all useful retrieval information belongs in embeddings. Structured facts should be stored as metadata and used as filters.

Useful metadata fields:

| Field | Example Use |
|-------|-------------|
| Document type | policy, lecture, paper, report |
| Date | retrieve only recent documents |
| Author | restrict to a named source |
| Department/category | HR, engineering, finance, AI systems |
| Access level | public, internal, confidential |
| Version | avoid outdated policies or docs |
| Language | retrieve only English, Irish, French, etc. |
| Source | website, PDF, database, wiki |

Best practice: **filter first, then retrieve**.

This reduces the candidate set before vector search, improving speed and relevance.

### Self-Query Retrieval

Self-query retrieval uses an LLM to convert natural language into:

- A semantic search query.
- Structured metadata filters.

Example:

> "Find recent machine learning papers about transformers"

The LLM might extract:

- Semantic query: `"transformers"`
- Metadata filters: `topic = "machine learning"`, `date = recent`

This is useful when users do not know the available metadata schema or when manually specifying filters would be clumsy.

Risk: the LLM can misinterpret the user's intent or generate unsupported filters, so metadata schemas should be explicit and validated.

### Designing Metadata Schemas

Good metadata schemas are driven by expected query patterns.

For a university course repository, useful metadata might include:

- Module code and module title.
- Lecture/week number.
- Topic.
- Academic year.
- Lecturer.
- File type.
- Assessment relevance.
- Difficulty level.
- Learning outcome.
- Access permission.
- Last updated date.

Design questions:

- What will users commonly filter by?
- Which filters improve performance most?
- Which metadata fields must be exact and reliable?
- Which fields affect access control or compliance?
- Which fields are likely to change over time?

### Multi-Modal Retrieval

Real knowledge bases often contain more than text:

- Images and diagrams.
- Charts and plots.
- Tables and spreadsheets.
- Audio and video.
- PDFs with mixed layouts.

Strategies:

| Content Type | Retrieval Strategy |
|--------------|-------------------|
| Images/diagrams | Generate captions; use CLIP-style image-text embeddings |
| Scanned documents | OCR, then embed extracted text |
| Tables | Preserve structure; create summaries or question-answer pairs |
| Audio/video | Transcribe with speech-to-text, then index transcript |
| Mixed PDFs | Route each section by modality and preserve source metadata |

Important principle: **do not flatten everything blindly into plain text**. Some information, especially tables and diagrams, loses meaning when structure is discarded.

### Handling Tables

Tables are difficult because row/column relationships carry meaning.

Possible strategies:

- Embed table title, caption, and surrounding text.
- Store the full table as context for generation.
- Convert table rows into natural-language statements.
- Generate likely question-answer pairs from the table.
- Use table-specific models where appropriate.
- Keep metadata about columns, units, source, and date.

Choose the strategy based on query type:

- Lookup queries need exact cells.
- Comparison queries need row/column relationships.
- Summary queries need natural-language descriptions.
- Analytical queries may need code execution or database querying rather than embedding search alone.

### Performance Bottlenecks in RAG

Time in a RAG pipeline is usually spent across several stages:

1. Query embedding generation.
2. Vector similarity search.
3. Document loading.
4. Context preparation.
5. Reranking or compression.
6. LLM generation.

Rule: **measure before optimising**. Different bottlenecks require different fixes.

### Optimising Embedding Generation

Strategies:

- Cache embeddings for common queries.
- Batch embeddings where possible.
- Use smaller embedding models when latency matters more than peak accuracy.
- Use asynchronous processing for non-blocking operations.
- Compare local and remote embedding models.

Trade-offs:

| Option | Benefit | Cost |
|--------|---------|------|
| Remote embeddings | Easy scaling, strong models | Network latency, API limits, recurring cost |
| Local embeddings | Privacy, lower marginal cost | Hardware limits, maintenance, batching complexity |
| Smaller models | Faster and cheaper | Lower retrieval quality |
| Larger models | Better semantic matching | Slower and more expensive |

### Vector Index Strategies

Vector databases use different index structures depending on scale and accuracy requirements.

| Index Type | Description | Trade-Off |
|------------|-------------|-----------|
| Flat index | Exact nearest-neighbour search | Accurate but slow at scale |
| HNSW | Graph-based approximate nearest neighbour search | Fast but memory-intensive |
| IVF | Cluster-based inverted file index | Faster search, possible recall loss |
| Product Quantisation | Compresses vectors | Lower memory, possible accuracy loss |

Choose based on corpus size, latency target, memory budget, and acceptable recall loss.

### Retriever Parameters

Common parameters to tune:

| Parameter | Meaning |
|-----------|---------|
| `k` | Number of final documents returned |
| `fetch_k` | Number of candidates fetched before reranking/filtering |
| `score_threshold` | Minimum similarity score required |
| Distance metric | Cosine, dot product, Euclidean, etc. |
| HNSW `M` | Graph connectivity |
| HNSW `ef` | Search/build accuracy-speed trade-off |

A common production pattern is:

1. Retrieve more candidates than needed.
2. Apply filters and reranking.
3. Keep only the best small set for generation.

### Reranking and Context Compression

LLM generation cost and latency grow with context length. Retrieval should not blindly stuff every matching chunk into the prompt.

Strategies:

- **Reranking**: retrieve many candidates, then keep the best few.
- **Compression**: summarise retrieved documents before generation.
- **Extraction**: include only relevant snippets.
- **Structured pruning**: remove boilerplate, navigation text, headers, footers, and duplicates.

Trade-off: smaller context is faster and cheaper, but excessive compression can remove evidence the LLM needs.

### Caching Strategies

Caching can reduce latency and cost, but stale data can damage trust.

| Cache Type | Key | Value |
|------------|-----|-------|
| Query cache | User query | Final answer or retrieval result |
| Embedding cache | Query string | Query embedding |
| Retrieval cache | Query embedding/hash | Retrieved documents |
| Generation cache | Query + retrieved context | LLM response |

Use TTL policies and invalidate caches when the corpus changes.

Caching fit by use case:

- Customer support: strong fit for repeated questions.
- Research search: use cautiously; freshness matters.
- Internal knowledge base: useful, but must respect permissions and document updates.

### Handling Corpus Updates

Production retrieval systems need update strategies.

Options:

- **Batch reindexing**: periodically rebuild the whole index.
- **Incremental updates**: add, update, or delete individual documents.
- **Shadow indices**: build a new index in parallel, then swap atomically.
- **Read replicas**: improve availability during heavy read traffic.

Challenges:

- Zero-downtime updates are difficult.
- Cache invalidation must align with index updates.
- Deleted or restricted documents must not remain retrievable.

### Building Evaluation Datasets

Retrieval quality cannot be improved reliably without ground truth.

Evaluation dataset sources:

- Manually curated question-answer pairs.
- LLM-generated synthetic queries, checked by humans.
- Real user query logs with feedback.
- Adversarial examples covering known failures.

Best practices:

- Cover diverse query types and difficulty levels.
- Match the production query distribution.
- Include edge cases and ambiguous queries.
- Refresh the dataset over time because corpora and user needs change.

### Retrieval Evaluation Metrics

| Metric | Question Answered |
|--------|-------------------|
| Precision@k | Of the top `k` retrieved documents, how many are relevant? |
| Recall@k | Of all relevant documents, how many were retrieved in the top `k`? |
| MRR | How high is the first relevant result ranked? |
| NDCG | Are more useful documents ranked higher than less useful ones? |

Retrieval evaluation requires labelled relevant documents. Without labels, tuning is mostly guesswork.

### A/B Testing Retrieval Strategies

Use A/B tests to compare retrieval configurations in production-like conditions.

Examples:

- Baseline vector search vs. hybrid retrieval.
- One embedding model vs. another.
- No reranker vs. reranker.
- Different chunk sizes or metadata filters.

Rollout pattern:

```
5% users → 20% users → 50% users → 100% users
```

Track:

- Retrieval precision and recall.
- Answer quality.
- Latency.
- Cost per query.
- User engagement or satisfaction.
- Escalation or failure rate.

### Query Planning

Complex questions often require more than one retrieval step.

Example:

> "Compare the performance of method A and method B"

A query planner might:

1. Retrieve documents about method A.
2. Retrieve documents about method B.
3. Retrieve evaluation or benchmark data.
4. Synthesize a comparison.
5. Verify the answer against retrieved evidence.

This leads to advanced RAG architectures:

```
Query
→ Query planning
→ Multiple retrievals
→ Filtering and reranking
→ Context compression
→ Structured generation
→ Verification/refinement
→ Answer
```

### Production Checklist

Production RAG systems need more than retrieval accuracy.

Checklist:

- Error handling for retrieval failures, empty results, and timeouts.
- Monitoring for latency, costs, quality, and failure rates.
- Rate limiting to protect APIs and infrastructure.
- Data privacy and access control.
- Compliance requirements such as GDPR, retention rules, and audit logs.
- Scalability testing at 10x or 100x expected load.
- Cache invalidation and stale-data handling.
- Evaluation datasets and regression tests.

### Cost Considerations

Costs compound quickly in retrieval systems:

- Embedding generation.
- Vector database storage and queries.
- Reranking.
- LLM generation.
- Monitoring and infrastructure.

Local models can reduce recurring API spend, but they introduce hardware, operations, and maintenance costs.

Decision factors:

- Query volume.
- Latency target.
- Privacy requirements.
- Team expertise.
- Hardware availability.
- Accuracy requirements.

### Choosing a Retrieval Strategy

| Use Case | Recommended Strategy |
|----------|---------------------|
| Technical docs with jargon | Hybrid dense + sparse retrieval |
| Exploratory search | Multi-query retrieval |
| Large corpus with fast queries | Strong metadata filtering |
| Mixed media content | Multi-modal embeddings and routing |
| High-volume, cost-sensitive system | Local models plus aggressive caching |
| Domain-specific, high-accuracy system | Reranking plus verification |

### Build vs. Buy

| Approach | Benefits | Costs/Risks |
|----------|----------|-------------|
| Build | Full control, customisation, less vendor lock-in | Requires expertise, maintenance, infrastructure |
| Buy | Faster deployment, managed scaling, lower operational burden | Ongoing costs, less control, vendor dependency |
| Hybrid | Managed vector store plus custom retrieval logic | Still requires integration and evaluation work |

Decision factors:

- Team size and skillset.
- Budget.
- Timeline.
- Compliance and privacy requirements.
- Required scale.
- Need for custom retrieval logic.

### Common Pitfalls

- **Over-engineering too early**: start simple, then add complexity when evaluation shows a need.
- **Ignoring evaluation**: without metrics, improvements are guesses.
- **Forgetting cost monitoring**: API and generation costs can grow quickly.
- **Weak error handling**: production failures often come from timeouts, empty retrievals, or dependency outages.
- **Bad cache invalidation**: stale answers erode user trust.
- **Optimising the wrong bottleneck**: profile before tuning.
- **Testing only on small corpora**: behaviour at 100 documents may not predict behaviour at 100,000.

### Key Takeaways

- Advanced retrieval is about **strategic selection**, not just nearest-neighbour search.
- Hybrid retrieval usually beats single-method retrieval when both meaning and exact wording matter.
- Metadata filtering is one of the simplest and highest-impact performance improvements.
- Multi-vector and parent-document retrieval help balance precision with useful context.
- Reranking and compression reduce context cost while preserving relevance.
- Evaluation datasets, A/B tests, and production monitoring are required for reliable improvement.
- Production RAG is a trade-off between precision, recall, latency, cost, privacy, and operational complexity.

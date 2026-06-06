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

---

## Lecture 08: Guardrails & Safety Mechanisms

> Synthesised from Week 8 PDF: Guardrails.

### The Alignment and Safety Problem

Production AI systems interact with real users, real data, and real organisational responsibilities. Safety is not just about avoiding obviously harmful text; it is about designing systems that behave predictably, minimise harm, comply with policy, and fail safely.

Safety has several dimensions:

| Safety Type | Meaning |
|-------------|---------|
| Technical safety | System stability, predictable behaviour, robustness |
| User safety | Avoiding harm to individual users |
| Societal safety | Avoiding systemic or aggregate harm |
| Organisational safety | Compliance, reputational risk, legal exposure |

Key principle: **safety is context-dependent**. A low-stakes study assistant and a medical triage assistant need very different guardrails.

### Common AI Safety Failure Modes

| Failure Mode | Description |
|--------------|-------------|
| False negative | Harmful content slips through |
| False positive | Legitimate content is incorrectly blocked |
| Adversarial attack | User deliberately tries to bypass safeguards |
| Emergent failure | System behaves unexpectedly at scale or in novel contexts |
| Over-refusal | System refuses too often and becomes unusable |
| Under-refusal | System complies with unsafe or disallowed requests |

The goal is not perfect prevention. Perfect prevention is unrealistic. The goal is **risk reduction, detection, graceful degradation, and continuous improvement**.

### Architectural Principles for Safe AI Systems

1. **Defence in depth**: use multiple independent safety layers because any single layer can fail.
2. **Fail safely**: when risk is high or uncertainty is large, refuse, escalate, or provide conservative guidance.
3. **Observable behaviour**: if safety behaviour is not measured, it cannot be improved or secured.
4. **Human in the loop**: high-stakes decisions require human review, escalation, or approval.
5. **Context-aware controls**: different domains require different safety thresholds.

Safe systems are not built by adding one moderation endpoint. They are built through layered architecture.

### Regulatory Context

Relevant regulatory patterns:

- **EU AI Act**: risk-based regulation with stricter requirements for high-risk systems.
- **UK approach**: sectoral and principles-based regulation.
- **US approach**: fragmented regulation with federal and state-level developments.

Practical implication: AI safety is both a technical and governance problem. Systems should be designed with auditability, access control, logging, and policy traceability from the start.

### LangChain as Guardrail Infrastructure

LangChain can support guardrails because its architecture is modular:

- Safety checks can be inserted before, during, and after model calls.
- Callbacks can monitor LLM behaviour.
- Chains can compose validation, generation, moderation, and fallback steps.
- It supports both local models such as Ollama and remote APIs such as OpenAI.

Useful insertion points:

| Pipeline Stage | Guardrail Examples |
|----------------|-------------------|
| Input | Prompt injection detection, content filters, rate limits |
| Retrieval | Permission filters, metadata filters, source restrictions |
| Prompt construction | Context limits, policy injection, sanitisation |
| Generation | Model callbacks, constrained decoding, tool restrictions |
| Output | Moderation, factual checks, schema validation |
| Post-output | Logging, user feedback, escalation, incident detection |

### Content Filtering

Content filtering is the first line of defence, but it is not sufficient alone.

Three main approaches:

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| Pattern matching | Fast, transparent, cheap | Brittle, easy to bypass, poor with nuance |
| Classification models | Better category detection | Needs training data, can still miss context |
| LLM-based moderation | Handles nuance and novel cases | Expensive, slower, moderator can fail too |

Best practice: combine approaches rather than relying on one.

### Pattern-Based Filters

Pattern filters use rules such as regular expressions or keyword lists.

Good for:

- Known banned phrases.
- Obvious policy violations.
- Simple data leakage patterns.
- Basic input sanitation.

Weak for:

- Encoded or obfuscated text.
- Context-sensitive meaning.
- Multi-turn manipulation.
- Distinguishing discussion of harm from promotion of harm.

Pattern matching is useful as a cheap first pass, not as the whole safety system.

### The False Positive Problem

False positives can cause harm when legitimate help-seeking is blocked.

Example: a depression-support chatbot blocking phrases such as:

- "I'm having dark thoughts"
- "I feel like giving up"
- "I don't want to be here anymore"

These may indicate a user needs support, not that the user should be blocked.

Design implication:

- Distinguish **expressing distress** from **promoting harm**.
- Prefer supportive escalation over blunt refusal in sensitive contexts.
- Use confidence thresholds and human escalation for high-risk categories.

### Classification-Based Moderation

Classification models detect categories of unsafe content.

Typical categories:

- Violence and physical harm.
- Hate speech and harassment.
- Sexual content and child safety.
- Self-harm and suicide.
- Illegal activities.
- Privacy and personal data leakage.

Good moderation systems usually use:

- Multiple classifiers or model ensembles.
- Confidence thresholds.
- Escalation when confidence is uncertain.
- Human review for high-impact decisions.

### LLM-Based Moderation

LLM-based moderation uses a second model to judge input or output from the primary model.

Advantages:

- Better contextual reasoning.
- Handles novel or ambiguous cases.
- Can explain the reason for a safety decision.

Disadvantages:

- Adds latency.
- Adds cost.
- Can hallucinate or misclassify.
- Can be attacked or manipulated itself.

Use LLM moderation where nuance matters, especially for high-stakes or ambiguous content.

### Prompt Injection and Jailbreaking

Prompt injection attacks attempt to override the system's intended behaviour.

Common attack vectors:

- "Ignore previous instructions..."
- Role-playing to bypass restrictions.
- Encoding unsafe instructions.
- Multi-turn attacks that gradually shift boundaries.
- Instructions hidden in retrieved documents.
- Exploiting weak system prompts.

Critical insight: **perfect prevention is impossible**. Design for detection, containment, and graceful degradation.

Mitigations:

- Treat user input and retrieved documents as untrusted.
- Separate system instructions from user-controllable content.
- Validate tool calls and outputs.
- Restrict what tools the model can access.
- Monitor repeated bypass attempts.
- Red-team the system continuously.

### Layered Guardrail Architecture

A practical guardrail pipeline:

```
User input
-> Input validation
-> Prompt injection checks
-> Retrieval with permission filters
-> Context sanitisation
-> LLM generation
-> Structured output validation
-> Output moderation
-> Grounding/factual checks
-> Fallback or escalation if needed
-> Logging and monitoring
-> User response
```

Each layer handles a different risk. No layer should be assumed perfect.

### Bias Detection

Bias detection is harder than content filtering because fairness is contextual, cultural, and contested.

Challenges:

- Protected characteristics vary by jurisdiction.
- Statistical fairness metrics can conflict.
- Ground truth may be disputed.
- Bias can appear through subtle wording, ranking, recommendations, or omissions.
- Intersectional effects may be missed by simple group comparisons.

Technical systems cannot decide what is fair on their own. They can make outcomes visible and measurable so policy decisions can be implemented.

### Categories of Algorithmic Bias

| Bias Type | Description |
|-----------|-------------|
| Training data bias | Historical patterns are encoded into data |
| Representation bias | Some groups are overrepresented, others absent |
| Measurement bias | Outcomes or labels are measured poorly |
| Aggregation bias | Diverse groups are treated as one uniform group |
| Evaluation bias | System is tested differently or insufficiently across groups |

Post-hoc filtering alone cannot fix these. Bias often enters earlier in data collection, task design, labelling, and evaluation.

### Technical Approaches to Bias Detection

Common methods:

- **Statistical parity analysis**: compare outcomes across demographic groups.
- **Demographic parity difference**: measure output-rate differences between groups.
- **Disparate impact analysis**: identify whether one group receives worse outcomes.
- **Intersectional analysis**: test combinations of identities rather than one attribute at a time.
- **Matched prompt testing**: compare outputs for prompts that differ only by demographic indicators.
- **Causal analysis**: investigate whether protected attributes causally affect outcomes.

Bias tests should be built into evaluation suites and rerun as models, prompts, and data change.

### Bias Mitigation Strategies

Bias mitigation can occur at different stages.

| Stage | Strategies |
|-------|------------|
| Pre-processing | Data augmentation, reweighting, additional data collection, balancing representation |
| In-processing | Fairness-aware prompting, constrained generation, adversarial debiasing |
| Post-processing | Output rewriting, calibration, threshold adjustment, review workflows |

Trade-offs:

- Accuracy vs. fairness.
- Simplicity vs. effectiveness.
- Individual fairness vs. group fairness.
- Consistency vs. context sensitivity.

Bottom line: **bias mitigation is policy work supported by technical tools**, not a purely technical problem.

### Output Validation

Output validation constrains model behaviour after generation and can also guide generation upfront.

Validation types:

| Validation Type | Example |
|-----------------|---------|
| Format validation | Must produce valid JSON |
| Schema validation | Must match a Pydantic model |
| Logical consistency | Dates, totals, or claims must not contradict |
| Business rules | Must include required disclaimers |
| Factual grounding | Claims must be supported by retrieved sources |
| Safety validation | Output must not contain disallowed content |

Principle: **make invalid outputs impossible where possible, not merely filtered after generation**.

### Structured Output

Structured output uses schemas to force responses into predictable formats.

Example fields for a safe response:

- `content`: the actual response.
- `confidence`: numeric confidence score.
- `safety_flags`: list of detected risks.
- `sources`: citations or document references.
- `requires_human_review`: boolean escalation flag.

Benefits:

- Easier validation.
- Easier downstream processing.
- Better monitoring.
- Clearer failure handling.

### Factual Grounding

LLMs can produce plausible but false claims. Guardrails should reduce hallucination risk.

Mitigation approaches:

- Retrieval-Augmented Generation to ground answers in documents.
- Citation requirements.
- External verification APIs for factual claims.
- Claim extraction followed by evidence checking.
- Confidence estimation and uncertainty language.
- Refusal or escalation when evidence is missing.

Important distinction:

- RAG can reduce hallucinations.
- RAG does not guarantee truth.
- Retrieved context can be wrong, incomplete, outdated, or malicious.

### Business Rule Enforcement

Domain-specific rules often matter more than generic moderation.

Examples:

| Domain | Rule |
|--------|------|
| Finance | Must not provide unsuitable financial recommendations |
| Healthcare | Must advise professional consultation for serious symptoms |
| Legal | Must state that output is not legal advice |
| Education | Must guide learning without completing assessed work |
| HR | Must respect confidentiality and employment policy |

Implementation options:

- Programmatic validators.
- Policy prompts.
- Structured output requirements.
- Human escalation.
- Tool access restrictions.

### Graceful Degradation and Fallbacks

When validation fails, the system needs a planned response.

Fallback options:

| Fallback | Use When |
|----------|----------|
| Hard block | Risk is high and no safe answer is possible |
| Conservative response | General safe guidance is acceptable |
| Explain and retry | User can reformulate safely |
| Human escalation | High-stakes or ambiguous cases |
| Safe alternative | Provide adjacent allowed help |

Example for education: instead of completing an assignment, the assistant can explain concepts, ask guiding questions, review a draft, or suggest a study plan.

### Monitoring and Observability

Guardrails must be observable in production.

Monitor:

- Guardrail trigger rates.
- False positive reports.
- False negative reports.
- Latency added by safety layers.
- Cost added by moderation or verification.
- Repeated bypass attempts.
- Bias and fairness metrics over time.
- Retrieval failures and grounding failures.
- Human escalation rates.

Safety metrics should be tracked alongside normal product metrics.

### Alerting

Alerts should be actionable, not merely informative.

Alert on:

- Sudden spikes in guardrail triggers.
- Repeated bypass attempts from the same user or source.
- Guardrail mechanisms failing open.
- Fairness metrics degrading.
- High-risk categories appearing unexpectedly.
- Sharp increases in user reports or escalations.

Do not alert on every individual guardrail trigger. That creates noise and alert fatigue.

### Incident Response

Safety incidents need documented playbooks before they happen.

Incident response stages:

1. **Contain**: disable affected functionality if necessary.
2. **Investigate**: identify what happened and which guardrail failed.
3. **Remediate**: fix the immediate issue.
4. **Learn**: update tests, filters, prompts, policies, and monitoring.
5. **Communicate**: inform affected users and stakeholders where appropriate.

Incident response is part of the safety system, not an afterthought.

### Cost-Benefit Analysis of Safety Mechanisms

Safety mechanisms impose costs:

- More latency.
- More compute.
- More model calls.
- More development work.
- More maintenance.
- More false positives.

They also provide benefits:

- Reduced harm.
- Regulatory compliance.
- User and stakeholder trust.
- Better understanding of system behaviour.
- Lower incident risk.

The target is not "maximum safety at all costs". The target is **appropriate safety for the use case**.

### Testing Guardrails

Guardrails should be tested adversarially.

Testing methods:

- **Red teaming**: deliberately attempt to bypass safety controls.
- **Fuzzing**: generate unusual or malformed inputs automatically.
- **Prompt injection tests**: test direct and indirect injection attempts.
- **Bias test suites**: use matched prompts across demographic groups.
- **Regression tests**: ensure old failures do not reappear.
- **External penetration testing**: use outside experts for high-stakes systems.
- **Bug bounties**: incentivise responsible reporting.

Mindset: assume guardrails can be bypassed and test that assumption continuously.

### Ethical Considerations and Transparency

Guardrails encode value judgments. Teams should be explicit about those judgments.

Questions to ask:

- Who decided what counts as safe?
- Were affected communities consulted?
- Are trade-offs between safety goals documented?
- Can users understand why content was blocked or modified?
- Are guardrails protecting users, the organisation, or both?
- Is the system fair across different groups and contexts?

Transparency principle: users should understand how safety mechanisms affect their experience, especially when content is refused, altered, escalated, or logged.

### High-Stakes Example: Medical Triage Assistant

For a medical triage assistant, high-priority guardrails include:

- Conservative advice under uncertainty.
- Clear escalation for urgent symptoms.
- Strong refusal boundaries for diagnosis certainty.
- Medical disclaimers and professional consultation guidance.
- Bias testing across demographic groups and access-to-care contexts.
- Monitoring for unsafe under-triage or over-triage.
- Human review for risky cases.
- Audit logs for safety and compliance.

This example shows why domain context matters: the cost of a false negative may be severe.

### Emerging Safety Challenges

Future guardrail problems include:

- **Multi-agent systems**: multiple models interacting can create new failure modes.
- **Real-time systems**: latency constraints may limit moderation and verification.
- **Multimodal models**: text, image, audio, and video introduce new attack surfaces.
- **Personalised models**: adaptation to users can create fairness and privacy risks.
- **Tool-using agents**: unsafe outputs may become unsafe actions.
- **Indirect prompt injection**: malicious content in retrieved documents or external tools can steer the model.

Safety designs must evolve as model capabilities and deployment patterns change.

### Key Takeaways

- Defence in depth is non-negotiable because any single guardrail can fail.
- Context determines appropriate safety: high-stakes domains need more conservative controls.
- Technical guardrails implement policy decisions; they do not remove the need for governance.
- Monitoring, red teaming, and incident response are continuous practices.
- Bias mitigation requires measurement, policy decisions, and technical support.
- Output validation, grounding, and business rules make AI systems more reliable.
- Transparency and accountability are as important as technical robustness.

---

## Lecture 09: Model Context Protocol (MCP) & Tool Integration

> Synthesised from Week 9 PDF: MCP & Tools.

### Why Models Need Tools

LLMs are powerful language systems, but they have hard limitations:

- They are frozen at training time.
- They cannot access real-time data by default.
- They cannot perform deterministic operations reliably on their own.
- They lack direct access to private databases, APIs, files, or business systems.
- They cannot safely perform real-world actions without controlled integration.

Tools extend LLMs from **text generators** into **systems that can retrieve, calculate, act, and interact with external services**.

Examples:

- Check current weather.
- Query a database.
- Search a product catalogue.
- Create a calendar event.
- Run code.
- Send a support ticket.
- Retrieve private enterprise data with permissions.

### RAG vs. MCP

RAG and MCP solve related but different problems.

| Aspect | RAG | MCP |
|--------|-----|-----|
| Primary goal | Improve answers using retrieved knowledge | Standardise how AI apps access tools, data, and services |
| Mechanism | Retrieve documents and add them to the prompt | Client-server protocol for tools, resources, and prompts |
| Typical output | Better grounded answer | Tool calls, actions, structured data access |
| Standardisation | Architectural pattern, not a universal protocol | Open protocol for reusable integrations |
| Best for | Knowledge lookup, Q&A, document grounding | Actions, APIs, real-time data, agent workflows |

Simple distinction:

- **RAG tells the model what to know.**
- **Tools/MCP let the model do things.**

### What Is MCP?

**Model Context Protocol (MCP)** is an open protocol developed by Anthropic for standardising how AI applications connect to external tools and data sources.

Core properties:

- Client-server architecture.
- Language and framework agnostic.
- Supports reusable tool integrations.
- Decouples tool providers from model providers.
- Enables AI hosts to discover and invoke external capabilities.

MCP is useful because it avoids every AI application needing custom one-off integrations for every tool or service.

### MCP Architecture

MCP has several key components:

| Component | Role |
|-----------|------|
| Host | AI application embedding the LLM, such as an IDE or desktop assistant |
| MCP Client | Protocol client inside the host that connects to an MCP server |
| MCP Server | Lightweight program exposing tools, resources, or prompts |
| External Resource | Data source such as files, databases, APIs, or records |
| Tool | Executable function the AI application can invoke |
| Resource | Read-only contextual data exposed to the AI application |
| Prompt | Reusable prompt template exposed by a server |

Typical flow:

```
User asks a question
-> Host receives request
-> Host/LLM decides a tool is needed
-> MCP client contacts MCP server
-> MCP server executes tool or returns resource
-> Tool result returns to host
-> LLM uses result to answer the user
```

Example:

```
User: What is the weather in Waterford?
Host -> MCP client -> Weather MCP server
Weather server -> external weather API
Weather result -> host
LLM -> final answer
```

### MCP Host, Client, and Server Responsibilities

The **host** orchestrates the user interaction:

- Manages the conversation.
- Sends prompts and context to the LLM.
- Decides when tools may be used.
- Routes tool calls through MCP clients.
- Feeds tool results back to the model.
- Returns the final response to the user.

The **MCP client** acts as a broker:

- Maintains connection to an MCP server.
- Discovers tools, resources, and prompts.
- Sends tool requests.
- Receives structured results.

The **MCP server** exposes capabilities:

- Implements tools.
- Provides resources.
- Provides prompt templates.
- Handles protocol requests.
- Connects to files, APIs, databases, or services.

### MCP vs. Traditional Function Calling

Traditional function calling usually means the host application defines a fixed set of local functions and exposes their schemas directly to the model API.

| Traditional Function Calling | MCP |
|------------------------------|-----|
| Tightly coupled to one application | Loosely coupled via protocol |
| Often vendor-specific | Model/provider agnostic |
| Tools live inside the host app | Tools can live in reusable servers |
| Limited composability | Mix-and-match server capabilities |
| Harder to reuse across apps | Designed for reuse |

MCP is not a replacement for function calling at the model level. It is a standardised integration layer around tool and resource access.

### MCP Server Primitives

MCP servers expose three main primitives.

| Primitive | Purpose | Examples |
|-----------|---------|----------|
| Tools | Executable functions with possible side effects | Send email, query API, update CRM, create file |
| Resources | Read-only contextual data | File contents, database record, API response |
| Prompts | Reusable prompt templates | Report template, analysis framework, few-shot prompt |

Choosing the right primitive matters:

- Use **resources** for read-only data.
- Use **tools** for actions or computations.
- Use **prompts** for reusable interaction patterns.

### MCP Client Primitives

MCP also defines primitives that clients can expose:

| Primitive | Purpose |
|-----------|---------|
| Sampling | Lets servers request model completions through the client |
| Elicitation | Lets servers ask the user for additional information |
| Logging | Lets servers send debugging or monitoring logs to the client |

These allow richer two-way interactions between hosts and servers.

### Tool Calling Fundamentals

Tool calling usually follows this loop:

1. Tool schemas are provided to the model.
2. User asks a request.
3. Model decides whether a tool is needed.
4. Model emits a structured tool call.
5. Application executes the tool.
6. Tool returns structured result or error.
7. Model incorporates the result into the final response.

Good tool schemas are critical because the model relies on descriptions to decide when and how to use tools.

### Tool Description Best Practices

A weak description:

> Gets weather.

A stronger description:

> Retrieves current weather for a specific city. Requires city name and optional country code. Returns temperature, conditions, humidity, and an error if the city is not found.

Good tool descriptions should:

- Say when to use the tool.
- Define required and optional parameters.
- Describe expected outputs.
- Explain constraints and failure modes.
- Include examples where useful.
- Avoid vague names and ambiguous behaviour.

Poor descriptions cause the model to call tools at the wrong time or with invalid inputs.

### Multi-Tool Reasoning

Models can use multiple tools to answer one request.

Example:

> "What's the weather like where my next meeting is?"

Possible tool chain:

1. Query calendar tool for next meeting.
2. Extract meeting location.
3. Query weather tool for that location.
4. Summarise weather in relation to the meeting time.

Tool calls can be:

- **Sequential** when later calls depend on earlier results.
- **Parallel** when calls are independent.
- **Conditional** when the next tool depends on previous output.

### Building Custom MCP Servers

Development lifecycle:

1. Define capabilities: tools, resources, prompts.
2. Choose a framework or language, such as Python or TypeScript.
3. Implement protocol handlers.
4. Add security controls and validation.
5. Test with MCP inspector or compatible clients.
6. Deploy and configure hosts/clients.
7. Monitor usage, failures, and cost.

For this course, the target maturity is production-ready basics: secure tools, proper schemas, validation, error handling, and monitoring.

### Resource vs. Tool Semantics

Use resources when the server is exposing information.

Examples:

- Read a document.
- Fetch a database row.
- Return API data.
- Provide configuration or metadata.

Use tools when the server performs an operation.

Examples:

- Send an email.
- Update a record.
- Create a file.
- Run a query.
- Trigger a workflow.

Rule of thumb: if it has side effects or performs an action, treat it as a tool and apply stricter permissions.

### Integrating External APIs

Common API integration patterns:

- REST APIs through HTTP requests.
- GraphQL query execution.
- WebSocket streams for real-time data.
- gRPC for high-performance RPC.
- Database connectors.
- Internal enterprise services.

Integration concerns:

- Authentication and secret management.
- Rate limits.
- Timeouts and retries.
- Response validation.
- Error handling.
- Cost tracking.

### Handling Tool and API Failures

Failures are expected in production tool use.

Common failure types:

- Network timeout.
- Invalid credentials.
- Rate limit exceeded.
- Service unavailable.
- Malformed response.
- Partial failure in batch operations.
- Permission denied.
- Invalid tool parameters.

Best practice: return structured errors the model and host can interpret.

Example structured error:

```json
{
  "ok": false,
  "error_type": "rate_limit",
  "message": "Weather API rate limit exceeded",
  "retry_after_seconds": 60
}
```

This is better than returning an unstructured exception string.

### Security Threat Model for Tool Access

Tool integration increases the attack surface because a probabilistic model can now access valuable systems.

Threats include:

- Prompt injection.
- Unauthorised data access.
- Privilege escalation.
- Data exfiltration.
- Tool abuse.
- Denial of service through expensive calls.
- SQL injection.
- Shell command injection.
- File system traversal.
- Leaking secrets or credentials.

Security must be designed before deployment, not added after incidents.

### Authentication and Authorisation

Tool systems need multi-layer access control.

| Layer | Question |
|-------|----------|
| User authentication | Who is making the request? |
| Tool permissions | Is this user allowed to call this tool? |
| Resource permissions | Is this user allowed to access this specific data? |
| Action permissions | Is this user allowed to perform this operation? |
| Audit trails | Can we reconstruct what happened? |
| Credential isolation | Are secrets protected from the model and user? |

Important rule: the model should not become a way to bypass normal access control.

### Input Validation and Sanitisation

Tool parameters must be validated against schemas.

Examples:

- Validate IDs are IDs, not arbitrary SQL.
- Use parameterised queries.
- Avoid shell execution where possible.
- Restrict file paths to allowed directories.
- Enforce input length limits.
- Rate-limit expensive operations.
- Validate enum values.

Dangerous pattern:

```python
cursor.execute(sql)
```

Safer pattern:

```python
cursor.execute(
    "SELECT * FROM users WHERE id = %s",
    (user_id,)
)
```

Never trust model-generated tool arguments without validation.

### Prompt Injection Mitigations for Tools

Prompt injection becomes more dangerous when tools can act.

Mitigations:

- Reinforce system-level instructions.
- Validate tool arguments programmatically.
- Require confirmation for sensitive actions.
- Use human-in-the-loop approval for critical operations.
- Restrict tools by user role and context.
- Log and monitor tool usage.
- Detect anomalous call patterns.
- Separate untrusted retrieved content from trusted instructions.
- Avoid giving broad tools when narrow tools would suffice.

Example: prefer `get_invoice_by_id(invoice_id)` over `run_arbitrary_sql(sql)`.

### Performance Optimisation for Tool Execution

Tool calls can dominate latency.

Optimisation strategies:

- Cache tool results.
- Run independent tools in parallel.
- Lazy-load resources only when needed.
- Optimise response size.
- Use connection pooling.
- Set timeouts.
- Avoid unnecessary tool calls.

Caching location depends on the architecture:

- **Host cache**: useful for user/session-level repeated requests.
- **Client cache**: useful across a specific server connection.
- **Server cache**: useful for shared external API calls and expensive computations.

The right choice depends on data freshness, permissions, and reuse patterns.

### Monitoring and Observability

Tool systems need detailed observability.

Track:

- Latency per tool: p90, p95, p99.
- Success and failure rates.
- Error categories.
- Cost per API call.
- Tool usage frequency.
- User and tenant usage patterns.
- Anomalous or suspicious tool usage.
- Distributed traces across host, client, server, and external API.

Observability helps debug failures, control costs, and identify security incidents.

### Advanced Tool Integration Patterns

Advanced systems use more than one-off tool calls.

Patterns:

- **Tool composition**: combine several tools into a workflow.
- **Tool chaining**: output of one tool becomes input to another.
- **Conditional execution**: choose tools based on intermediate results.
- **Parallel execution**: run independent calls concurrently.
- **Result validation**: verify tool outputs before using them.
- **Retry logic**: retry transient failures safely.
- **Dynamic discovery**: discover available tools from MCP servers.
- **Tool versioning**: manage changes without breaking clients.

These patterns improve capability but increase complexity, so they should be justified by use case needs.

### Error Recovery Strategies

Production tools should degrade gracefully.

Recovery techniques:

- Fallback tools or alternate data sources.
- Partial success handling.
- Circuit breakers for failing services.
- Exponential backoff with jitter.
- Clear user notification when a tool is unavailable.
- Retry only when safe and idempotent.
- Escalation for critical failures.

Example: if a live stock-price API fails, a system might return the latest cached value with a timestamp and warning.

### Testing Tool Integrations

Testing should cover correctness, reliability, security, and scale.

| Test Type | Purpose |
|-----------|---------|
| Unit tests | Validate individual tool logic |
| Integration tests | Verify live API/database behaviour |
| Mock tests | Test without depending on external services |
| Adversarial tests | Find injection and abuse paths |
| Load tests | Measure behaviour under high traffic |
| Chaos tests | Verify resilience to failures |
| Regression tests | Ensure fixed bugs do not return |

Tools are production software components and should be tested like production software.

### Evaluating Tool Effectiveness

Metrics for tool quality:

- **Precision**: how often the tool returns correct results.
- **Recall**: whether the tool finds all relevant information.
- **Latency**: especially p95 and p99 response times.
- **Cost efficiency**: value per API call.
- **Robustness**: behaviour on edge cases and failures.
- **Usefulness**: whether tool use improves final answer or task completion.
- **Safety**: whether tool access respects permissions and policy.

Tool evaluation should measure both tool output and final model behaviour after using the tool.

### Ethical Considerations

Responsible tool integration requires:

- Transparency: users should know when AI uses tools.
- Accountability: tool actions should have audit trails.
- Consent: sensitive operations should require approval.
- Privacy: tools should not expose unnecessary data.
- Bias awareness: tools can amplify existing system biases.
- Environmental awareness: high-volume tool/model calls consume resources.

Sensitive example: an AI assistant with access to employee performance reviews, salary information, medical leave records, and calendars needs strict access limits, auditing, and explicit policies.

### Integration Patterns Summary

Key architectural choices:

| Choice | Options |
|--------|---------|
| Coupling | Tight local function calls vs. loose MCP integration |
| Execution | Synchronous vs. asynchronous |
| State | Stateless tools vs. stateful sessions |
| Scope | Single-purpose tools vs. multi-capability tools |
| Deployment | Local process vs. remote service |
| Discovery | Static tool list vs. dynamic MCP discovery |

Trade-offs must be evaluated for the use case rather than chosen abstractly.

### Common Pitfalls

- Over-engineering tool ecosystems.
- Insufficient error handling.
- Ignoring security until too late.
- Poor tool descriptions that confuse models.
- Not monitoring costs and performance.
- Coupling tool implementation too tightly to one model provider.
- Giving models broad tools when narrow tools would be safer.
- Returning unstructured errors that the model cannot recover from.

### Tool Integration Maturity Model

| Level | Description |
|-------|-------------|
| 1. Basic | Hardcoded functions, little or no error handling |
| 2. Functional | Proper schemas and basic validation |
| 3. Production | Security, monitoring, error recovery |
| 4. Advanced | Dynamic discovery, versioning, optimisation |
| 5. Autonomous | Adaptive and self-improving tool ecosystems |

For this course, aim for **Level 3: Production**.

That means:

- Clear tool schemas.
- Input validation.
- Access control.
- Structured errors.
- Monitoring.
- Retries and graceful degradation.
- Documentation.
- Tests.

### Cross-Cutting Concerns

Every production tool integration must address:

| Concern | Requirement |
|---------|-------------|
| Security | Validate inputs, control access, log actions |
| Reliability | Handle failures, retry safely, degrade gracefully |
| Performance | Manage latency, caching, response size, and cost |
| Observability | Trace execution and collect useful metrics |
| Maintainability | Document tools, test thoroughly, design for change |

Ignoring any one of these usually creates production risk.

### When Are Tools the Right Solution?

Use tools when the system needs:

- Real-time information.
- Deterministic computation.
- External API access.
- Private data access with permissions.
- Actions with side effects.
- Workflow automation.

Prefer RAG when the main need is:

- Answering from documents.
- Reducing hallucination with knowledge grounding.
- Searching static or semi-static knowledge bases.

Prefer fine-tuning when the main need is:

- Changing style or behaviour.
- Teaching task format.
- Improving domain-specific response patterns.

Do not add tools just because they are available. Tool integration adds complexity, security risk, latency, and operational cost.

### Preparing Tool Systems for Production

Production readiness includes:

- Hosting and scaling plans.
- CI/CD for tool updates.
- Versioning and compatibility management.
- Monitoring and alerting.
- Incident response procedures.
- Cost forecasting.
- Security review.
- Permission model.
- Audit trails.
- Documentation for operators and users.

Prototype success does not imply production readiness.

### Key Takeaways

- MCP standardises AI-tool communication and enables reusable, interoperable tool ecosystems.
- RAG retrieves knowledge; MCP/tools enable actions, real-time data, and structured service access.
- Good tool schemas are essential because models rely on descriptions to choose and call tools correctly.
- Security is fundamental: validate inputs, enforce access control, isolate credentials, and audit tool calls.
- Failure is expected, so tools need structured errors, retries, fallbacks, and graceful degradation.
- Observability enables improvement: measure latency, cost, success rates, failures, and anomalous use.
- Architecture choices such as coupling, state, scope, and deployment determine long-term system quality.

---

## Lecture 10: AI Agents & Orchestration

> Synthesised from Week 10 PDF: Agents.

### What Makes an Agent Different?

Traditional LLM interactions are usually:

- Stateless.
- Request-response based.
- Driven by a human deciding the next step.
- Focused on single-turn problem solving.

Agent-based systems add:

- State across interactions.
- Autonomous decision-making.
- Multi-step reasoning and planning.
- Tool use.
- Environment interaction.
- Memory and adaptation over time.

Simple distinction:

- **LLM call**: "Answer this prompt."
- **Agent**: "Work toward this goal using reasoning, tools, memory, and feedback."

### Core Agent Components

| Component | Role |
|-----------|------|
| Perception | Understand current state, input, available tools, and environment |
| Reasoning | Decide what to do next based on goals and context |
| Action | Execute decisions through tools, API calls, or responses |
| Memory | Maintain context, past experience, user preferences, and task state |

Agents are systems, not just prompts. Their behaviour emerges from how these components interact.

### The ReAct Pattern

**ReAct = Reasoning + Acting**.

The agent cycles through:

1. **Thought**: reason about the current situation.
2. **Action**: choose a tool or response.
3. **Observation**: inspect the result.
4. Repeat until the goal is achieved or a stop condition is met.

Typical loop:

```
User goal
-> Thought
-> Action/tool call
-> Observation
-> Thought
-> Action/tool call
-> Observation
-> Final answer
```

Strengths:

- Good for multi-step tasks.
- Makes tool use explicit.
- Supports iterative problem solving.

Risks:

- Can loop indefinitely.
- Can misuse tools.
- Can lose context.
- Can become expensive due to repeated LLM calls.

Mitigations:

- Maximum iteration limits.
- Better tool descriptions.
- Tool argument validation.
- Loop detection.
- Logging intermediate steps.

### Agent Architecture Patterns

| Pattern | Structure | Best For |
|---------|-----------|----------|
| Single-agent with tools | One LLM coordinates multiple tools | Focused tasks with clear tool boundaries |
| Chain-of-agents | Agents run sequentially like a pipeline | Workflows with distinct stages |
| Hierarchical agents | Manager delegates to specialist agents | Complex tasks requiring different expertise |
| Collaborative agents | Peer agents coordinate on shared goals | Problems benefiting from diverse perspectives |

Architecture matters because it determines how work is decomposed, coordinated, debugged, and controlled.

### Autonomy vs. Control

More autonomy gives agents flexibility:

- Handles unexpected situations.
- Can pursue creative solutions.
- Requires less human intervention.

But autonomy adds risk:

- Less predictable behaviour.
- Harder debugging.
- Higher chance of tool misuse.
- More complex safety requirements.

More control gives:

- Bounded behaviour.
- Easier validation.
- Lower misuse risk.
- Better auditability.

But too much control limits adaptability.

Best practice: use **task-appropriate autonomy**, not maximum autonomy.

### Planning in Agent Systems

Agents can plan in different ways.

| Planning Type | Description | Trade-Off |
|---------------|-------------|-----------|
| Reactive planning | Decide based only on current state | Fast but limited foresight |
| Forward planning | Generate action sequence before execution | Better foresight but more expensive |
| Hierarchical planning | Break goals into subgoals recursively | Balanced and human-like |

Planning is useful when tasks require multiple dependent steps, tool selection, or coordination.

### ReWOO Pattern

**ReWOO = Reasoning Without Observation**.

It separates planning from execution:

1. **Plan phase**: generate a complete action plan before executing tools.
2. **Execution phase**: run planned actions, in parallel where possible.
3. **Response phase**: synthesise results into final answer.

Advantages:

- Fewer LLM calls.
- Lower latency and cost.
- Easier parallelisation.
- Execution failures are separated from reasoning.

Trade-off: because the plan is created before observations, it may be less adaptive than ReAct.

### Agent Reasoning Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| Chain-of-Thought | Step-by-step reasoning | Multi-step logic and calculations |
| Tree-of-Thoughts | Explore and evaluate multiple reasoning paths | Problems with several possible approaches |
| Self-Consistency | Generate multiple reasoning chains and aggregate | Reducing occasional reasoning errors |
| Reflexion | Agent critiques and improves based on past attempts | Learning from failure and iterative improvement |

Reasoning strategies can improve reliability but often increase cost and latency.

### Multi-Agent Systems

Multiple agents are useful when one monolithic agent becomes hard to manage.

Benefits:

- Specialisation by domain or task.
- Scalability through distributed work.
- Modularity: individual agents can be updated separately.
- Robustness: one agent failure may not break the whole system.

Challenges:

- Coordination overhead.
- Inconsistent outputs between agents.
- Harder debugging.
- Shared state conflicts.
- Higher cost.
- More complex evaluation.

Use multi-agent designs only when the task complexity justifies them.

### Communication Patterns in Multi-Agent Systems

| Pattern | Description | Use When |
|---------|-------------|----------|
| Broadcast | One agent sends information to all agents | All agents need the same update |
| Direct messaging | Agent-to-agent communication | Specific agent needs specific information |
| Blackboard | Shared memory all agents can read/write | Collaborative problem solving |
| Manager-worker | Central coordinator delegates subtasks | Clear hierarchy and task decomposition |

Example manager-worker flow:

```
Manager receives task
-> decomposes into subtasks
-> assigns subtasks to specialist agents
-> specialists return results
-> manager synthesises final answer
```

### Agent Orchestration Patterns

Orchestration coordinates agents and tools across workflows.

| Pattern | Description |
|---------|-------------|
| Sequential orchestration | Agents execute in predefined order |
| Parallel orchestration | Multiple agents work simultaneously |
| Conditional orchestration | Runtime conditions decide which agent runs next |
| Event-driven orchestration | Agents respond asynchronously to events |

LangGraph-style state graphs are useful for orchestration because they make workflow state and transitions explicit.

### State Management in Multi-Agent Workflows

State management is difficult because workflows evolve over time and multiple agents may update shared state.

Challenges:

- **Consistency**: agents need a coherent view of state.
- **Concurrency**: several agents may update state simultaneously.
- **Persistence**: state should survive failures and restarts.
- **Size**: state grows as workflows progress.
- **Relevance**: not all old state should remain active.

Patterns:

- **Immutable state**: agents produce new state versions rather than mutating existing state.
- **Event sourcing**: record changes as events.
- **Checkpointing**: periodically save workflow snapshots.
- **Selective persistence**: store only critical state.

### Memory in Agents

Memory lets agents behave coherently over time.

Why memory matters:

- Maintains context across turns.
- Learns from previous successes and failures.
- Supports personalisation.
- Enables long-term behaviour.
- Helps avoid repeating past mistakes.

Types of memory:

| Memory Type | Description |
|-------------|-------------|
| Short-term memory | Current conversation, task state, scratchpad |
| Long-term memory | Persistent cross-session information |
| Episodic memory | Specific past experiences or interactions |
| Semantic memory | General learned facts, patterns, or preferences |

### Memory Implementations

Common implementations:

| Memory Method | Strength | Weakness |
|---------------|----------|----------|
| Buffer memory | Keeps recent conversation exactly | Grows quickly, token expensive |
| Summary memory | Compresses older context | May lose detail |
| Vector memory | Retrieves semantically relevant past interactions | May miss important but semantically distant context |
| Profile memory | Stores explicit user facts/preferences | Requires update and privacy controls |

A strong architecture often combines short-term and long-term memory.

### Memory Retrieval Strategies

| Strategy | Benefit | Limitation |
|----------|---------|------------|
| Recency-based | Maintains conversational coherence | Misses older relevant information |
| Relevance-based | Finds semantically similar past interactions | May miss critical context outside similarity |
| Importance-weighted | Prioritises explicitly important memories | Requires scoring importance |
| Hybrid | Combines recent, relevant, and important memory | More complex to implement |

Practical pattern:

1. Always include recent conversation.
2. Retrieve semantically relevant past memory.
3. Add explicitly important user preferences or constraints.
4. Compress or summarise when context grows too large.

### Agent Memory Lifecycle

Memory lifecycle:

```
Input
-> Encode
-> Store
-> Retrieve
-> Use
-> Update or forget
```

Memory systems should also support forgetting:

- Remove outdated information.
- Respect privacy requests.
- Avoid retaining sensitive data unnecessarily.
- Prevent irrelevant memories from polluting context.

### Production Considerations for Agents

Agents introduce production risks beyond normal LLM applications.

| Area | Requirements |
|------|--------------|
| Reliability | Graceful degradation, timeouts, retries, recovery |
| Safety | Rate limits, approvals, audit logs, sandboxing |
| Cost | Token monitoring, caching, batching, smaller models for subtasks |
| Observability | Logs, traces, metrics, feedback loops, continuous evaluation |
| Security | Tool permissions, secret isolation, input validation |
| Scalability | Load balancing, distributed memory, multi-region deployment where needed |

Agents should not be given broad autonomy in production without guardrails and monitoring.

### Ethical Considerations in Agent Systems

Autonomous agents raise ethical questions:

- Who is accountable when an agent makes a mistake?
- Should the agent disclose that it is AI?
- Can the agent amplify bias?
- What personal data can the agent access or remember?
- Could the agent be misused for spam, fraud, manipulation, or surveillance?
- What decisions require human oversight?

Example: a hiring agent reviewing CVs and screening candidates needs bias testing, transparency, appeal processes, privacy controls, and human decision-making authority.

### Agent Evaluation Challenges

Agent evaluation is harder than ordinary model evaluation because:

- Outputs are non-deterministic.
- Tasks involve multiple steps.
- Failures can happen at any step.
- External tools affect behaviour.
- Agents may use different paths for the same goal.
- Open-ended tasks may not have one correct answer.
- Emergent behaviour is hard to predict.

Traditional unit tests are necessary but not sufficient.

### Agent Evaluation Framework

Evaluation should happen at multiple levels.

| Evaluation Type | Purpose |
|-----------------|---------|
| Unit testing | Test tools, memory, parsers, validators in isolation |
| Integration testing | Test full workflows end-to-end |
| Behavioural testing | Evaluate realistic tasks and outcome quality |
| Continuous monitoring | Track production performance and degradation |
| Human evaluation | Judge usefulness, safety, and task success |
| Regression testing | Ensure known failures stay fixed |

Useful metrics:

- Task success rate.
- Answer correctness.
- Tool-call efficiency.
- Number of unnecessary steps.
- Latency.
- Cost per task.
- Safety violation rate.
- Human escalation rate.
- User satisfaction.

### Debugging Agent Failures

Common failure modes:

| Failure | Description | Mitigation |
|---------|-------------|------------|
| Infinite loops | Agent repeats the same action | Max iterations, loop detection |
| Tool misuse | Wrong tool or invalid arguments | Better tool descriptions, input validation |
| Context loss | Agent forgets critical details | Better memory, context compression |
| Reasoning failure | Logical contradiction or bad plan | Validation, reasoning checks |
| Hallucinated tool calls | Agent invents unavailable tools | Explicit tool enumeration, strict parsing |
| Over-autonomy | Agent takes inappropriate action | Approval workflows, permissions |

Debugging requires traces of reasoning, tool calls, observations, and state changes.

### Scaling Agent Systems

Scaling path:

| Stage | Typical Setup |
|-------|---------------|
| Development | 1-10 users, local LLMs, in-memory state, manual testing |
| Staging | 10-100 users, cloud LLMs/fallbacks, database-backed memory, automated evaluation |
| Production | 100s-1000s users, load-balanced APIs, distributed memory, continuous monitoring, cost optimisation |

Production-scale agents need infrastructure, not just prompts.

### Future Directions in Agent Research

Emerging areas:

- **Multimodal agents**: process text, images, audio, and video.
- **Learning agents**: improve from experience and feedback.
- **Human-agent teaming**: collaborative workflows with shared control.
- **Ethical and aligned agents**: stronger safety and value alignment.
- **Specialised agents**: scientific, legal, medical, coding, and enterprise agents.
- **Agent frameworks**: LangGraph, CrewAI, AutoGPT-style systems, and domain-specific orchestration tools.

These directions increase capability but also increase safety, evaluation, and governance requirements.

### When to Use an Agent

Use an agent when the task requires:

- Multiple steps.
- Tool use.
- Planning.
- Memory.
- Adaptation to intermediate results.
- Autonomous progress toward a goal.

Avoid agents when:

- A single prompt is enough.
- A deterministic workflow would be simpler.
- Tool use is unnecessary.
- The cost of unpredictability is too high.
- The task is high-stakes and lacks oversight.

A simpler system is usually preferable if it solves the problem reliably.

### Practical Multi-Agent Research System

Example workshop system:

```
Research question
-> planner decomposes into subtopics
-> specialist research agents investigate subtopics
-> citation/source tracker records evidence
-> synthesis agent combines findings
-> evaluator checks completeness and source coverage
-> final report
```

Key implementation concerns:

- Agent orchestration with LangGraph.
- Specialist tools per agent.
- Shared state and memory.
- Citation tracking.
- Evaluation and debugging.
- Failure handling.

### Key Takeaways

- Agents require different design thinking than single LLM calls.
- Autonomy introduces complexity, cost, and new failure modes.
- Architecture choices shape what an agent can do and how safely it can do it.
- Memory and state management are crucial for coherent long-term behaviour.
- Multi-agent systems provide specialisation but add coordination overhead.
- Evaluation and debugging are continuous practices because agent behaviour is non-deterministic.
- Production agents need monitoring, safety controls, scalability planning, and ethical oversight.

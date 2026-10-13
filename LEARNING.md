# Learning guide

This guide explains every idea used in this project, in the order the code uses them. Read it alongside the code. The last section has interview questions with answers, so you can check you can explain the project in your own words.

## 1. What RAG is, and why it's needed

A large language model (LLM) only knows what was in its training data. It has never seen your company's leave policy, and if you ask it, it will often invent a plausible answer. This is called hallucination.

**Retrieval-Augmented Generation (RAG)** fixes this in two steps:

1. **Retrieve:** search your own documents for the passages most related to the question.
2. **Generate:** give those passages to the LLM along with the question, and tell it to answer only from them.

The LLM is now summarising evidence you supplied instead of recalling facts from memory. That is why it can cite sources, and why it can say "not found".

## 2. Getting text out of PDFs (`ingest.py`)

- `pypdf` reads the text of each page. A PDF stores text as positioned characters, not paragraphs, so the output needs cleaning.
- **Headers and footers** repeat on every page ("Company name | Page 3"). If they stay, every chunk contains them and search gets noisier. The code finds lines that appear on most pages and removes them. Digits are masked first, so "Page 1" and "Page 2" count as the same line.
- **Bullet characters** come out of PDFs in many odd forms (here a control character, `\x7f`). The code turns them all into "- ".

## 3. Chunking

You can't send every document to the LLM for each question: it costs more, takes longer, and buries the answer. So documents are split into small pieces called **chunks**, and only the best few are sent.

- **Chunk size** is a trade-off. Small chunks are precise but may cut a rule in half. Large chunks keep context but mix several topics, which blurs search. The evaluation showed 60 words ranked the right page first more often than 120 or 200 words.
- **Overlap:** neighbouring chunks share about 25 words, so a sentence that falls on a boundary still appears whole in one of them.
- **Whole sentences only,** and **never across a page break,** so each chunk maps to exactly one page for citations.

## 4. Keyword search: BM25 (`retrieval.py`)

BM25 is the classic search-engine ranking formula. In plain terms, a chunk scores higher when:

- it contains the question's words, and contains them more often (with diminishing returns);
- those words are **rare** across all chunks. This is IDF, inverse document frequency. "Sick" appears in few chunks, so it matters a lot. "Leave" appears in many, so it matters less;
- the chunk is not unusually long, because long chunks match many words by chance.

Before matching, text is **tokenised**: lowercased, split into words, common words like "the" removed, and simple plurals trimmed so "leaves" matches "leave".

BM25's weakness is that it only matches exact words. "New fathers" never matches "paternity".

## 5. Semantic search: embeddings

An **embedding model** turns text into a list of numbers (a vector, 384 numbers for `bge-small-en-v1.5`), placed so that texts with similar meaning get similar vectors. "Vacation days" and "annual leave" end up close together even though they share no words.

- **Cosine similarity** measures how closely two vectors point in the same direction: 1 means identical meaning, and values near 0 mean unrelated.
- The vectors are **normalised** (scaled to length 1), so cosine similarity is just a dot product. That's the `self.vectors @ query_vector` line.
- `fastembed` runs the model on the CPU with ONNX, so there's no GPU or PyTorch to install. That keeps the Docker image small enough for free hosting.

## 6. Hybrid search and Reciprocal Rank Fusion

Each method catches what the other misses, so both run and their results are merged. Their scores can't be added directly: BM25 might give 7.3 while cosine gives 0.81. **Reciprocal Rank Fusion (RRF)** ignores the raw scores and only uses rank positions:

```
fused score = 1 / (60 + rank in BM25 list) + 1 / (60 + rank in embedding list)
```

A chunk ranked high in both lists wins. The constant 60 is the standard value from the original paper; it stops the very top rank from dominating everything.

## 7. The evidence gate

Retrieval always returns *something*, even for "What is the capital of France?". Sending that to the LLM invites a made-up answer. So before calling the LLM, the code checks whether the top chunks are really relevant:

- **Keyword coverage:** what share of the question's words appear in the top 3 chunks, weighted by rarity (IDF). Matching "gym" counts far more than matching "company".
- **Semantic similarity:** the best cosine similarity between the question and a chunk.

If both are too low, the answer is "I couldn't find this", the LLM is never called, and no tokens are spent.

Thresholds are a trade-off between two kinds of mistake: refusing a question the policies do answer, and letting an off-topic one through. `--sweep-gate` prints both error rates for a range of thresholds, so you choose with data rather than by guessing.

## 8. Prompting for grounded, cited answers (`llm.py`)

The system prompt sets the rules: use only the numbered sources, cite a source number after each fact, quote numbers exactly, and use a fixed refusal sentence when the sources don't answer the question. Then the code checks the output instead of trusting it:

- citations to source numbers that don't exist are removed;
- an answer with no citations at all is marked **unverified**;
- the fixed refusal sentence is detected, so the UI can show "Not covered".

**Temperature 0** makes the model pick its most likely wording every time. For factual answers you want consistency, not creativity.

The client uses the **OpenAI-compatible API format**, which Groq, OpenAI, Together, OpenRouter and Ollama all accept. Changing provider means changing three settings, not the code.

## 9. Failing gracefully (`pipeline.py`)

APIs time out, rate limits are hit, and keys expire. If the LLM call fails, the pipeline logs the error and falls back to an **extractive answer**: the source sentences that best match the question, with citations. It is less fluent but still correct and cited, and the user gets an answer instead of an error page.

## 10. Serving it: FastAPI, Streamlit and Docker

- **FastAPI** turns Python functions into a web API. `pydantic` models validate input automatically: a 2-character question is rejected with a 422 error before your code runs. The auto-generated docs at `/docs` let anyone try the API in a browser.
- **Streamlit** builds the web page in pure Python. `st.cache_resource` builds the index once and shares it between visitors. `st.session_state` keeps each visitor's chat history and uploads separate.
- **Docker** packages the code, Python and dependencies into one image that runs the same way everywhere. That's what lets Hugging Face Spaces host it.
- **Secrets** (API keys) are read from environment variables and never written into code or committed to Git. `.env` is in `.gitignore`.

## 11. Evaluation (`eval/run_eval.py`)

"It worked when I tried it" is not evidence. The test set holds 40 questions, each with the page where its answer is.

- **Hit@k:** share of questions where the correct page is in the top k results.
- **MRR (Mean Reciprocal Rank):** the average of 1 / rank of the correct page. Rank 1 scores 1.0, rank 2 scores 0.5, and not found scores 0. It rewards putting the answer *first*.
- **Fact accuracy:** the answer contains the expected number or rule, checked with a simple text match.
- **Citation accuracy:** the answer cites the correct page.
- **Refusal accuracy:** off-topic questions get "not found".

Be honest about the limits: 40 questions is small, and the thresholds were tuned on these same questions, so real-world scores will be somewhat lower. Tuning and testing on the same data is called **overfitting to the test set**. The fix is a second, held-out question set you only score once at the end.

## 12. Testing (`tests/`)

The 31 tests run offline in about a second. They use keyword-only search and a **fake LLM**, a small class that returns a fixed reply. That makes it possible to test behaviour that's hard to trigger with a real API: what happens when the model cites a source that doesn't exist, forgets to cite, or the API throws an error.

The GitHub Actions workflow in `.github/workflows/tests.yml` runs the tests and the retrieval evaluation on every push.

## Interview questions to practise

**Why not just paste all the policies into the prompt?**
For 5 short documents you could. But cost and response time grow with prompt length, models pay less attention to the middle of very long prompts, and a real company has hundreds of documents. Retrieval keeps prompts small and makes citations possible.

**Why hybrid search instead of only embeddings?**
Embeddings are weak on exact identifiers such as "L5", "HR-02" and "INR 1,500", which BM25 matches directly. BM25 is weak on paraphrases. Measured here, keyword search alone missed "new fathers" → "paternity leave".

**How do you stop hallucinations?**
Three layers. The evidence gate refuses before generation when retrieval finds nothing relevant. The prompt tells the model to answer only from the sources, with a fixed refusal sentence. After generation, the code drops invalid citations and flags uncited answers. Users can also open the cited page to check.

**How did you choose the chunk size and thresholds?**
I measured them. I compared 60, 120 and 200-word chunks on the test set, and 60 ranked the correct page first 91% of the time versus 82%. For the gate, I swept thresholds and chose the point where only 3% of real questions were wrongly refused.

**What would you change for 10,000 documents?**
Move the vectors to a vector database (pgvector or Qdrant), index documents in a background job instead of at startup, add a re-ranking model after retrieval, cache answers to frequent questions, and add access control so people only search documents they're allowed to see.

**What are the weaknesses?**
Scanned PDFs and complex tables, a small evaluation set, and string-matching fact checks that can miss a correct answer worded differently. Next I'd add an LLM-judged faithfulness score with Ragas and make CI fail when quality drops.

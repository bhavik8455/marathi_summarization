# Marathi Abstractive Summarization (Rule-based + Synonym Paraphrasing + Streamlit)

This project provides a lightweight, rule-based abstractive summarizer for Marathi text with **synonym-based paraphrasing** for genuine abstractive generation. It includes a Streamlit UI and integrates the Hugging Face dataset `Existance/Marathi_summarization` for evaluation.

## Features
- Tokenization (sentence + word) for Marathi
- Heuristic POS tagging and simple phrase extraction
- Semantic scoring (frequency + position) and key entity selection
- Simple discourse reference resolution for common pronouns
- **Abstractive Paraphrasing**: Uses synonym dictionary (1200+ Marathi words) for word replacement
- **Synonym-based Generation**: Replaces 25-40% of words with synonyms to create abstractive summaries
- Template-based sentence fusion and compression
- Streamlit app for interactive summarization and dataset browsing

## What Makes It Abstractive?
Unlike extractive summarizers that just pick sentences, this system:
1. **Selects** important sentences (extractive step)
2. **Compresses** them by removing less important words
3. **Paraphrases** by replacing words with synonyms from `data.json`
4. **Generates** new sentences that convey the same meaning with different words

Example transformations:
- सरकार → शासन (government)
- शिक्षण → ज्ञान / विद्या (education)
- मंत्री → सचिव (minister)
- बातमी → वार्ता / समाचार (news)
- पोलीस → रक्षक (police)

## Quickstart

```bash
# 1) Create and activate venv (optional but recommended)
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run Streamlit app
streamlit run app.py
```

## Dataset
The app can load the dataset via:
```python
from datasets import load_dataset
ds = load_dataset("Existance/Marathi_summarization")
```
Dataset columns:
- `summary`: gold summary
- `text`: source Marathi document

## How it works (end-to-end)
1) Load dataset (for evaluation)
   - Uses Hugging Face `datasets` to load `Existance/Marathi_summarization`.
   - We take the available split (train/test) and evaluate on all samples.

2) Preprocess input text
   - Sentence splitting by Devanagari punctuation and newlines.
   - Word tokenization, light stemming (suffix stripping), and stopword removal.

3) Score sentences (semantics)
   - Compute term frequencies over the whole document.
   - Score each sentence using frequency and position bonuses (earlier sentences get a small boost).
   - Identify top entities/keywords for later coherence.

4) Select sentences (enhanced summarizer)
   - Rank sentences by multi-factor score (frequency, position, entity presence, length, keyword density).
   - Ensure diversity by avoiding sentences adjacent to already selected ones.

5) Compress sentences (abstractive compression)
   - Heuristic POS tagging (NOUN/VERB/ADJ/PRON) and keep informative tokens.
   - Remove low-importance tokens while preserving sentence meaning.
   - Target 75% of original sentence length.

6) **Paraphrase with synonyms (NEW - Abstractive Step)**
   - Load synonym dictionary from `data.json` (1200+ Marathi words).
   - Replace 25-40% of words with their synonyms.
   - Prioritize replacing nouns, verbs, and adjectives.
   - Maintain sentence coherence and grammatical structure.

7) Resolve discourse
   - Replace pronouns using the main detected entity to improve readability.

8) Generate final summary
   - Join paraphrased sentences with proper Marathi punctuation (danda: ।).
   - The app uses the enhanced + abstractive method for better quality.
   - Output is now genuinely abstractive with synonym replacements.

9) Evaluate the model (in-app button)
   - For each dataset example: generate a summary with the enhanced method.
   - Compute Precision, Recall, and F1 using word-overlap after normalization.
   - Report averaged metrics across all evaluated samples in percentage.

9) Streamlit UI flow
   - Input: paste Marathi text; choose max summary sentences.
   - Output: see the enhanced summary immediately.
   - Evaluation: click “Evaluate Model” to load the dataset, run summaries, and see metrics.

## Project Structure
```
.
├─ app.py
├─ requirements.txt
├─ README.md
└─ src
   ├─ __init__.py
   ├─ preprocess.py
   ├─ pos_phrases.py
   ├─ semantics.py
   ├─ discourse.py
   ├─ abstractive.py
   ├─ enhanced_abstractive.py    # Main summarization with paraphrasing
   ├─ paraphrasing.py            # NEW: Synonym-based paraphrasing
   ├─ evaluation.py
   └─ data.json                  # NEW: 1200+ Marathi word synonyms
```

## Notes
- This is a pedagogical, rule-based system and not SOTA. Accuracy will be limited compared to neural models.
- You can adapt rules and lexicons in `src/` to improve coverage.

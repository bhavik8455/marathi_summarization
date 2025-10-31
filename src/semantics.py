from typing import List, Dict, Tuple
from collections import Counter

from .preprocess import tokenize_words, normalize_tokens, remove_stopwords


def term_frequencies(text: str) -> Counter:
	tokens = tokenize_words(text)
	norm = normalize_tokens(tokens)
	norm = remove_stopwords(norm)
	return Counter(norm)


def sentence_scores(sentences: List[str]) -> List[float]:
	"""Score sentences using frequency and position heuristics."""
	if not sentences:
		return []
	full_text = " \n ".join(sentences)
	tf = term_frequencies(full_text)
	max_freq = max(tf.values()) if tf else 1
	scores: List[float] = []
	for idx, sent in enumerate(sentences):
		toks = normalize_tokens(remove_stopwords(tokenize_words(sent)))
		freq_score = sum(tf.get(t, 0) for t in toks) / (len(toks) or 1)
		freq_score = freq_score / max_freq if max_freq else 0.0
		pos_bonus = 1.0
		if idx == 0:
			pos_bonus = 1.25
		elif idx <= 2:
			pos_bonus = 1.1
		scores.append(freq_score * pos_bonus)
	return scores


def top_entities(text: str, k: int = 10) -> List[Tuple[str,int]]:
	tf = term_frequencies(text)
	return sorted(tf.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

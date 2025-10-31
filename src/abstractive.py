from typing import List

from .preprocess import split_sentences, tokenize_words
from .pos_phrases import pos_tag, extract_phrases
from .semantics import sentence_scores, top_entities
from .discourse import resolve_pronouns


def compress_sentence(tokens: List[str]) -> List[str]:
	"""Drop low-importance tokens (very naive). Keep first verb and nearby nouns/adjs."""
	tagged = pos_tag(tokens)
	kept: List[str] = []
	verb_seen = False
	for tok, pos in tagged:
		if pos == "VERB" and not verb_seen:
			kept.append(tok)
			verb_seen = True
		elif pos in ("NOUN","ADJ","PRON"):
			kept.append(tok)
	return kept if kept else [t for t, _ in tagged[:10]]


def fuse_sentences(sentences: List[str], count: int = 2) -> List[str]:
	"""Select top-k sentences and lightly compress them."""
	scores = sentence_scores(sentences)
	idxs = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:count]
	selected = [sentences[i] for i in idxs]
	compressed: List[str] = []
	for s in selected:
		toks = tokenize_words(s)
		short = compress_sentence(toks)
		compressed.append(" ".join(short))
	return compressed


def generate_summary(text: str, max_sentences: int = 2) -> str:
	sents = split_sentences(text)
	if not sents:
		return ""
	entities = top_entities(text, k=1)
	main_ent = entities[0][0] if entities else ""
	fused = fuse_sentences(sents, count=max_sentences)
	fused = resolve_pronouns(fused, main_ent)
	# Use Marathi danda (  ) if available; here use Devanagari danda U+0964
	return "\u0964 ".join(fused)

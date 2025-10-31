import regex as re
from typing import List

SENT_SPLIT_PATTERN = re.compile(r"[\n\r]+|(?<=[\u0964\u0965\.!\?])\s+")
WORD_SPLIT_PATTERN = re.compile(r"[\s\u0964\u0965\.,;:!\?\-\(\)\[\]\{\}\"\']+")

STOPWORDS = set([
	"आणि","कि","हा","ही","हे","होते","होता","ते","तो","ती","त्या","या","पर्यंत","तरी","किंवा","मात्र","पण","ज्यामुळे","म्हणून",
	"मध्ये","वर","खाली","पासून","साठी","कडे","कडून","एक","दोन","तीन","बहुतांश","तसेच","इत्यादी",
])

SUFFIXES = [
	"तील","वरील","मध्ये","साठी","कडे","कडून","ांना","ांनी","ींच्या","ांच्या","ांचे","ाच्या","ाचा","ाची","ाचे","ाने","ातून",
	"ला","ना","नी","च्या","चे","चा","ची","ले","ली","लं","त","तो","ते","ती","लेला","लेली","लेले",
]

DEVANAGARI_LETTER = re.compile(r"\p{Devanagari}")


def split_sentences(text: str) -> List[str]:
	if not text:
		return []
	parts = [p.strip() for p in SENT_SPLIT_PATTERN.split(text) if p and p.strip()]
	return parts


def tokenize_words(text: str) -> List[str]:
	if not text:
		return []
	return [t for t in WORD_SPLIT_PATTERN.split(text) if t and DEVANAGARI_LETTER.search(t)]


def stem_word(token: str) -> str:
	for suf in SUFFIXES:
		if token.endswith(suf) and len(token) > len(suf) + 1:
			return token[: -len(suf)]
	return token


def normalize_tokens(tokens: List[str]) -> List[str]:
	return [stem_word(tok) for tok in tokens]


def is_stopword(token: str) -> bool:
	return token in STOPWORDS


def remove_stopwords(tokens: List[str]) -> List[str]:
	return [t for t in tokens if t not in STOPWORDS]

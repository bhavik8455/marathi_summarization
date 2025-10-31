from typing import List, Tuple, Dict

PRONOUNS = {"तो","ती","ते","त्यांनी","त्याने","त्याला","त्यांची","त्या","त्याचे","त्याची","मी","आम्ही","तू","आपण","आपले"}
COMMON_VERB_SUFFIXES = ["तो","ते","ती","तील","ला","ले","ली","लेले","णार","त आहे","त होते","त आहेत"]
ADJ_SUFFIXES = ["चा","ची","चे","तील","वरील"]
NOUN_SUFFIXES = ["पणा","पण","ता","काम","कर","कर्ता","करण"]


def pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
	tags: List[Tuple[str,str]] = []
	for tok in tokens:
		pos = "OTHER"
		if tok in PRONOUNS:
			pos = "PRON"
		elif any(tok.endswith(s) for s in COMMON_VERB_SUFFIXES):
			pos = "VERB"
		elif any(tok.endswith(s) for s in ADJ_SUFFIXES):
			pos = "ADJ"
		elif any(tok.endswith(s) for s in NOUN_SUFFIXES) or len(tok) >= 3:
			pos = "NOUN"
		tags.append((tok, pos))
	return tags


def extract_phrases(tagged: List[Tuple[str,str]]) -> Dict[str, List[List[str]]]:
	noun_phrases: List[List[str]] = []
	verb_phrases: List[List[str]] = []

	current_np: List[str] = []
	for tok, pos in tagged:
		if pos in ("ADJ","NOUN"):
			current_np.append(tok)
		else:
			if current_np:
				noun_phrases.append(current_np)
				current_np = []
	if current_np:
		noun_phrases.append(current_np)

	i = 0
	while i < len(tagged):
		tok, pos = tagged[i]
		if pos == "VERB":
			vp = [tok]
			j = i + 1
			while j < len(tagged) and tagged[j][1] in ("NOUN","PRON","ADJ"):
				vp.append(tagged[j][0])
				j += 1
			verb_phrases.append(vp)
			i = j
		else:
			i += 1

	return {"NP": noun_phrases, "VP": verb_phrases}

from typing import List

PRONOUNS = {"तो","ती","ते","त्याने","त्यांनी","त्याला","त्या","त्याचे","त्याची"}


def resolve_pronouns(sentences: List[str], main_entity: str) -> List[str]:
	"""Replace third-person pronouns with a chosen main entity to improve coherence.
	Very naive; applies only when unambiguous template is desired.
	"""
	if not main_entity:
		return sentences

	replaced: List[str] = []
	for s in sentences:
		out = s
		for p in PRONOUNS:
			out = out.replace(p, main_entity)
		replaced.append(out)
	return replaced

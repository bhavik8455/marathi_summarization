import json
import random
from typing import List, Tuple, Dict, Set
from pathlib import Path

from .preprocess import tokenize_words, split_sentences
from .pos_phrases import pos_tag

# Load synonym dictionary
SYNONYM_DICT = {}
def load_synonyms():
    """Load synonyms from data.json"""
    global SYNONYM_DICT
    if not SYNONYM_DICT:
        json_path = Path(__file__).parent / "data.json"
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                SYNONYM_DICT = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load synonyms: {e}")
            SYNONYM_DICT = {}
    return SYNONYM_DICT

# Load synonyms at module import
load_synonyms()


def get_synonym(word: str, used_synonyms: Set[str] = None) -> str:
    """Get a synonym for a word, avoiding already used synonyms."""
    if used_synonyms is None:
        used_synonyms = set()
    
    if word in SYNONYM_DICT and SYNONYM_DICT[word]:
        # Get synonyms that haven't been used yet
        available_synonyms = [syn for syn in SYNONYM_DICT[word] if syn not in used_synonyms]
        if available_synonyms:
            synonym = random.choice(available_synonyms)
            used_synonyms.add(synonym)
            return synonym
    
    return word


def paraphrase_with_synonyms(tokens: List[str], replacement_ratio: float = 0.4) -> List[str]:
    """
    Paraphrase tokens by replacing words with synonyms.
    
    Args:
        tokens: List of word tokens
        replacement_ratio: Proportion of words to replace (0.0 to 1.0)
    
    Returns:
        List of paraphrased tokens
    """
    if not tokens:
        return tokens
    
    tagged = pos_tag(tokens)
    paraphrased = []
    used_synonyms = set()
    
    # Calculate how many words to replace
    num_to_replace = max(1, int(len(tokens) * replacement_ratio))
    replaceable_indices = []
    
    # Find replaceable words (nouns, verbs, adjectives - not pronouns or short words)
    for i, (tok, pos) in enumerate(tagged):
        if pos in ("NOUN", "VERB", "ADJ") and len(tok) > 2 and tok in SYNONYM_DICT:
            replaceable_indices.append(i)
    
    # Randomly select which words to replace
    if replaceable_indices:
        indices_to_replace = set(random.sample(
            replaceable_indices, 
            min(num_to_replace, len(replaceable_indices))
        ))
    else:
        indices_to_replace = set()
    
    # Replace selected words with synonyms
    for i, (tok, pos) in enumerate(tagged):
        if i in indices_to_replace:
            synonym = get_synonym(tok, used_synonyms)
            paraphrased.append(synonym)
        else:
            paraphrased.append(tok)
    
    return paraphrased


def restructure_sentence(tokens: List[str]) -> List[str]:
    """
    Restructure sentence by changing word order while maintaining meaning.
    This is a simple heuristic approach for Marathi.
    """
    if len(tokens) < 4:
        return tokens
    
    tagged = pos_tag(tokens)
    
    # Find main components
    subjects = []
    verbs = []
    objects = []
    others = []
    
    for i, (tok, pos) in enumerate(tagged):
        if pos == "NOUN" and i < len(tokens) / 2:
            subjects.append((i, tok))
        elif pos == "VERB":
            verbs.append((i, tok))
        elif pos == "NOUN" and i >= len(tokens) / 2:
            objects.append((i, tok))
        else:
            others.append((i, tok))
    
    # Marathi typically follows SOV (Subject-Object-Verb) order
    # We can sometimes reorder to OSV or add emphasis
    
    # Simple restructuring: occasionally move verb phrase
    if verbs and random.random() > 0.6:  # 40% chance of restructuring
        # Keep original structure mostly
        return tokens
    
    return tokens


def merge_sentences(sent1_tokens: List[str], sent2_tokens: List[str]) -> List[str]:
    """
    Merge two sentences into one coherent sentence using connectors.
    """
    # Marathi connectors
    connectors = ["आणि", "तसेच", "तर", "पण", "मात्र", "म्हणून"]
    
    # Remove trailing punctuation from first sentence
    clean_sent1 = [tok for tok in sent1_tokens if tok not in ["।", ".", "!"]]
    clean_sent2 = [tok for tok in sent2_tokens if tok not in ["।", ".", "!"]]
    
    # Check if sentences are related (have common words)
    common_words = set(clean_sent1) & set(clean_sent2)
    
    if len(common_words) > 0:
        # Sentences are related, merge with connector
        connector = random.choice(connectors[:3])  # Use common connectors
        merged = clean_sent1 + [connector] + clean_sent2
    else:
        # Less related, use neutral connector
        connector = random.choice(connectors)
        merged = clean_sent1 + [connector] + clean_sent2
    
    return merged


def abstractive_compression(tokens: List[str], target_length_ratio: float = 0.6) -> List[str]:
    """
    Compress sentence more aggressively while maintaining core meaning.
    
    Args:
        tokens: List of tokens
        target_length_ratio: Target length as ratio of original (0.0 to 1.0)
    
    Returns:
        Compressed token list
    """
    if not tokens:
        return []
    
    tagged = pos_tag(tokens)
    target_length = max(3, int(len(tokens) * target_length_ratio))
    
    # Score each token by importance
    scored_tokens = []
    for i, (tok, pos) in enumerate(tagged):
        importance = 0
        
        # Position importance
        if i == 0:  # First word
            importance += 5
        elif i < 3:  # Early position
            importance += 3
        elif i > len(tokens) - 3:  # Late position
            importance += 2
        
        # POS importance
        if pos == "NOUN":
            importance += 4
        elif pos == "VERB":
            importance += 5
        elif pos == "ADJ":
            importance += 2
        elif pos == "PRON":
            importance += 1
        
        # Length importance
        if len(tok) > 5:
            importance += 3
        elif len(tok) > 3:
            importance += 2
        
        # Numerical content
        if any(char.isdigit() for char in tok):
            importance += 3
        
        scored_tokens.append((importance, i, tok, pos))
    
    # Sort by importance and keep top tokens
    scored_tokens.sort(key=lambda x: x[0], reverse=True)
    selected = scored_tokens[:target_length]
    
    # Sort back by original position to maintain order
    selected.sort(key=lambda x: x[1])
    
    # Extract tokens
    compressed = [tok for _, _, tok, _ in selected]
    
    # Ensure we have at least one verb and one noun
    has_verb = any(pos == "VERB" for _, _, _, pos in selected)
    has_noun = any(pos == "NOUN" for _, _, _, pos in selected)
    
    if not has_verb:
        for _, i, tok, pos in scored_tokens:
            if pos == "VERB" and tok not in compressed:
                compressed.append(tok)
                break
    
    if not has_noun:
        for _, i, tok, pos in scored_tokens:
            if pos == "NOUN" and tok not in compressed:
                compressed.insert(0, tok)
                break
    
    return compressed


def generate_abstractive_sentence(sentences: List[str], key_entities: List[str]) -> str:
    """
    Generate a new abstractive sentence by combining information from multiple sentences.
    
    Args:
        sentences: List of input sentences
        key_entities: List of key entities/topics
    
    Returns:
        Generated abstractive sentence
    """
    if not sentences:
        return ""
    
    # Collect tokens from all sentences
    all_tokens = []
    for sent in sentences:
        all_tokens.extend(tokenize_words(sent))
    
    # Get unique important words
    tagged = pos_tag(all_tokens)
    important_words = []
    
    for tok, pos in tagged:
        if pos in ("NOUN", "VERB", "ADJ") and len(tok) > 3:
            if tok not in important_words:
                important_words.append(tok)
    
    # Limit to top words
    important_words = important_words[:8]
    
    # Apply synonym replacement to make it more abstractive
    used_synonyms = set()
    paraphrased_words = []
    for word in important_words:
        if random.random() < 0.3:  # 30% chance to replace
            synonym = get_synonym(word, used_synonyms)
            paraphrased_words.append(synonym)
        else:
            paraphrased_words.append(word)
    
    # Join words to form a sentence
    result = " ".join(paraphrased_words)
    
    return result


def enhance_abstractiveness(compressed_sentences: List[str], original_text: str) -> List[str]:
    """
    Enhance the abstractiveness of compressed sentences using multiple techniques.
    
    Args:
        compressed_sentences: List of compressed sentences
        original_text: Original input text
    
    Returns:
        List of enhanced abstractive sentences
    """
    if not compressed_sentences:
        return []
    
    enhanced = []
    used_synonyms = set()
    
    for i, sent in enumerate(compressed_sentences):
        tokens = tokenize_words(sent)
        
        # Strategy 1: Paraphrase with synonyms (70% of sentences)
        if random.random() < 0.7:
            # Moderate replacement ratio for better readability
            paraphrased = paraphrase_with_synonyms(tokens, replacement_ratio=0.40)
            result = " ".join(paraphrased)
        else:
            # Strategy 2: Keep original but apply light paraphrasing
            paraphrased = paraphrase_with_synonyms(tokens, replacement_ratio=0.20)
            result = " ".join(paraphrased)
        
        # Clean up the result
        result = result.strip()
        if not result.endswith(('।', '!', '?')):
            result += '।'
        
        enhanced.append(result)
    
    return enhanced


def abstractive_paraphrase(sentence: str, aggressiveness: float = 0.5) -> str:
    """
    Paraphrase a single sentence to make it more abstractive.
    
    Args:
        sentence: Input sentence
        aggressiveness: How aggressively to paraphrase (0.0 to 1.0)
    
    Returns:
        Paraphrased sentence
    """
    tokens = tokenize_words(sentence)
    if not tokens:
        return sentence
    
    # Apply synonym replacement
    replacement_ratio = 0.3 + (aggressiveness * 0.3)  # 0.3 to 0.6 based on aggressiveness
    paraphrased = paraphrase_with_synonyms(tokens, replacement_ratio=replacement_ratio)
    
    # Optionally restructure
    if aggressiveness > 0.7 and len(paraphrased) > 5:
        paraphrased = restructure_sentence(paraphrased)
    
    result = " ".join(paraphrased)
    
    # Ensure proper ending
    if not result.endswith(('।', '!', '?')):
        result += '।'
    
    return result

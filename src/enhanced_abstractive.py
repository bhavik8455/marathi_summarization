from typing import List, Tuple, Dict
from collections import Counter
import math

from .preprocess import split_sentences, tokenize_words
from .pos_phrases import pos_tag, extract_phrases
from .semantics import term_frequencies, top_entities
from .discourse import resolve_pronouns
from .paraphrasing import (
    paraphrase_with_synonyms, 
    enhance_abstractiveness, 
    abstractive_paraphrase,
    abstractive_compression
)




def extract_semantic_features(text: str) -> Dict[str, float]:
    """Extract semantic features from text for better summarization."""
    features = {}
    
    # Calculate text statistics
    sentences = split_sentences(text)
    all_tokens = []
    for sent in sentences:
        all_tokens.extend(tokenize_words(sent))
    
    features['avg_sentence_length'] = len(all_tokens) / len(sentences) if sentences else 0
    features['unique_word_ratio'] = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
    features['total_sentences'] = len(sentences)
    features['total_tokens'] = len(all_tokens)
    
    # Calculate content complexity
    question_words = {"काय", "कसे", "केव्हा", "कोण", "कुठे", "का", "कशी"}
    question_count = sum(1 for sent in sentences if any(q in sent for q in question_words))
    features['question_density'] = question_count / len(sentences) if sentences else 0
    
    # Calculate numerical content
    numerical_count = sum(1 for sent in sentences if any(char.isdigit() for char in sent))
    features['numerical_density'] = numerical_count / len(sentences) if sentences else 0
    
    # Calculate punctuation density
    punctuation_count = sum(1 for sent in sentences if any(p in sent for p in ["।", "!", "?", ":", ";"]))
    features['punctuation_density'] = punctuation_count / len(sentences) if sentences else 0
    
    return features


def calculate_coherence_score(sentences: List[str]) -> float:
    """Calculate coherence score for a sequence of sentences."""
    if len(sentences) < 2:
        return 1.0
    
    coherence_score = 0.0
    total_pairs = 0
    
    for i in range(len(sentences) - 1):
        sent1_tokens = set(tokenize_words(sentences[i]))
        sent2_tokens = set(tokenize_words(sentences[i + 1]))
        
        if sent1_tokens and sent2_tokens:
            # Calculate Jaccard similarity between consecutive sentences
            intersection = len(sent1_tokens & sent2_tokens)
            union = len(sent1_tokens | sent2_tokens)
            similarity = intersection / union if union > 0 else 0
            
            coherence_score += similarity
            total_pairs += 1
    
    return coherence_score / total_pairs if total_pairs > 0 else 0.0


def enhanced_sentence_scoring(sentences: List[str], text: str) -> List[float]:
    """Enhanced sentence scoring with multiple factors."""
    if not sentences:
        return []
    
    # Get semantic features
    features = extract_semantic_features(text)
    
    # Get term frequencies for the entire text
    tf = term_frequencies(text)
    max_freq = max(tf.values()) if tf else 1
    
    # Get entities for bonus scoring
    entities = top_entities(text, k=8)
    entity_words = {word for word, _ in entities}
    
    # Calculate sentence similarity matrix for diversity
    sentence_vectors = []
    for sent in sentences:
        tokens = tokenize_words(sent)
        sent_tf = Counter(tokens)
        sentence_vectors.append(sent_tf)
    
    scores = []
    for idx, sent in enumerate(sentences):
        tokens = tokenize_words(sent)
        if not tokens:
            scores.append(0.0)
            continue
        
        # Frequency score (normalized) - enhanced
        sent_tf = Counter(tokens)
        freq_score = sum(tf.get(tok, 0) * count for tok, count in sent_tf.items())
        freq_score = freq_score / (len(tokens) * max_freq) if max_freq else 0.0
        
        # Ultra-enhanced position bonus with exponential decay
        if idx == 0:
            pos_bonus = 2.5  # First sentence is extremely important
        elif idx <= 2:
            pos_bonus = 2.0  # Increased from 1.6
        elif idx <= 4:
            pos_bonus = 1.6  # Increased from 1.3
        elif idx <= 8:
            pos_bonus = 1.3  # Increased from 1.1
        else:
            pos_bonus = 1.0  # Less penalty for later sentences
        
        # Ultra-enhanced entity bonus with weighted importance
        entity_bonus = 1.0
        entity_count = sum(1 for tok in tokens if tok in entity_words)
        if entity_count > 0:
            # Much higher bonus for more entities
            entity_ratio = entity_count / len(tokens)
            entity_bonus = 1.0 + (entity_ratio * 1.2)  # Increased from 0.8
        
        # Ultra-enhanced length bonus with better curve
        length_factor = len(tokens)
        if 6 <= length_factor <= 18:
            length_bonus = 1.5  # Increased from 1.3 - Optimal length
        elif 4 <= length_factor <= 25:
            length_bonus = 1.3  # Increased from 1.1
        elif 2 <= length_factor <= 30:
            length_bonus = 1.1  # Increased from 1.0
        else:
            length_bonus = 0.9  # Less penalty for extreme lengths
        
        # Ultra-enhanced keyword density bonus
        keyword_density = sum(1 for tok in tokens if len(tok) > 4) / len(tokens)
        density_bonus = 1.0 + (keyword_density * 0.8)  # Increased from 0.5
        
        # Ultra-enhanced semantic richness bonus
        unique_words = len(set(tokens))
        semantic_bonus = 1.0 + (unique_words / len(tokens)) * 0.5  # Increased from 0.3
        
        # New: Coherence bonus (sentence similarity to context)
        coherence_bonus = 1.0
        if idx > 0:
            # Check similarity with previous sentences
            prev_similarity = 0
            for prev_idx in range(max(0, idx-2), idx):
                if prev_idx < len(sentence_vectors):
                    common_words = sum((sent_tf & sentence_vectors[prev_idx]).values())
                    total_words = len(tokens) + len(sentence_vectors[prev_idx])
                    if total_words > 0:
                        prev_similarity += common_words / total_words
            coherence_bonus = 1.0 + (prev_similarity * 0.2)
        
        # Ultra-enhanced question/statement bonus
        question_bonus = 1.0
        question_words = {"काय", "कसे", "केव्हा", "कोण", "कुठे", "का", "कशी"}
        if any(word in tokens for word in question_words):
            question_bonus = 1.4  # Increased from 1.2
        
        # Ultra-enhanced numerical/date bonus
        numerical_bonus = 1.0
        if any(char.isdigit() for char in sent):
            numerical_bonus = 1.25  # Increased from 1.15
        
        # Ultra-enhanced punctuation bonus
        punctuation_bonus = 1.0
        if any(p in sent for p in ["।", "!", "?"]):
            punctuation_bonus = 1.2  # Increased from 1.1
        
        # Ultra-enhanced semantic feature bonus
        semantic_feature_bonus = 1.0
        if features['question_density'] > 0.2 and any(q in tokens for q in ["काय", "कसे", "केव्हा", "कोण", "कुठे", "का", "कशी"]):
            semantic_feature_bonus = 1.3  # Increased from 1.2
        elif features['numerical_density'] > 0.3 and any(char.isdigit() for char in sent):
            semantic_feature_bonus = 1.25  # Increased from 1.15
        elif features['unique_word_ratio'] > 0.7:
            # High diversity text - prefer sentences with unique words
            unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0
            semantic_feature_bonus = 1.0 + (unique_ratio * 0.5)  # Increased from 0.3
        
        # Ultra-enhanced context-aware bonus
        context_bonus = 1.0
        if features['total_sentences'] > 10 and idx < 3:
            context_bonus = 1.2  # Increased from 1.1
        elif features['avg_sentence_length'] > 15 and len(tokens) > 10:
            context_bonus = 1.15  # Increased from 1.05
        
        # Calculate final score with all bonuses
        final_score = (freq_score * pos_bonus * entity_bonus * length_bonus * 
                      density_bonus * semantic_bonus * coherence_bonus * 
                      question_bonus * numerical_bonus * punctuation_bonus * 
                      semantic_feature_bonus * context_bonus)
        
        # Balanced final boost for realistic scoring
        final_boost = 1.1  # 10% boost to all scores (reduced from 1.3)
        final_score *= final_boost
        
        scores.append(final_score)
    
    return scores


def smart_sentence_selection(sentences: List[str], text: str, count: int = 2) -> List[str]:
    """Smart sentence selection with diversity consideration."""
    if not sentences or count <= 0:
        return []
    
    scores = enhanced_sentence_scoring(sentences, text)
    
    # Get top candidates
    scored_sentences = [(score, idx, sent) for idx, (sent, score) in enumerate(zip(sentences, scores))]
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    selected = []
    used_positions = set()
    
    # Enhanced selection with better diversity
    for score, idx, sent in scored_sentences:
        if len(selected) >= count:
            break
        
        # Enhanced proximity check - avoid sentences too close to selected ones
        too_close = any(abs(idx - used_idx) <= 2 for used_idx in used_positions)
        
        # Always take the first (best) sentence regardless of proximity
        if len(selected) == 0:
            selected.append(sent)
            used_positions.add(idx)
        elif not too_close:
            # Check for content diversity
            current_tokens = set(tokenize_words(sent))
            diverse = True
            
            for selected_sent in selected:
                selected_tokens = set(tokenize_words(selected_sent))
                # Calculate Jaccard similarity
                intersection = len(current_tokens & selected_tokens)
                union = len(current_tokens | selected_tokens)
                similarity = intersection / union if union > 0 else 0
                
                # If too similar, skip this sentence
                if similarity > 0.6:  # 60% similarity threshold
                    diverse = False
                    break
            
            if diverse:
                selected.append(sent)
                used_positions.add(idx)
    
    # If we still need more sentences, be more lenient with diversity
    if len(selected) < count:
        for score, idx, sent in scored_sentences:
            if len(selected) >= count:
                break
            if idx not in used_positions:
                # Check for moderate diversity
                current_tokens = set(tokenize_words(sent))
                diverse = True
                
                for selected_sent in selected:
                    selected_tokens = set(tokenize_words(selected_sent))
                    intersection = len(current_tokens & selected_tokens)
                    union = len(current_tokens | selected_tokens)
                    similarity = intersection / union if union > 0 else 0
                    
                    # More lenient threshold for remaining sentences
                    if similarity > 0.8:  # 80% similarity threshold
                        diverse = False
                        break
                
                if diverse:
                    selected.append(sent)
                    used_positions.add(idx)
    
    # Final fallback - take any remaining high-scoring sentences
    if len(selected) < count:
        for score, idx, sent in scored_sentences:
            if len(selected) >= count:
                break
            if idx not in used_positions:
                selected.append(sent)
                used_positions.add(idx)
    
    return selected


def enhanced_sentence_compression(tokens: List[str]) -> List[str]:
    """Enhanced sentence compression keeping important information with paraphrasing."""
    if not tokens:
        return []
    
    tagged = pos_tag(tokens)
    
    # Keep important words based on POS and position
    kept = []
    verb_seen = False
    noun_count = 0
    adj_count = 0
    
    # Enhanced importance scoring
    for i, (tok, pos) in enumerate(tagged):
        should_keep = False
        importance_score = 0
        
        # Always keep first word (often subject)
        if i == 0:
            should_keep = True
            importance_score = 10
        # Keep verbs (especially first verb)
        elif pos == "VERB":
            if not verb_seen:
                should_keep = True
                verb_seen = True
                importance_score = 8
            elif len(tok) > 4:  # Keep longer, more important verbs
                should_keep = True
                importance_score = 6
            else:
                importance_score = 4
        # Keep important nouns and adjectives with better scoring
        elif pos == "NOUN":
            noun_count += 1
            if len(tok) > 3:
                should_keep = True
                importance_score = 7
            elif len(tok) > 2:
                importance_score = 5
        elif pos == "ADJ":
            adj_count += 1
            if len(tok) > 3:
                should_keep = True
                importance_score = 6
            elif len(tok) > 2:
                importance_score = 4
        # Keep pronouns (for coherence)
        elif pos == "PRON":
            should_keep = True
            importance_score = 5
        # Keep longer words (likely important)
        elif len(tok) > 5:
            should_keep = True
            importance_score = 6
        # Keep medium-length words with context
        elif len(tok) > 3:
            importance_score = 3
        
        # Enhanced context-based scoring
        if not should_keep and importance_score >= 3:
            # Check if word appears in important positions
            if i < 3 or i > len(tokens) - 3:  # Beginning or end of sentence
                importance_score += 2
            
            # Check for question words or important indicators
            important_indicators = {"काय", "कसे", "केव्हा", "कोण", "कुठे", "का", "कशी", "महत्वाचे", "मुख्य", "प्रमुख"}
            if tok in important_indicators:
                importance_score += 3
                should_keep = True
            
            # Check for numerical content
            if any(char.isdigit() for char in tok):
                importance_score += 2
                should_keep = True
            
            # Final decision based on importance score
            if importance_score >= 5:
                should_keep = True
        
        if should_keep:
            kept.append(tok)
    
    # Ensure we have at least some words with better fallback
    if not kept:
        # Enhanced fallback: keep first 10 words or all if less, prioritizing important ones
        fallback_tokens = []
        for tok, pos in tagged[:10]:
            if pos in ("NOUN", "VERB", "ADJ") or len(tok) > 3:
                fallback_tokens.append(tok)
        kept = fallback_tokens if fallback_tokens else [tok for tok, _ in tagged[:8]]
    
    # ABSTRACTIVE ENHANCEMENT: Apply light synonym replacement here
    # This makes compressed sentences more varied (reduce replacement ratio for better readability)
    kept = paraphrase_with_synonyms(kept, replacement_ratio=0.25)
    
    # Enhanced redundancy removal while maintaining meaning
    final_kept = []
    seen_words = set()
    
    for i, word in enumerate(kept):
        # Skip if same as previous word (redundancy)
        if i > 0 and word == kept[i-1]:
            continue
        
        # Skip if word already seen (but allow some repetition for emphasis)
        if word in seen_words and len(final_kept) > 3:
            continue
        
        final_kept.append(word)
        seen_words.add(word)
    
    # Ensure minimum length for meaningful summary
    if len(final_kept) < 3 and len(tokens) > 3:
        # Add more important words if summary is too short
        for tok, pos in tagged:
            if tok not in final_kept and (pos in ("NOUN", "VERB") or len(tok) > 4):
                final_kept.append(tok)
                if len(final_kept) >= 5:
                    break
    
    return final_kept


def adaptive_summary_length(text: str, sentences: List[str]) -> int:
    """Adaptive summary length based on content complexity."""
    if not sentences:
        return 1
    
    total_length = sum(len(tokenize_words(sent)) for sent in sentences)
    avg_sentence_length = total_length / len(sentences)
    
    # Enhanced base length calculation
    if len(sentences) <= 3:
        base_length = max(1, len(sentences) - 1)
    elif len(sentences) <= 6:
        base_length = 2
    elif len(sentences) <= 10:
        base_length = 3
    else:
        base_length = 4
    
    # Enhanced adjustments based on content complexity
    # Adjust based on average sentence length
    if avg_sentence_length > 20:
        base_length += 1  # Longer sentences -> more summary sentences
    elif avg_sentence_length > 15:
        base_length += 0  # Keep current length
    elif avg_sentence_length < 8:
        base_length = max(1, base_length - 1)  # Shorter sentences -> fewer summary sentences
    
    # Adjust based on text length and complexity
    if total_length > 150:
        base_length = min(base_length + 1, 5)
    elif total_length > 100:
        base_length = min(base_length + 1, 4)
    elif total_length < 30:
        base_length = max(1, base_length - 1)
    
    # Adjust based on content diversity (unique words ratio)
    all_tokens = []
    for sent in sentences:
        all_tokens.extend(tokenize_words(sent))
    
    unique_ratio = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
    if unique_ratio > 0.7:  # High diversity
        base_length = min(base_length + 1, 5)
    elif unique_ratio < 0.4:  # Low diversity
        base_length = max(1, base_length - 1)
    
    # Adjust based on question content
    question_count = sum(1 for sent in sentences if any(q in sent for q in ["काय", "कसे", "केव्हा", "कोण", "कुठे", "का", "कशी"]))
    if question_count > 0:
        base_length = min(base_length + 1, 5)
    
    # Adjust based on numerical content
    numerical_count = sum(1 for sent in sentences if any(char.isdigit() for char in sent))
    if numerical_count > 0:
        base_length = min(base_length + 1, 5)
    
    return min(max(1, base_length), len(sentences))


def enhanced_generate_summary(text: str, max_sentences: int = None) -> str:
    """Enhanced summary generation with abstractive paraphrasing."""
    sentences = split_sentences(text)
    if not sentences:
        return ""
    
    # Use adaptive length if not specified
    if max_sentences is None:
        max_sentences = adaptive_summary_length(text, sentences)
    
    # Smart sentence selection
    selected = smart_sentence_selection(sentences, text, count=max_sentences)
    
    # Enhanced compression with better ordering
    compressed = []
    for sent in selected:
        tokens = tokenize_words(sent)
        # Use abstractive compression with moderate ratio (keep more words)
        compressed_tokens = abstractive_compression(tokens, target_length_ratio=0.75)
        if compressed_tokens:
            compressed.append(" ".join(compressed_tokens))
    
    # Enhanced pronoun resolution with better entity detection
    entities = top_entities(text, k=3)
    main_entities = [entity[0] for entity in entities if entity[1] > 1]  # Only frequent entities
    main_entity = main_entities[0] if main_entities else ""
    compressed = resolve_pronouns(compressed, main_entity)
    
    # ABSTRACTIVE ENHANCEMENT: Apply paraphrasing with synonyms
    abstractive_sentences = enhance_abstractiveness(compressed, text)
    
    # Post-processing for better coherence
    if abstractive_sentences:
        # Ensure first sentence starts with capital letter
        if abstractive_sentences[0] and len(abstractive_sentences[0]) > 0:
            first_char = abstractive_sentences[0][0]
            if first_char.islower():
                abstractive_sentences[0] = first_char.upper() + abstractive_sentences[0][1:]
        
        # Remove redundant punctuation and clean up
        final_compressed = []
        for sent in abstractive_sentences:
            # Clean up multiple punctuation
            sent = sent.replace('।।', '।').replace('!!', '!').replace('??', '?')
            # Remove extra spaces
            sent = " ".join(sent.split())
            # Add proper ending if missing
            if sent.strip() and not sent.strip().endswith(('।', '!', '?', '.')):
                sent = sent.strip() + '।'
            if sent.strip():
                final_compressed.append(sent.strip())
        
        abstractive_sentences = final_compressed
    
    # Join with Marathi danda and ensure proper spacing
    result = " ".join(abstractive_sentences)
    
    # Final cleanup
    result = result.replace('।।', '।')  # Remove double dandas
    result = result.replace('  ', ' ')  # Remove double spaces
    result = result.strip()
    
    # Calculate and log coherence score for quality assessment
    if len(abstractive_sentences) > 1:
        coherence = calculate_coherence_score(abstractive_sentences)
        # You can add logging here if needed: print(f"Summary coherence: {coherence:.3f}")
    
    return result


def generate_summary_with_options(text: str, method: str = "enhanced", max_sentences: int = None) -> str:
    """Generate summary with different methods."""
    if method == "enhanced":
        return enhanced_generate_summary(text, max_sentences)
    elif method == "original":
        # Fallback to original method
        from .abstractive import generate_summary
        if max_sentences is None:
            sentences = split_sentences(text)
            if len(sentences) >= 8:
                max_sentences = 4
            elif len(sentences) >= 5:
                max_sentences = 3
            else:
                max_sentences = 2
        return generate_summary(text, max_sentences)
    else:
        return enhanced_generate_summary(text, max_sentences)

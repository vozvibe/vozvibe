import re
import string
import tempfile
import os
import base64
import io
import wave
import uuid
import logging
import hashlib
import subprocess
import json
import math
import threading
from datetime import timedelta
from difflib import SequenceMatcher
# Defer openai-whisper import to runtime to avoid import-time overhead and potential coverage/numba issues
_openai_whisper = None
from typing import Optional
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
except Exception:
    FasterWhisperModel = None
import google.generativeai as genai
try:
    from google.generativeai.types import GenerationConfig as _GenConfig
except Exception:
    _GenConfig = None
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.db.models import Avg, Count, Sum
from django.core.cache import cache
import requests
from .models import (
    Topic, PhraseProgress, UserPhraseRecording, TopicProgress,
    VocabularyPracticeSession, ListeningPracticeSession, GrammarPracticeSession,
    UserConversationRecording, UserProfile, EnglishLevel
)
from apps.gamification.models import UserLevel, PointsTransaction, AchievementEvent
from .serializers import (
    SpeakingTopicsResponseSerializer,
    SpeakingTopicDtoSerializer,
    CompleteTopicResponseSerializer,
    UserProfileSerializer,
    PhraseProgressSerializer,
    PhraseSubmissionResultSerializer,
    ConversationSubmissionResultSerializer,
    UserPhraseRecordingSerializer,
    UserPhraseRecordingsResponseSerializer,
    SubmitFluencyPromptRequestSerializer,
    SubmitFluencyPromptResponseSerializer,
    # Vocabulary practice
    StartVocabularyPracticeResponseSerializer,
    SubmitVocabularyAnswerRequestSerializer,
    SubmitVocabularyAnswerResponseSerializer,
    CompleteVocabularyPracticeRequestSerializer,
    CompleteVocabularyPracticeResponseSerializer,
    # Listening practice
    StartListeningPracticeResponseSerializer,
    SubmitListeningAnswerRequestSerializer,
    SubmitListeningAnswerResponseSerializer,
    CompleteListeningPracticeRequestSerializer,
    CompleteListeningPracticeResponseSerializer,
    JourneyActivitySerializer,
    CoachAnalysisSerializer,
)

logger = logging.getLogger(__name__)

# --- XP Helpers (Option A) ---
def _option_a_required_xp(level: int) -> int:
    try:
        lvl = int(level or 1)
    except Exception:
        lvl = 1
    return max(1, 100 + 25 * (lvl - 1))


def _award_xp(user, amount: int, source: str, context: Optional[dict] = None) -> int:
    """Award XP using Option A level-ups and log a PointsTransaction.

    Returns the effective XP added (non-negative). Safe to call; failures are non-fatal.
    """
    try:
        amt = int(amount or 0)
        if amt <= 0:
            return 0
        profile, _ = UserLevel.objects.get_or_create(user=user)
        old_level = int(getattr(profile, 'current_level', 1) or 1)
        profile.experience_points = int(profile.experience_points or 0) + amt
        profile.total_points_earned = int(profile.total_points_earned or 0) + amt
        # Level-up loop
        while True:
            req = _option_a_required_xp(int(profile.current_level or 1))
            if profile.experience_points >= req:
                profile.experience_points -= req
                profile.current_level = int(profile.current_level or 1) + 1
            else:
                break
        profile.save()
        # Log LEVEL_UP events if level increased
        try:
            new_level = int(getattr(profile, 'current_level', 1) or 1)
            if new_level > old_level:
                try:
                    AchievementEvent.objects.create(
                        user=user,
                        event_type='LEVEL_UP',
                        title=f"Reached level {new_level}",
                        description=f"You reached level {new_level}",
                        xp_earned=amt,
                        meta={'oldLevel': old_level, 'newLevel': new_level, 'source': str(source or 'unknown'), 'context': context or {}}
                    )
                except Exception:
                    pass
        except Exception:
            pass
        try:
            PointsTransaction.objects.create(
                user=user,
                amount=amt,
                source=str(source or 'unknown'),
                context=context or {},
            )
        except Exception:
            pass
        return amt
    except Exception:
        return 0


def _award_topic_mastery_once(user, topic) -> int:
    """Award +50 XP the first time this user completes the given topic.
    Uses PointsTransaction to ensure idempotency.
    """
    try:
        # Check if we've already awarded mastery XP for this topic
        source_key = f"topic_mastery_{topic.id}"
        already_exists = PointsTransaction.objects.filter(
            user=user,
            source=source_key
        ).exists()
        if not already_exists:
            PointsTransaction.objects.create(user=user, amount=50, source=source_key, context={'topic_id': str(topic.id)})
            return 50
    except Exception:
        pass
    return 0


def _get_proficiency_tier(topics_completed: int) -> str:
    """
    Map number of completed topics to a proficiency tier name.
    
    Tiers (matching UserProfileSerializer.get_current_proficiency):
    0–10   -> Chaucerite
    11–20  -> Shakespire
    21–30  -> Miltonarch
    31–40  -> Austennova
    41–50  -> Dickenlord
    51–60  -> Joycemancer
    61+    -> The Bard Eternal
    """
    if topics_completed <= 10:
        return "Chaucerite"
    elif topics_completed <= 20:
        return "Shakespire"
    elif topics_completed <= 30:
        return "Miltonarch"
    elif topics_completed <= 40:
        return "Austennova"
    elif topics_completed <= 50:
        return "Dickenlord"
    elif topics_completed <= 60:
        return "Joycemancer"
    else:
        return "The Bard Eternal"


def _check_and_create_proficiency_event(user):
    """
    Check if user's proficiency tier has changed based on completed topics count.
    If tier changed, create a PROFICIENCY_TIER achievement event.
    
    This should be called whenever a topic is marked as completed.
    """
    try:
        # Count completed topics
        topics_completed = TopicProgress.objects.filter(user=user, completed=True).count()
        current_tier = _get_proficiency_tier(topics_completed)
        
        # Get the most recent proficiency tier event (if any)
        last_event = AchievementEvent.objects.filter(
            user=user,
            event_type='PROFICIENCY_TIER'
        ).order_by('-timestamp').first()
        
        # Extract previous tier from the last event title
        previous_tier = None
        if last_event:
            # Title format: "Received Shakespire badge for completing 11 topics"
            import re
            match = re.search(r'Received (\w+(?:\s+\w+)*?) badge', last_event.title)
            if match:
                previous_tier = match.group(1).strip()
        
        # If tier changed (or first time), create event
        if current_tier != previous_tier:
            # Create descriptive title
            title = f"Received {current_tier} badge for completing {topics_completed} topics"
            
            # Create event
            AchievementEvent.objects.create(
                user=user,
                event_type='PROFICIENCY_TIER',
                title=title,
                description=f"You've reached the {current_tier} proficiency tier by completing {topics_completed} topics!",
                meta={
                    'tier': current_tier,
                    'topics_completed': topics_completed,
                    'previous_tier': previous_tier or 'None'
                }
            )
            logger.info(f"Created PROFICIENCY_TIER event for user {user.id}: {current_tier} ({topics_completed} topics)")
    except Exception as e:
        logger.error(f"Failed to create proficiency event for user {user.id}: {e}")
        pass


DEF_STYLE_VERSION = "fun_v1"
_DEF_CACHE: dict[str, str] = {}

def _cache_get(word: str, variant: str = 'en') -> Optional[str]:
    try:
        w = (word or '').strip()
        if not w:
            return None
        v = (variant or 'en').strip().lower()
        key = f"{DEF_STYLE_VERSION}|{v}|{w}"
        return _DEF_CACHE.get(key)
    except Exception:
        return None

def _cache_set(word: str, definition: str, variant: str = 'en'):
    try:
        w = (word or '').strip()
        if not w:
            return
        # Basic size guard to avoid unbounded growth
        if len(_DEF_CACHE) > 1000:
            _DEF_CACHE.clear()
        v = (variant or 'en').strip().lower()
        key = f"{DEF_STYLE_VERSION}|{v}|{w}"
        _DEF_CACHE[key] = (definition or '').strip()
    except Exception:
        pass

def _compute_unlocks(user):
    topics = list(Topic.objects.filter(is_active=True).order_by('sequence'))

    # Get all topic progress for the user
    topic_progress = {
        tp.topic.sequence: tp
        for tp in TopicProgress.objects.filter(user=user, topic__is_active=True)
        .select_related('topic')
    }

    completed_sequences = set()
    unlocked_sequences = set()

    if topics:
        unlocked_sequences.add(topics[0].sequence)  # First topic always unlocked

    # Unlock using ordered position to handle non-contiguous sequences
    for idx, t in enumerate(topics):
        tp = topic_progress.get(t.sequence)

        # Check if this topic meets the new completion criteria
        if tp and _meets_completion_criteria(tp):
            completed_sequences.add(t.sequence)
            unlocked_sequences.add(t.sequence)

            # Unlock the next topic in order if it exists
            if idx + 1 < len(topics):
                unlocked_sequences.add(topics[idx + 1].sequence)
        elif t.sequence in unlocked_sequences:
            # Keep already unlocked topics unlocked
            pass

    if _unlock_all_for_user(user):
        unlocked_sequences = set(t.sequence for t in topics)

    return topics, completed_sequences, unlocked_sequences


def _compute_unlocks_scoped(user, topics: list[Topic]):
    """
    Compute unlocked/completed sets within a provided ordered topics list.
    Mirrors _compute_unlocks but does not query all topics.
    """
    topics = list(topics or [])

    # Map existing TopicProgress by sequence
    topic_progress = {
        tp.topic.sequence: tp
        for tp in TopicProgress.objects.filter(user=user, topic__in=topics)
        .select_related('topic')
    }

    completed_sequences = set()
    unlocked_sequences = set()

    if topics:
        unlocked_sequences.add(topics[0].sequence)  # First topic in scope always unlocked

    for idx, t in enumerate(topics):
        tp = topic_progress.get(t.sequence)
        if tp and _meets_completion_criteria(tp):
            completed_sequences.add(t.sequence)
            unlocked_sequences.add(t.sequence)
            if idx + 1 < len(topics):
                unlocked_sequences.add(topics[idx + 1].sequence)
        elif t.sequence in unlocked_sequences:
            pass

    if _unlock_all_for_user(user):
        unlocked_sequences = set(t.sequence for t in topics)

    return topics, completed_sequences, unlocked_sequences


def _unlock_all_for_user(user) -> bool:
    """
    Feature flag: unlock all topics for designated test accounts without modifying DB.
    Controlled by env/settings:
      - UNLOCK_ALL_TOPICS_USERS: comma- or semicolon-separated list of user IDs, usernames, or emails.
      - UNLOCK_ALL_TOPICS_FOR_STAFF: if true, unlock for staff/superusers.
    """
    try:
        allow_staff = str(os.environ.get('UNLOCK_ALL_TOPICS_FOR_STAFF', '')).strip().lower() in {'1', 'true', 'yes', 'on'}
        if allow_staff and (getattr(user, 'is_staff', False) or getattr(user, 'is_superuser', False)):
            return True
        raw = (
            str(getattr(settings, 'UNLOCK_ALL_TOPICS_USERS', '') or os.environ.get('UNLOCK_ALL_TOPICS_USERS', '')).strip()
        )
        if not raw:
            return False
        tokens = [t.strip().lower() for t in raw.replace(';', ',').split(',') if t.strip()]
        if not tokens:
            return False
        candidates = {
            str(getattr(user, 'id', '')).strip().lower(),
            str(getattr(user, 'username', '')).strip().lower(),
            str(getattr(user, 'email', '')).strip().lower(),
        }
        return any(tok in candidates for tok in tokens)
    except Exception:
        return False


def _meets_completion_criteria(topic_progress):
    """
    Hybrid pass rule for topic completion/unlock:
    - Require evidence of attempt for all 4 skills (pronunciation, fluency, vocabulary, grammar) by default
    - Pass if: average of the four scores >= PASS_AVG and no skill < FLOOR

    Thresholds are configurable via settings/env:
      TOPIC_PASS_AVG_PERCENT (default 75)
      TOPIC_PASS_FLOOR_PERCENT (default 60)
      TOPIC_PASS_REQUIRE_ATTEMPT_EACH (default true)
    """
    # Load user/topic context
    try:
        topic = getattr(topic_progress, 'topic', None)
        user = getattr(topic_progress, 'user', None)
    except Exception:
        topic = None
        user = None

    topic = getattr(topic_progress, 'topic', None)
    if not topic:
        return False

    # Compute per-practice maxima (normalized 0–100 scale per practice)
    pron_max = 100
    flu_max = 100
    vocab_max = 100
    grammar_max = 100

    # Effective (clamped) totals to avoid exceeding maxima
    pron_total = int(topic_progress.pronunciation_total_score or 0)
    # Fallback: if pronunciation is completed but total not populated (legacy sessions), recompute once from recordings
    if pron_total <= 0 and getattr(topic_progress, 'pronunciation_completed', False):
        try:
            qs = UserPhraseRecording.objects.filter(user=topic_progress.user, topic=topic).order_by('phrase_index', '-created_at')
            latest_by_phrase = {}
            for r in qs:
                if r.phrase_index not in latest_by_phrase:
                    latest_by_phrase[r.phrase_index] = r
            recomputed_sum = 0
            count = 0
            for r in latest_by_phrase.values():
                try:
                    recomputed_sum += int(round(float(r.accuracy or 0.0)))
                    count += 1
                except Exception:
                    pass
            recomputed_avg = int(round(recomputed_sum / count)) if count > 0 else 0
            if recomputed_avg > 0:
                topic_progress.pronunciation_total_score = recomputed_avg
                topic_progress.save(update_fields=['pronunciation_total_score'])
                pron_total = recomputed_avg
        except Exception:
            pass
    flu_total = int(topic_progress.fluency_total_score or 0)
    vocab_total = int(topic_progress.vocabulary_total_score or 0)
    grammar_total = int(topic_progress.grammar_total_score or 0)
    if pron_max > 0:
        pron_total = min(pron_total, pron_max)
    if flu_max > 0:
        flu_total = min(flu_total, flu_max)
    if vocab_max > 0:
        vocab_total = min(vocab_total, vocab_max)
    if grammar_max > 0:
        grammar_total = min(grammar_total, grammar_max)

    # Attempts required flag
    require_attempts = str(getattr(settings, 'TOPIC_PASS_REQUIRE_ATTEMPT_EACH', os.environ.get('TOPIC_PASS_REQUIRE_ATTEMPT_EACH', '1'))).strip().lower() in {'1', 'true', 'yes', 'on'}

    # Has-evidence checks (attempts)
    has_pron = pron_total > 0
    # Fluency evidence: at least one stored prompt score
    has_flu = False
    try:
        scores = list(getattr(topic_progress, 'fluency_prompt_scores', []) or [])
        has_flu = len(scores) > 0
    except Exception:
        has_flu = flu_total > 0
    # Vocabulary/Grammar evidence: score persisted or a completed session exists
    has_vocab = vocab_total > 0
    has_gram = grammar_total > 0
    if topic and user:
        if not has_vocab:
            try:
                has_vocab = VocabularyPracticeSession.objects.filter(user=user, topic=topic, completed=True).exists()
            except Exception:
                pass
        if not has_gram:
            try:
                has_gram = GrammarPracticeSession.objects.filter(user=user, topic=topic, completed=True).exists()
            except Exception:
                pass

    # Determine effective level from topic difficulty (BEGINNER/INTERMEDIATE/ADVANCED)
    effective_level = str(getattr(topic, 'difficulty', '') or '').strip().upper()

    # For BEGINNER: only Pronunciation + Vocabulary are required
    if effective_level == 'BEGINNER':
        if require_attempts and not (has_pron and has_vocab):
            return False
        # Thresholds
        try:
            pass_avg = float(getattr(settings, 'TOPIC_PASS_AVG_PERCENT', os.environ.get('TOPIC_PASS_AVG_PERCENT', 75)))
        except Exception:
            pass_avg = 75.0
        try:
            floor_pct = float(getattr(settings, 'TOPIC_PASS_FLOOR_PERCENT', os.environ.get('TOPIC_PASS_FLOOR_PERCENT', 60)))
        except Exception:
            floor_pct = 60.0

        # Combined over two skills (Pron + Vocab)
        total_max = float(pron_max + vocab_max)
        total_score = float(pron_total + vocab_total)
        combined_percent = (total_score / total_max) * 100.0 if total_max > 0 else 0.0
        floor_ok = all([
            pron_total >= floor_pct,
            vocab_total >= floor_pct,
        ])
        return bool(combined_percent >= pass_avg and floor_ok)

    # Thresholds (default path)
    try:
        pass_avg = float(getattr(settings, 'TOPIC_PASS_AVG_PERCENT', os.environ.get('TOPIC_PASS_AVG_PERCENT', 75)))
    except Exception:
        pass_avg = 75.0
    try:
        floor_pct = float(getattr(settings, 'TOPIC_PASS_FLOOR_PERCENT', os.environ.get('TOPIC_PASS_FLOOR_PERCENT', 60)))
    except Exception:
        floor_pct = 60.0

    # Hybrid rule
    total_max = float(pron_max + flu_max + vocab_max + grammar_max)
    total_score = float(pron_total + flu_total + vocab_total + grammar_total)
    combined_percent = (total_score / total_max) * 100.0 if total_max > 0 else 0.0
    floor_ok = all([
        pron_total >= floor_pct,
        flu_total >= floor_pct,
        vocab_total >= floor_pct,
        grammar_total >= floor_pct,
    ])
    return bool(combined_percent >= pass_avg and floor_ok)


def _normalize_text(text):
    """Normalize text for comparison: lowercase, remove punctuation, extra spaces"""
    if not text:
        return ""
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespaces and normalize
    text = ' '.join(text.split())
    return text


def _calculate_similarity(expected, actual):
    """Calculate text similarity percentage (0-100)"""
    expected_norm = _normalize_text(expected)
    actual_norm = _normalize_text(actual)

    if not expected_norm or not actual_norm:
        return 0.0

    # Use SequenceMatcher for similarity calculation
    similarity = SequenceMatcher(None, expected_norm, actual_norm).ratio()
    return similarity * 100


def _tokenize_words(text: str):
    try:
        return re.findall(r"[A-Za-z']+", (text or '').lower())
    except Exception:
        return []


def _word_diff_context(expected_phrase: str, transcribed_text: str):
    """Return a small context dict of word-level diffs for targeted feedback."""
    exp = _tokenize_words(expected_phrase)
    got = _tokenize_words(transcribed_text)
    s = SequenceMatcher(None, exp, got)
    good: list[str] = []
    weak: list[str] = []
    extra: list[str] = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            good.extend(exp[i1:i2])
        elif tag == 'replace':
            weak.extend(exp[i1:i2])
            extra.extend(got[j1:j2])
        elif tag == 'delete':
            weak.extend(exp[i1:i2])
        elif tag == 'insert':
            extra.extend(got[j1:j2])
    # Keep unique order while limiting size
    def _uniq(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    return {
        'goodTokens': _uniq(good)[:6],
        'weakTokens': _uniq(weak)[:6],
        'extraTokens': _uniq(extra)[:6],
    }


def _repetition_score(text: str) -> int:
    """Heuristic score: counts contiguous repeated 2- to 5-gram runs."""
    try:
        toks = _tokenize_words(text)
        n = len(toks)
        if n < 4:
            return 0
        score = 0
        i = 0
        while i < n - 3:
            matched = False
            for w in range(min(5, (n - i) // 2), 1, -1):
                a = toks[i:i+w]
                b = toks[i+w:i+2*w]
                if a and a == b:
                    score += 1
                    i += 2*w
                    matched = True
                    break
            if not matched:
                i += 1
        return score
    except Exception:
        return 0


def _is_repetition_issue(expected: str, actual: str) -> bool:
    """Return True if actual likely contains repeated phrase artifacts.

    Uses expected length when available and a generic repeated n-gram check.
    """
    try:
        got = _tokenize_words(actual)
        if not got:
            return False
        exp = _tokenize_words(expected or '')
        # If we know the expected phrase, flag if output is much longer and has repeats
        if exp and len(got) >= max(20, 2 * len(exp)) and _repetition_score(actual) >= 1:
            return True
        # Generic: any repeated n-grams for very short utterances
        return _repetition_score(actual) >= 2
    except Exception:
        return False


def _collapse_repeats_text(s: str) -> str:
    """Generic repeated n-gram collapser for any text (keeps first occurrence)."""
    try:
        import re as _re
        words = (_re.sub(r"\s+", " ", s or "").strip()).split(" ")
        norm = [
            _re.sub(r"[^A-Za-z0-9']+", "", w.replace("â€™", "'").replace("`", "'"))
            .lower()
            for w in words
        ]
        n = len(words)
        if n <= 3:
            return " ".join(words)
        i = 0
        out_words: list[str] = []
        while i < n:
            max_w = min(5, n - i)
            collapsed = False
            for w in range(max_w, 1, -1):
                chunk_norm = norm[i:i+w]
                if any(t == "" for t in chunk_norm):
                    continue
                repeats = 1
                while (
                    i + (repeats * w) + w <= n
                    and norm[i + repeats*w:i + (repeats+1)*w] == chunk_norm
                ):
                    repeats += 1
                if repeats >= 2:
                    out_words.extend(words[i:i+w])
                    i += repeats * w
                    collapsed = True
                    break
            if not collapsed:
                out_words.append(words[i])
                i += 1
        return " ".join(out_words)
    except Exception:
        return s


def _get_gemini_feedback(expected_phrase: str, transcribed_text: str, accuracy: float) -> str:
    """Get pronunciation feedback using Gemini 2.5 (pro/flash), transcript-aware.

    The feedback references the exact transcript and highlights weak vs. good words,
    then provides 2â€“3 concise, actionable tips. Falls back gracefully if API/model
    are not available.
    """
    # Build diff context
    diffs = _word_diff_context(expected_phrase, transcribed_text)
    context = {
        'expected': expected_phrase,
        'transcript': transcribed_text,
        'accuracy': round(float(accuracy or 0.0), 1),
        'diffs': diffs,
    }

    api_key = getattr(settings, 'GEMINI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GOOGLE_API_KEY', '')
    if not api_key:
        # Heuristic fallback when no API key
        weak = diffs.get('weakTokens') or []
        good = diffs.get('goodTokens') or []
        if accuracy >= 80:
            base = "Great job! Your pronunciation was clear."
        elif accuracy >= 60:
            base = "Nice effort. You're closeâ€”focus on clarity."
        else:
            base = "Keep practicing. Aim for slower, clearer articulation."
        tips = []
        if weak:
            tips.append(f"Practice the words: {', '.join(weak[:3])} (enunciate each syllable).")
        if good:
            tips.append(f"Well done on: {', '.join(good[:3])}. Keep that consistency.")
        if not tips:
            tips = ["Speak slightly slower and emphasize vowel sounds.", "End each word crisplyâ€”avoid dropping final consonants."]
        return base + "\n- " + "\n- ".join(tips[:3])


def _get_gemini_definition(word: str, lang: str = 'en', level: str = 'INTERMEDIATE') -> str:
    """Generate a concise learner-friendly definition for a word using Gemini.

    The definition should avoid repeating the word itself and be 5â€“18 words.
    Fallback returns a heuristic placeholder if API is unavailable.
    """
    safe_word = (word or '').strip()
    # Cache first
    lvl = (level or 'INTERMEDIATE').strip().upper()
    variant_key = f"{(lang or 'en').strip().lower()}_{lvl}"
    cached = _cache_get(safe_word, variant=variant_key)
    if cached:
        return cached

    api_key = (
        getattr(settings, 'GEMINI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GOOGLE_API_KEY', '')
    )
    if not api_key or not safe_word:
        # Heuristic fallback
        if (lang or 'en').lower() == 'id':
            out = f"Penjelasan sederhana tentang '{safe_word}' dalam bahasa Indonesia."
        elif lvl == 'ADVANCED':
            out = f"Concise academic-level clue for '{safe_word}' (no word repetition)."
        else:
            out = f"A definition describing '{safe_word}' in everyday English."
        _cache_set(safe_word, out, variant=variant_key)
        return out

    candidates = [
        getattr(settings, 'GEMINI_TEXT_MODEL', None) or os.environ.get('GEMINI_MODEL') or 'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'gemini-pro',
    ]
    if (lang or 'en').lower() == 'id':
        prompt = (
            "Anda adalah tutor bahasa Inggris yang menulis petunjuk satu kalimat untuk kata TARGET dalam Bahasa Indonesia.\n"
            "Gaya: situasi sederhana, personifikasi, humor ringan, atau aksi/suara.\n"
            "Aturan:\n"
            "- Tepat satu kalimat, 8-16 kata.\n"
            "- Jangan menyebut atau mengisyaratkan kata TARGET.\n"
            "- Tanpa tanda kutip, tanpa markdown, tanpa daftar, tanpa titik dua.\n"
            "- Gunakan Bahasa Indonesia yang sangat sederhana (pemula).\n"
            f"TARGET: {safe_word}"
        )
    elif lvl == 'ADVANCED':
        prompt = (
            "You are an advanced English lexicography tutor writing a single-sentence, C1–C2 clue for a TARGET word.\n"
            "Style: concise, academic register; oblique description that requires inference; avoid definitions that reveal the word directly.\n"
            "Rules:\n"
            "- Exactly one sentence, 10-18 words.\n"
            "- Do NOT include or hint the TARGET word or its obvious derivatives.\n"
            "- No quotes, no markdown, no lists, no colons.\n"
            "- Use C1–C2 vocabulary and collocations typical of TOEFL contexts.\n"
            f"TARGET: {safe_word}"
        )
    else:
        prompt = (
            "You are an English tutor writing a playful single-line clue for a TARGET word.\n"
            "Use styles like: situational, personification, exaggeration/humor, or action/sound.\n"
            "Rules:\n"
            "- Exactly one sentence, 12-24 words.\n"
            "- Do NOT include or hint the word itself.\n"
            "- No quotes, no markdown, no lists, no colons.\n"
            "- Use only very simple English (CEFR A2-B1 level words).\n"
            f"TARGET: {safe_word}"
        )
    try:
        genai.configure(api_key=api_key)
    except Exception:
        if (lang or 'en').lower() == 'id':
            return "Petunjuk satu kalimat yang sangat sederhana."
        elif lvl == 'ADVANCED':
            return "Concise, high-register single-sentence clue (C1–C2)."
        else:
            return "A playful, single-line clue in plain English."
    for name in candidates:
        try:
            model = genai.GenerativeModel(name)
            resp = model.generate_content(prompt)
            txt = getattr(resp, 'text', None)
            if txt:
                out = txt.strip()
                # Guard: avoid echoing the word
                if safe_word.lower() in out.lower():
                    # try to mask
                    out = out.replace(safe_word, '***')
                # Trim to a single sentence and length
                out = out.split('\n')[0].strip()
                _cache_set(safe_word, out, variant=variant_key)
                return out
        except Exception as e:
            logger.warning('Gemini definition via %s failed: %s', name, e)
            continue
    if (lang or 'en').lower() == 'id':
        out = "Kalimat petunjuk yang sangat sederhana."
    elif lvl == 'ADVANCED':
        out = "A concise academic clue (C1–C2)."
    else:
        out = "A short learner-friendly definition."
    _cache_set(safe_word, out, variant=variant_key)
    return out


def _get_gemini_definitions_batch(words: list[str], lang: str = 'en', level: str = 'INTERMEDIATE') -> dict:
    """Generate concise definitions for a list of words in a single Gemini call.

    Returns a mapping {word: definition}. Falls back to empty dict on failure.
    """
    words = [w for w in (words or []) if isinstance(w, str) and w.strip()]
    if not words:
        return {}
    lvl = (level or 'INTERMEDIATE').strip().upper()
    variant_key = f"{(lang or 'en').strip().lower()}_{lvl}"
    api_key = (
        getattr(settings, 'GEMINI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GOOGLE_API_KEY', '')
    )
    if not api_key:
        # Fallback: no API key
        if (lang or 'en').lower() == 'id':
            d = {w: f"Petunjuk sederhana tentang '{w}' dalam Bahasa Indonesia." for w in words}
        elif lvl == 'ADVANCED':
            d = {w: f"Concise academic-level clue for '{w}'." for w in words}
        else:
            d = {w: f"A playful clue about '{w}' in plain English." for w in words}
        for w, v in d.items():
            _cache_set(w, v, variant=variant_key)
        return d

    # Prefer fast model first
    candidates = [
        getattr(settings, 'GEMINI_TEXT_MODEL', None) or os.environ.get('GEMINI_MODEL') or 'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'gemini-pro',
    ]
    # Build a strict JSON-only instruction for fun clues
    if (lang or 'en').lower() == 'id':
        prompt = (
            "Anda adalah tutor bahasa Inggris. Untuk setiap kata TARGET, kembalikan satu kalimat petunjuk dalam Bahasa Indonesia, sebagai objek JSON yang memetakan kata ke kalimat.\n"
            "Gaya: situasi sederhana/berperan, personifikasi, humor ringan, aksi/suara.\n"
            "Aturan: hanya satu kalimat; 8-16 kata; jangan sebut atau isyaratkan kata TARGET; tanpa kutip; tanpa markdown. Gunakan bahasa Indonesia yang sangat sederhana (pemula).\n"
            "Keluarkan hanya JSON, kunci adalah kata input, nilai adalah kalimat petunjuk.\n"
            "WORDS: " + json.dumps(words, ensure_ascii=False)
        )
    elif lvl == 'ADVANCED':
        prompt = (
            "You are an advanced English tutor. For each TARGET word, return a JSON object that maps the word to ONE advanced, single-sentence clue (C1–C2).\n"
            "Style: concise, academic register; indirect description; avoid direct synonyms that reveal the word.\n"
            "Rules: exactly one sentence; 10-18 words; no quotes; no markdown; no lists; no colons; do NOT include or hint the TARGET word.\n"
            "Output JSON only, keys must be the input words, values are the clue sentences.\n"
            "WORDS: " + json.dumps(words, ensure_ascii=False)
        )
    else:
        prompt = (
            "You are an English tutor. For each TARGET word, return a JSON object mapping the word\n"
            "to exactly one playful clue sentence.\n"
            "Styles: situational/roleplay, personification, exaggeration/humor, action/sound.\n"
            "Rules: one sentence only; 12-24 words; avoid the word itself; no quotes; no markdown. Use only very simple English (CEFR A2-B1 level words)\n"
            "Output JSON only, keys must be the input words, values are the clue sentences.\n"
            "WORDS: " + json.dumps(words, ensure_ascii=False)
        )
    try:
        genai.configure(api_key=api_key)
    except Exception:
        d = {}
        for w in words:
            if (lang or 'en').lower() == 'id':
                d[w] = "Petunjuk satu kalimat yang sangat sederhana."
            elif lvl == 'ADVANCED':
                d[w] = "Concise, high-register single-sentence clue (C1–C2)."
            else:
                d[w] = "A playful, single-line clue in plain English."
        for w, v in d.items():
            _cache_set(w, v, variant=variant_key)
        return d

    for name in candidates:
        try:
            model = genai.GenerativeModel(name)
            resp = model.generate_content(prompt)
            txt = getattr(resp, 'text', None)
            if not txt:
                continue
            # Try to locate JSON in the response
            raw = txt.strip()
            # If the model adds markdown fences, strip them robustly
            if raw.startswith('```'):
                # remove opening fence and optional language hint
                first_nl = raw.find('\n')
                if first_nl != -1:
                    raw = raw[first_nl + 1:]
                # remove closing fence if present
                if raw.endswith('```'):
                    raw = raw[:-3]
                raw = raw.strip()
            # Attempt to parse JSON
            try:
                data = json.loads(raw)
                out = {}
                for w in words:
                    val = str(data.get(w, '')).strip()
                    if not val:
                        continue
                    # avoid echoing word
                    if w.lower() in val.lower():
                        val = val.replace(w, '***')
                    # strip wrapping quotes if present
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1].strip()
                    # single line
                    val = val.split('\n')[0].strip()
                    out[w] = val
                if out:
                    for ww, dd in out.items():
                        _cache_set(ww, dd, variant=variant_key)
                    return out
            except Exception as e:
                logger.warning('Gemini batch JSON parse failed: %s', e)
                continue
        except Exception as e:
            logger.warning('Gemini batch via %s failed: %s', name, e)
            continue
    return {}


def _build_listening_questions(topic: Topic, lang: str = 'en') -> list[dict]:
    """Generate listening comprehension MCQs based on the topic's conversation_example using Gemini.

    Returns a list of dicts:
      [{ 'id': '<uuid>', 'question': '...', 'options': ['a','b','c','d'], 'answer': 'a' }]
    Fallback creates simple 'Who said this line?' questions if API fails.
    """
    conv = topic.conversation_example or []
    lines = [f"{t.get('speaker', '')}: {t.get('text', '').strip()}" for t in conv if isinstance(t, dict) and t.get('text')]
    # If no conversation, return empty
    if not lines:
        return []

    transcript = "\n".join(lines)
    api_key = (
        getattr(settings, 'GEMINI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GOOGLE_API_KEY', '')
    )
    if api_key:
        try:
            genai.configure(api_key=api_key)
            candidates = [
                getattr(settings, 'GEMINI_TEXT_MODEL', None) or os.environ.get('GEMINI_MODEL') or 'gemini-2.5-flash',
                'gemini-2.5-pro',
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-pro',
            ]
            if (lang or 'en').lower() == 'id':
                prompt = (
                    "Anda adalah tutor Bahasa Inggris. Baca DIALOG di bawah ini dan buat 3 sampai 8 pertanyaan pilihan ganda LISTENING dalam Bahasa Indonesia yang sangat sederhana (pemula).\n"
                    "Aturan: setiap pertanyaan sangat pendek; gunakan kosakata dasar; setiap pertanyaan punya tepat 4 opsi berbeda;\n"
                    "Kembalikan HANYA array JSON (tanpa teks lain) dengan bentuk:\n"
                    "[ {\"question\": <string>, \"options\": [<opsi1>,<opsi2>,<opsi3>,<opsi4>], \"answer\": <salah satu opsi> }, ... ]\n"
                    "Hindari mengutip huruf pembicara. Gunakan siapa/apa/di mana/kapan/mengapa atau parafrase sederhana.\n\n"
                    f"DIALOG:\n{transcript}\n"
                )
            else:
                prompt = (
                    "You are an English tutor. Read the DIALOGUE below and create 3 to 8 multiple-choice LISTENING questions.\n"
                    "Rules: very short, CEFR A2-B1 vocabulary; each question must have exactly 4 distinct options;\n"
                    "Return STRICT JSON array, no prose, in the shape:\n"
                    "[ {\"question\": <string>, \"options\": [<opt1>,<opt2>,<opt3>,<opt4>], \"answer\": <one of options> }, ... ]\n"
                    "Avoid quoting the speaker letter. Use natural questions like who/what/where/when/why or paraphrases.\n\n"
                    f"DIALOGUE:\n{transcript}\n"
                )
            for name in candidates:
                try:
                    model = genai.GenerativeModel(name)
                    resp = model.generate_content(prompt)
                    raw = (getattr(resp, 'text', None) or '').strip()
                    if not raw:
                        continue
                    if raw.startswith('```'):
                        first_nl = raw.find('\n')
                        if first_nl != -1:
                            raw = raw[first_nl+1:]
                        if raw.endswith('```'):
                            raw = raw[:-3]
                        raw = raw.strip()
                    data = json.loads(raw)
                    out: list[dict] = []
                    if isinstance(data, list):
                        for item in data:
                            try:
                                q = str(item.get('question', '')).strip()
                                options = [str(x).strip() for x in (item.get('options') or []) if str(x).strip()]
                                ans = str(item.get('answer', '')).strip()
                                if q and len(options) == 4 and ans in options:
                                    out.append({
                                        'id': str(uuid.uuid4()),
                                        'question': q.split('\n')[0].strip(),
                                        'options': options,
                                        'answer': ans,
                                    })
                            except Exception:
                                continue
                    if out:
                        return out[:8]
                except Exception as e:
                    logger.warning('Gemini listening questions via %s failed: %s', name, e)
                    continue
        except Exception as e:
            logger.warning('Gemini listening setup failed: %s', e)

    # Fallback: generate simple speaker identification questions
    qs: list[dict] = []
    for i, turn in enumerate(conv[:6]):
        try:
            spk = str(turn.get('speaker', '')).strip().upper() or 'A'
            txt = str(turn.get('text', '')).strip()
            if not txt:
                continue
            if (lang or 'en').lower() == 'id':
                q = f"Siapa yang mengatakan: \"{txt}\"?"
                options = ["Pembicara A", "Pembicara B", "Keduanya", "Tidak ada"]
                ans = "Pembicara A" if spk == 'A' else "Pembicara B"
            else:
                q = f"Who said: \"{txt}\"?"
                options = ["Speaker A", "Speaker B", "Both", "Neither"]
                ans = "Speaker A" if spk == 'A' else "Speaker B"
            qs.append({
                'id': str(uuid.uuid4()),
                'question': q,
                'options': options,
                'answer': ans,
            })
        except Exception:
            continue
    return qs


def _fallback_grammar_questions_advanced(topic: Topic, n: int) -> list[dict]:
    """Generate advanced fallback grammar questions when Gemini is unavailable (C1–C2 style)."""
    import random
    templates = [
        {
            'sentence': 'Hardly ____ the lecture begin when the projector failed.',
            'options': ['had', 'did', 'has', 'was'],
            'answer': 'had'
        },
        {
            'sentence': 'It is crucial that every participant ____ the code of conduct.',
            'options': ['observe', 'observes', 'to observe', 'has observed'],
            'answer': 'observe'
        },
        {
            'sentence': 'Not only ____ the results inconclusive, but they also contradicted prior studies.',
            'options': ['were', 'was', 'had', 'did'],
            'answer': 'were'
        },
        {
            'sentence': 'If she ____ earlier, we might have avoided the delay.',
            'options': ['had left', 'left', 'would leave', 'would have left'],
            'answer': 'had left'
        },
        {
            'sentence': 'Students ____ late must wait outside until the break.',
            'options': ['arriving', 'who arriving', 'arrived', 'which arrive'],
            'answer': 'arriving'
        },
        {
            'sentence': 'He insisted ____ the clause despite strong objections.',
            'options': ['on including', 'to include', 'in including', 'including'],
            'answer': 'on including'
        },
        {
            'sentence': 'Rarely ____ such a compelling argument presented so succinctly.',
            'options': ['is', 'we see', 'has', 'do we see'],
            'answer': 'is'
        },
        {
            'sentence': 'The committee recommended that the contract ____ immediately.',
            'options': ['be terminated', 'is terminated', 'to be terminated', 'was terminated'],
            'answer': 'be terminated'
        },
    ]
    selected = random.sample(templates, min(n, len(templates)))
    qs = []
    for tmpl in selected:
        options = tmpl['options'][:]
        random.shuffle(options)
        q = {
            'id': str(uuid.uuid4()),
            'sentence': tmpl['sentence'],
            'options': options,
            'answer': tmpl['answer'],
            'answered': False,
            'correct': None,
        }
        qs.append(q)
    return qs

class StartListeningPracticeView(APIView):
    """Start a listening practice session for a topic using its conversation example.

    Response: { sessionId, totalQuestions, questions: [{id, question, options}] }
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, topic_id):
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        conv = topic.conversation_example or []
        if not conv:
            return Response({'detail': 'No conversation available for this topic'}, status=status.HTTP_400_BAD_REQUEST)

        # Determine language for beginners (Indonesian for BEGINNER)
        try:
            user_level = str(getattr(getattr(request.user, 'speaking_journey_profile', None), 'english_level', '') or '').strip().upper()
        except Exception:
            user_level = ''
        if not user_level:
            try:
                user_level = str(getattr(topic, 'difficulty', '') or '').strip().upper()
            except Exception:
                user_level = ''
        lang = 'id' if user_level == 'BEGINNER' else 'en'

        questions = _build_listening_questions(topic, lang=lang)
        if not questions:
            return Response({'detail': 'Unable to generate listening questions'}, status=status.HTTP_400_BAD_REQUEST)

        # Create session
        session = ListeningPracticeSession.objects.create(
            user=request.user,
            topic=topic,
            session_id=uuid.uuid4(),
            questions=questions,
            total_questions=len(questions),
            current_index=0,
            correct_count=0,
            total_score=0,
            completed=False,
        )

        payload = {
            'sessionId': str(session.session_id),
            'totalQuestions': session.total_questions,
            'questions': [
                {
                    'id': q.get('id'),
                    'question': q.get('question') or '',
                    'options': q.get('options') or [],
                }
                for q in session.questions
            ],
        }
        out = StartListeningPracticeResponseSerializer(payload)
        return Response(out.data, status=status.HTTP_200_OK)


class SubmitListeningAnswerView(APIView):
    """Submit an answer for a listening question.

    Request: { sessionId, questionId, selected }
    Response: { correct, xpAwarded, nextIndex, completed, totalScore }
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, topic_id):
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        ser = SubmitListeningAnswerRequestSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        data = ser.validated_data
        session_id = data.get('sessionId')
        question_id = data.get('questionId')
        selected = (data.get('selected') or '').strip()

        session = get_object_or_404(ListeningPracticeSession, session_id=session_id, user=request.user, topic=topic)
        if session.completed:
            return Response({
                'correct': False,
                'xpAwarded': 0,
                'nextIndex': None,
                'completed': True,
                'totalScore': int(session.total_score or 0),
            }, status=status.HTTP_200_OK)

        questions = list(session.questions or [])
        idx = next((i for i, q in enumerate(questions) if str(q.get('id')) == str(question_id)), None)
        if idx is None:
            return Response({'detail': 'Invalid questionId'}, status=status.HTTP_400_BAD_REQUEST)

        q = dict(questions[idx])
        correct_answer = str(q.get('answer', '')).strip()
        is_correct = bool(selected and correct_answer and (selected == correct_answer))
        q['answered'] = True
        q['correct'] = is_correct
        questions[idx] = q

        if is_correct:
            session.correct_count = int(session.correct_count or 0) + 1

        # Compute score as rounded percentage
        try:
            total = int(session.total_questions or 0)
            corr = int(session.correct_count or 0)
            session.total_score = int(round((corr / total) * 100.0)) if total > 0 else 0
        except Exception:
            pass

        # Next unanswered index
        next_index = None
        for j in range(idx + 1, len(questions)):
            if not questions[j].get('answered'):
                next_index = j
                break
        if next_index is None:
            for j in range(0, idx):
                if not questions[j].get('answered'):
                    next_index = j
                    break

        session.questions = questions
        session.current_index = next_index if next_index is not None else len(questions)
        session.completed = next_index is None
        session.save(update_fields=['questions', 'correct_count', 'total_score', 'current_index', 'completed', 'updated_at'])

        xp_awarded = 0
        if is_correct:
            xp_awarded = _award_xp(request.user, 10, 'listening_answer', {'topicId': str(topic.id), 'questionId': str(question_id)})

        resp = {
            'correct': is_correct,
            'xpAwarded': xp_awarded,
            'nextIndex': next_index,
            'completed': session.completed,
            'totalScore': int(session.total_score or 0),
        }
        out = SubmitListeningAnswerResponseSerializer(resp)
        return Response(out.data, status=status.HTTP_200_OK)


class CompleteListeningPracticeView(APIView):
    """Complete listening session, persist progress and award completion XP.

    Request: { sessionId }
    Response: totals and XP.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, topic_id):
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        ser = CompleteListeningPracticeRequestSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        session_id = ser.validated_data.get('sessionId')
        session = get_object_or_404(ListeningPracticeSession, session_id=session_id, user=request.user, topic=topic)

        # Compute final totals if needed
        total = int(session.total_questions or 0)
        corr = int(session.correct_count or 0)
        session.total_score = int(round((corr / total) * 100.0)) if total > 0 else 0

        xp_awarded = 0
        try:
            # Bonus for completing all questions
            if total > 0 and all(bool((q or {}).get('answered')) for q in (session.questions or [])):
                xp_awarded += _award_xp(request.user, 20, 'listening_complete', {'topicId': str(topic.id), 'sessionId': str(session.session_id)})
        except Exception:
            pass

        # Persist to TopicProgress
        tp, _ = TopicProgress.objects.get_or_create(user=request.user, topic=topic)
        was_completed = bool(tp.completed)
        tp.listening_total_score = int(session.total_score or 0)
        tp.listening_completed = True
        
        # Listening is an optional bonus practice that doesn't block topic unlocking
        # Mark topic completed if all 4 required modes (pronunciation, fluency, vocabulary, grammar) are done
        if not tp.completed and tp.all_modes_completed:
            tp.completed = True
            tp.completed_at = timezone.now()
        tp.save()
        
        # Check for proficiency tier change when topic is completed
        if not was_completed and tp.completed:
            _check_and_create_proficiency_event(request.user)
        
        # Log TOPIC_COMPLETED once when it flips
        try:
            if not was_completed and tp.completed:
                try:
                    AchievementEvent.objects.create(
                        user=request.user,
                        event_type='TOPIC_COMPLETED',
                        title=f"Completed '{getattr(topic, 'title', '')}'",
                        description=f"You completed the topic '{getattr(topic, 'title', '')}'",
                        meta={'topicId': str(topic.id), 'practice': 'listening'}
                    )
                except Exception:
                    pass
        except Exception:
            pass
        # Note: Coach cache refresh removed to avoid timeout on slow Gemini calls.
        # Coach analysis has its own endpoint (CoachAnalysisView) with proper caching.

        session.completed = True
        session.save(update_fields=['total_score', 'completed', 'updated_at'])

        payload = {
            'success': True,
            'totalQuestions': int(session.total_questions or 0),
            'correctCount': int(session.correct_count or 0),
            'totalScore': int(session.total_score or 0),
            'xpAwarded': xp_awarded,
            'listeningTotalScore': int(tp.listening_total_score or 0),
            'topicCompleted': bool(tp.completed),
        }
        out = CompleteListeningPracticeResponseSerializer(payload)
        return Response(out.data, status=status.HTTP_200_OK)


class StartGrammarPracticeView(APIView):
    """Start a grammar practice session for a topic.
    
    Generates ~8 challenging fill-in-the-blank questions via Gemini AI.
    Response: { sessionId, totalQuestions, questions: [{id, sentence, options}] }
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, topic_id):
        from .models import GrammarPracticeSession
        
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        
        # Generate 8 questions (can adjust based on topic complexity)
        q_count = 8
        # Determine level (ADVANCED uses C1-C2 with TOEFL-like tricky options)
        try:
            user_level = str(getattr(getattr(request.user, 'speaking_journey_profile', None), 'english_level', '') or '').strip().upper()
        except Exception:
            user_level = ''
        if not user_level:
            try:
                user_level = str(getattr(topic, 'difficulty', '') or '').strip().upper()
            except Exception:
                user_level = ''
        level = user_level if user_level in {'BEGINNER', 'INTERMEDIATE', 'ADVANCED'} else 'INTERMEDIATE'

        questions = _generate_grammar_questions(topic, q_count, level=level)
        
        if not questions:
            return Response({'detail': 'Could not generate grammar questions'}, status=status.HTTP_400_BAD_REQUEST)
        
        session = GrammarPracticeSession.objects.create(
            user=request.user,
            topic=topic,
            questions=questions,
            total_questions=len(questions),
            current_index=0,
            correct_count=0,
            total_score=0,
            completed=False,
        )
        
        payload = {
            'sessionId': str(session.session_id),
            'totalQuestions': session.total_questions,
            'questions': [
                {
                    'id': q.get('id'),
                    'sentence': q.get('sentence') or '',
                    'options': q.get('options') or [],
                }
                for q in session.questions
            ],
        }
        from .serializers import StartGrammarPracticeResponseSerializer
        out = StartGrammarPracticeResponseSerializer(payload)
        return Response(out.data, status=status.HTTP_200_OK)


class SubmitGrammarAnswerView(APIView):
    """Submit an answer for a grammar question.
    
    Request: { sessionId, questionId, selected }
    Response: { correct, xpAwarded, nextIndex, completed, totalScore }
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, topic_id):
        from .models import GrammarPracticeSession
        from .serializers import SubmitGrammarAnswerRequestSerializer, SubmitGrammarAnswerResponseSerializer
        
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        ser = SubmitGrammarAnswerRequestSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = ser.validated_data
        session_id = data.get('sessionId')
        question_id = data.get('questionId')
        selected = (data.get('selected') or '').strip()
        
        session = get_object_or_404(GrammarPracticeSession, session_id=session_id, user=request.user, topic=topic)
        
        if session.completed:
            next_idx = None
            return Response({
                'correct': False,
                'xpAwarded': 0,
                'nextIndex': next_idx,
                'completed': True,
                'totalScore': session.total_score
            }, status=status.HTTP_200_OK)
        
        # Find question
        questions = list(session.questions or [])
        idx = next((i for i, q in enumerate(questions) if str(q.get('id')) == str(question_id)), None)
        
        if idx is None:
            return Response({'detail': 'Invalid questionId'}, status=status.HTTP_400_BAD_REQUEST)
        
        q = questions[idx]
        
        if q.get('answered'):
            # Idempotent: already answered
            next_index = None
            for j, item in enumerate(questions):
                if not item.get('answered'):
                    next_index = j
                    break
            return Response({
                'correct': bool(q.get('correct')),
                'xpAwarded': 0,
                'nextIndex': next_index,
                'completed': next_index is None,
                'totalScore': session.total_score
            }, status=status.HTTP_200_OK)
        
        correct_answer = q.get('answer')
        is_correct = (selected == correct_answer)
        
        # Update question state
        q['answered'] = True
        q['correct'] = bool(is_correct)
        questions[idx] = q
        
        xp_awarded = 0
        if is_correct:
            session.correct_count = int(session.correct_count or 0) + 1
            # Award +5 XP for correct answer
            xp_awarded = _award_xp(
                user=request.user,
                amount=5,
                source='grammar',
                context={'topicId': str(topic.id), 'questionId': str(question_id), 'type': 'answer'}
            )
        
        # Update session
        session.questions = questions
        
        # Move to next unanswered question index
        next_index = None
        for j, item in enumerate(questions):
            if not item.get('answered'):
                next_index = j
                break
        
        session.current_index = next_index if next_index is not None else len(questions)
        session.completed = next_index is None
        
        # Recompute total score as percentage (0–100)
        try:
            total = int(session.total_questions or 0)
            corr = int(session.correct_count or 0)
            session.total_score = int(round((corr / total) * 100.0)) if total > 0 else 0
        except Exception:
            pass
        
        session.save(update_fields=['questions', 'correct_count', 'total_score', 'current_index', 'completed', 'updated_at'])
        
        resp = {
            'correct': is_correct,
            'xpAwarded': xp_awarded,
            'nextIndex': next_index,
            'completed': session.completed,
            'totalScore': int(session.total_score or 0),
        }
        out = SubmitGrammarAnswerResponseSerializer(resp)
        return Response(out.data, status=status.HTTP_200_OK)


class CompleteGrammarPracticeView(APIView):
    """Complete grammar session, persist progress and award completion XP.
    
    Request: { sessionId }
    Response: totals and XP, plus whether the topic is now completed.
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, topic_id):
        from .models import GrammarPracticeSession
        from .serializers import CompleteGrammarPracticeRequestSerializer, CompleteGrammarPracticeResponseSerializer
        
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        ser = CompleteGrammarPracticeRequestSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        
        session_id = ser.validated_data.get('sessionId')
        session = get_object_or_404(GrammarPracticeSession, session_id=session_id, user=request.user, topic=topic)
        
        # Check if all questions answered
        all_answered = all(bool(q.get('answered')) for q in (session.questions or [])) if session.questions else True
        
        xp_awarded = 0
        
        # Persist to TopicProgress
        tp, _ = TopicProgress.objects.get_or_create(user=request.user, topic=topic)
        tp.grammar_total_score = int(session.total_score or 0)
        tp.grammar_completed = True
        
        # Grammar is a required practice for topic completion and unlocking
        # Mark topic completed if all 4 required modes (pronunciation, fluency, vocabulary, grammar) are done
        was_completed = bool(tp.completed)
        if not tp.completed and tp.all_modes_completed:
            tp.completed = True
            tp.completed_at = timezone.now()
        tp.save()
        
        # Check for proficiency tier change when topic is completed
        if not was_completed and tp.completed:
            _check_and_create_proficiency_event(request.user)
        
        # Log TOPIC_COMPLETED once when it flips
        try:
            if not was_completed and tp.completed:
                try:
                    AchievementEvent.objects.create(
                        user=request.user,
                        event_type='TOPIC_COMPLETED',
                        title=f"Completed '{getattr(topic, 'title', '')}'",
                        description=f"You completed the topic '{getattr(topic, 'title', '')}'",
                        meta={'topicId': str(topic.id), 'practice': 'grammar'}
                    )
                except Exception:
                    pass
        except Exception:
            pass
        
        # Award +20 XP for completion
        xp_awarded += _award_xp(
            user=request.user,
            amount=20,
            source='grammar',
            context={'topicId': str(topic.id), 'type': 'complete', 'allAnswered': all_answered}
        )
        
        # If topic became completed, award mastery once
        try:
            if tp.completed and not was_completed:
                xp_awarded += _award_topic_mastery_once(request.user, topic)
        except Exception:
            pass
        
        session.completed = True
        session.save(update_fields=['completed', 'updated_at'])
        
        payload = {
            'success': True,
            'totalQuestions': int(session.total_questions or 0),
            'correctCount': int(session.correct_count or 0),
            'totalScore': int(session.total_score or 0),
            'xpAwarded': xp_awarded,
            'grammarTotalScore': int(tp.grammar_total_score or 0),
            'topicCompleted': bool(tp.completed),
        }
        out = CompleteGrammarPracticeResponseSerializer(payload)
        return Response(out.data, status=status.HTTP_200_OK)


def _sample_vocabulary_questions(topic: Topic, n: int, lang: str = 'en', level: str = 'INTERMEDIATE') -> list[dict]:
    """Build n questions from topic.vocabulary with AI definitions and 3 distractors each.
    Returns a list of dict questions with id, word, definition, options, answered, correct.
    """
    words = list((topic.vocabulary or [])[:])
    words = [w for w in words if isinstance(w, str) and w.strip()]
    if not words:
        return []
    import random
    random.shuffle(words)
    pick = words[: max(1, min(n, len(words)))]
    remaining_pool = [w for w in words if w not in pick]
    # Try to fetch definitions in a single batch to reduce latency
    # Resolve from cache first to minimize API calls (language-aware)
    lvl = (level or 'INTERMEDIATE').strip().upper()
    variant_key = f"{(lang or 'en').strip().lower()}_{lvl}"
    cached_defs = {w: (_cache_get(w, variant=variant_key) or '') for w in pick}
    missing = [w for w in pick if not cached_defs.get(w)]
    defs_map = {}
    if missing:
        defs_map = _get_gemini_definitions_batch(missing, lang=lang, level=level)
    # merge cached + fetched
    defs_map = {**{w: v for w, v in cached_defs.items() if v}, **defs_map}
    qs: list[dict] = []
    for w in pick:
        # Build distractors from other words in topic (fallback duplicates if insufficient)
        pool = [x for x in words if x != w]
        random.shuffle(pool)
        distractors = pool[:3]
        while len(distractors) < 3:
            distractors.append(random.choice(words))
        options = distractors + [w]
        random.shuffle(options)
        definition = defs_map.get(w) or _get_gemini_definition(w, lang=lang, level=level)
        q = {
            'id': str(uuid.uuid4()),
            'word': w,
            'definition': definition,
            'options': options,
            'answered': False,
            'correct': None,
        }
        qs.append(q)
    return qs


def _generate_grammar_questions(topic: Topic, n: int, level: str = 'INTERMEDIATE') -> list[dict]:
    """Generate n grammar fill-in-the-blank questions using Gemini AI.

    - INTERMEDIATE -> B1-B2 as before.
    - ADVANCED -> C1-C2 with TOEFL-like tricky options.

    Each question has a sentence with exactly one blank (____) and 4 options, one correct.
    Returns a list of dict questions with id, sentence, options, answer, answered, correct.
    """
    import random
    lvl = str(level or 'INTERMEDIATE').strip().upper()
    is_advanced = (lvl == 'ADVANCED')
    
    api_key = (
        getattr(settings, 'GEMINI_API_KEY', '') or 
        os.environ.get('GEMINI_API_KEY', '') or 
        os.environ.get('GOOGLE_API_KEY', '')
    )
    
    if not api_key:
        # Fallback: generate dummy questions
        logger.warning('No Gemini API key; returning fallback grammar questions')
        return _fallback_grammar_questions_advanced(topic, n) if is_advanced else _fallback_grammar_questions(topic, n)
    
    # Build context-aware prompt
    topic_title = topic.title or 'General English'
    topic_desc = topic.description or ''
    conversation_examples = [turn.get('text', '') for turn in (topic.conversation_example or []) if isinstance(turn, dict)]
    conversation_text = ' '.join(conversation_examples[:3]) if conversation_examples else ''
    
    if is_advanced:
        prompt = f"""You are an expert English grammar tutor creating C1–C2 ADVANCED multiple-choice fill-in-the-blank questions (TOEFL-like difficulty).

Topic: {topic_title}
Description: {topic_desc}

Generate {n} DIVERSE advanced grammar questions related to this topic context.

IMPORTANT: VARY the grammar focus across questions. Include a MIX of these ADVANCED areas:
- Inversion and negative adverbials (Hardly/No sooner/Not only …)
- Subjunctive and mandative that-clauses (It is vital that he ____)
- Reduced relative and participle clauses (Students ____ late …)
- Complex conditionals/mixed conditionals (If he ____ earlier, …)
- Ellipsis and parallelism; correlative conjunctions (not only … but also …)
- Advanced prepositions/collocations (insist ____ doing, pride ____ something)
- Modifier placement; dangling/misplaced modifiers detection
- Cleft/it-cleft/wh-cleft sentences; emphasis with do-support
- Reported speech tense consistency; sequence of tenses

Each question MUST:
1) Have EXACTLY one blank: ____
2) Provide EXACTLY 4 options with ONE correct answer and 3 TRICKY distractors.
   Distractors should be plausible minimal pairs: subtle preposition changes, tense/aspect shifts,
   article/determiner differences, collocation/register mismatches, or punctuation/word-order differences.
3) Use natural, formal-academic register typical of TOEFL contexts. Do NOT be ambiguous.
4) NOT copy or closely paraphrase any provided conversation lines.

IMPORTANT: Do NOT copy or closely paraphrase these conversation examples:
{conversation_text}

Return VALID JSON array ONLY (no markdown, no code blocks), e.g.:
[
  {{
    "sentence": "Hardly ____ the committee convene when objections were raised.",
    "options": ["had", "did", "has", "was"],
    "answer": "had"
  }},
  {{
    "sentence": "It is essential that the proposal ____ before Friday.",
    "options": ["be submitted", "is submitted", "to be submitted", "was submitted"],
    "answer": "be submitted"
  }}
]

Generate {n} ADVANCED questions now."""
    else:
        prompt = f"""You are an expert English grammar tutor creating B1-B2 intermediate level multiple-choice fill-in-the-blank questions.

Topic: {topic_title}
Description: {topic_desc}

Generate {n} DIVERSE grammar questions suitable for B1-B2 (CEFR intermediate) level learners related to this topic context. 

IMPORTANT: VARY the grammar topics across all questions. Include a MIX of these different grammar areas:
- Verb tenses (present simple, past simple, present perfect, future)
- Subject-verb agreement
- Articles (a, an, the)
- Prepositions (time and place)
- Comparatives and superlatives
- Modal verbs (can, should, must, might, would)
- Question forms
- Gerunds vs infinitives
- Pronouns (subject, object, possessive)
- Conjunctions (and, but, because, although)
- Quantifiers (some, any, much, many, few)
- Adverbs of frequency

Each question should:
1. Test a DIFFERENT grammar point from the others - ensure variety!
2. Be clear and straightforward - focus on practical, everyday grammar usage
3. NOT use any sentences from the actual conversation examples
4. Have exactly 4 options: one correct answer and 3 plausible but clearly wrong distractors
5. Use simple, natural English sentences from everyday situations
6. Avoid advanced or tricky grammar constructions (no subjunctive, complex conditionals, or rare tenses)

IMPORTANT: Do NOT copy or closely paraphrase these conversation examples:
{conversation_text}

Format your response as valid JSON array ONLY (no markdown, no code blocks):
[
  {{
    "sentence": "She ____ taller than her brother.",
    "options": ["is", "are", "be", "am"],
    "answer": "is"
  }},
  {{
    "sentence": "I enjoy ____ books in my free time.",
    "options": ["reading", "to read", "read", "reads"],
    "answer": "reading"
  }}
]

Generate {n} VARIED questions now."""
    
    candidates = [
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-1.5-flash',
        'gemini-1.5-pro',
    ]
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        logger.error(f'Gemini configure failed: {e}')
        return _fallback_grammar_questions(topic, n)
    
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            txt = getattr(resp, 'text', None)
            
            if txt:
                # Clean markdown code blocks if present
                txt = txt.strip()
                if txt.startswith('```'):
                    lines = txt.split('\n')
                    txt = '\n'.join(lines[1:-1]) if len(lines) > 2 else txt
                txt = txt.strip()
                
                # Parse JSON
                import json
                questions_data = json.loads(txt)
                
                if not isinstance(questions_data, list):
                    continue
                
                # Build structured questions
                qs = []
                for q_data in questions_data[:n]:
                    if not isinstance(q_data, dict):
                        continue
                    
                    sentence = q_data.get('sentence', '').strip()
                    options = q_data.get('options', [])
                    answer = q_data.get('answer', '').strip()
                    
                    if not sentence or not options or not answer:
                        continue
                    
                    if '____' not in sentence and '__' not in sentence:
                        # Try to insert blank at answer position if format is wrong
                        if answer in sentence:
                            sentence = sentence.replace(answer, '____', 1)
                        else:
                            continue
                    
                    # Normalize blank to ____
                    sentence = sentence.replace('__', '____')
                    
                    # Ensure 4 options
                    if len(options) < 4:
                        continue
                    options = options[:4]
                    
                    # Shuffle options
                    random.shuffle(options)
                    
                    q = {
                        'id': str(uuid.uuid4()),
                        'sentence': sentence,
                        'options': options,
                        'answer': answer,
                        'answered': False,
                        'correct': None,
                    }
                    qs.append(q)
                
                if len(qs) >= n:
                    logger.info(f'Generated {len(qs)} grammar questions via {model_name}')
                    return qs[:n]
                    
        except Exception as e:
            logger.warning(f'Gemini grammar generation via {model_name} failed: {e}')
            continue
    
    # All models failed, use fallback
    logger.warning('All Gemini models failed; using fallback grammar questions')
    return _fallback_grammar_questions_advanced(topic, n) if is_advanced else _fallback_grammar_questions(topic, n)


def _fallback_grammar_questions(topic: Topic, n: int) -> list[dict]:
    """Generate diverse fallback grammar questions when Gemini is unavailable."""
    import random
    
    templates = [
        # Past simple
        {
            'sentence': 'I ____ to the market yesterday.',
            'options': ['go', 'went', 'have gone', 'will go'],
            'answer': 'went'
        },
        # Present perfect
        {
            'sentence': 'She ____ English for three years.',
            'options': ['studies', 'studied', 'has studied', 'will study'],
            'answer': 'has studied'
        },
        # Comparatives
        {
            'sentence': 'This book is ____ than that one.',
            'options': ['more interesting', 'most interesting', 'interesting', 'interest'],
            'answer': 'more interesting'
        },
        # Articles
        {
            'sentence': 'She is ____ excellent teacher.',
            'options': ['an', 'a', 'the', '---'],
            'answer': 'an'
        },
        # Gerunds
        {
            'sentence': 'I enjoy ____ music in my free time.',
            'options': ['listening to', 'listen to', 'to listen', 'listened to'],
            'answer': 'listening to'
        },
        # Prepositions of time
        {
            'sentence': 'The meeting is ____ Monday morning.',
            'options': ['on', 'in', 'at', 'for'],
            'answer': 'on'
        },
        # Subject-verb agreement
        {
            'sentence': 'Everyone ____ to arrive on time.',
            'options': ['needs', 'need', 'needing', 'to need'],
            'answer': 'needs'
        },
        # Quantifiers
        {
            'sentence': 'There are ____ people at the party.',
            'options': ['many', 'much', 'a lot', 'few of'],
            'answer': 'many'
        },
        # Adverbs of frequency
        {
            'sentence': 'She ____ goes to the gym on weekends.',
            'options': ['usually', 'usual', 'use to', 'used'],
            'answer': 'usually'
        },
        # Conjunctions
        {
            'sentence': 'I wanted to go out, ____ it was raining.',
            'options': ['but', 'and', 'because', 'so'],
            'answer': 'but'
        },
        # Question forms
        {
            'sentence': '____ does the train leave?',
            'options': ['What time', 'What', 'Which time', 'How time'],
            'answer': 'What time'
        },
        # Pronouns
        {
            'sentence': 'This is ____ favorite restaurant.',
            'options': ['my', 'me', 'mine', 'I'],
            'answer': 'my'
        },
    ]
    
    selected = random.sample(templates, min(n, len(templates)))
    
    qs = []
    for tmpl in selected:
        options = tmpl['options'][:]
        random.shuffle(options)
        q = {
            'id': str(uuid.uuid4()),
            'sentence': tmpl['sentence'],
            'options': options,
            'answer': tmpl['answer'],
            'answered': False,
            'correct': None,
        }
        qs.append(q)
    
    return qs


def _transcribe_audio_with_whisper(audio_file):
    """Transcribe audio using OpenAI Whisper tiny.en model"""
    try:
        # Treat DISABLE_WHISPER as disabling only the local openai-whisper fallback, not the hosted API.
        disable_local_whisper = str(os.environ.get('DISABLE_WHISPER', '')).strip().lower() in {"1", "true", "yes"}

        # Lazy-import local openai-whisper only when allowed and available; keep it optional.
        global _openai_whisper
        model = None
        if not disable_local_whisper:
            if _openai_whisper is None:
                try:
                    # Patch coverage types for numba/coverage cross-version compatibility
                    try:
                        import coverage.types as _cov_types  # type: ignore
                        # Map Tracer <-> TTracer
                        if not hasattr(_cov_types, 'Tracer') and hasattr(_cov_types, 'TTracer'):
                            setattr(_cov_types, 'Tracer', getattr(_cov_types, 'TTracer'))
                        if not hasattr(_cov_types, 'TTracer') and hasattr(_cov_types, 'Tracer'):
                            setattr(_cov_types, 'TTracer', getattr(_cov_types, 'Tracer'))
                        # Map ShouldTraceFn <-> TShouldTraceFn
                        if not hasattr(_cov_types, 'TShouldTraceFn') and hasattr(_cov_types, 'ShouldTraceFn'):
                            setattr(_cov_types, 'TShouldTraceFn', getattr(_cov_types, 'ShouldTraceFn'))
                        if not hasattr(_cov_types, 'ShouldTraceFn') and hasattr(_cov_types, 'TShouldTraceFn'):
                            setattr(_cov_types, 'ShouldTraceFn', getattr(_cov_types, 'TShouldTraceFn'))
                        # Define a minimal fallback when neither exists (older coverage.py versions)
                        if not hasattr(_cov_types, 'TShouldTraceFn') and not hasattr(_cov_types, 'ShouldTraceFn'):
                            try:
                                _T = _cov_types.Callable[[str, _cov_types.FrameType], _cov_types.TFileDisposition]  # type: ignore[attr-defined]
                            except Exception:
                                _T = object  # type: ignore
                            setattr(_cov_types, 'TShouldTraceFn', _T)
                            setattr(_cov_types, 'ShouldTraceFn', _T)
                    except Exception:
                        pass
                    import whisper as _w
                    _openai_whisper = _w
                except Exception:
                    _openai_whisper = None
            if _openai_whisper is not None:
                # Load Whisper model once and reuse (tiny.en is ~75MB)
                if not hasattr(_transcribe_audio_with_whisper, '_model'):
                    _transcribe_audio_with_whisper._model = _openai_whisper.load_model("tiny.en")
                model = _transcribe_audio_with_whisper._model

        # Persist upload to a temp file using original extension, then convert to 16kHz mono WAV
        try:
            orig_name = getattr(audio_file, 'name', '')
            ext = os.path.splitext(orig_name)[1] or '.m4a'
        except Exception:
            ext = '.m4a'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as src:
            for chunk in audio_file.chunks():
                src.write(chunk)
            src_path = src.name

        conv_path = src_path + '.wav'
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', src_path, '-ac', '1', '-ar', '16000', conv_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        except Exception as e:
            logger.error('ffmpeg conversion failed for Whisper ASR: %s', e)
            try:
                if os.path.exists(src_path):
                    os.unlink(src_path)
            except Exception:
                pass
            return ""

        # Optional short-audio guard to avoid empty/invalid transcripts
        try:
            with wave.open(conv_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate() or 16000
                duration = frames / float(rate)
                if duration < 0.25:
                    try:
                        if os.path.exists(conv_path):
                            os.unlink(conv_path)
                    except Exception:
                        pass
                    try:
                        if os.path.exists(src_path):
                            os.unlink(src_path)
                    except Exception:
                        pass
                    return ""
        except Exception:
            pass

        try:
            # Try OpenAI Whisper API first if key present and not disabled
            api_text = ""
            try:
                # Prefer WHISPER_API_KEY (same as WhisperService) and strip whitespace
                api_key = (os.environ.get('WHISPER_API_KEY') or os.environ.get('OPENAI_API_KEY') or '').strip()
                if api_key and str(os.environ.get('DISABLE_OPENAI_WHISPER_API', '')).strip().lower() not in {'1','true','yes'}:
                    headers = {'Authorization': 'Bearer ' + api_key}
                    data = {
                        'model': os.environ.get('OPENAI_WHISPER_MODEL', 'whisper-1'),
                        'response_format': 'json',
                        'language': 'en'
                    }
                    with open(conv_path, 'rb') as f:
                        files = {'file': ('audio.wav', f, 'audio/wav')}
                        resp = requests.post('https://api.openai.com/v1/audio/transcriptions', headers=headers, data=data, files=files, timeout=60)
                    if getattr(resp, 'status_code', 0) == 200:
                        try:
                            j = resp.json()
                            api_text = (j.get('text') or '').strip()
                            if not api_text:
                                try:
                                    body = resp.text
                                    logger.warning('OpenAI Whisper API 200 but empty text (speaking_journey): %s', body[:200])
                                except Exception:
                                    logger.warning('OpenAI Whisper API 200 but empty text (speaking_journey)')
                        except Exception as je:
                            try:
                                body = resp.text
                            except Exception:
                                body = ''
                            logger.warning('OpenAI Whisper API JSON parse failed (speaking_journey): %s | body: %s', je, body[:200])
                            api_text = ''
                    else:
                        try:
                            status_code = getattr(resp, 'status_code', 'unknown')
                            body = resp.text
                            logger.warning('OpenAI Whisper API error %s (speaking_journey): %s', status_code, body[:200])
                        except Exception:
                            logger.warning('OpenAI Whisper API error (speaking_journey): non-200 response')
            except Exception as e:
                logger.warning('OpenAI Whisper API call failed: %s', e)

            if api_text:
                try:
                    logger.info('ASR used: openai-api (speaking_journey), chars=%d', len(api_text))
                except Exception:
                    pass
                return api_text

            # Fallback to local Whisper only when a model is available and not disabled
            if model is None:
                return ""

            result = model.transcribe(conv_path, language='en')
            transcription = (result.get("text") or "").strip()
            try:
                logger.info('ASR used: openai-whisper (local, speaking_journey), chars=%d', len(transcription))
            except Exception:
                pass
            return transcription
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(conv_path):
                    os.unlink(conv_path)
            except Exception:
                pass
            try:
                if os.path.exists(src_path):
                    os.unlink(src_path)
            except Exception:
                pass
    except Exception as e:
        logger.error('Whisper transcription error: %s', e)
        return ""


def _transcribe_audio_with_speechbrain(audio_file):
    """Transcribe audio using SpeechBrain ASR (if available). Returns empty string on failure.

    This function makes a temporary copy of the uploaded file, converts it to 16kHz mono WAV via ffmpeg,
    then performs ASR with a pre-trained SpeechBrain model.
    """
    # Global feature flag guardrails
    # - Disable by default unless ENABLE_SPEECHBRAIN is set, to avoid torchaudio dependency/cold-start on Railway.
    if not os.environ.get('ENABLE_SPEECHBRAIN'):
        return ""
    if os.environ.get('DISABLE_SPEECHBRAIN') in ('1', 'true', 'True', 'YES', 'yes') or os.environ.get('SB_DISABLE') in ('1', 'true', 'True', 'YES', 'yes'):
        logger.info('SpeechBrain ASR disabled by environment flag (DISABLE_SPEECHBRAIN/SB_DISABLE).')
        return ""
    try:
        # Try to reset pointer in case it was read before
        if hasattr(audio_file, 'seek'):
            try:
                audio_file.seek(0)
            except Exception:
                pass

        # Persist chunks to a temp file using original extension when available
        try:
            orig_name = getattr(audio_file, 'name', '')
            ext = os.path.splitext(orig_name)[1] or '.m4a'
        except Exception:
            ext = '.m4a'

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as src:
            for chunk in audio_file.chunks():
                src.write(chunk)
            src_path = src.name

        # Convert to 16kHz mono WAV for ASR consumption
        conv_path = src_path + '.wav'
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', src_path, '-ac', '1', '-ar', '16000', conv_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        except Exception as e:
            logger.error('ffmpeg conversion failed for SpeechBrain ASR: %s', e)
            return ""

        # Lazy-load SpeechBrain ASR
        try:
            from speechbrain.pretrained import EncoderDecoderASR
        except Exception as e:
            logger.warning('SpeechBrain not installed or failed to import: %s', e)
            return ""

        try:
            if not hasattr(_transcribe_audio_with_speechbrain, '_asr_model'):
                _transcribe_audio_with_speechbrain._asr_model = EncoderDecoderASR.from_hparams(
                    source="speechbrain/asr-crdnn-rnnlm-librispeech",
                    run_opts={"device": "cpu"}
                )
            model = _transcribe_audio_with_speechbrain._asr_model
            text = model.transcribe_file(conv_path)
            return (text or "").strip()
        except Exception as e:
            msg = str(e)
            if 'WinError 1314' in msg or 'privilege' in msg.lower():
                logger.warning('SpeechBrain ASR skipped due to Windows symlink privilege error: %s', e)
            elif 'CoverageScorer' in msg or 'speechbrain.decoders.scorer.CoverageScorer' in msg:
                # Version mismatch between model hyperparams and installed speechbrain.
                # Downgrade to warning so request continues with Whisper fallback.
                logger.warning('SpeechBrain ASR transcription failed due to CoverageScorer mismatch: %s', e)
            else:
                logger.error('SpeechBrain ASR transcription failed: %s', e)
            return ""
        finally:
            # Cleanup temp files
            try:
                if os.path.exists(conv_path):
                    os.unlink(conv_path)
            except Exception:
                pass
            try:
                if os.path.exists(src_path):
                    os.unlink(src_path)
            except Exception:
                pass
    except Exception as e:
        logger.error('SpeechBrain transcription error: %s', e)
        return ""

def _transcribe_audio_with_faster_whisper(audio_file):
    """Transcribe audio using faster-whisper tiny.en model (CPU, int8).

    Returns empty string on failure or if the model is unavailable.
    """
    # Allow disabling faster-whisper completely via environment flag
    if str(os.environ.get('DISABLE_FASTER_WHISPER', '')).strip().lower() in {"1", "true", "yes"}:
        return ""
    if FasterWhisperModel is None:
        return ""
    try:
        # Lazily load model once
        if not hasattr(_transcribe_audio_with_faster_whisper, '_model'):
            _transcribe_audio_with_faster_whisper._model = FasterWhisperModel(
                "tiny.en", device="cpu", compute_type="int8"
            )
        model = _transcribe_audio_with_faster_whisper._model

        # Write uploaded audio to a temporary file; let ffmpeg decode container
        with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as tmp:
            for chunk in getattr(audio_file, 'chunks', lambda: [audio_file.read()])():
                if chunk:
                    tmp.write(chunk)
            temp_path = tmp.name

        try:
            # Transcribe with speed-optimized params (no VAD for short clips, greedy decode)
            # Notes:
            # - beam_size=1 and best_of=1 avoids beam search overhead
            # - vad_filter=False removes pyannote-like pre-VAD which can add latency
            # - temperature=0 for deterministic greedy
            segments, info = model.transcribe(
                temp_path,
                language="en",
                vad_filter=False,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                max_new_tokens=64,
                log_prob_threshold=-1.0,
                hallucination_silence_threshold=0.5,
            )
            # Join text from segments
            texts = []
            for seg in segments:
                try:
                    if seg and getattr(seg, 'text', None):
                        texts.append(seg.text)
                except Exception:
                    continue
            out = ' '.join([t.strip() for t in texts if t and t.strip()]).strip()

            # Collapse simple repeated phrase patterns for short utterances
            def _collapse_repeats(s: str) -> str:
                try:
                    import re as _re
                    # Keep original words for output, but compare using normalized tokens
                    words = (_re.sub(r"\s+", " ", s or "").strip()).split(" ")
                    norm = [
                        _re.sub(r"[^A-Za-z0-9']+", "", w.replace("â€™", "'").replace("`", "'"))
                        .lower()
                        for w in words
                    ]
                    n = len(words)
                    if n <= 3:
                        return " ".join(words)
                    i = 0
                    out_words: list[str] = []
                    while i < n:
                        max_w = min(5, n - i)
                        collapsed = False
                        for w in range(max_w, 1, -1):
                            chunk_norm = norm[i:i+w]
                            if any(t == "" for t in chunk_norm):
                                continue
                            repeats = 1
                            while (
                                i + (repeats * w) + w <= n
                                and norm[i + repeats*w:i + (repeats+1)*w] == chunk_norm
                            ):
                                repeats += 1
                            if repeats >= 2:
                                # Keep only the first occurrence in original formatting
                                out_words.extend(words[i:i+w])
                                i += repeats * w
                                collapsed = True
                                break
                        if not collapsed:
                            out_words.append(words[i])
                            i += 1
                    return " ".join(out_words)
                except Exception:
                    return s

            if len(out.split()) <= 30:
                out = _collapse_repeats(out)
            return out
        finally:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
    except Exception as e:
        logger.error('faster-whisper transcription error: %s', e)
        return ""


def _transcribe_audio_with_faster_whisper_vad(audio_file):
    """Transcribe using faster-whisper with VAD enabled; shares the same model instance."""
    if FasterWhisperModel is None:
        return ""
    try:
        # Reuse or create model
        if not hasattr(_transcribe_audio_with_faster_whisper, '_model'):
            _transcribe_audio_with_faster_whisper._model = FasterWhisperModel(
                "tiny.en", device="cpu", compute_type="int8"
            )
        model = _transcribe_audio_with_faster_whisper._model

        with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as tmp:
            for chunk in getattr(audio_file, 'chunks', lambda: [audio_file.read()])():
                if chunk:
                    tmp.write(chunk)
            temp_path = tmp.name

        try:
            segments, info = model.transcribe(
                temp_path,
                language="en",
                vad_filter=True,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                max_new_tokens=64,
                log_prob_threshold=-1.0,
                hallucination_silence_threshold=0.5,
            )
            texts = []
            for seg in segments:
                try:
                    if seg and getattr(seg, 'text', None):
                        texts.append(seg.text)
                except Exception:
                    continue
            out = ' '.join([t.strip() for t in texts if t and t.strip()]).strip()
            # Reuse the same collapser as above
            def _collapse_repeats(s: str) -> str:
                try:
                    import re as _re
                    words = (_re.sub(r"\s+", " ", s or "").strip()).split(" ")
                    norm = [
                        _re.sub(r"[^A-Za-z0-9']+", "", w.replace("â€™", "'").replace("`", "'"))
                        .lower()
                        for w in words
                    ]
                    n = len(words)
                    if n <= 3:
                        return " ".join(words)
                    i = 0
                    out_words: list[str] = []
                    while i < n:
                        max_w = min(5, n - i)
                        collapsed = False
                        for w in range(max_w, 1, -1):
                            chunk_norm = norm[i:i+w]
                            if any(t == "" for t in chunk_norm):
                                continue
                            repeats = 1
                            while (
                                i + (repeats * w) + w <= n
                                and norm[i + repeats*w:i + (repeats+1)*w] == chunk_norm
                            ):
                                repeats += 1
                            if repeats >= 2:
                                out_words.extend(words[i:i+w])
                                i += repeats * w
                                collapsed = True
                                break
                        if not collapsed:
                            out_words.append(words[i])
                            i += 1
                    return " ".join(out_words)
                except Exception:
                    return s

            if len(out.split()) <= 30:
                out = _collapse_repeats(out)
            return out
        finally:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
    except Exception as e:
        logger.error('faster-whisper (VAD) transcription error: %s', e)
        return ""


class SpeakingTopicsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        # Determine effective English Level for topic scoping:
        # 1) Query param (englishLevel or english_level) if provided and valid
        # 2) Else user's stored profile.english_level if set
        # 3) Else default to INTERMEDIATE for legacy clients
        raw_level = (request.query_params.get('englishLevel') or request.query_params.get('english_level') or '').strip().upper()

        # Get or create user profile (for welcome personalization and to persist settings elsewhere)
        profile, _ = UserProfile.objects.get_or_create(
            user=request.user,
            defaults={'first_visit': True}
        )

        valid_levels = {c for c, _ in EnglishLevel.choices}
        if raw_level in valid_levels:
            effective_level = raw_level
        else:
            prof_level = str(getattr(profile, 'english_level', '') or '').strip().upper()
            effective_level = prof_level if prof_level in valid_levels else EnglishLevel.INTERMEDIATE

        scoped_qs = Topic.objects.filter(is_active=True, difficulty=effective_level).order_by('sequence')
        topics_list = list(scoped_qs)
        topics, completed_sequences, unlocked_sequences = _compute_unlocks_scoped(request.user, topics_list)
        # Test flag: unlock all topics for designated test accounts
        if _unlock_all_for_user(request.user):
            unlocked_sequences = {t.sequence for t in topics}

        # Profile already fetched/created above

        # Fetch all phrase progress for this user
        phrase_progress_dict = {}
        for topic in topics:
            phrase_progress, _ = PhraseProgress.objects.get_or_create(
                user=request.user,
                topic=topic,
                defaults={'current_phrase_index': 0, 'completed_phrases': []}
            )
            phrase_progress_dict[topic.id] = phrase_progress

        payload = []
        for t in topics:
            # Get phrase progress for this topic
            phrase_prog = phrase_progress_dict.get(t.id)
            phrase_progress_data = None
            if phrase_prog:
                phrase_progress_data = {
                    'currentPhraseIndex': phrase_prog.current_phrase_index,
                    'completedPhrases': phrase_prog.completed_phrases or [],
                    'totalPhrases': len(t.material_lines or []),
                    'isAllPhrasesCompleted': phrase_prog.is_all_phrases_completed,
                }

            # Build fluency progress view
            fprompt = t.fluency_practice_prompt or ''
            tp, _ = TopicProgress.objects.get_or_create(user=request.user, topic=t)
            stored_scores = list(tp.fluency_prompt_scores or [])
            # For single prompt, just get the first score if it exists
            prompt_score = None
            if len(stored_scores) > 0:
                try:
                    prompt_score = int(stored_scores[0])
                except Exception:
                    prompt_score = None
            
            # Single prompt is completed if score exists
            fluency_completed = fprompt and prompt_score is not None
            fluency_total_score = int(prompt_score or 0)

            # Build practice scores data
            pronunciation_score = int(tp.pronunciation_total_score or 0)
            logger.info(f"Topic {t.title} - pronunciation_total_score from DB: {tp.pronunciation_total_score}, using: {pronunciation_score}")
            # Fallback recompute for legacy sessions where pron_total was not persisted (use average 0â€“100)
            if pronunciation_score <= 0:
                try:
                    qs = UserPhraseRecording.objects.filter(user=request.user, topic=t).order_by('phrase_index', '-created_at')
                    latest_by_phrase = {}
                    for r in qs:
                        if r.phrase_index not in latest_by_phrase:
                            latest_by_phrase[r.phrase_index] = r
                    total_acc = 0
                    cnt = 0
                    for r in latest_by_phrase.values():
                        try:
                            total_acc += int(round(float(r.accuracy or 0.0)))
                            cnt += 1
                        except Exception:
                            pass
                    recomputed_avg = int(round(total_acc / cnt)) if cnt > 0 else 0
                    if recomputed_avg > 0:
                        tp.pronunciation_total_score = recomputed_avg
                        tp.save(update_fields=['pronunciation_total_score'])
                        pronunciation_score = recomputed_avg
                except Exception:
                    pass
            vocabulary_score = int(tp.vocabulary_total_score or 0)

            # Grammar total score (optional mode). Prefer persisted field if present; otherwise
            # fall back to latest completed GrammarPracticeSession to avoid schema dependency during rollout.
            try:
                grammar_score = int(getattr(tp, 'grammar_total_score', 0) or 0)
                if grammar_score <= 0:
                    from .models import GrammarPracticeSession
                    latest_gram = (
                        GrammarPracticeSession.objects
                        .filter(user=request.user, topic=t, completed=True)
                        .order_by('-created_at')
                        .first()
                    )
                    if latest_gram:
                        grammar_score = int(latest_gram.total_score or 0)
            except Exception:
                grammar_score = int(getattr(tp, 'grammar_total_score', 0) or 0)

            # Compute per-practice maxima (normalized 0â€“100)
            pron_max = 100
            flu_max = 100
            vocab_max = 100
            grammar_max = 100

            # Clamp totals to maxima for fair percentage/progress
            eff_pron = min(pronunciation_score, pron_max) if pron_max > 0 else 0
            eff_flu = min(fluency_total_score, flu_max) if flu_max > 0 else 0
            eff_vocab = min(vocabulary_score, vocab_max) if vocab_max > 0 else 0
            eff_grammar = min(grammar_score, grammar_max) if grammar_max > 0 else 0
            
            # Pass rule thresholds (configurable via env/settings)
            try:
                pass_avg = float(getattr(settings, 'TOPIC_PASS_AVG_PERCENT', os.environ.get('TOPIC_PASS_AVG_PERCENT', 75)))
            except Exception:
                pass_avg = 75.0
            try:
                floor_pct = float(getattr(settings, 'TOPIC_PASS_FLOOR_PERCENT', os.environ.get('TOPIC_PASS_FLOOR_PERCENT', 60)))
            except Exception:
                floor_pct = 60.0

            # Combined progress: BEGINNER uses Pronunciation + Vocabulary only; others use 4 practices
            level_str = str(getattr(t, 'difficulty', '') or '').strip().upper()
            if level_str == 'BEGINNER':
                total_max = int(pron_max + vocab_max)
                total_score = int(eff_pron + eff_vocab)
                combined_percent = round((total_score / total_max) * 100.0, 1) if total_max > 0 else 0.0
                # Beginner floor checks apply only to Pron + Vocab
                floor_ok = all([
                    eff_pron >= floor_pct,
                    eff_vocab >= floor_pct,
                ])
            else:
                total_max = int(pron_max + flu_max + vocab_max + grammar_max)
                total_score = int(eff_pron + eff_flu + eff_vocab + eff_grammar)
                combined_percent = round((total_score / total_max) * 100.0, 1) if total_max > 0 else 0.0
                # Hybrid pass: average >= pass_avg AND no skill below floor
                floor_ok = all([
                    eff_pron >= floor_pct,
                    eff_flu >= floor_pct,
                    eff_vocab >= floor_pct,
                    eff_grammar >= floor_pct,
                ])
            meets_requirement = bool(combined_percent >= pass_avg and floor_ok)

            combined_threshold_score = int(math.ceil((pass_avg / 100.0) * total_max)) if total_max > 0 else 0

            # 'Met' flags reflect floor threshold
            pron_met = eff_pron >= floor_pct
            flu_met = eff_flu >= floor_pct
            vocab_met = eff_vocab >= floor_pct
            grammar_met = eff_grammar >= floor_pct

            practice_scores_data = {
                'pronunciation': pronunciation_score,
                'fluency': fluency_total_score,
                'vocabulary': vocabulary_score,
                # Grammar is now part of unlock criteria (4th required practice)
                'grammar': grammar_score,
                # Listening practice (optional; not part of unlock criteria)
                'listening': int(getattr(tp, 'listening_total_score', 0) or 0),
                # Keep average as a normalized combined percent for clearer UI
                'average': combined_percent,
                'meetsRequirement': meets_requirement,
                # New fields for UI progress bar and clarity
                'maxPronunciation': pron_max,
                'maxFluency': flu_max,
                'maxVocabulary': vocab_max,
                'maxGrammar': grammar_max,
                'maxListening': 100,
                'pronunciationMet': pron_met,
                'fluencyMet': flu_met,
                'vocabularyMet': vocab_met,
                'grammarMet': grammar_met,
                'listeningMet': False,
                'totalScore': total_score,
                'totalMaxScore': total_max,
                'combinedThresholdScore': combined_threshold_score,
                'combinedPercent': combined_percent,
                'thresholdPercent': int(pass_avg),
            }

            payload.append({
                'id': str(t.id),
                'title': t.title,
                'description': t.description or "",
                'material': t.material_lines or [],
                'vocabulary': t.vocabulary or [],
                'conversation': t.conversation_example or [],
                'fluencyPracticePrompts': [t.fluency_practice_prompt] if t.fluency_practice_prompt else [],
                'difficulty': getattr(t, 'difficulty', None) or None,
                'fluencyProgress': {
                    'promptsCount': 1 if fprompt else 0,
                    'promptScores': [int(prompt_score)] if prompt_score is not None else [],
                    'totalScore': fluency_total_score,
                    'nextPromptIndex': 0 if not fluency_completed and fprompt else None,
                    'completed': fluency_completed,
                },
                'phraseProgress': phrase_progress_data,
                'practiceScores': practice_scores_data,
                'conversationScore': int(getattr(tp, 'conversation_total_score', 0) or 0),
                'conversationCompleted': bool(getattr(tp, 'conversation_completed', False)),
                'unlocked': t.sequence in unlocked_sequences,
                'completed': t.sequence in completed_sequences,
            })

        # Build user profile data
        user_profile_data = {
            'firstVisit': profile.first_visit,
            'lastVisitedTopicId': str(profile.last_visited_topic.id) if profile.last_visited_topic else None,
            'lastVisitedTopicTitle': profile.last_visited_topic.title if profile.last_visited_topic else "",
            'englishLevel': profile.english_level if getattr(profile, 'english_level', None) else None,
        }

        # Mark as not first visit after this request
        if profile.first_visit:
            profile.first_visit = False
            profile.save(update_fields=['first_visit'])

        response_data = {
            'topics': payload,
            'userProfile': user_profile_data
        }
        serializer = SpeakingTopicsResponseSerializer(response_data)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        topic_id = request.data.get('lastVisitedTopicId')
        raw_level = (request.data.get('englishLevel') or request.data.get('english_level') or '').strip().upper()

        if not topic_id and not raw_level:
            return Response({'detail': 'Missing lastVisitedTopicId or englishLevel'}, status=status.HTTP_400_BAD_REQUEST)

        profile, _ = UserProfile.objects.get_or_create(
            user=request.user,
            defaults={'first_visit': False}
        )

        if topic_id:
            topic = get_object_or_404(Topic, id=topic_id, is_active=True)
            profile.last_visited_topic = topic

        if raw_level in {c for c, _ in EnglishLevel.choices}:
            profile.english_level = raw_level

        profile.save()

        return Response({'success': True}, status=status.HTTP_200_OK)


class SubmitPhraseRecordingView(APIView):
    """Submit phrase recording for transcription and accuracy evaluation"""
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, topic_id):
        # Get topic and validate access
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)

        # Get required parameters
        phrase_index = request.data.get('phraseIndex')
        audio_file = request.data.get('audio')

        if phrase_index is None:
            return Response({'detail': 'Missing phraseIndex'}, status=status.HTTP_400_BAD_REQUEST)

        if not audio_file:
            return Response({'detail': 'Missing audio file'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            phrase_index = int(phrase_index)
        except (ValueError, TypeError):
            return Response({'detail': 'Invalid phraseIndex'}, status=status.HTTP_400_BAD_REQUEST)

        # Validate phrase index
        material_lines = topic.material_lines or []
        if phrase_index < 0 or phrase_index >= len(material_lines):
            return Response({'detail': 'Invalid phraseIndex for this topic'}, status=status.HTTP_400_BAD_REQUEST)

        expected_phrase = material_lines[phrase_index]

        # Get or create phrase progress
        phrase_progress, _ = PhraseProgress.objects.get_or_create(
            user=request.user,
            topic=topic,
            defaults={'current_phrase_index': 0, 'completed_phrases': []}
        )
        # Also ensure TopicProgress exists so we can persist running totals
        topic_progress, _ = TopicProgress.objects.get_or_create(
            user=request.user,
            topic=topic
        )

        # Transcribe engines. Optionally prefer faster-whisper first via env flag, unless disabled.
        prefer_fw = (
            str(os.environ.get('PREFER_FASTER_WHISPER', '')).strip().lower() in {"1", "true", "yes"}
            and str(os.environ.get('DISABLE_FASTER_WHISPER', '')).strip().lower() not in {"1", "true", "yes"}
        )
        whisper_transcription = ''
        sb_transcription = ''
        fw_transcription = ''

        if prefer_fw:
            fw_transcription = _transcribe_audio_with_faster_whisper(audio_file)
            if hasattr(audio_file, 'seek'):
                try:
                    audio_file.seek(0)
                except Exception:
                    pass
            whisper_transcription = _transcribe_audio_with_whisper(audio_file)
            if hasattr(audio_file, 'seek'):
                try:
                    audio_file.seek(0)
                except Exception:
                    pass
            sb_transcription = _transcribe_audio_with_speechbrain(audio_file)
        else:
            # Original order: Whisper, SpeechBrain, then faster-whisper fallback
            whisper_transcription = _transcribe_audio_with_whisper(audio_file)
            if hasattr(audio_file, 'seek'):
                try:
                    audio_file.seek(0)
                except Exception:
                    pass
            sb_transcription = _transcribe_audio_with_speechbrain(audio_file)
            if not whisper_transcription and not sb_transcription and str(os.environ.get('DISABLE_FASTER_WHISPER', '')).strip().lower() not in {"1", "true", "yes"}:
                if hasattr(audio_file, 'seek'):
                    try:
                        audio_file.seek(0)
                    except Exception:
                        pass
                fw_transcription = _transcribe_audio_with_faster_whisper(audio_file)

        # If all failed, continue gracefully: persist attempt with zero accuracy so progress/state updates
        all_failed = (not whisper_transcription and not sb_transcription and not fw_transcription)

        # Compute accuracies for available transcripts
        scores = []
        whisper_accuracy = None
        if whisper_transcription:
            whisper_accuracy = _calculate_similarity(expected_phrase, whisper_transcription)
            scores.append(whisper_accuracy)
        sb_accuracy = None
        if sb_transcription:
            sb_accuracy = _calculate_similarity(expected_phrase, sb_transcription)
            scores.append(sb_accuracy)
        fw_accuracy = None
        if fw_transcription:
            fw_accuracy = _calculate_similarity(expected_phrase, fw_transcription)
            scores.append(fw_accuracy)

        # Combine scores (average) and choose best transcription for feedback
        combined_accuracy = sum(scores) / len(scores) if scores else 0.0
        # Pick the best transcription among those available
        best_transcription = whisper_transcription
        best_score = whisper_accuracy or -1.0
        if (sb_accuracy or -1.0) > best_score:
            best_transcription = sb_transcription
            best_score = sb_accuracy or best_score
        if (fw_accuracy or -1.0) > best_score:
            best_transcription = fw_transcription

        # Final sanitary collapse for short utterances to remove accidental repetitions
        try:
            if (best_transcription or '').strip():
                if len((best_transcription or '').split()) <= 30:
                    best_transcription = _collapse_repeats_text(best_transcription)
        except Exception:
            pass

        # Determine pass/fail for UI feedback (80% threshold for "Phrase Passed!" message)
        passed = combined_accuracy >= 80.0
        next_phrase_index = None
        topic_completed = False
        
        # Calculate proportional score: 100% = 10 points, 90-99% = 9 points, etc.
        proportional_score = int(min(10, max(0, combined_accuracy // 10)))
        
        # Award XP based on proportional score (0-20 XP, scaled from 0-10 score)
        xp_awarded = 0
        if proportional_score > 0:
            # Scale proportional score (0-10) to XP amount (0-20)
            xp_amount = proportional_score * 2
            xp_awarded += _award_xp(
                user=request.user,
                amount=xp_amount,
                source='pronunciation',
                context={
                    'topicId': str(topic.id),
                    'phraseIndex': phrase_index,
                    'accuracy': round(float(combined_accuracy or 0.0), 1),
                    'proportionalScore': proportional_score,
                }
            )

        # Always mark phrase as completed and advance for progression (regardless of accuracy)
        # The pass/fail logic affects UI feedback and XP, but not progression blocking
        phrase_progress.mark_phrase_completed(phrase_index)
        next_phrase_index = phrase_progress.current_phrase_index

        # If all phrases completed, mark pronunciation as completed
        # (Listening is an optional bonus practice that users can do separately)
        if phrase_progress.is_all_phrases_completed:
            topic_progress.pronunciation_completed = True

        # If now all modes are completed, mark topic completed and award mastery later
        was_completed = bool(topic_progress.completed)
        if not topic_progress.completed and topic_progress.all_modes_completed:
            topic_progress.completed = True
            topic_progress.completed_at = timezone.now()
        topic_progress.save()
        
        # Check for proficiency tier change when topic is completed
        if not was_completed and topic_progress.completed:
            _check_and_create_proficiency_event(request.user)
        
        topic_completed = topic_progress.completed
        # Log TOPIC_COMPLETED once when it flips
        try:
            if not was_completed and topic_progress.completed:
                try:
                    AchievementEvent.objects.create(
                        user=request.user,
                        event_type='TOPIC_COMPLETED',
                        title=f"Completed '{getattr(topic, 'title', '')}'",
                        description=f"You completed the topic '{getattr(topic, 'title', '')}'",
                        meta={'topicId': str(topic.id), 'practice': 'pronunciation'}
                    )
                except Exception:
                    pass
        except Exception:
            pass
        if topic_completed and phrase_progress.is_all_phrases_completed:
            # Award topic mastery bonus once (+50)
            xp_awarded += _award_topic_mastery_once(request.user, topic)

        # Get feedback from Gemini based on the best transcription and combined accuracy
        feedback = _get_gemini_feedback(expected_phrase, best_transcription, combined_accuracy)
        # Ensure feedback is never null to avoid DB constraint violation
        if feedback is None:
            feedback = ""

        # Update pronunciation_total_score BEFORE saving recording so it's available in response
        # Temporarily create the recording object to include it in the calculation
        temp_recording = UserPhraseRecording(
            user=request.user,
            topic=topic,
            phrase_index=phrase_index,
            transcription=best_transcription,
            accuracy=round(combined_accuracy, 1),
            feedback=feedback,
        )
        
        # Calculate updated pronunciation_total_score including this new recording
        try:
            # Get existing recordings for this topic
            qs = (
                UserPhraseRecording.objects
                .filter(user=request.user, topic=topic)
                .order_by('phrase_index', '-created_at')
            )
            latest_by_phrase = {}
            for r in qs:
                if r.phrase_index not in latest_by_phrase:
                    latest_by_phrase[r.phrase_index] = r
            
            # Include the new recording in the calculation
            latest_by_phrase[phrase_index] = temp_recording
            
            total_acc = 0
            cnt = 0
            for r in latest_by_phrase.values():
                try:
                    total_acc += int(round(float(r.accuracy or 0.0)))
                    cnt += 1
                except Exception:
                    pass
            if cnt > 0:
                avg_acc = int(round(total_acc / cnt))
                logger.info(f"Updating pronunciation_total_score for topic {topic.id}: {topic_progress.pronunciation_total_score} -> {avg_acc} (based on {cnt} recordings, including new one)")
                topic_progress.pronunciation_total_score = avg_acc
                topic_progress.save(update_fields=['pronunciation_total_score'])
        except Exception as e:
            logger.warning('Failed updating pronunciation totals (pre-save): %s', e)

        # Persist user recording with audio and metadata
        recording_id = None
        audio_url = ''
        try:
            # Reset file pointer if needed before saving
            if hasattr(audio_file, 'seek'):
                try:
                    audio_file.seek(0)
                except Exception:
                    pass

            upr = UserPhraseRecording(
                user=request.user,
                topic=topic,
                phrase_index=phrase_index,
                transcription=best_transcription,
                accuracy=round(combined_accuracy, 1),
                feedback=feedback,
            )
            # Save file to storage (uses upload_to path)
            try:
                filename = getattr(audio_file, 'name', 'recording.m4a')
                upr.audio_file.save(filename, audio_file, save=False)
            except Exception:
                # As a fallback, try reading content to a ContentFile
                try:
                    if hasattr(audio_file, 'seek'):
                        audio_file.seek(0)
                    content = audio_file.read()
                except Exception:
                    content = b''
                if content:
                    upr.audio_file.save('recording.m4a', ContentFile(content), save=False)
                else:
                    # Final fallback: save a short silent WAV so the record persists
                    try:
                        import struct
                        duration_s = 0.2
                        sr = 16000
                        n_samples = int(sr * duration_s)
                        buf = io.BytesIO()
                        with wave.open(buf, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(sr)
                            # Write n_samples of silence
                            silence = struct.pack('<' + 'h'*n_samples, *([0]*n_samples))
                            wf.writeframes(silence)
                        upr.audio_file.save('placeholder.wav', ContentFile(buf.getvalue()), save=False)
                    except Exception:
                        # If even this fails, allow outer handler to proceed without audio
                        pass
            upr.save()
            recording_id = str(upr.id)
            try:
                if upr.audio_file:
                    audio_url = request.build_absolute_uri(upr.audio_file.url)
            except Exception:
                audio_url = ''

        except Exception as e:
            # Ensure outer try is closed to avoid SyntaxError, and continue with safe defaults
            logger.warning('Failed to persist user phrase recording or build audio URL: %s', e)

        # pronunciation_total_score already updated above before saving recording

        # Build response
        result = {
            'success': passed,
            'accuracy': round(combined_accuracy, 1),
            'transcription': best_transcription,
            'feedback': feedback,
            'nextPhraseIndex': next_phrase_index,
            'topicCompleted': topic_completed,
            'xpAwarded': xp_awarded,
            'recordingId': recording_id,
            'audioUrl': audio_url,
        }
        
        # Debug logging to track accuracy and scoring
        logger.info(f"Pronunciation submission - Expected: '{expected_phrase}', Got: '{best_transcription}', Combined Accuracy: {combined_accuracy:.1f}%, Proportional Score: {proportional_score}/10, XP: {xp_awarded}, Passed UI: {passed}")
        
        serializer = PhraseSubmissionResultSerializer(result)
        return Response(serializer.data, status=status.HTTP_200_OK)


class SubmitConversationTurnView(APIView):
    """Submit a conversation turn recording for transcription and accuracy scoring."""
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, topic_id):
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        turn_index = request.data.get('turnIndex')
        role = (request.data.get('role') or '').strip()
        audio_file = request.data.get('audio')

        if turn_index is None:
            return Response({'detail': 'Missing turnIndex'}, status=status.HTTP_400_BAD_REQUEST)
        if not audio_file:
            return Response({'detail': 'Missing audio file'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            turn_index = int(turn_index)
        except (TypeError, ValueError):
            return Response({'detail': 'Invalid turnIndex'}, status=status.HTTP_400_BAD_REQUEST)

        conv = topic.conversation_example or []
        if turn_index < 0 or turn_index >= len(conv):
            return Response({'detail': 'Invalid turnIndex for this topic'}, status=status.HTTP_400_BAD_REQUEST)

        # Expected text for this turn (handle dict or simple list)
        try:
            expected_text = ''
            item = conv[turn_index]
            if isinstance(item, dict):
                expected_text = (item.get('text') or '').strip()
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                expected_text = str(item[1]).strip()
            else:
                expected_text = str(item).strip()
        except Exception:
            expected_text = ''

        tp, _ = TopicProgress.objects.get_or_create(user=request.user, topic=topic)

        # Transcribe (optionally prefer faster-whisper first), unless disabled
        prefer_fw = (
            str(os.environ.get('PREFER_FASTER_WHISPER', '')).strip().lower() in {"1", "true", "yes"}
            and str(os.environ.get('DISABLE_FASTER_WHISPER', '')).strip().lower() not in {"1", "true", "yes"}
        )
        whisper_tx = ''
        sb_tx = ''
        fw_tx = ''
        if prefer_fw:
            fw_tx = _transcribe_audio_with_faster_whisper(audio_file)
            if hasattr(audio_file, 'seek'):
                try:
                    audio_file.seek(0)
                except Exception:
                    pass
            whisper_tx = _transcribe_audio_with_whisper(audio_file)
            if hasattr(audio_file, 'seek'):
                try:
                    audio_file.seek(0)
                except Exception:
                    pass
            sb_tx = _transcribe_audio_with_speechbrain(audio_file)
        else:
            whisper_tx = _transcribe_audio_with_whisper(audio_file)
            if hasattr(audio_file, 'seek'):
                try:
                    audio_file.seek(0)
                except Exception:
                    pass
            sb_tx = _transcribe_audio_with_speechbrain(audio_file)
            if not whisper_tx and not sb_tx and str(os.environ.get('DISABLE_FASTER_WHISPER', '')).strip().lower() not in {"1", "true", "yes"}:
                if hasattr(audio_file, 'seek'):
                    try:
                        audio_file.seek(0)
                    except Exception:
                        pass
                fw_tx = _transcribe_audio_with_faster_whisper(audio_file)

        # Scores and best transcript
        scores = []
        wa = _calculate_similarity(expected_text, whisper_tx) if whisper_tx else None
        if wa is not None:
            scores.append(wa)
        sa = _calculate_similarity(expected_text, sb_tx) if sb_tx else None
        if sa is not None:
            scores.append(sa)
        fa = _calculate_similarity(expected_text, fw_tx) if fw_tx else None
        if fa is not None:
            scores.append(fa)
        combined_accuracy = sum(scores) / len(scores) if scores else 0.0

        best_tx = whisper_tx
        best_score = wa if wa is not None else -1.0
        if sa is not None and sa > best_score:
            best_tx = sb_tx
            best_score = sa
        if fa is not None and fa > best_score:
            best_tx = fw_tx

        # Repetition-aware tie-breaker: if top choices are close, prefer the one without repetition
        try:
            cand = []
            if whisper_tx:
                cand.append((whisper_tx, wa or 0.0, _is_repetition_issue(expected_text, whisper_tx)))
            if sb_tx:
                cand.append((sb_tx, sa or 0.0, _is_repetition_issue(expected_text, sb_tx)))
            if fw_tx:
                cand.append((fw_tx, fa or 0.0, _is_repetition_issue(expected_text, fw_tx)))
            if cand:
                # sort by score desc first
                cand.sort(key=lambda x: x[1], reverse=True)
                top_score = cand[0][1]
                # among near-top (within 5 pts), pick the one with no repetition if available
                near = [c for c in cand if (top_score - c[1]) <= 5.0]
                if any(not c[2] for c in near):
                    chosen = next(c for c in near if not c[2])
                    best_tx, best_score, _ = chosen
        except Exception:
            pass

        # If still looks repetitive, try a quick VAD-enabled faster-whisper pass and adopt if better
        try:
            if _is_repetition_issue(expected_text, best_tx or ''):
                if hasattr(audio_file, 'seek'):
                    try:
                        audio_file.seek(0)
                    except Exception:
                        pass
                fw_vad_tx = _transcribe_audio_with_faster_whisper_vad(audio_file)
                if fw_vad_tx:
                    new_score = _calculate_similarity(expected_text, fw_vad_tx)
                    # Accept if better score, or if similar (within 3 pts) but fixes repetition
                    if (new_score > (best_score or -1.0)) or (
                        abs(new_score - (best_score or 0.0)) <= 3.0 and not _is_repetition_issue(expected_text, fw_vad_tx)
                    ):
                        best_tx = fw_vad_tx
                        best_score = new_score
        except Exception:
            pass

        # Final sanitary collapse for short utterances
        try:
            if (best_tx or '').strip():
                if len((best_tx or '').split()) <= 30:
                    best_tx = _collapse_repeats_text(best_tx)
        except Exception:
            pass

        xp_awarded = 0
        if combined_accuracy >= 80.0:
            xp_awarded += _award_xp(
                user=request.user,
                amount=20,
                source='conversation',
                context={'topicId': str(topic.id), 'turnIndex': turn_index, 'accuracy': round(float(combined_accuracy or 0.0), 1)}
            )

        feedback = _get_gemini_feedback(expected_text, best_tx or '', combined_accuracy)

        # Persist conversation recording
        rec_id = None
        audio_url = ''
        try:
            if hasattr(audio_file, 'seek'):
                try:
                    audio_file.seek(0)
                except Exception:
                    pass
            ucr = UserConversationRecording(
                user=request.user,
                topic=topic,
                turn_index=turn_index,
                role=role,
                transcription=best_tx or '',
                accuracy=round(float(combined_accuracy or 0.0), 1),
                feedback=feedback or '',
            )
            try:
                filename = getattr(audio_file, 'name', 'conversation.m4a')
                ucr.audio_file.save(filename, audio_file, save=False)
            except Exception:
                try:
                    if hasattr(audio_file, 'seek'):
                        audio_file.seek(0)
                    content = audio_file.read()
                except Exception:
                    content = b''
                if content:
                    ucr.audio_file.save('conversation.m4a', ContentFile(content), save=False)
            ucr.save()
            rec_id = str(ucr.id)
            try:
                if ucr.audio_file:
                    audio_url = request.build_absolute_uri(ucr.audio_file.url)
            except Exception:
                audio_url = ''
        except Exception as e:
            logger.warning('Failed to persist conversation recording: %s', e)

        # Update totals: average accuracy across user's turns only (not all turns)
        try:
            qs = UserConversationRecording.objects.filter(user=request.user, topic=topic).order_by('turn_index', '-created_at')
            latest_by_turn = {}
            for r in qs:
                if r.turn_index not in latest_by_turn:
                    latest_by_turn[r.turn_index] = r
            
            # Only count recorded turns (user's turns), not total conversation turns
            total = 0
            cnt = 0
            individual_accuracies = []
            for r in latest_by_turn.values():
                try:
                    accuracy = int(round(float(r.accuracy or 0.0)))
                    total += accuracy
                    cnt += 1
                    individual_accuracies.append(f"Turn {r.turn_index}: {accuracy}%")
                except Exception:
                    total += 0
                    individual_accuracies.append(f"Turn {r.turn_index}: 0% (error)")
            avg = int(round(total / cnt)) if cnt > 0 else 0
            
            logger.info(f"Individual turn accuracies: {', '.join(individual_accuracies)}, Average: {avg}%")
            tp.conversation_total_score = int(avg)
            
            # Determine completion: check if user has recorded all their turns
            # Count how many turns belong to the user's role based on existing recordings
            user_role_from_recordings = None
            if latest_by_turn:
                # Infer user's role from their recordings
                sample_recording = next(iter(latest_by_turn.values()))
                user_role_from_recordings = getattr(sample_recording, 'role', None)
            
            if user_role_from_recordings and conv:
                # Clean user role (remove quotes and whitespace)
                clean_user_role = str(user_role_from_recordings or '').strip().strip('"').strip("'").upper()
                
                # Count total turns for this role in the conversation
                role_turns_total = 0
                for turn in conv:
                    if isinstance(turn, dict):
                        speaker = str(turn.get('speaker', '')).strip().upper()
                        if speaker == clean_user_role:
                            role_turns_total += 1
                    elif isinstance(turn, (list, tuple)) and len(turn) >= 1:
                        speaker = str(turn[0]).strip().upper()
                        if speaker == clean_user_role:
                            role_turns_total += 1
                
                user_recorded_turns = len(latest_by_turn)
                tp.conversation_completed = (user_recorded_turns >= role_turns_total and role_turns_total > 0)
                
                # Debug logging for role counting  
                logger.info(f"Conversation analysis - Topic: {topic.title} (ID: {topic.id}), User role: {repr(user_role_from_recordings)}, Total conversation turns: {len(conv)}, Role turns expected: {role_turns_total}, User recorded: {user_recorded_turns}")
                logger.info(f"Cleaned user role: {repr(clean_user_role)}")
                
                for i, turn in enumerate(conv):
                    speaker = turn.get('speaker', 'Unknown') if isinstance(turn, dict) else str(turn[0]) if isinstance(turn, (list, tuple)) and len(turn) > 0 else 'Unknown'
                    clean_speaker = str(speaker).strip().upper()
                    is_user_turn = clean_speaker == clean_user_role
                    logger.info(f"  Turn {i}: Speaker '{speaker}' (clean: '{clean_speaker}') - {'USER TURN' if is_user_turn else 'OTHER'}")
                    
                # Also log conversation data structure for debugging
                logger.info(f"Raw conversation data sample: {conv[:2] if len(conv) >= 2 else conv}")
            else:
                # Fallback: use original logic if role inference fails
                tp.conversation_completed = (len(latest_by_turn) >= len(conv) and len(conv) > 0)
                logger.info(f"Conversation analysis fallback - No role inferred, total turns: {len(conv)}, recorded: {len(latest_by_turn)}")
            
            tp.save(update_fields=['conversation_total_score', 'conversation_completed'])
            
            logger.info(f"Conversation score updated - User role: {user_role_from_recordings}, Recorded: {len(latest_by_turn)}, Avg accuracy: {avg}%, Completed: {tp.conversation_completed}")
            
        except Exception as e:
            logger.warning('Failed updating conversation totals: %s', e)

        # Determine success based on accuracy threshold (80% for passing)
        success = combined_accuracy >= 80.0
        next_turn_index = turn_index + 1 if (turn_index + 1) < len(conv) else None
        payload = {
            'success': success,
            'accuracy': round(float(combined_accuracy or 0.0), 1),
            'transcription': best_tx or '',
            'feedback': feedback or '',
            'nextTurnIndex': next_turn_index,
            'topicCompleted': bool(tp.completed),
            'xpAwarded': int(xp_awarded or 0),
            'recordingId': rec_id,
            'audioUrl': audio_url,
        }
        out = ConversationSubmissionResultSerializer(payload)
        return Response(out.data, status=status.HTTP_200_OK)


class CompleteTopicView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, topic_id):
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        progress, created = TopicProgress.objects.get_or_create(user=request.user, topic=topic)
        was_completed = bool(progress.completed)

        # Only allow marking topic as completed if the 4-practice criteria are met.
        if not _meets_completion_criteria(progress):
            resp = {
                'success': False,
                'message': 'Cannot complete topic: requirements not met (Pronunciation, Fluency, Vocabulary, Grammar each >= 75)',
                'completedTopicId': str(topic.id),
                'unlockedTopicId': None,
            }
            serializer = CompleteTopicResponseSerializer(resp)
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST)

        if not progress.completed:
            progress.completed = True
            progress.completed_at = timezone.now()
            progress.save(update_fields=['completed', 'completed_at'])
            
            # Check for proficiency tier change when topic is manually completed
            _check_and_create_proficiency_event(request.user)

        # Determine next topic to unlock
        next_topic = (
            Topic.objects.filter(is_active=True, sequence__gt=topic.sequence)
            .order_by('sequence')
            .first()
        )
        resp = {
            'success': True,
            'message': 'Topic marked as completed',
            'completedTopicId': str(topic.id),
            'unlockedTopicId': str(next_topic.id) if next_topic else None,
        }
        serializer = CompleteTopicResponseSerializer(resp)
        return Response(serializer.data, status=status.HTTP_200_OK)


class UserPhraseRecordingsView(APIView):
    """List user's recordings for a topic, optionally filtered by phraseIndex"""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, topic_id):
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        qs = UserPhraseRecording.objects.filter(user=request.user, topic=topic).order_by('-created_at')

        phrase_index = request.query_params.get('phraseIndex')
        if phrase_index is not None:
            try:
                idx = int(phrase_index)
                qs = qs.filter(phrase_index=idx)
            except (TypeError, ValueError):
                return Response({'detail': 'Invalid phraseIndex'}, status=status.HTTP_400_BAD_REQUEST)

        serializer = UserPhraseRecordingSerializer(qs, many=True, context={'request': request})
        data = {'recordings': serializer.data}
        return Response(data, status=status.HTTP_200_OK)


class SubmitFluencyRecordingView(APIView):
    """Submit a fluency recording for comprehensive Gemini-based evaluation.

    Request body (multipart/form-data):
      - audio: audio file (m4a/wav)
      - promptIndex: 0 (for single prompt)
      - recordingDuration: actual seconds spoken (for timing scoring)
    Response JSON body:
      { "success": true, "score": 85, "feedback": "...", "suggestions": [...], "sessionId": "uuid" }
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, topic_id):
        logger.info(f"🔥 NEW GEMINI ENDPOINT CALLED - SubmitFluencyRecordingView for topic {topic_id} by user {request.user}")
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        prompt = topic.fluency_practice_prompt or ''
        if not prompt:
            return Response({'detail': 'No fluency prompt configured for this topic'}, status=status.HTTP_400_BAD_REQUEST)

        # Get form data
        audio_file = request.FILES.get('audio')
        prompt_index = request.data.get('promptIndex', 0)
        recording_duration = request.data.get('recordingDuration', 0)

        if not audio_file:
            return Response({'detail': 'Audio file is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            prompt_index = int(prompt_index)
            recording_duration = float(recording_duration)
        except (TypeError, ValueError):
            return Response({'detail': 'Invalid promptIndex or recordingDuration'}, status=status.HTTP_400_BAD_REQUEST)

        if prompt_index != 0:  # Only allow index 0 for single prompt
            return Response({'detail': 'Invalid promptIndex - only 0 allowed for single prompt'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Use the existing transcription and evaluation infrastructure
            from ..ai_evaluation.services import WhisperService, LLMEvaluationService
            import asyncio
            import tempfile
            import os

            # Save audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_file:
                for chunk in audio_file.chunks():
                    temp_file.write(chunk)
                temp_path = temp_file.name

            try:
                # Read audio file content for transcription
                with open(temp_path, 'rb') as f:
                    audio_bytes = f.read()
                
                # Transcribe audio
                whisper_service = WhisperService()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Use faster-whisper specifically for Fluency Practice without changing env
                    transcription = loop.run_until_complete(
                        whisper_service.transcribe_audio(
                            audio_bytes,
                            prefer_faster_whisper=True,
                            disable_openai_api=True,  # avoid hosted API so FW runs first
                            disable_faster_whisper=False,  # ensure FW is allowed even if env disabled
                        )
                    )
                finally:
                    loop.close()

                # Normalize transcription to string
                transcribed_text = (
                    transcription.get("text", "") if isinstance(transcription, dict) else str(transcription or "")
                )

                # Calculate comprehensive score using Gemini
                score, feedback, suggestions = self._evaluate_fluency_with_gemini(
                    prompt=prompt,
                    transcription=transcribed_text,
                    recording_duration=recording_duration,
                    target_duration=30.0  # 30 seconds target
                )

                # Create a session ID for tracking
                session_id = str(uuid.uuid4())

                # Also update the topic progress with the Gemini score
                try:
                    tp, _ = TopicProgress.objects.get_or_create(user=request.user, topic=topic)
                    
                    # Persist per-prompt score list (index 0 for single-prompt flow)
                    scores = list(tp.fluency_prompt_scores or [])
                    if prompt_index is None:
                        prompt_index = 0
                    # Ensure list capacity
                    while len(scores) <= int(prompt_index):
                        scores.append(None)
                    scores[int(prompt_index)] = int(score)
                    tp.fluency_prompt_scores = scores

                    # Update total (for single prompt equals the first score)
                    tp.fluency_total_score = int(score)

                    # Mark fluency practice as completed once a score exists (threshold handled by meetsRequirement)
                    tp.fluency_completed = True
                    
                    tp.save(update_fields=['fluency_prompt_scores', 'fluency_total_score', 'fluency_completed'])
                    logger.info(f"✅ Updated TopicProgress: fluency_total_score={tp.fluency_total_score}, fluency_completed={tp.fluency_completed}, prompt_scores={tp.fluency_prompt_scores}")
                
                except Exception as e:
                    logger.error(f"Failed to update TopicProgress: {e}")

                return Response({
                    'success': True,
                    'score': score,
                    'feedback': feedback,
                    'suggestions': suggestions,
                    'sessionId': session_id,
                    'transcription': transcribed_text
                }, status=status.HTTP_200_OK)

            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Fluency recording evaluation error: {str(e)}")
            return Response(
                {'detail': f'Evaluation failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _evaluate_fluency_with_gemini(self, prompt, transcription, recording_duration, target_duration=30.0):
        """Use Gemini to evaluate fluency based on relevance, fluency metrics, and timing."""
        import google.generativeai as genai
        import os
        import json
        
        api_key = (
            getattr(settings, 'GEMINI_API_KEY', '') or
            getattr(settings, 'GOOGLE_API_KEY', '') or
            os.environ.get('GEMINI_API_KEY', '') or
            os.environ.get('GOOGLE_API_KEY', '')
        )
        
        if not api_key:
            # Fallback to simple scoring
            logger.warning(f"🚨 GEMINI FALLBACK: No API key available, using simple scoring")
            return 80, "Basic evaluation completed.", ["Keep practicing!"]
        
        genai.configure(api_key=api_key)
        
        # Calculate timing score (closer to 30 seconds = better)
        timing_score = self._calculate_timing_score(recording_duration, target_duration)
        
        # Basic fluency metrics
        word_count = len(transcription.split())
        wpm = (word_count / recording_duration * 60) if recording_duration > 0 else 0
        
        # Count potential fluency issues
        pause_indicators = transcription.count('...') + transcription.count('um') + transcription.count('uh')
        stutter_indicators = len([w for w in transcription.split() if '-' in w or w.count(w[0] if w else '') > 2])
        
        prompt_text = f"""You are an encouraging English fluency coach AI. Analyze this speaking performance and provide a supportive score (0-100). Be very generous and focus on what the user did well.

FLUENCY PROMPT: "{prompt}"

USER'S RESPONSE: "{transcription}"

PERFORMANCE METRICS:
- Recording Duration: {recording_duration:.1f} seconds (Target: {target_duration} seconds)
- Word Count: {word_count} words
- Words Per Minute: {wpm:.1f}
- Pause Indicators: {pause_indicators}
- Potential Stutters: {stutter_indicators}
- Timing Score: {timing_score}/25 (based on how close to {target_duration}s target)

EVALUATION CRITERIA (be very lenient and encouraging):
1. RELEVANCE (0-35 points): Award points generously if the response relates to the prompt in ANY way. Even vague relevance deserves 28+ points. Any attempt to address the topic should get 30+ points.
2. FLUENCY (0-40 points): Focus on effort and courage to speak. Any natural speaking attempt deserves 32+ points. Reward attempts at complete sentences. Only unintelligible speech should score below 28.
3. TIMING (0-25 points): Be flexible with duration. Any attempt of 8+ seconds deserves at least 18 points. Reward longer attempts generously.

IMPORTANT: Your goal is to motivate learners and build confidence. Most genuine attempts should score 78-88. Even basic attempts should reach 70-75 easily. Reserve scores below 65 only for extremely poor quality or off-topic responses.

Provide your response in this JSON format:
{{
  "relevance_score": <0-35>,
  "fluency_score": <0-40>, 
  "timing_score": <0-25>,
  "total_score": <0-100>,
  "feedback": "<encouraging, positive feedback highlighting strengths>",
  "suggestions": ["<gentle suggestion 1>", "<gentle suggestion 2>", "<gentle suggestion 3>"]
}}

Be very encouraging and supportive to help build the user's confidence!"""

        try:
            logger.info(f"🤖 GEMINI EVALUATION: Starting evaluation for {word_count} words, {recording_duration:.1f}s duration")

            # Select model: prefer settings/env, otherwise try a list of candidates
            preferred = (
                getattr(settings, 'GEMINI_MODEL', '')
                or os.environ.get('GEMINI_MODEL', '')
            ).strip()
            candidates = [m for m in [
                preferred or None,
                'gemini-2.5-flash',
                'gemini-2.0-flash',
                'gemini-1.5-flash',
                'gemini-1.5-flash-latest',
            ] if m]

            response_text = None
            last_err = None

            # Try new google-genai client first (supports 2.x models)
            use_new_client = False
            try:
                from google import genai as genai_v1
                client = genai_v1.Client(api_key=api_key)
                use_new_client = True
            except Exception:
                use_new_client = False

            if use_new_client:
                for model_name in candidates:
                    try:
                        logger.info(f"Trying Gemini model (google-genai): {model_name}")
                        resp = client.models.generate_content(model=model_name, contents=prompt_text)
                        # Prefer output_text, fallback to text
                        response_text = getattr(resp, 'output_text', None) or getattr(resp, 'text', None)
                        if response_text:
                            logger.info(f"Gemini model succeeded (google-genai): {model_name}")
                            break
                    except Exception as me:
                        last_err = me
                        logger.warning(f"Gemini model failed ({model_name}) via google-genai: {me}")
                        continue

            # Fallback to legacy google-generativeai client
            if not response_text:
                for model_name in candidates:
                    try:
                        logger.info(f"Trying Gemini model (google-generativeai): {model_name}")
                        model = genai.GenerativeModel(model_name)
                        resp = model.generate_content(prompt_text)
                        response_text = getattr(resp, 'text', None)
                        if response_text:
                            logger.info(f"Gemini model succeeded (google-generativeai): {model_name}")
                            break
                    except Exception as me:
                        last_err = me
                        logger.warning(f"Gemini model failed ({model_name}) via google-generativeai: {me}")
                        continue

            if not response_text and last_err is not None:
                raise last_err
            
            # Parse JSON response
            response_text = (response_text or "").strip()
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].strip()
            
            result = json.loads(response_text)
            raw_score = int(result.get('total_score', 75))
            
            # Apply a gentle boost to make scoring more achievable (18% boost)
            # This helps users feel more encouraged and motivated
            boost_percentage = 0.18  # 18% boost (increased from 12%)
            boosted_score = min(100, int(raw_score * (1 + boost_percentage)))
            
            # Ensure minimum score of 70 for genuine attempts (word_count > 5)
            if word_count > 5 and boosted_score < 70:
                boosted_score = 70
            
            logger.info(f"✅ GEMINI SUCCESS: Raw score {raw_score} -> Boosted score {boosted_score}, Relevance: {result.get('relevance_score')}, Fluency: {result.get('fluency_score')}, Timing: {result.get('timing_score')}")
            
            return (
                boosted_score,
                result.get('feedback', 'Good effort! Keep practicing.'),
                result.get('suggestions', ['Keep practicing regularly', 'Focus on speaking clearly', 'Try to speak for the full 30 seconds'])
            )
            
        except Exception as e:
            logger.warning(f"🚨 GEMINI FALLBACK: Evaluation failed: {e}")
            # Fallback scoring - very generous to encourage users
            base_score = 72  # Increased from 70
            if word_count > 5:  # Basic relevance check
                base_score += 18  # Increased from 15
            if word_count > 20:  # Extra bonus for longer responses
                base_score += 8  # Increased from 5
            base_score += min(timing_score, 18)  # Add timing bonus (increased cap)
            fallback_score = min(base_score, 100)
            logger.info(f"⚠️ FALLBACK SCORE: {fallback_score} (base={base_score}, words={word_count}, timing={timing_score})")
            
            return (
                fallback_score,
                "Evaluation completed. Keep practicing to improve your fluency!",
                ["Try to speak more clearly", "Aim for about 30 seconds", "Stay focused on the topic"]
            )
    
    def _calculate_timing_score(self, actual_duration, target_duration):
        """Calculate timing score based on how close to target duration. More forgiving version."""
        if actual_duration <= 0:
            return 0
        
        # Perfect score at exactly target duration
        # More forgiving score as we get further from target
        difference = abs(actual_duration - target_duration)
        
        if difference == 0:
            return 25
        elif difference <= 5:
            return 24  # Very close (was 22)
        elif difference <= 10:
            return 22  # Close enough (was 18)
        elif difference <= 15:
            return 19  # Acceptable (was 12)
        elif difference <= 20:
            return 16  # Still reasonable (was 8)
        elif difference <= 30:
            return 12  # Give credit for effort (new tier)
        else:
            return 8  # Minimum score increased (was 5)


class SubmitFluencyPromptView(APIView):
    """Submit a fluency prompt completion with a score, enforcing sequential unlocking.

    Request JSON body:
      { "promptIndex": 0, "score": 78, "sessionId": "optional" }
    Response JSON body:
      { "success": true, "nextPromptIndex": 1, "fluencyTotalScore": 78, "fluencyCompleted": false, "promptScores": [78] }
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, topic_id):
        logger.info(f"⚠️ OLD ENDPOINT CALLED - SubmitFluencyPromptView for topic {topic_id} by user {request.user}")
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        prompt = topic.fluency_practice_prompt or ''
        if not prompt:
            return Response({'detail': 'No fluency prompt configured for this topic'}, status=status.HTTP_400_BAD_REQUEST)

        # Validate request body
        serializer = SubmitFluencyPromptRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        data = serializer.validated_data
        prompt_index: int = int(data.get('promptIndex'))
        score: int = int(data.get('score'))

        if prompt_index != 0:  # Only allow index 0 for single prompt
            return Response({'detail': 'Invalid promptIndex - only 0 allowed for single prompt'}, status=status.HTTP_400_BAD_REQUEST)

        # Load or init progress
        tp, _ = TopicProgress.objects.get_or_create(user=request.user, topic=topic)
        was_completed = bool(tp.completed)
        scores = list(tp.fluency_prompt_scores or [])

        # For single prompt, just check if it's already completed
        already_completed = len(scores) > 0 and isinstance(scores[0], int)

        # For single prompt, allow resubmission to improve score
        if already_completed:
            # Allow resubmission but just update the score
            pass

        # Persist new score (ensure scores list has at least one element)
        if len(scores) == 0:
            scores = [int(score)]
        else:
            scores[0] = int(score)

        # For single prompt, total score is just the single score
        tp.fluency_prompt_scores = scores
        tp.fluency_total_score = int(score)
        tp.fluency_completed = True  # Single prompt is always completed once score is set
        # If all modes completed, mark topic completed
        was_completed = bool(tp.completed)
        if tp.all_modes_completed and not tp.completed:
            tp.completed = True
            tp.completed_at = timezone.now()
        tp.save()
        
        # Check for proficiency tier change when topic is completed
        if not was_completed and tp.completed:
            _check_and_create_proficiency_event(request.user)
        
        # Log TOPIC_COMPLETED once when it flips
        try:
            if not was_completed and tp.completed:
                try:
                    AchievementEvent.objects.create(
                        user=request.user,
                        event_type='TOPIC_COMPLETED',
                        title=f"Completed '{getattr(topic, 'title', '')}'",
                        description=f"You completed the topic '{getattr(topic, 'title', '')}'",
                        meta={'topicId': str(topic.id), 'practice': 'fluency'}
                    )
                except Exception:
                    pass
        except Exception:
            pass
        # Note: Coach cache refresh removed to avoid timeout on slow Gemini calls.
        # Coach analysis has its own endpoint (CoachAnalysisView) with proper caching.

        # For single prompt, next is always None (no more prompts after the first)
        new_next = None

        # XP awarding logic (Option A)
        xp_awarded = 0
        # +10 for this prompt if score >= 80
        if score >= 80:
            xp_awarded += _award_xp(
                user=request.user,
                amount=10,
                source='fluency',
                context={
                    'topicId': str(topic.id),
                    'promptIndex': prompt_index,
                    'score': int(score),
                    'type': 'prompt'
                }
            )

        # Bonus +50 if all prompts are now completed
        if new_next is None:
            xp_awarded += _award_xp(
                user=request.user,
                amount=50,
                source='fluency',
                context={
                    'topicId': str(topic.id),
                    'bonus': 'complete_all_prompts',
                    'promptScores': [int(s) for s in scores if isinstance(s, int)]
                }
            )
            # If all prompts complete contributes to topic completion, award mastery once
            try:
                tp.refresh_from_db()
                if tp.completed:
                    xp_awarded += _award_topic_mastery_once(request.user, topic)
            except Exception:
                pass

        resp = {
            'success': True,
            'nextPromptIndex': new_next,
            'fluencyTotalScore': tp.fluency_total_score,
            'fluencyCompleted': tp.fluency_completed,
            'promptScores': [int(s) for s in scores if isinstance(s, int)],
            'xpAwarded': xp_awarded,
        }
        out = SubmitFluencyPromptResponseSerializer(resp)
        return Response(out.data, status=status.HTTP_200_OK)


class GenerateTTSView(APIView):
    """Generate TTS audio via Gemini and return a temporary WAV URL.

    Request JSON body:
      { "text": "Hello world", "voiceName": "Kore" }
    Response JSON body:
      { "audioUrl": "https://.../media/speaking_journey/tts/<id>.wav", "sampleRate": 24000, "voiceName": "Kore" }
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        text = request.data.get('text')
        voice_name = request.data.get('voiceName') or 'Kore'

        if not text or not isinstance(text, str) or not text.strip():
            return Response({'detail': 'Missing or invalid "text"'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Configuration and deterministic cache
            sample_rate = 24000
            text_norm = ' '.join(text.split())

            # API key
            api_key = (
                getattr(settings, 'GEMINI_API_KEY', '') or
                getattr(settings, 'GOOGLE_API_KEY', '') or
                os.environ.get('GEMINI_API_KEY', '') or
                os.environ.get('GOOGLE_API_KEY', '')
            )
            if not api_key:
                logger.error('Server misconfiguration: GEMINI_API_KEY/GOOGLE_API_KEY not set')
                return Response({'detail': 'Server misconfiguration: GEMINI_API_KEY/GOOGLE_API_KEY not set'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Candidate models: prefer override, then stable TTS-capable models
            preferred_model = (
                getattr(settings, 'GEMINI_TTS_MODEL', '') or
                os.environ.get('GEMINI_TTS_MODEL', '') or
                'gemini-2.5-flash-preview-tts'
            )
            candidate_models = [
                preferred_model,
                'gemini-2.5-flash-tts',
                'gemini-2.5-flash',
                'gemini-2.5-flash-latest',
                'gemini-1.5-flash',
            ]
            # de-duplicate while preserving order
            seen = set()
            models = []
            for m in candidate_models:
                if m and m not in seen:
                    seen.add(m)
                    models.append(m)

            # 1) Return from cache if any prior model variant already generated this text
            for m in models:
                key_str = f"{m}|voice={voice_name}|rate={sample_rate}|text={text_norm}"
                cache_hash = hashlib.sha256(key_str.encode('utf-8')).hexdigest()
                relative_path = f"speaking_journey/tts/{cache_hash[:2]}/{cache_hash}.wav"
                if default_storage.exists(relative_path):
                    try:
                        file_url = request.build_absolute_uri(default_storage.url(relative_path))
                    except Exception:
                        base_url = request.build_absolute_uri(getattr(settings, 'MEDIA_URL', '/media/'))
                        file_url = base_url.rstrip('/') + '/' + relative_path.lstrip('/')
                    return Response(
                        {
                            'audioUrl': file_url,
                            'sampleRate': sample_rate,
                            'voiceName': voice_name,
                            'cached': True,
                        },
                        status=status.HTTP_200_OK
                    )

            last_status = None
            last_body = None

            # 2) Try each candidate model until one succeeds
            for m in models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
                payload = {
                    'model': m,
                    'contents': [{
                        'role': 'user',
                        'parts': [{ 'text': text.strip() }]
                    }],
                    'generationConfig': {
                        'responseModalities': ['AUDIO'],
                        'speechConfig': {
                            'voiceConfig': {
                                'prebuiltVoiceConfig': { 'voiceName': voice_name }
                            }
                        }
                    }
                }
                headers = {
                    'x-goog-api-key': api_key,
                    'Content-Type': 'application/json'
                }

                resp = requests.post(url, json=payload, headers=headers, timeout=30)
                if resp.status_code != 200:
                    last_status = resp.status_code
                    last_body = resp.text
                    # Include status/body snippet in log message for easier Railway debugging
                    snippet = (resp.text or '')[:300]
                    try:
                        logger.error('Gemini TTS request failed: status=%s model=%s body=%s', resp.status_code, m, snippet)
                    except Exception:
                        logger.error('Gemini TTS request failed for model=%s (status=%s)', m, resp.status_code)
                    continue

                data = resp.json()
                # Extract audio inlineData from parts
                b64 = None
                try:
                    parts = data['candidates'][0]['content']['parts']
                    for p in parts:
                        if isinstance(p, dict) and 'inlineData' in p and 'data' in p['inlineData']:
                            b64 = p['inlineData']['data']
                            break
                except Exception:
                    b64 = None
                if not b64:
                    # Legacy single-part shape fallback
                    try:
                        b64 = data['candidates'][0]['content']['parts'][0]['inlineData']['data']
                    except Exception:
                        try:
                            snippet = json.dumps(data)[:300]
                        except Exception:
                            snippet = str(data)[:300]
                        logger.error('Gemini TTS response missing inlineData for model=%s: %s', m, snippet)
                        last_status = 502
                        last_body = snippet
                        continue

                # Decode PCM (s16le, 24kHz, mono) to WAV
                pcm_bytes = base64.b64decode(b64)
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit PCM
                    wf.setframerate(sample_rate)
                    wf.writeframes(pcm_bytes)
                wav_bytes = buf.getvalue()

                # Persist to media storage under deterministic cache path tied to the working model
                key_str = f"{m}|voice={voice_name}|rate={sample_rate}|text={text_norm}"
                cache_hash = hashlib.sha256(key_str.encode('utf-8')).hexdigest()
                relative_path = f"speaking_journey/tts/{cache_hash[:2]}/{cache_hash}.wav"
                saved_path = default_storage.save(relative_path, ContentFile(wav_bytes))
                try:
                    file_url = request.build_absolute_uri(default_storage.url(saved_path))
                except Exception:
                    # Fallback to constructing from MEDIA_URL
                    base_url = request.build_absolute_uri(getattr(settings, 'MEDIA_URL', '/media/'))
                    file_url = base_url.rstrip('/') + '/' + saved_path.lstrip('/')

                return Response(
                    {
                        'audioUrl': file_url,
                        'sampleRate': sample_rate,
                        'voiceName': voice_name,
                        'cached': False,
                    },
                    status=status.HTTP_200_OK
                )

            # All candidates failed
            if last_status == 429:
                # Rate limit: surface as 503 to encourage client retry/backoff
                return Response(
                    {'detail': 'Gemini TTS rate limited', 'status': last_status, 'body': (last_body or '')[:800]},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            return Response(
                {'detail': 'Gemini TTS request failed', 'status': last_status, 'body': (last_body or '')[:800]},
                status=status.HTTP_502_BAD_GATEWAY
            )

        except Exception as e:
            logger.exception('TTS generation failed')
            return Response({'detail': 'TTS generation failed', 'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class StartVocabularyPracticeView(APIView):
    """Start a vocabulary practice session for a topic.

    Computes ~60% of topic vocabulary as questions.
    Response: { sessionId, totalQuestions, questions: [{id, definition, options}] }
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, topic_id):
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        vocab = topic.vocabulary or []
        total = len([w for w in vocab if isinstance(w, str) and w.strip()])
        if total < 1:
            return Response({'detail': 'No vocabulary available for this topic'}, status=status.HTTP_400_BAD_REQUEST)
        # 60% rule
        import math
        q_count = max(1, int(round(0.6 * total)))

        # Create session and generate questions
        # Determine english level from USER profile first, then topic difficulty
        try:
            user_level = str(getattr(getattr(request.user, 'speaking_journey_profile', None), 'english_level', '') or '').strip().upper()
        except Exception:
            user_level = ''
        if not user_level:
            try:
                user_level = str(getattr(topic, 'difficulty', '') or '').strip().upper()
            except Exception:
                user_level = ''
        eff_level = user_level if user_level in {'BEGINNER', 'INTERMEDIATE', 'ADVANCED'} else 'INTERMEDIATE'
        lang = 'id' if eff_level == 'BEGINNER' else 'en'
        questions = _sample_vocabulary_questions(topic, q_count, lang=lang, level=eff_level)
        session = VocabularyPracticeSession.objects.create(
            user=request.user,
            topic=topic,
            questions=questions,
            total_questions=len(questions),
            current_index=0,
            correct_count=0,
            total_score=0,
            completed=False,
        )

        payload = {
            'sessionId': str(session.session_id),
            'totalQuestions': session.total_questions,
            'questions': [
                {
                    'id': q.get('id'),
                    'definition': q.get('definition') or '',
                    'options': q.get('options') or [],
                }
                for q in session.questions
            ],
        }
        out = StartVocabularyPracticeResponseSerializer(payload)
        return Response(out.data, status=status.HTTP_200_OK)


class SubmitVocabularyAnswerView(APIView):
    """Submit an answer for a vocabulary question.

    Request: { sessionId, questionId, selected }
    Response: { correct, xpAwarded, nextIndex, completed, totalScore }
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, topic_id):
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        ser = SubmitVocabularyAnswerRequestSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        data = ser.validated_data
        session_id = data.get('sessionId')
        question_id = data.get('questionId')
        selected = (data.get('selected') or '').strip()

        session = get_object_or_404(VocabularyPracticeSession, session_id=session_id, user=request.user, topic=topic)
        if session.completed:
            # Already done
            next_idx = None
            return Response({'correct': False, 'xpAwarded': 0, 'nextIndex': next_idx, 'completed': True, 'totalScore': session.total_score}, status=status.HTTP_200_OK)

        # Find question
        questions = list(session.questions or [])
        idx = next((i for i, q in enumerate(questions) if str(q.get('id')) == str(question_id)), None)
        if idx is None:
            return Response({'detail': 'Invalid questionId'}, status=status.HTTP_400_BAD_REQUEST)
        q = questions[idx]
        if q.get('answered'):
            # Idempotent: do not double-award
            # Move to next unanswered index
            next_index = None
            for j, item in enumerate(questions):
                if not item.get('answered'):
                    next_index = j
                    break
            return Response({'correct': bool(q.get('correct')), 'xpAwarded': 0, 'nextIndex': next_index, 'completed': next_index is None, 'totalScore': session.total_score}, status=status.HTTP_200_OK)

        correct_word = q.get('word')
        is_correct = (selected == correct_word)

        # Update question state
        q['answered'] = True
        q['correct'] = bool(is_correct)
        questions[idx] = q

        xp_awarded = 0
        if is_correct:
            session.correct_count = int(session.correct_count or 0) + 1
            # Award +5 XP for correct answer (Option A)
            xp_awarded = _award_xp(
                user=request.user,
                amount=5,
                source='vocabulary',
                context={'topicId': str(topic.id), 'questionId': str(question_id), 'type': 'answer'}
            )

        # Update session
        session.questions = questions
        # Move to next unanswered question index
        next_index = None
        for j, item in enumerate(questions):
            if not item.get('answered'):
                next_index = j
                break
        session.current_index = next_index if next_index is not None else len(questions)
        # Mark completed if no more
        session.completed = next_index is None
        # Recompute total score as percentage (0â€“100)
        try:
            total = int(session.total_questions or 0)
            corr = int(session.correct_count or 0)
            session.total_score = int(round((corr / total) * 100.0)) if total > 0 else 0
        except Exception:
            pass
        session.save(update_fields=['questions', 'correct_count', 'total_score', 'current_index', 'completed', 'updated_at'])

        resp = {
            'correct': is_correct,
            'xpAwarded': xp_awarded,
            'nextIndex': next_index,
            'completed': session.completed,
            'totalScore': int(session.total_score or 0),
        }
        out = SubmitVocabularyAnswerResponseSerializer(resp)
        return Response(out.data, status=status.HTTP_200_OK)


class CompleteVocabularyPracticeView(APIView):
    """Complete the vocabulary session, persist topic progress and award completion XP.

    Request: { sessionId }
    Response: totals and XP, plus whether the topic is now completed.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, topic_id):
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        ser = CompleteVocabularyPracticeRequestSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        session_id = ser.validated_data.get('sessionId')
        session = get_object_or_404(VocabularyPracticeSession, session_id=session_id, user=request.user, topic=topic)

        # If some questions unanswered, mark as completed but do not award bonus XP (still persist score)
        all_answered = all(bool(q.get('answered')) for q in (session.questions or [])) if session.questions else True
        xp_awarded = 0
        # Persist to TopicProgress
        tp, _ = TopicProgress.objects.get_or_create(user=request.user, topic=topic)
        tp.vocabulary_total_score = int(session.total_score or 0)
        tp.vocabulary_completed = True
        # Mark topic completed if all modes completed
        was_completed = bool(tp.completed)
        if not tp.completed and tp.all_modes_completed:
            tp.completed = True
            tp.completed_at = timezone.now()
        tp.save()
        
        # Check for proficiency tier change when topic is completed
        if not was_completed and tp.completed:
            _check_and_create_proficiency_event(request.user)
        
        # Note: Coach cache refresh removed to avoid timeout on slow Gemini calls.
        # Coach analysis has its own endpoint (CoachAnalysisView) with proper caching.

        # Award +20 XP for completion of this practice (Option A)
        xp_awarded += _award_xp(
            user=request.user,
            amount=20,
            source='vocabulary',
            context={'topicId': str(topic.id), 'type': 'complete', 'allAnswered': all_answered}
        )

        # If topic became completed due to this practice, award mastery once
        try:
            if tp.completed:
                xp_awarded += _award_topic_mastery_once(request.user, topic)
        except Exception:
            pass

        session.completed = True
        session.save(update_fields=['completed', 'updated_at'])

        payload = {
            'success': True,
            'totalQuestions': int(session.total_questions or 0),
            'correctCount': int(session.correct_count or 0),
            'totalScore': int(session.total_score or 0),
            'xpAwarded': xp_awarded,
            'vocabularyTotalScore': int(tp.vocabulary_total_score or 0),
            'topicCompleted': bool(tp.completed),
        }
        out = CompleteVocabularyPracticeResponseSerializer(payload)
        return Response(out.data, status=status.HTTP_200_OK)


class DebugTopicStatusView(APIView):
    """Debug endpoint to see exact completion status and scores for a topic"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request, topic_id):
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        tp, _ = TopicProgress.objects.get_or_create(user=request.user, topic=topic)
        
        # Check PhraseProgress
        try:
            pp = PhraseProgress.objects.get(user=request.user, topic=topic)
            phrase_data = {
                'current_phrase_index': pp.current_phrase_index,
                'completed_phrases': pp.completed_phrases or [],
                'total_phrases': len(topic.material_lines or []),
                'is_all_phrases_completed': pp.is_all_phrases_completed
            }
        except PhraseProgress.DoesNotExist:
            phrase_data = {'error': 'No PhraseProgress found'}
        
        # Check VocabularyPracticeSession
        try:
            vocab_sessions = list(VocabularyPracticeSession.objects.filter(
                user=request.user, topic=topic
            ).values('completed', 'total_score', 'created_at'))
        except Exception:
            vocab_sessions = []
            
        # Check fluency prompts vs scores
        fprompts = topic.fluency_practice_prompt or []
        fscores = tp.fluency_prompt_scores or []
        
        # Compute strict all_prompts_scored: require full length and all ints
        all_prompts_scored = False
        try:
            all_prompts_scored = bool(len(fprompts) > 0 and len(fscores) >= len(fprompts) and all(isinstance(fscores[i], int) for i in range(len(fprompts))))
        except Exception:
            all_prompts_scored = False

        # Pronunciation recompute details for debugging
        pron_details = {}
        try:
            qs = UserPhraseRecording.objects.filter(user=request.user, topic=topic).order_by('phrase_index', '-created_at')
            latest_by_phrase = {}
            for r in qs:
                if r.phrase_index not in latest_by_phrase:
                    latest_by_phrase[r.phrase_index] = r
            latest_map = {int(k): int(round(float(v.accuracy or 0.0))) for k, v in latest_by_phrase.items()}
            rec_cnt = len(latest_map)
            recomputed_avg = int(round(sum(latest_map.values()) / rec_cnt)) if rec_cnt > 0 else 0
            pron_details = {
                'recordings_count': len(list(qs)),
                'latest_per_phrase_accuracies': latest_map,
                'recomputed_avg': recomputed_avg,
            }
        except Exception:
            pron_details = {}

        debug_data = {
            'topic_id': str(topic.id),
            'topic_title': topic.title,
            'topic_progress_flags': {
                'pronunciation_completed': tp.pronunciation_completed,
                'fluency_completed': tp.fluency_completed,
                'vocabulary_completed': tp.vocabulary_completed,
                'listening_completed': tp.listening_completed,
                'grammar_completed': tp.grammar_completed,
                'completed': tp.completed
            },
            'scores': {
                'pronunciation_total_score': tp.pronunciation_total_score,
                'fluency_total_score': tp.fluency_total_score,
                'vocabulary_total_score': tp.vocabulary_total_score
            },
            'phrase_progress': phrase_data,
            'fluency_details': {
                'prompts_count': len(fprompts),
                'scores_count': len([s for s in fscores if isinstance(s, int)]),
                'fluency_prompt_scores': fscores,
                'all_prompts_scored': all_prompts_scored
            },
            'pronunciation_details': pron_details,
            'vocabulary_sessions': vocab_sessions,
            'meets_completion_criteria': _meets_completion_criteria(tp),
            'all_modes_completed': tp.all_modes_completed
        }
        
        return Response(debug_data, status=status.HTTP_200_OK)


class SeedPerfectScoresView(APIView):
    """Seed perfect scores and completion flags for testing unlock mechanism"""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, topic_id):
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        tp, _ = TopicProgress.objects.get_or_create(user=request.user, topic=topic)
        
        # Set perfect scores
        tp.pronunciation_total_score = 100
        tp.fluency_total_score = 100
        tp.vocabulary_total_score = 100
        
        # Set all completion flags
        tp.pronunciation_completed = True
        tp.fluency_completed = True
        tp.vocabulary_completed = True
        # Listening is an optional bonus practice. Grammar is required.
        tp.listening_completed = True
        tp.grammar_completed = True
        
        # Set fluency prompt scores to perfect
        fprompts = topic.fluency_practice_prompt or []
        if fprompts:
            tp.fluency_prompt_scores = [100] * len(fprompts)
            
        # Mark topic as completed if all modes are now completed
        was_completed = bool(tp.completed)
        if tp.all_modes_completed and not tp.completed:
            tp.completed = True
            tp.completed_at = timezone.now()
            
        tp.save()
        
        # Check for proficiency tier change when topic is completed
        if not was_completed and tp.completed:
            _check_and_create_proficiency_event(request.user)
        # Log TOPIC_COMPLETED once when it flips
        try:
            if not was_completed and tp.completed:
                try:
                    AchievementEvent.objects.create(
                        user=request.user,
                        event_type='TOPIC_COMPLETED',
                        title=f"Completed '{getattr(topic, 'title', '')}'",
                        description=f"You completed the topic '{getattr(topic, 'title', '')}'",
                        meta={'topicId': str(topic.id), 'practice': 'conversation'}
                    )
                except Exception:
                    pass
        except Exception:
            pass
        
        # Ensure phrase progress shows all completed
        try:
            pp, _ = PhraseProgress.objects.get_or_create(
                user=request.user,
                topic=topic,
                defaults={'current_phrase_index': 0, 'completed_phrases': []}
            )
            total_phrases = len(topic.material_lines or [])
            if total_phrases > 0:
                pp.completed_phrases = list(range(total_phrases))
                pp.current_phrase_index = total_phrases
                pp.save()
        except Exception as e:
            pass
            
        # Create a completed vocabulary session if none exists
        try:
            if not VocabularyPracticeSession.objects.filter(
                user=request.user, topic=topic, completed=True
            ).exists():
                VocabularyPracticeSession.objects.create(
                    user=request.user,
                    topic=topic,
                    total_questions=5,
                    correct_count=5,
                    total_score=100,
                    completed=True
                )
        except Exception:
            pass
        
        return Response({
            'success': True,
            'message': f'Seeded perfect scores for topic: {topic.title}',
            'topic_id': str(topic.id),
            'meets_criteria_after_seed': _meets_completion_criteria(tp)
        }, status=status.HTTP_200_OK)


class RecomputeTopicAggregatesView(APIView):
    """Recompute and persist aggregate scores for a topic from existing data.

    - Pronunciation: average of latest accuracy per phrase (0â€“100)
    - Fluency: average of per-prompt scores (ints), completed if all prompts have scores
    - Vocabulary: latest completed session's total_score
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, topic_id):
        topic = get_object_or_404(Topic, id=topic_id, is_active=True)
        tp, _ = TopicProgress.objects.get_or_create(user=request.user, topic=topic)

        # Pronunciation aggregate from latest per-phrase recordings
        pron_total = 0
        try:
            qs = (
                UserPhraseRecording.objects
                .filter(user=request.user, topic=topic)
                .order_by('phrase_index', '-created_at')
            )
            latest_by_phrase = {}
            for r in qs:
                if r.phrase_index not in latest_by_phrase:
                    latest_by_phrase[r.phrase_index] = r
            total_acc = 0
            cnt = 0
            for r in latest_by_phrase.values():
                try:
                    total_acc += int(round(float(r.accuracy or 0.0)))
                    cnt += 1
                except Exception:
                    pass
            pron_total = int(round(total_acc / cnt)) if cnt > 0 else 0
        except Exception:
            pron_total = 0

        # Fluency aggregate from stored per-prompt scores
        flu_scores = []
        try:
            scores = list(tp.fluency_prompt_scores or [])
            for s in scores:
                if isinstance(s, int):
                    flu_scores.append(int(s))
        except Exception:
            pass
        flu_total = int(round(sum(flu_scores) / len(flu_scores))) if flu_scores else 0

        # Vocabulary aggregate from latest completed session
        vocab_total = 0
        try:
            last_v = (
                VocabularyPracticeSession.objects
                .filter(user=request.user, topic=topic, completed=True)
                .order_by('-updated_at', '-created_at')
                .first()
            )
            if last_v:
                vocab_total = int(last_v.total_score or 0)
        except Exception:
            vocab_total = int(tp.vocabulary_total_score or 0)

        # Persist aggregates
        tp.pronunciation_total_score = pron_total
        tp.fluency_total_score = flu_total
        tp.vocabulary_total_score = vocab_total

        # Sync completion flags from data
        try:
            pp = PhraseProgress.objects.filter(user=request.user, topic=topic).first()
            if pp and pp.is_all_phrases_completed:
                tp.pronunciation_completed = True
        except Exception:
            pass
        try:
            prompts = list(topic.fluency_practice_prompt or [])
            tp.fluency_completed = bool(prompts) and all(
                (i < len(tp.fluency_prompt_scores or [])) and isinstance((tp.fluency_prompt_scores or [None])[i], int)
                for i in range(len(prompts))
            )
        except Exception:
            pass
        try:
            tp.vocabulary_completed = VocabularyPracticeSession.objects.filter(user=request.user, topic=topic, completed=True).exists()
        except Exception:
            pass

        # Mark topic completed if all modes completed
        was_completed = bool(tp.completed)
        if not tp.completed and tp.all_modes_completed:
            tp.completed = True
            tp.completed_at = timezone.now()
        tp.save()
        # Check for proficiency tier change when topic is completed
        if not was_completed and tp.completed:
            _check_and_create_proficiency_event(request.user)
        # Log TOPIC_COMPLETED once when it flips
        try:
            if not was_completed and tp.completed:
                try:
                    AchievementEvent.objects.create(
                        user=request.user,
                        event_type='TOPIC_COMPLETED',
                        title=f"Completed '{getattr(topic, 'title', '')}'",
                        description=f"You completed the topic '{getattr(topic, 'title', '')}'",
                        meta={'topicId': str(topic.id), 'practice': 'unknown'}
                    )
                except Exception:
                    pass
        except Exception:
            pass

        data = {
            'success': True,
            'topicId': str(topic.id),
            'title': topic.title,
            'aggregates': {
                'pronunciation': int(tp.pronunciation_total_score or 0),
                'fluency': int(tp.fluency_total_score or 0),
                'vocabulary': int(tp.vocabulary_total_score or 0),
            },
            'flags': {
                'pronunciation_completed': tp.pronunciation_completed,
                'fluency_completed': tp.fluency_completed,
                'vocabulary_completed': tp.vocabulary_completed,
                'listening_completed': tp.listening_completed,
                'grammar_completed': tp.grammar_completed,
                'completed': tp.completed,
            },
            'meetsCompletionCriteria': _meets_completion_criteria(tp),
        }
        return Response(data, status=status.HTTP_200_OK)

class SpeakingActivitiesView(APIView):
    """Return a consolidated list of speaking journey activities for the current user.

    Includes:
    - Topic completions (TopicProgress.completed_at)
    - Vocabulary sessions (VocabularyPracticeSession updated_at; completed vs in-progress)
    - Conversation practice recordings (UserConversationRecording created_at)
    - Pronunciation recordings (UserPhraseRecording created_at)
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        try:
            limit = int(request.query_params.get('limit') or 50)
        except Exception:
            limit = 50

        # Support viewing another user's activity feed when a userId/user_id is provided
        target_user = request.user
        try:
            raw_uid = request.query_params.get('userId') or request.query_params.get('user_id')
            if raw_uid:
                UserModel = get_user_model()
                tu = UserModel.objects.filter(id=raw_uid).first()
                if tu is not None:
                    target_user = tu
        except Exception:
            # Fallback silently to current user if lookup fails
            pass

        user = target_user

        events: list[dict] = []

        # Topic completions
        try:
            tps = (
                TopicProgress.objects.select_related('topic')
                .filter(user=user, completed=True, completed_at__isnull=False)
            )
            for tp in tps:
                try:
                    events.append({
                        'id': f"topic_completed:{tp.topic_id}:{int(tp.completed_at.timestamp())}",
                        'type': 'LESSON_COMPLETED',
                        'title': f"Completed '{tp.topic.title}' topic",
                        'description': 'All required practice modes completed',
                        'timestamp': tp.completed_at,
                        'xpEarned': None,
                    })
                except Exception:
                    continue
        except Exception:
            pass

        # Vocabulary practice sessions (use updated_at; label based on completed)
        try:
            vs = (
                VocabularyPracticeSession.objects.select_related('topic')
                .filter(user=user)
                .order_by('-updated_at')[: max(50, limit)]
            )
            for s in vs:
                try:
                    title = f"Vocabulary Practice - {s.topic.title}"
                    desc = 'Completed session' if s.completed else 'Practiced vocabulary'
                    events.append({
                        'id': f"vocab:{s.session_id}:{int(s.updated_at.timestamp())}",
                        'type': 'PRACTICE_SESSION',
                        'title': title,
                        'description': desc,
                        'timestamp': s.updated_at,
                        'xpEarned': None,
                    })
                except Exception:
                    continue
        except Exception:
            pass

        # Conversation recordings
        try:
            cr = (
                UserConversationRecording.objects.select_related('topic')
                .filter(user=user)
                .order_by('-created_at')[: max(50, limit)]
            )
            for r in cr:
                try:
                    events.append({
                        'id': f"conversation:{r.id}",
                        'type': 'PRACTICE_SESSION',
                        'title': f"Conversation Practice - {r.topic.title}",
                        'description': 'Practiced conversation',
                        'timestamp': r.created_at,
                        'xpEarned': None,
                    })
                except Exception:
                    continue
        except Exception:
            pass

        # Pronunciation recordings
        try:
            pr = (
                UserPhraseRecording.objects.select_related('topic')
                .filter(user=user)
                .order_by('-created_at')[: max(50, limit)]
            )
            for r in pr:
                try:
                    events.append({
                        'id': f"pronunciation:{r.id}",
                        'type': 'PRACTICE_SESSION',
                        'title': f"Pronunciation Practice - {r.topic.title}",
                        'description': 'Practiced pronunciation',
                        'timestamp': r.created_at,
                        'xpEarned': None,
                    })
                except Exception:
                    continue
        except Exception:
            pass

        # Sort and limit
        try:
            events.sort(key=lambda x: x.get('timestamp') or timezone.now(), reverse=True)
        except Exception:
            pass
        events = events[:limit]

        ser = JourneyActivitySerializer(events, many=True)
        return Response(ser.data, status=status.HTTP_200_OK)


class LingoLeagueView(APIView):
    """Provide cross-user rankings based on Speaking Journey data.

    Categories:
      - PRONUNCIATION: average accuracy from UserPhraseRecording.accuracy (0-100)
      - FLUENCY: average of all values in TopicProgress.fluency_prompt_scores (0-100)
      - VOCABULARY: average percent across completed VocabularyPracticeSession (0-100)
      - TOPICS_COMPLETED: count of TopicProgress entries with completed=True

    Returns Android-friendly LeaderboardData-like payload to match existing mobile domain models.
    """
    permission_classes = [permissions.IsAuthenticated]

    def _resolve_avatar_url(self, user, request):
        try:
            profile = getattr(user, 'profile', None)
            if not profile:
                return None
            avatar_field = getattr(profile, 'avatar', None)
            if avatar_field and hasattr(avatar_field, 'url'):
                try:
                    return request.build_absolute_uri(avatar_field.url) if request else avatar_field.url
                except Exception:
                    return getattr(avatar_field, 'url', None)
            legacy = getattr(profile, 'avatar_url', '') or ''
            return legacy or None
        except Exception:
            return None

    def get(self, request):
        User = get_user_model()
        category = str(request.query_params.get('category') or 'PRONUNCIATION').strip().upper()
        try:
            limit = int(request.query_params.get('limit') or 50)
        except Exception:
            limit = 50
        limit = max(1, min(limit, 200))

        # Compute scores per user based on category
        scores: list[tuple[int, float]] = []  # list of (user_id, score)

        if category == 'PRONUNCIATION':
            try:
                qs = (
                    UserPhraseRecording.objects
                    .filter(accuracy__isnull=False)
                    .values('user_id')
                    .annotate(score=Avg('accuracy'))
                    .order_by('-score')
                )
                scores = [(row['user_id'], float(row['score'] or 0.0)) for row in qs]
            except Exception:
                scores = []
        elif category == 'FLUENCY':
            try:
                raw = TopicProgress.objects.values('user_id', 'fluency_prompt_scores')
                agg: dict[int, list[float]] = {}
                for row in raw:
                    uid = int(row['user_id'])
                    arr = row.get('fluency_prompt_scores') or []
                    for v in arr:
                        try:
                            val = float(v)
                        except Exception:
                            continue
                        if uid not in agg:
                            agg[uid] = []
                        agg[uid].append(val)
                temp = []
                for uid, vals in agg.items():
                    if vals:
                        avg = sum(vals) / len(vals)
                        temp.append((uid, avg))
                scores = sorted(temp, key=lambda x: x[1], reverse=True)
            except Exception:
                scores = []
        elif category == 'VOCABULARY':
            try:
                raw = VocabularyPracticeSession.objects.filter(completed=True).values('user_id', 'total_questions', 'total_score')
                sums: dict[int, float] = {}
                counts: dict[int, int] = {}
                for r in raw:
                    uid = int(r['user_id'])
                    tq = int(r.get('total_questions') or 0)
                    ts = int(r.get('total_score') or 0)
                    if tq <= 0:
                        continue
                    max_score = tq * 10
                    perc = max(0.0, min(100.0, (ts / max(1, max_score)) * 100.0))
                    sums[uid] = sums.get(uid, 0.0) + perc
                    counts[uid] = counts.get(uid, 0) + 1
                items = []
                for uid, s in sums.items():
                    c = counts.get(uid, 0)
                    if c > 0:
                        items.append((uid, s / c))
                scores = sorted(items, key=lambda x: x[1], reverse=True)
            except Exception:
                scores = []
        else:  # TOPICS_COMPLETED
            try:
                qs = (
                    TopicProgress.objects.filter(completed=True)
                    .values('user_id')
                    .annotate(score=Count('id'))
                    .order_by('-score')
                )
                scores = [(row['user_id'], float(row['score'] or 0.0)) for row in qs]
                category = 'TOPICS_COMPLETED'
            except Exception:
                scores = []

        # Trim and gather user info
        scores = scores[:limit]
        user_ids = [uid for uid, _ in scores]
        users = {u.id: u for u in User.objects.filter(id__in=user_ids).select_related('profile', 'level_profile')}
        levels = {u.id: getattr(u, 'level_profile', None) for u in users.values()}

        entries = []
        current_user_entry = None
        for rank, (uid, score) in enumerate(scores, 1):
            user = users.get(uid)
            if not user:
                continue
            # Display name prefers full name if available
            try:
                name_val = (user.get_full_name() or '').strip() or user.username
            except Exception:
                name_val = getattr(user, 'username', str(uid))
            lvl = levels.get(uid)
            level_val = int(getattr(lvl, 'current_level', 0) or 0)
            streak_val = int(getattr(lvl, 'streak_days', 0) or 0)
            entry = {
                'rank': rank,
                'userId': str(uid),
                'username': getattr(user, 'username', str(uid)),
                'displayName': name_val,
                'avatarUrl': self._resolve_avatar_url(user, request),
                'score': int(round(score)),
                'level': level_val,
                'streakDays': streak_val,
                'country': None,
                'countryCode': None,
                'isCurrentUser': uid == getattr(request.user, 'id', None),
                'change': 'NONE',
                'achievements': 0,
                'weeklyXp': 0,
                'monthlyXp': 0,
                'badge': None,
            }
            if entry['isCurrentUser']:
                current_user_entry = entry
            entries.append(entry)

        data = {
            'type': 'LINGO_LEAGUE',
            'filter': category,
            'entries': entries,
            'currentUserEntry': current_user_entry,
            'lastUpdated': timezone.now(),
            'totalParticipants': len(entries),
        }
        return Response(data, status=status.HTTP_200_OK)


class TopicLeaderboardView(APIView):
    """Per-topic leaderboard based on combined mastery across required practices
    (Pronunciation, Fluency, Vocabulary, Grammar). Returns Android-friendly
    LeaderboardData-like payload to match the mobile domain model.
    """
    permission_classes = [permissions.IsAuthenticated]

    def _resolve_avatar_url(self, user, request):
        try:
            profile = getattr(user, 'profile', None)
            if not profile:
                return None
            avatar_field = getattr(profile, 'avatar', None)
            if avatar_field and hasattr(avatar_field, 'url'):
                try:
                    return request.build_absolute_uri(avatar_field.url) if request else avatar_field.url
                except Exception:
                    return getattr(avatar_field, 'url', None)
            legacy = getattr(profile, 'avatar_url', '') or ''
            return legacy or None
        except Exception:
            return None

    def get(self, request, topic_id):
        try:
            limit = int(request.query_params.get('limit') or 50)
        except Exception:
            limit = 50
        limit = max(1, min(limit, 200))

        topic = get_object_or_404(Topic, id=topic_id, is_active=True)

        tps = TopicProgress.objects.filter(topic=topic).select_related('user', 'user__profile', 'user__level_profile')

        entries_raw = []
        for tp in tps:
            user = getattr(tp, 'user', None)
            if not user:
                continue
            try:
                pron = int(getattr(tp, 'pronunciation_total_score', 0) or 0)
                flu = int(getattr(tp, 'fluency_total_score', 0) or 0)
                vocab = int(getattr(tp, 'vocabulary_total_score', 0) or 0)
                grammar = int(getattr(tp, 'grammar_total_score', 0) or 0)
                pron = min(max(pron, 0), 100)
                flu = min(max(flu, 0), 100)
                vocab = min(max(vocab, 0), 100)
                grammar = min(max(grammar, 0), 100)
                total_max = 400
                total_score = pron + flu + vocab + grammar
                combined_percent = int(round((total_score / max(1, total_max)) * 100.0))
            except Exception:
                combined_percent = 0

            if combined_percent <= 0:
                continue

            # Display name prefers full name if available
            try:
                display_name = (user.get_full_name() or '').strip() or user.username
            except Exception:
                display_name = getattr(user, 'username', str(getattr(user, 'id', '')))

            # Level and streak
            try:
                level_profile = getattr(user, 'level_profile', None)
                level_val = int(getattr(level_profile, 'current_level', 0) or 0)
                streak_val = int(getattr(level_profile, 'streak_days', 0) or 0)
            except Exception:
                level_val = 0
                streak_val = 0

            entries_raw.append(
                (
                    combined_percent,
                    {
                        'rank': 0,  # temporary
                        'userId': str(user.id),
                        'username': getattr(user, 'username', str(user.id)),
                        'displayName': display_name,
                        'avatarUrl': self._resolve_avatar_url(user, request),
                        'score': combined_percent,
                        'level': level_val,
                        'streakDays': streak_val,
                        'country': None,
                        'countryCode': None,
                        'isCurrentUser': user.id == getattr(request.user, 'id', None),
                        'change': 'NONE',
                        'achievements': 0,
                        'weeklyXp': 0,
                        'monthlyXp': 0,
                        'badge': None,
                    }
                )
            )

        # Sort and rank
        entries_raw.sort(key=lambda x: x[0], reverse=True)
        entries = []
        current_user_entry = None
        for idx, (_, entry) in enumerate(entries_raw[:limit], 1):
            entry['rank'] = idx
            if entry.get('isCurrentUser'):
                current_user_entry = entry
            entries.append(entry)

        data = {
            'type': 'ALL_TIME',
            'filter': 'OVERALL_XP',
            'entries': entries,
            'currentUserEntry': current_user_entry,
            'lastUpdated': timezone.now(),
            'totalParticipants': len(entries_raw),
        }
        return Response(data, status=status.HTTP_200_OK)


# ---------------------------
# AI Coach - LLM-based Adaptive Learning System
# Uses Gemini to simulate GRU-like temporal modeling for personalized recommendations
# ---------------------------

def _get_cached_analysis(user):
    """Retrieve cached analysis from database if exists and not expired."""
    from .models import CoachAnalysisCache
    try:
        cache = CoachAnalysisCache.objects.get(user=user)
        logger.info(f"Coach cache found for user {user.id}: AI={cache.is_ai_generated}, stale={cache.is_stale}")
        return cache
    except CoachAnalysisCache.DoesNotExist:
        logger.info(f"No coach cache found for user {user.id}")
        return None

def _save_cached_analysis(user, data: dict, is_ai: bool = False, ttl_hours: int = 12):
    """Save analysis to database cache with expiration."""
    from .models import CoachAnalysisCache
    expires_at = timezone.now() + timedelta(hours=ttl_hours)
    cache, created = CoachAnalysisCache.objects.update_or_create(
        user=user,
        defaults={
            'analysis_data': data,
            'expires_at': expires_at,
            'is_ai_generated': is_ai,
        }
    )
    logger.info(f"Coach cache {'created' if created else 'updated'} for user {user.id}: AI={is_ai}, expires={expires_at}")
    return cache


# Lightweight background enqueue (deduped via Django cache)
_BG_REFRESH_KEY = "coach_refresh:{user_id}"

def _is_refresh_in_progress(user_id) -> bool:
    try:
        return bool(cache.get(_BG_REFRESH_KEY.format(user_id=user_id)))
    except Exception:
        return False

def _enqueue_coach_refresh(user_id: int) -> bool:
    """Start a background thread to refresh coach analysis if not already running.
    Returns True if a refresh was enqueued, False if one is already in progress or enqueue failed.
    """
    key = _BG_REFRESH_KEY.format(user_id=user_id)
    try:
        # cache.add is atomic: only first caller within TTL succeeds
        if not cache.add(key, True, timeout=300):
            logger.info(f"Coach background refresh already in progress for user {user_id}")
            return False
    except Exception:
        # If cache backend is unavailable, still proceed without dedupe
        logger.warning("Cache add failed for coach refresh key; proceeding without dedupe")

    def _worker():
        try:
            UserModel = get_user_model()
            user = UserModel.objects.get(id=user_id)
            logger.info(f"Starting background Gemini coach refresh for user {user_id}")
            data = _call_gemini_coach(user)
            is_ai = data.get('_is_ai_generated', True)
            _save_cached_analysis(user, data, is_ai=is_ai, ttl_hours=12)
            logger.info(f"Background Gemini coach refresh completed for user {user_id}; AI={is_ai}")
        except Exception as e:
            logger.error(f"Background coach refresh failed for user {user_id}: {e}", exc_info=True)
        finally:
            try:
                cache.delete(key)
            except Exception:
                pass

    try:
        t = threading.Thread(target=_worker, name=f"coach_refresh_{user_id}", daemon=True)
        t.start()
        return True
    except Exception as e:
        logger.error(f"Failed to start background thread for coach refresh: {e}")
        try:
            cache.delete(key)
        except Exception:
            pass
        return False


def _json_safe(obj):
    """Recursively convert objects to JSON-serializable types (UUID -> str, datetime -> ISO, Decimal -> float)."""
    import uuid as _uuid
    from datetime import datetime, date
    from decimal import Decimal
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, _uuid.UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return obj


def _build_coach_features(user) -> dict:
    """Aggregate temporal features for the LLM-based Adaptive Coach.
    
    Simulates GRU-like sequential state tracking by including:
      - Current skill levels (perTopic, totals)
      - Historical trends (improvement rates over time)
      - Temporal context (recency weights, session patterns)
      - User engagement metrics (streaks, session history)

    Returns a dict with:
      - perTopic: [{title, pron, flu, vocab, completed}]
      - totals: {pronunciation, fluency, vocabulary, topicsCompleted, recencyLatest}
      - historicalTrends: {pronunciation, fluency, vocabulary} with recent progression
      - recentTranscripts: [{type, text, topic, accuracy, daysAgo, recencyWeight}]
      - sessionHistory: Last 10 practice sessions with timestamps and XP
      - streakData: {current, longest, daysSinceBreak}
    """
    from apps.gamification.models import UserLevel, PointsTransaction
    from datetime import timedelta as td
    from statistics import mean
    
    features: dict = {
        'perTopic': [],
        'totals': {
            'pronunciation': 0,
            'fluency': 0,
            'vocabulary': 0,
            'listening': 0,
            'grammar': 0,
            'topicsCompleted': 0,
            'recencyLatest': None,
        },
        'historicalTrends': {
            'pronunciation': [],
            'fluency': [],
            'vocabulary': [],
            'listening': [],
            'grammar': [],
        },
        'recentTranscripts': [],
        'sessionHistory': [],
        'streakData': {
            'current': 0,
            'longest': 0,
            'daysSinceBreak': 0,
        },
    }
    try:
        tps = TopicProgress.objects.select_related('topic').filter(user=user)
        logger.info(f"Coach features: Found {tps.count()} TopicProgress records for user {user.id}")
        pron_total = 0
        flu_total = 0
        vocab_total = 0
        listen_total = 0
        grammar_total = 0
        count = 0
        completed_cnt = 0
        latest_dt = None
        for tp in tps:
            try:
                pron = int(tp.pronunciation_total_score or 0)
                flu = int(tp.fluency_total_score or 0)
                vocab = int(tp.vocabulary_total_score or 0)
                listen = int(tp.listening_total_score or 0)
                grammar = int(tp.grammar_total_score or 0)
                logger.info(f"Topic {tp.topic.title}: pron={pron}, flu={flu}, vocab={vocab}, listen={listen}, grammar={grammar}")
                features['perTopic'].append({
                    'topicId': str(tp.topic.id),
                    'title': tp.topic.title,
                    'pron': pron,
                    'flu': flu,
                    'vocab': vocab,
                    'listen': listen,
                    'grammar': grammar,
                    'completed': bool(tp.completed),
                })
                pron_total += pron
                flu_total += flu
                vocab_total += vocab
                listen_total += listen
                grammar_total += grammar
                count += 1
                if tp.completed and tp.completed_at:
                    completed_cnt += 1
                    if latest_dt is None or tp.completed_at > latest_dt:
                        latest_dt = tp.completed_at
            except Exception:
                continue
        if count > 0:
            features['totals']['pronunciation'] = int(round(pron_total / count))
            features['totals']['fluency'] = int(round(flu_total / count))
            features['totals']['vocabulary'] = int(round(vocab_total / count))
            features['totals']['listening'] = int(round(listen_total / count))
            features['totals']['grammar'] = int(round(grammar_total / count))
        logger.info(f"Coach totals: pron={features['totals']['pronunciation']}, flu={features['totals']['fluency']}, vocab={features['totals']['vocabulary']}, count={count}")
        features['totals']['topicsCompleted'] = completed_cnt
        features['totals']['recencyLatest'] = (latest_dt.isoformat() if latest_dt else None)
    except Exception as e:
        logger.error(f"Error building coach features: {e}", exc_info=True)

    # Historical trends: Calculate improvement rates over time windows
    # Compare recent (7 days) vs older (8-14 days) averages to detect learning velocity
    try:
        now = timezone.now()
        cutoff_recent = now - td(days=7)
        cutoff_older = now - td(days=14)
        
        # Recent pronunciation scores (last 7 days)
        recent_pron = list(
            UserPhraseRecording.objects.filter(user=user, created_at__gte=cutoff_recent)
            .values_list('accuracy', flat=True)
        )
        older_pron = list(
            UserPhraseRecording.objects.filter(
                user=user, created_at__gte=cutoff_older, created_at__lt=cutoff_recent
            ).values_list('accuracy', flat=True)
        )
        
        if recent_pron:
            features['historicalTrends']['pronunciation'].append({
                'period': 'recent_7d',
                'avgScore': int(round(mean(recent_pron))),
                'count': len(recent_pron)
            })
        if older_pron:
            features['historicalTrends']['pronunciation'].append({
                'period': 'prior_7d',
                'avgScore': int(round(mean(older_pron))),
                'count': len(older_pron)
            })
        
        # Same for fluency (conversation recordings as proxy)
        recent_flu = list(
            UserConversationRecording.objects.filter(user=user, created_at__gte=cutoff_recent)
            .values_list('accuracy', flat=True)
        )
        older_flu = list(
            UserConversationRecording.objects.filter(
                user=user, created_at__gte=cutoff_older, created_at__lt=cutoff_recent
            ).values_list('accuracy', flat=True)
        )
        
        if recent_flu:
            features['historicalTrends']['fluency'].append({
                'period': 'recent_7d',
                'avgScore': int(round(mean(recent_flu))),
                'count': len(recent_flu)
            })
        if older_flu:
            features['historicalTrends']['fluency'].append({
                'period': 'prior_7d',
                'avgScore': int(round(mean(older_flu))),
                'count': len(older_flu)
            })
        
        # Vocabulary trends from sessions
        recent_vocab_sessions = VocabularyPracticeSession.objects.filter(
            user=user, completed=True, updated_at__gte=cutoff_recent
        )
        older_vocab_sessions = VocabularyPracticeSession.objects.filter(
            user=user, completed=True, updated_at__gte=cutoff_older, updated_at__lt=cutoff_recent
        )
        
        if recent_vocab_sessions.exists():
            recent_vocab_scores = [s.total_score for s in recent_vocab_sessions]
            features['historicalTrends']['vocabulary'].append({
                'period': 'recent_7d',
                'avgScore': int(round(mean(recent_vocab_scores))),
                'count': len(recent_vocab_scores)
            })
        if older_vocab_sessions.exists():
            older_vocab_scores = [s.total_score for s in older_vocab_sessions]
            features['historicalTrends']['vocabulary'].append({
                'period': 'prior_7d',
                'avgScore': int(round(mean(older_vocab_scores))),
                'count': len(older_vocab_scores)
            })
        
        # Listening trends from sessions
        recent_listen_sessions = ListeningPracticeSession.objects.filter(
            user=user, completed=True, updated_at__gte=cutoff_recent
        )
        older_listen_sessions = ListeningPracticeSession.objects.filter(
            user=user, completed=True, updated_at__gte=cutoff_older, updated_at__lt=cutoff_recent
        )
        
        if recent_listen_sessions.exists():
            recent_listen_scores = [s.total_score for s in recent_listen_sessions]
            features['historicalTrends']['listening'].append({
                'period': 'recent_7d',
                'avgScore': int(round(mean(recent_listen_scores))),
                'count': len(recent_listen_scores)
            })
        if older_listen_sessions.exists():
            older_listen_scores = [s.total_score for s in older_listen_sessions]
            features['historicalTrends']['listening'].append({
                'period': 'prior_7d',
                'avgScore': int(round(mean(older_listen_scores))),
                'count': len(older_listen_scores)
            })
        
        # Grammar trends from sessions
        recent_grammar_sessions = GrammarPracticeSession.objects.filter(
            user=user, completed=True, updated_at__gte=cutoff_recent
        )
        older_grammar_sessions = GrammarPracticeSession.objects.filter(
            user=user, completed=True, updated_at__gte=cutoff_older, updated_at__lt=cutoff_recent
        )
        
        if recent_grammar_sessions.exists():
            recent_grammar_scores = [s.total_score for s in recent_grammar_sessions]
            features['historicalTrends']['grammar'].append({
                'period': 'recent_7d',
                'avgScore': int(round(mean(recent_grammar_scores))),
                'count': len(recent_grammar_scores)
            })
        if older_grammar_sessions.exists():
            older_grammar_scores = [s.total_score for s in older_grammar_sessions]
            features['historicalTrends']['grammar'].append({
                'period': 'prior_7d',
                'avgScore': int(round(mean(older_grammar_scores))),
                'count': len(older_grammar_scores)
            })
    except Exception as e:
        logger.debug(f'Historical trends calculation failed: {e}')

    # Recent transcripts with temporal weighting (limit small for token budget)
    try:
        pr = list(
            UserPhraseRecording.objects.select_related('topic')
            .filter(user=user).order_by('-created_at')[:5]
        )
        for r in pr:
            try:
                txt = _collapse_repeats_text((r.transcription or '').strip())[:220]
                if txt:
                    days_ago = (now - r.created_at).days
                    # Recency weight: more recent = higher weight (exponential decay)
                    recency_weight = round(1.0 / (1.0 + days_ago * 0.2), 2)
                    features['recentTranscripts'].append({
                        'type': 'phrase',
                        'text': txt,
                        'topic': r.topic.title,
                        'accuracy': int(round(float(r.accuracy or 0))),
                        'daysAgo': days_ago,
                        'recencyWeight': recency_weight
                    })
            except Exception:
                continue
    except Exception:
        pass
    try:
        cr = list(
            UserConversationRecording.objects.select_related('topic')
            .filter(user=user).order_by('-created_at')[:5]
        )
        for r in cr:
            try:
                txt = _collapse_repeats_text((r.transcription or '').strip())[:220]
                if txt:
                    days_ago = (now - r.created_at).days
                    recency_weight = round(1.0 / (1.0 + days_ago * 0.2), 2)
                    features['recentTranscripts'].append({
                        'type': 'conversation',
                        'text': txt,
                        'topic': r.topic.title,
                        'accuracy': int(round(float(r.accuracy or 0))),
                        'daysAgo': days_ago,
                        'recencyWeight': recency_weight
                    })
            except Exception:
                continue
    except Exception:
        pass

    # Session history: Last 10 practice sessions from PointsTransaction
    try:
        recent_xp_txns = PointsTransaction.objects.filter(
            user=user,
            amount__gt=0,  # Only positive XP gains
            source__in=['pronunciation_practice', 'fluency_practice', 'vocabulary_practice', 
                       'listening_practice', 'grammar_practice', 'conversation_practice', 
                       'topic_completion']
        ).order_by('-created_at')[:10]
        
        for txn in recent_xp_txns:
            features['sessionHistory'].append({
                'date': txn.created_at.isoformat(),
                'source': txn.source,
                'xpEarned': txn.amount,
                'daysAgo': (now - txn.created_at).days
            })
    except Exception as e:
        logger.debug(f'Session history failed: {e}')

    # Streak data from UserLevel
    try:
        level_profile = UserLevel.objects.filter(user=user).first()
        if level_profile:
            days_since_break = 0
            if level_profile.last_activity_date:
                days_since_break = (now.date() - level_profile.last_activity_date).days
            
            features['streakData'] = {
                'current': level_profile.streak_days,
                'longest': level_profile.longest_streak,
                'daysSinceBreak': days_since_break,
            }
    except Exception as e:
        logger.debug(f'Streak data failed: {e}')

    return features


def _heuristic_coach(user) -> dict:
    """Build a minimal coach analysis without LLM based on topic progress averages for all 5 skills."""
    feats = _build_coach_features(user)
    pron = int(feats.get('totals', {}).get('pronunciation') or 0)
    flu = int(feats.get('totals', {}).get('fluency') or 0)
    vocab = int(feats.get('totals', {}).get('vocabulary') or 0)
    listen = int(feats.get('totals', {}).get('listening') or 0)
    grammar = int(feats.get('totals', {}).get('grammar') or 0)
    
    skills = [
        {'id': 'pronunciation', 'name': 'Pronunciation', 'mastery': pron, 'confidence': 0.6, 'trend': 'flat', 'evidence': []},
        {'id': 'fluency', 'name': 'Fluency & Coherence', 'mastery': flu, 'confidence': 0.6, 'trend': 'flat', 'evidence': []},
        {'id': 'vocabulary', 'name': 'Vocabulary', 'mastery': vocab, 'confidence': 0.6, 'trend': 'flat', 'evidence': []},
        {'id': 'listening', 'name': 'Comprehension (Listening)', 'mastery': listen, 'confidence': 0.6, 'trend': 'flat', 'evidence': []},
        {'id': 'grammar', 'name': 'Grammar', 'mastery': grammar, 'confidence': 0.6, 'trend': 'flat', 'evidence': []},
    ]
    
    # Detect new user (all scores are 0)
    total_score = pron + flu + vocab + listen + grammar
    is_new_user = total_score == 0
    
    # strengths/weaknesses by sorting (only if user has data)
    if is_new_user:
        # New users: suggest starting with pronunciation (foundation skill)
        strengths = []
        weaknesses = []
        weakest = 'pronunciation'
        coach_msg = 'Welcome! Let\'s start your English learning journey. Begin with pronunciation basics to build a strong foundation.'
    else:
        ordered = sorted(skills, key=lambda s: s['mastery'], reverse=True)
        # Only show as strength if score > 60
        strengths = [s['id'] for s in ordered if s['mastery'] > 60][:2]
        # Only show as weakness if score < 50
        weaknesses = [s['id'] for s in ordered if s['mastery'] < 50][-2:]
        weakest = weaknesses[0] if weaknesses else ordered[-1]['id']
        
        # Dynamic coach message based on performance
        avg_score = total_score // 5
        if avg_score >= 75:
            coach_msg = f'Excellent progress! Your average score is {avg_score}%. Keep pushing to maintain your momentum.'
        elif avg_score >= 50:
            coach_msg = f'You\'re making steady progress (avg: {avg_score}%). Let\'s focus on your weaker areas to level up faster.'
        else:
            coach_msg = f'Good start! Your average is {avg_score}%. Consistent practice will help you improve quickly.'
    
    nba_title = {
        'pronunciation': 'Practice Pronunciation Now',
        'fluency': 'Do a Fluency Prompt',
        'vocabulary': 'Study Vocabulary Lesson',
        'listening': 'Practice Listening Comprehension',
        'grammar': 'Complete Grammar Exercise',
    }.get(weakest, 'Master Current Topic')
    
    # Choose the most relevant topic based on user's weakest skill performance
    mode_map = {
        'pronunciation': 'master',
        'fluency': 'conversation',
        'vocabulary': 'vocab',
        'listening': 'listening',
        'grammar': 'grammar',
    }
    
    # Find the topic where user is weakest in their identified weak skill
    target_topic = None
    if not is_new_user and feats.get('perTopic'):
        # Map weakest skill to the corresponding field in perTopic
        skill_field_map = {
            'pronunciation': 'pron',
            'fluency': 'flu',
            'vocabulary': 'vocab',
            'listening': 'listen',
            'grammar': 'grammar',
        }
        field_key = skill_field_map.get(weakest, 'pron')
        
        # Find topic with lowest score in the weak skill (excluding completed topics)
        per_topic_list = feats['perTopic']
        incomplete_topics = [t for t in per_topic_list if not t.get('completed', False)]
        
        if incomplete_topics:
            # Sort by the weak skill score (ascending) to find the weakest
            sorted_topics = sorted(incomplete_topics, key=lambda t: t.get(field_key, 0))
            weakest_topic_data = sorted_topics[0]
            # Find the actual Topic object by ID (more reliable than title)
            try:
                topic_id = weakest_topic_data.get('topicId')
                if topic_id:
                    target_topic = Topic.objects.filter(
                        id=topic_id,
                        is_active=True
                    ).first()
                # Fallback to title if topicId not available
                if not target_topic:
                    target_topic = Topic.objects.filter(
                        title=weakest_topic_data['title'],
                        is_active=True
                    ).first()
            except Exception:
                pass
    
    # Fallback: use first unlocked topic or first active topic
    if not target_topic:
        # Try to get user's first unlocked but incomplete topic
        try:
            user_progress = TopicProgress.objects.select_related('topic').filter(
                user=user,
                topic__is_active=True,
                completed=False
            ).order_by('topic__sequence').first()
            if user_progress:
                target_topic = user_progress.topic
        except Exception:
            pass
    
    # Final fallback: first active topic
    if not target_topic:
        target_topic = Topic.objects.filter(is_active=True).order_by('sequence').first()
    
    deeplink = f"app://voicevibe/speaking/topic/{target_topic.id if target_topic else 'current'}/" + mode_map.get(weakest, 'master')
    
    out = {
        'currentVersion': 1,
        'generatedAt': timezone.now().isoformat(),
        'skills': skills,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'nextBestActions': [
            {
                'id': f'nba:{weakest}',
                'title': nba_title,
                'rationale': 'Based on your recent scores, focusing here will improve your overall progress the fastest.',
                'deeplink': deeplink,
                'expectedGain': 'medium',
            }
        ],
        'difficultyCalibration': {
            'pronunciation': 'baseline',
            'fluency': 'baseline',
            'vocabulary': 'baseline',
            'listening': 'baseline',
            'grammar': 'baseline',
        },
        'schedule': [],
        'coachMessage': coach_msg,
        'cacheForHours': 12,
        '_is_ai_generated': False,  # Mark as heuristic, not AI
    }
    return out


def _call_gemini_coach(user) -> dict:
    """Call Gemini LLM to generate adaptive learning recommendations simulating GRU behavior."""
    api_key = (
        getattr(settings, 'GEMINI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GOOGLE_API_KEY', '')
    )
    if not api_key:
        return _heuristic_coach(user)
    feats = _build_coach_features(user)
    prompt = (
        "You are an LLM-based Adaptive Coach simulating a Gated Recurrent Unit (GRU) for personalized English learning.\n\n"
        "ANALYZE ALL 5 CORE ENGLISH SKILLS:\n"
        "1. Pronunciation - clear articulation and accent\n"
        "2. Fluency & Coherence - speaking smoothly and logically\n"
        "3. Vocabulary - word choice and lexical resource\n"
        "4. Grammar - sentence structure and accuracy\n"
        "5. Comprehension (Listening) - understanding spoken English\n\n"
        "GRU BEHAVIOR REQUIREMENTS:\n"
        "1. Temporal Awareness: Weight recent performance (historicalTrends.recent_7d) MORE heavily than older data (prior_7d)\n"
        "2. Learning Velocity Detection: Compare recent vs prior scores to identify if user is improving, plateauing, or declining\n"
        "3. Recency Bias: Use recencyWeight in recentTranscripts - higher weight = more influence on recommendations\n"
        "4. Skill Dependencies (Cascade Model):\n"
        "   - Pronunciation affects Fluency (poor pronunciation blocks fluent speech)\n"
        "   - Vocabulary affects both Fluency and Comprehension (limited vocab limits understanding)\n"
        "   - Grammar affects Fluency (poor grammar disrupts coherence)\n"
        "   - If pronunciation < 50, ALWAYS prioritize it over other skills regardless of their scores\n"
        "   - If grammar < 40, prioritize it alongside pronunciation\n"
        "5. Adaptive Difficulty: Calibrate based on mastery levels:\n"
        "   - If skill mastery > 75: suggest 'harder' difficulty\n"
        "   - If skill mastery < 45: suggest 'easier' difficulty\n"
        "   - Otherwise: 'baseline'\n"
        "6. Confidence Calculation: Higher variance in recent scores = lower confidence\n"
        "7. Evidence-Based: Always cite specific examples from recentTranscripts in evidence[] arrays\n"
        "8. Actionable Schedule: Generate 3-5 day practice plan in schedule[] using spaced repetition\n"
        "   - Alternate skill focus (don't practice same skill 2 days in a row)\n"
        "   - Include microSkills like 'th sound', 'past tense', 'question forms'\n"
        "9. Streak Reinforcement: Acknowledge current streak in coachMessage if streakData.current > 0\n\n"
        "SCHEMA (return STRICT JSON only):\n"
        "{\n"
        "  currentVersion: int,\n"
        "  generatedAt: isoDateTime,\n"
        "  skills: [\n"
        "    {id: 'pronunciation', name: 'Pronunciation', mastery: 0-100, confidence: 0-1, trend: 'up'|'down'|'flat', evidence: string[]},\\n"
        "    {id: 'fluency', name: 'Fluency and Coherence', mastery: 0-100, confidence: 0-1, trend: 'up'|'down'|'flat', evidence: string[]},\\n"
        "    {id: 'vocabulary', name: 'Vocabulary', mastery: 0-100, confidence: 0-1, trend: 'up'|'down'|'flat', evidence: string[]},\\n"
        "    {id: 'grammar', name: 'Grammar', mastery: 0-100, confidence: 0-1, trend: 'up'|'down'|'flat', evidence: string[]},\\n"
        "    {id: 'listening', name: 'Comprehension Listening', mastery: 0-100, confidence: 0-1, trend: 'up'|'down'|'flat', evidence: string[]}\\n"
        "  ],\n"
        "  strengths: string[],  // skill IDs of top performers\n"
        "  weaknesses: string[],  // skill IDs needing improvement\n"
        "  nextBestActions: [{id: string, title: string, rationale: string, deeplink: string, expectedGain: 'small'|'medium'|'large'}],\n"
        "  difficultyCalibration: {\n"
        "    pronunciation: 'easier'|'baseline'|'harder',\n"
        "    fluency: 'slower'|'baseline'|'faster',\n"
        "    vocabulary: 'fewer_terms'|'baseline'|'more_terms',\n"
        "    listening: 'easier'|'baseline'|'harder',\n"
        "    grammar: 'easier'|'baseline'|'harder'\n"
        "  },\n"
        "  schedule: [{date: 'YYYY-MM-DD', focus: string, microSkills: string[], reason: string}],\n"
        "  coachMessage: string,\n"
        "  cacheForHours: int\n"
        "}\n\n"
        "Deeplink format: app://voicevibe/speaking/topic/<topicId>/(master|conversation|vocab|listening|grammar)\n"
        "IMPORTANT: Analyze perTopic data to identify which specific topic the user needs to practice.\n"
        "- For nextBestActions, use the topicId from perTopic where the user has the lowest score in their weakest skill\n"
        "- Prioritize incomplete topics (completed: false) over completed ones\n"
        "- If all topics are completed or no data exists, use 'current' as topicId\n"
        "- Example: If user is weak in pronunciation and 'Daily Activities' has lowest pron score, use that topic's ID\n\n"
        f"USER FEATURES:\n{json.dumps(_json_safe(feats), ensure_ascii=False, indent=2)}\n\n"
        "Return ONLY valid JSON matching the schema above. Ensure all 5 skills are analyzed.\n"
        "CRITICAL: In nextBestActions, specify the actual topic ID where user needs most improvement, not just 'current'.\n"
    )
    candidates = [
        getattr(settings, 'GEMINI_TEXT_MODEL', None) or os.environ.get('GEMINI_MODEL') or 'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-1.5-flash',
        'gemini-1.5-pro',
    ]
    try:
        genai.configure(api_key=api_key)
    except Exception:
        return _heuristic_coach(user)
    gen_config = None
    try:
        if _GenConfig:
            max_tokens = int(os.environ.get('COACH_MAX_OUTPUT_TOKENS', '1024'))
            gen_config = _GenConfig(max_output_tokens=max_tokens, temperature=0.3, top_p=0.9)
    except Exception:
        gen_config = None
    for name in candidates:
        try:
            model = genai.GenerativeModel(name)
            if gen_config is not None:
                resp = model.generate_content(prompt, generation_config=gen_config)
            else:
                resp = model.generate_content(prompt)
            raw = (getattr(resp, 'text', None) or '').strip()
            if not raw:
                continue
            if raw.startswith('```'):
                first_nl = raw.find('\n')
                if first_nl != -1:
                    raw = raw[first_nl+1:]
                if raw.endswith('```'):
                    raw = raw[:-3]
                raw = raw.strip()
            data = json.loads(raw)
            # Validate with serializer shape
            ser = CoachAnalysisSerializer(data=data)
            if ser.is_valid():
                # Return serialized representation to ensure datetimes are strings
                return ser.data
        except Exception as e:
            logger.warning('Gemini coach via %s failed: %s', name, e)
            continue
    return _heuristic_coach(user)


class CoachAnalysisView(APIView):
    """AI Coach analysis endpoint with persistent 12-hour cache."""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user
        logger.info(f"CoachAnalysisView GET called for user {user.id}")
        
        # Try to get cached analysis from database
        cache = _get_cached_analysis(user)
        
        if cache:
            # Serve cached data (even if stale) to prevent timeout
            try:
                ser = CoachAnalysisSerializer(cache.analysis_data)
                response_data = ser.data.copy() if hasattr(ser.data, 'copy') else dict(ser.data)
                # Add metadata for client
                response_data['_cache_stale'] = cache.is_stale
                response_data['_is_ai_generated'] = cache.is_ai_generated
                response_data['_generated_at'] = cache.generated_at.isoformat()
                response_data['_next_refresh_at'] = cache.expires_at.isoformat()
                response_data['_in_progress'] = _is_refresh_in_progress(user.id)
                logger.info(f"Returning cached analysis for user {user.id}: stale={cache.is_stale}, AI={cache.is_ai_generated}")
                return Response(response_data, status=status.HTTP_200_OK)
            except Exception as e:
                logger.error(f"Failed to serialize cached analysis for user {user.id}: {e}")
        
        # Cache miss: return heuristic response immediately and save to DB
        # NEVER call Gemini synchronously from GET endpoint to avoid timeout
        logger.info(f"Cache miss for user {user.id}, generating heuristic analysis")
        data = _heuristic_coach(user)
        _save_cached_analysis(user, data, is_ai=False, ttl_hours=12)
        
        # Get the cache entry to retrieve expires_at timestamp
        cache = _get_cached_analysis(user)
        
        ser = CoachAnalysisSerializer(data)
        response_data = ser.data.copy() if hasattr(ser.data, 'copy') else dict(ser.data)
        response_data['_cache_stale'] = False
        response_data['_is_ai_generated'] = False
        if cache:
            response_data['_next_refresh_at'] = cache.expires_at.isoformat()
        response_data['_in_progress'] = _is_refresh_in_progress(user.id)
        return Response(response_data, status=status.HTTP_200_OK)


class CoachAnalysisRefreshView(APIView):
    """Manual refresh endpoint for AI Coach analysis.
    
    This endpoint performs a synchronous Gemini call and may take 10-45 seconds.
    Only call this when the user explicitly requests a refresh.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        user = request.user
        logger.info(f"CoachAnalysisRefreshView POST called for user {user.id} - starting Gemini analysis")
        cache_entry = _get_cached_analysis(user)
        if cache_entry and not cache_entry.is_stale:
            try:
                ser = CoachAnalysisSerializer(cache_entry.analysis_data)
                response_data = ser.data.copy() if hasattr(ser.data, 'copy') else dict(ser.data)
                response_data['_cache_stale'] = False
                response_data['_is_ai_generated'] = cache_entry.is_ai_generated
                response_data['_generated_at'] = cache_entry.generated_at.isoformat()
                response_data['_next_refresh_at'] = cache_entry.expires_at.isoformat()
                response_data['_skipped_refresh'] = True
                response_data['_in_progress'] = _is_refresh_in_progress(user.id)
                logger.info(f"Coach analysis refresh skipped for user {user.id}: cache valid until {cache_entry.expires_at}")
                return Response(response_data, status=status.HTTP_200_OK)
            except Exception as e:
                logger.error(f"Failed to serialize cached analysis during refresh for user {user.id}: {e}")
        enqueued = _enqueue_coach_refresh(user.id)
        try:
            if cache_entry:
                ser = CoachAnalysisSerializer(cache_entry.analysis_data)
                response_data = ser.data.copy() if hasattr(ser.data, 'copy') else dict(ser.data)
                response_data['_cache_stale'] = cache_entry.is_stale
                response_data['_is_ai_generated'] = cache_entry.is_ai_generated
                response_data['_generated_at'] = cache_entry.generated_at.isoformat()
                response_data['_next_refresh_at'] = cache_entry.expires_at.isoformat()
            else:
                baseline = _heuristic_coach(user)
                _save_cached_analysis(user, baseline, is_ai=False, ttl_hours=12)
                ser = CoachAnalysisSerializer(baseline)
                response_data = ser.data.copy() if hasattr(ser.data, 'copy') else dict(ser.data)
                response_data['_cache_stale'] = False
                response_data['_is_ai_generated'] = False
                ce2 = _get_cached_analysis(user)
                if ce2:
                    response_data['_next_refresh_at'] = ce2.expires_at.isoformat()
            response_data['_refresh_enqueued'] = bool(enqueued)
            response_data['_in_progress'] = _is_refresh_in_progress(user.id)
            logger.info(f"Coach analysis background refresh {'enqueued' if enqueued else 'already in progress'} for user {user.id}")
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Coach analysis non-blocking refresh response failed for user {user.id}: {e}", exc_info=True)
            return Response({'error': 'Refresh request accepted but response formatting failed.'}, status=status.HTTP_202_ACCEPTED)

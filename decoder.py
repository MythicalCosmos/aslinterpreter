import time
from typing import Optional, Tuple, List, Set
from difflib import get_close_matches
from core.constants import (
    WORD_GAP, CONFIDENCE_THRESHOLD, AUTOCORRECT_TOGGLE, AUTOCORRECT_THRESHOLD
)


class WordDecoder:
    def __init__(self, word_set: Set[str]) -> None:
        self.wordSet   = word_set
        self.buffer:   List[str] = []
        self.lastTime: Optional[float] = None

    # ── Public API ────────────────────────────────────────────────────────

    def addLetter(self, letter: str, t: Optional[float] = None) -> str:
        """Append one recognized letter. Returns current preview string."""
        t = t or time.time()
        self.lastTime = t
        self.buffer.append(letter.upper())
        return "".join(self.buffer)

    def shouldFlush(self) -> bool:
        """Return True when the inter-letter gap exceeds WORD_GAP seconds."""
        return bool(
            self.buffer
            and self.lastTime is not None
            and time.time() - self.lastTime >= WORD_GAP
        )

    def flush(self) -> Tuple[str, float, str]:
        """
        Finalize the current letter buffer into a word.

        Returns (text, confidence, tag) where tag is one of:
            'auto'    - autocorrected dictionary match
            'word'    - direct dictionary word above threshold
            'letters' - low-confidence fallback, space-separated letters
        """
        raw = "".join(self.buffer)
        self.buffer.clear()
        self.lastTime = None

        if AUTOCORRECT_TOGGLE:
            corrected = self._autocorrect(raw)
            if corrected:
                return corrected, self._wordConfidence(corrected), "auto"
            deduped = self._deduplicate(raw)
            if deduped != raw:
                corrected = self._autocorrect(deduped)
                if corrected:
                    return corrected, self._wordConfidence(corrected), "auto"

        conf = self._wordConfidence(raw)
        if conf >= CONFIDENCE_THRESHOLD:
            return raw, conf, "word"

        return raw, 0.0, "letters"

    def clear(self) -> None:
        """Discard the current buffer without flushing."""
        self.buffer.clear()
        self.lastTime = None

    def deleteLast(self) -> str:
        """Remove the last letter from the buffer. Returns updated preview."""
        if self.buffer:
            self.buffer.pop()
        return "".join(self.buffer)

    # ── Private helpers ───────────────────────────────────────────────────

    def _autocorrect(self, word: str) -> Optional[str]:
        matches = get_close_matches(word, self.wordSet, n=1, cutoff=AUTOCORRECT_THRESHOLD)
        return matches[0] if matches else None

    def _deduplicate(self, word: str) -> str:
        if not word:
            return word
        result = word[0]
        for ch in word[1:]:
            if ch != result[-1]:
                result += ch
        return result

    def _wordConfidence(self, word: str, max_len: int = 12) -> float:
        """Score 0.6-1.0 for known words, 0.0 for unknown."""
        if word not in self.wordSet:
            return 0.0
        return 0.6 + 0.4 * min(len(word) / max_len, 1.0)
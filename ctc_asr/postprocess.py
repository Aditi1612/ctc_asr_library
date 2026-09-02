"""Repo-specific jamo normalization and Hangul composition.

Vendored from ``service_ctc/transcribe_bundle/transcribe.py`` so this library has
no dependency on the API repository. Keep these two copies in sync if the
serving-side normalization ever changes.
"""

from __future__ import annotations


def normalize_jamo(decoded: str) -> str:
    """Collapse CTC spacing and split on the word-delimiter token."""
    cleaned = decoded.replace(" ", "")
    words = cleaned.split("|")
    words = [w for w in words if w != ""]
    return " ".join(words)


# Hangul composition helpers
CHO = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
JUNG = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
DOUBLE_FINAL = {
    ("ㄱ", "ㅅ"): "ㄳ", ("ㄴ", "ㅈ"): "ㄵ", ("ㄴ", "ㅎ"): "ㄶ", ("ㄹ", "ㄱ"): "ㄺ", ("ㄹ", "ㅁ"): "ㄻ",
    ("ㄹ", "ㅂ"): "ㄼ", ("ㄹ", "ㅅ"): "ㄽ", ("ㄹ", "ㅌ"): "ㄾ", ("ㄹ", "ㅍ"): "ㄿ", ("ㄹ", "ㅎ"): "ㅀ", ("ㅂ", "ㅅ"): "ㅄ",
}
DOUBLE_FINAL_REVERSE = {value: key for key, value in DOUBLE_FINAL.items()}
IMPLICIT_VOWEL = "ㅡ"


def compose_hangul_from_jamo(jamo_sentence: str) -> str:
    """Compose a space-separated jamo string into Hangul syllables."""

    def compose_word(word: str) -> str:
        expanded = []
        for ch in word:
            if ch in DOUBLE_FINAL_REVERSE:
                expanded.extend(DOUBLE_FINAL_REVERSE[ch])
            else:
                expanded.append(ch)
        word = "".join(expanded)

        out = []
        idx = 0
        while idx < len(word):
            ch = word[idx]
            if ch in JUNG:
                # Standalone vowels must be composed with an implicit ㅇ onset.
                onset = "ㅇ"
                vowel = ch
                next_idx = idx + 1
                final = ""
                if next_idx < len(word):
                    c1 = word[next_idx]
                    if next_idx + 1 < len(word) and (c1, word[next_idx + 1]) in DOUBLE_FINAL and (next_idx + 2 == len(word) or word[next_idx + 2] not in JUNG):
                        final = DOUBLE_FINAL[(c1, word[next_idx + 1])]
                        next_idx += 2
                    elif c1 in JONG and (next_idx + 1 == len(word) or word[next_idx + 1] not in JUNG):
                        final = c1
                        next_idx += 1
                code = 0xAC00 + CHO.index(onset) * 588 + JUNG.index(vowel) * 28 + JONG.index(final)
                out.append(chr(code))
                idx = next_idx
                continue
            if ch not in CHO:
                out.append(ch)
                idx += 1
                continue
            onset = ch
            if idx + 1 >= len(word) or word[idx + 1] not in JUNG:
                # Emit a readable syllable for standalone consonants.
                vowel = IMPLICIT_VOWEL
                code = 0xAC00 + CHO.index(onset) * 588 + JUNG.index(vowel) * 28
                out.append(chr(code))
                idx += 1
                continue
            vowel = word[idx + 1]
            next_idx = idx + 2
            final = ""
            if next_idx < len(word):
                c1 = word[next_idx]
                if next_idx + 1 < len(word) and (c1, word[next_idx + 1]) in DOUBLE_FINAL and (next_idx + 2 == len(word) or word[next_idx + 2] not in JUNG):
                    final = DOUBLE_FINAL[(c1, word[next_idx + 1])]
                    next_idx += 2
                elif c1 in JONG and (next_idx + 1 == len(word) or word[next_idx + 1] not in JUNG):
                    final = c1
                    next_idx += 1
            code = 0xAC00 + CHO.index(onset) * 588 + JUNG.index(vowel) * 28 + JONG.index(final)
            out.append(chr(code))
            idx = next_idx
        return "".join(out)

    words = [w for w in jamo_sentence.split() if w != ""]
    return " ".join(compose_word(w) for w in words)

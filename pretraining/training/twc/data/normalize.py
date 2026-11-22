"""Deterministic text normalization for the Phase 2A filtering foundation."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass, field
from functools import cached_property


NORMALIZER_VERSION = "phase2a_normalizer_v1"
MAX_HTML_UNESCAPE_PASSES = 3

CANONICAL_STEPS = (
    "line_endings_to_lf",
    "bounded_html_entity_unescape",
    "unicode_nfkc",
    "quote_apostrophe_dash_canonicalization",
    "control_format_cleanup",
    "casefold",
    "whitespace_collapse_preserve_double_lf",
)

HTML_ENTITY_RE = re.compile(
    r"&(?:#[0-9]{1,7}|#x[0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});"
)
ESCAPE_RE = re.compile(
    r"\\(?:u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{2})"
)

QUOTE_DASH_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)


@dataclass(frozen=True)
class _Unit:
    text: str
    raw_start: int
    raw_end: int


@dataclass(frozen=True)
class CanonicalNormalization:
    raw_text: str
    canonical_text: str
    raw_sha256: str
    canonical_sha256: str
    normalization_version: str = NORMALIZER_VERSION


@dataclass
class NormalizedDocument:
    raw_text: str
    canonical_text: str
    source_spans: tuple[tuple[int, int], ...]
    raw_sha256: str
    canonical_sha256: str
    normalization_version: str = NORMALIZER_VERSION
    steps_applied: tuple[str, ...] = CANONICAL_STEPS
    stats: dict[str, int] = field(default_factory=dict)

    @cached_property
    def _views(self) -> dict[str, str]:
        return {
            "canonical": self.canonical_text,
            "ascii_folded": _ascii_fold(self.canonical_text),
            "depunct_compact": _depunct_compact(self.canonical_text),
            "escape_decoded": _decode_backslash_escapes(self.canonical_text),
        }

    @cached_property
    def view_sha256s(self) -> dict[str, str]:
        return {name: _sha256_text(text) for name, text in self._views.items()}

    def view(self, name: str) -> str:
        if name == "canonical":
            return self.canonical_text
        try:
            return self._views[name]
        except KeyError as exc:
            raise KeyError(f"unknown normalized view: {name}") from exc

    def project_canonical_span(
        self,
        start: int,
        end: int,
        *,
        context_chars: int = 80,
    ) -> tuple[tuple[int, int] | None, str]:
        if start < 0 or end > len(self.source_spans) or start >= end:
            return None, ""

        raw_start = min(span[0] for span in self.source_spans[start:end])
        raw_end = max(span[1] for span in self.source_spans[start:end])
        raw_context_start = max(0, raw_start - context_chars)
        raw_context_end = min(len(self.raw_text), raw_end + context_chars)
        raw_context = self.raw_text[raw_context_start:raw_context_end]

        projected = normalize_text(self.raw_text[raw_start:raw_end]).canonical_text
        expected = self.canonical_text[start:end]
        raw_span = (raw_start, raw_end) if projected == expected else None
        return raw_span, raw_context


def normalize_text(raw: str) -> NormalizedDocument:
    if not isinstance(raw, str):
        raise TypeError("normalize_text requires a str input")

    stats = {
        "raw_chars": len(raw),
        "html_unescape_passes": 0,
        "removed_format_or_control_chars": 0,
        "collapsed_whitespace_chars": 0,
    }
    units = _line_endings_to_lf(raw)
    units = _html_unescape_units(units, stats)
    units = _canonicalize_units(units, stats)
    units = _collapse_whitespace(units, stats)

    canonical = "".join(unit.text for unit in units)
    spans = tuple((unit.raw_start, unit.raw_end) for unit in units)
    return NormalizedDocument(
        raw_text=raw,
        canonical_text=canonical,
        source_spans=spans,
        raw_sha256=_sha256_text(raw),
        canonical_sha256=_sha256_text(canonical),
        stats=stats,
    )


def normalize_canonical_text(raw: str) -> CanonicalNormalization:
    if not isinstance(raw, str):
        raise TypeError("normalize_canonical_text requires a str input")

    text = _line_endings_to_lf_text(raw)
    text = _html_unescape_text(text)
    text = _canonicalize_text(text)
    canonical = _collapse_whitespace_text(text)
    return CanonicalNormalization(
        raw_text=raw,
        canonical_text=canonical,
        raw_sha256=_sha256_text(raw),
        canonical_sha256=_sha256_text(canonical),
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _line_endings_to_lf_text(raw: str) -> str:
    return raw.replace("\r\n", "\n").translate(
        {
            ord("\r"): "\n",
            ord("\u0085"): "\n",
            ord("\u2028"): "\n",
            ord("\u2029"): "\n",
        }
    )


def _html_unescape_text(text: str) -> str:
    current = text
    for _pass_index in range(MAX_HTML_UNESCAPE_PASSES):
        changed = False

        def replace_match(match: re.Match[str]) -> str:
            nonlocal changed
            token = match.group(0)
            replacement = html.unescape(token)
            if replacement != token:
                changed = True
            return replacement

        current = HTML_ENTITY_RE.sub(replace_match, current)
        if not changed:
            break
    return current


def _canonicalize_text(text: str) -> str:
    output: list[str] = []
    for unit in text:
        canonical = unicodedata.normalize("NFKC", unit).translate(QUOTE_DASH_TRANSLATION)
        canonical = canonical.casefold()
        for char in canonical:
            category = unicodedata.category(char)
            if char == "\n":
                output.append(char)
            elif category == "Cf" or category.startswith("C"):
                if char.isspace():
                    output.append(" ")
            elif char.isspace():
                output.append(" ")
            else:
                output.append(char)
    return "".join(output)


def _collapse_whitespace_text(text: str) -> str:
    output: list[str] = []
    newline_run = 0

    for char in text:
        if char == " ":
            if not output or output[-1] in {" ", "\n"}:
                continue
            output.append(char)
            newline_run = 0
            continue

        if char == "\n":
            while output and output[-1] == " ":
                output.pop()
            if not output:
                continue
            if newline_run < 2:
                output.append(char)
            newline_run += 1
            continue

        output.append(char)
        newline_run = 0

    while output and output[-1] in {" ", "\n"}:
        output.pop()

    return "".join(output)


def _line_endings_to_lf(raw: str) -> list[_Unit]:
    units: list[_Unit] = []
    index = 0
    while index < len(raw):
        char = raw[index]
        if char == "\r" and index + 1 < len(raw) and raw[index + 1] == "\n":
            units.append(_Unit("\n", index, index + 2))
            index += 2
            continue
        if char in {"\r", "\u0085", "\u2028", "\u2029"}:
            units.append(_Unit("\n", index, index + 1))
        else:
            units.append(_Unit(char, index, index + 1))
        index += 1
    return units


def _html_unescape_units(units: list[_Unit], stats: dict[str, int]) -> list[_Unit]:
    current = units
    for pass_index in range(MAX_HTML_UNESCAPE_PASSES):
        text = "".join(unit.text for unit in current)
        output: list[_Unit] = []
        changed = False
        cursor = 0
        for match in HTML_ENTITY_RE.finditer(text):
            output.extend(current[cursor : match.start()])
            token = match.group(0)
            replacement = html.unescape(token)
            if replacement != token:
                raw_start = current[match.start()].raw_start
                raw_end = current[match.end() - 1].raw_end
                output.extend(_Unit(char, raw_start, raw_end) for char in replacement)
                changed = True
            else:
                output.extend(current[match.start() : match.end()])
            cursor = match.end()
        output.extend(current[cursor:])
        current = output
        if not changed:
            break
        stats["html_unescape_passes"] = pass_index + 1
    return current


def _canonicalize_units(units: list[_Unit], stats: dict[str, int]) -> list[_Unit]:
    output: list[_Unit] = []
    for unit in units:
        text = unicodedata.normalize("NFKC", unit.text).translate(QUOTE_DASH_TRANSLATION)
        text = text.casefold()
        for char in text:
            category = unicodedata.category(char)
            if char == "\n":
                output.append(_Unit(char, unit.raw_start, unit.raw_end))
            elif category == "Cf" or category.startswith("C"):
                if char.isspace():
                    output.append(_Unit(" ", unit.raw_start, unit.raw_end))
                else:
                    stats["removed_format_or_control_chars"] += 1
            elif char.isspace():
                output.append(_Unit(" ", unit.raw_start, unit.raw_end))
            else:
                output.append(_Unit(char, unit.raw_start, unit.raw_end))
    return output


def _collapse_whitespace(units: list[_Unit], stats: dict[str, int]) -> list[_Unit]:
    output: list[_Unit] = []
    newline_run = 0

    for unit in units:
        char = unit.text
        if char == " ":
            if not output or output[-1].text in {" ", "\n"}:
                stats["collapsed_whitespace_chars"] += 1
                continue
            output.append(unit)
            newline_run = 0
            continue

        if char == "\n":
            while output and output[-1].text == " ":
                output.pop()
                stats["collapsed_whitespace_chars"] += 1
            if not output:
                stats["collapsed_whitespace_chars"] += 1
                continue
            if newline_run < 2:
                output.append(unit)
            else:
                stats["collapsed_whitespace_chars"] += 1
            newline_run += 1
            continue

        output.append(unit)
        newline_run = 0

    while output and output[-1].text in {" ", "\n"}:
        output.pop()
        stats["collapsed_whitespace_chars"] += 1

    return output


def _ascii_fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(
        char for char in decomposed if not unicodedata.category(char).startswith("M")
    )
    return without_marks.encode("ascii", errors="ignore").decode("ascii")


def _depunct_compact(text: str) -> str:
    return "".join(char for char in text if char.isalnum())


def _decode_backslash_escapes(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            if token.startswith("\\u"):
                codepoint = int(token[2:], 16)
            elif token.startswith("\\U"):
                codepoint = int(token[2:], 16)
            else:
                codepoint = int(token[2:], 16)
            char = chr(codepoint)
            if unicodedata.category(char) == "Cs":
                return token
            return char
        except (ValueError, OverflowError):
            return token

    decoded = ESCAPE_RE.sub(replace, text)
    return normalize_text(decoded).canonical_text if decoded != text else decoded

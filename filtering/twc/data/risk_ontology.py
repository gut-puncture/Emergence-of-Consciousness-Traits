"""Strict risk-ontology loading and rule compilation."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import regex
import yaml

from twc.data.decision_schema import stable_json_hash


SCHEMA_VERSION = 1
MATCH_TIMEOUT_MS = 25
ALLOWED_TOP_LEVEL_KEYS = {"version", "category", "description", "rules"}
ALLOWED_RULE_KEYS = {
    "id",
    "severity",
    "type",
    "view",
    "value",
    "pattern",
    "flags",
    "reason",
    "examples",
    "weight",
    "threshold",
}
ALLOWED_SEVERITIES = {
    "hard_reject",
    "density_reject",
    "phrase_reject",
    "audit_only",
    "source_reject",
}
ALLOWED_TYPES = {"literal", "phrase", "regex", "url_regex", "wrapper_marker"}
ALLOWED_VIEWS = {
    "canonical",
    "ascii_folded",
    "depunct_compact",
    "escape_decoded",
    "source_metadata",
    "url",
}
ALLOWED_FLAGS = {"MULTILINE", "DOTALL", "WORD"}


class OntologyLoadError(RuntimeError):
    """Raised when ontology artifacts are invalid or unsafe."""


class _StrictYamlLoader(yaml.SafeLoader):
    pass


_StrictYamlLoader.yaml_implicit_resolvers = {
    key: list(value) for key, value in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _strip_implicit_resolvers(loader_cls: type[yaml.SafeLoader]) -> None:
    for key, resolvers in list(loader_cls.yaml_implicit_resolvers.items()):
        loader_cls.yaml_implicit_resolvers[key] = [
            (tag, regexp)
            for tag, regexp in resolvers
            if tag
            not in {
                "tag:yaml.org,2002:bool",
                "tag:yaml.org,2002:timestamp",
                "tag:yaml.org,2002:null",
            }
        ]


def _construct_mapping_no_duplicates(loader: yaml.SafeLoader, node: yaml.Node) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key == "<<":
            raise OntologyLoadError("YAML merge keys are forbidden")
        if key in mapping:
            raise OntologyLoadError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node)
    return mapping


_strip_implicit_resolvers(_StrictYamlLoader)
_StrictYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_no_duplicates,
)


@dataclass(frozen=True)
class RiskRule:
    id: str
    category: str
    severity: str
    type: str
    view: str
    reason: str
    value: str | None = None
    pattern: str | None = None
    flags: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    weight: float = 1.0
    threshold: int = 1
    compiled: Any | None = field(default=None, compare=False, repr=False)

    @property
    def expression(self) -> str:
        return self.pattern if self.pattern is not None else self.value or ""

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "severity": self.severity,
            "type": self.type,
            "view": self.view,
            "reason": self.reason,
            "value": self.value,
            "pattern": self.pattern,
            "flags": list(self.flags),
            "examples": list(self.examples),
            "weight": self.weight,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class RiskOntology:
    rules: tuple[RiskRule, ...]
    file_hashes: dict[str, str]
    source_tree_hash: str
    semantic_hash: str
    compiled_rule_manifest: tuple[dict[str, Any], ...]
    regex_engine: str
    regex_engine_version: str
    yaml_parser_version: str
    python_version: str
    python_unicode_version: str
    match_timeout_ms: int = MATCH_TIMEOUT_MS


def load_ontology(path: str | Path) -> RiskOntology:
    root = Path(path)
    files = _ontology_files(root)
    if not files:
        raise OntologyLoadError(f"no ontology YAML files found under {root}")

    all_rules: list[RiskRule] = []
    file_hashes: dict[str, str] = {}
    category_ids: set[str] = set()
    rule_ids: set[str] = set()

    for file_path in files:
        relative = file_path.relative_to(root if root.is_dir() else root.parent).as_posix()
        text = file_path.read_text(encoding="utf-8")
        _reject_yaml_aliases_and_anchors(text)
        file_hashes[relative] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        parsed = _strict_yaml_load(text, relative)
        rules = _validate_ontology_file(parsed, relative, category_ids, rule_ids)
        all_rules.extend(rules)

    all_rules = sorted(all_rules, key=lambda rule: rule.id)
    semantic_payload = {
        "schema_version": SCHEMA_VERSION,
        "files": sorted(file_hashes),
        "rules": [rule.semantic_payload() for rule in all_rules],
    }
    compiled_manifest = tuple(
        {
            "id": rule.id,
            "category": rule.category,
            "severity": rule.severity,
            "type": rule.type,
            "view": rule.view,
            "flags": list(rule.flags),
            "expression_sha256": hashlib.sha256(
                rule.expression.encode("utf-8")
            ).hexdigest(),
        }
        for rule in all_rules
    )
    source_tree_hash = stable_json_hash(file_hashes)
    semantic_hash = stable_json_hash(semantic_payload)

    return RiskOntology(
        rules=tuple(all_rules),
        file_hashes=file_hashes,
        source_tree_hash=source_tree_hash,
        semantic_hash=semantic_hash,
        compiled_rule_manifest=compiled_manifest,
        regex_engine="regex",
        regex_engine_version=getattr(regex, "__version__", "unknown"),
        yaml_parser_version=getattr(yaml, "__version__", "unknown"),
        python_version=sys.version.split()[0],
        python_unicode_version=unicodedata.unidata_version,
    )


def _ontology_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        [
            path
            for path in root.rglob("*.yaml")
            if path.is_file() and not any(part.startswith(".") for part in path.parts)
        ],
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _reject_yaml_aliases_and_anchors(text: str) -> None:
    try:
        for event in yaml.parse(text, Loader=_StrictYamlLoader):
            if isinstance(event, yaml.AliasEvent):
                raise OntologyLoadError("YAML aliases are forbidden")
            if getattr(event, "anchor", None):
                raise OntologyLoadError("YAML anchors are forbidden")
        node = yaml.compose(text, Loader=_StrictYamlLoader)
        if node is not None:
            _reject_plain_implicit_traps(node)
    except yaml.YAMLError as exc:
        raise OntologyLoadError(f"invalid YAML syntax: {exc}") from exc


def _reject_plain_implicit_traps(node: yaml.Node) -> None:
    if isinstance(node, yaml.ScalarNode):
        value = str(node.value)
        lowered = value.casefold()
        if node.style is None and (
            lowered in {"on", "off", "yes", "no", "true", "false", "null", "~"}
            or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
        ):
            raise OntologyLoadError(f"ambiguous plain YAML scalar is forbidden: {value}")
        return
    if isinstance(node, yaml.SequenceNode):
        for child in node.value:
            _reject_plain_implicit_traps(child)
        return
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            _reject_plain_implicit_traps(key_node)
            _reject_plain_implicit_traps(value_node)


def _strict_yaml_load(text: str, relative: str) -> dict[str, Any]:
    try:
        parsed = yaml.load(text, Loader=_StrictYamlLoader)
    except OntologyLoadError:
        raise
    except yaml.YAMLError as exc:
        raise OntologyLoadError(f"{relative}: invalid YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise OntologyLoadError(f"{relative}: ontology file must be a mapping")
    return parsed


def _validate_ontology_file(
    parsed: dict[str, Any],
    relative: str,
    category_ids: set[str],
    rule_ids: set[str],
) -> list[RiskRule]:
    unknown = set(parsed) - ALLOWED_TOP_LEVEL_KEYS
    if unknown:
        raise OntologyLoadError(f"{relative}: unknown top-level keys: {sorted(unknown)}")

    version = parsed.get("version")
    if version != SCHEMA_VERSION:
        raise OntologyLoadError(f"{relative}: version must be {SCHEMA_VERSION}")

    category = parsed.get("category")
    description = parsed.get("description")
    rules = parsed.get("rules")
    if not isinstance(category, str) or not category:
        raise OntologyLoadError(f"{relative}: category must be a non-empty string")
    if not isinstance(description, str) or not description:
        raise OntologyLoadError(f"{relative}: description must be a non-empty string")
    if category in category_ids:
        raise OntologyLoadError(f"{relative}: duplicate category id: {category}")
    category_ids.add(category)
    if not isinstance(rules, list) or not rules:
        raise OntologyLoadError(f"{relative}: rules must be a non-empty list")

    validated: list[RiskRule] = []
    for index, raw_rule in enumerate(rules):
        if not isinstance(raw_rule, dict):
            raise OntologyLoadError(f"{relative}: rule {index} must be a mapping")
        validated.append(_validate_rule(raw_rule, relative, category, rule_ids))
    return validated


def _validate_rule(
    raw_rule: dict[str, Any],
    relative: str,
    category: str,
    rule_ids: set[str],
) -> RiskRule:
    unknown = set(raw_rule) - ALLOWED_RULE_KEYS
    if unknown:
        raise OntologyLoadError(f"{relative}: unknown rule keys: {sorted(unknown)}")

    rule_id = _required_str(raw_rule, "id", relative)
    if rule_id in rule_ids:
        raise OntologyLoadError(f"{relative}: duplicate rule id: {rule_id}")
    rule_ids.add(rule_id)

    severity = _required_str(raw_rule, "severity", relative)
    rule_type = _required_str(raw_rule, "type", relative)
    view = _required_str(raw_rule, "view", relative)
    reason = _required_str(raw_rule, "reason", relative)

    if severity not in ALLOWED_SEVERITIES:
        raise OntologyLoadError(f"{relative}: invalid severity for {rule_id}")
    if rule_type not in ALLOWED_TYPES:
        raise OntologyLoadError(f"{relative}: invalid type for {rule_id}")
    if view not in ALLOWED_VIEWS:
        raise OntologyLoadError(f"{relative}: invalid view for {rule_id}")

    value = raw_rule.get("value")
    pattern = raw_rule.get("pattern")
    if rule_type in {"literal", "phrase", "wrapper_marker"}:
        if not isinstance(value, str) or not value:
            raise OntologyLoadError(f"{relative}: {rule_id} requires value")
        if pattern is not None:
            raise OntologyLoadError(f"{relative}: {rule_id} must not set pattern")
    else:
        if not isinstance(pattern, str) or not pattern:
            raise OntologyLoadError(f"{relative}: {rule_id} requires pattern")
        if value is not None:
            raise OntologyLoadError(f"{relative}: {rule_id} must not set value")

    flags = raw_rule.get("flags", [])
    if flags is None:
        flags = []
    if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
        raise OntologyLoadError(f"{relative}: {rule_id} flags must be string list")
    invalid_flags = set(flags) - ALLOWED_FLAGS
    if invalid_flags:
        raise OntologyLoadError(f"{relative}: invalid flags for {rule_id}: {invalid_flags}")

    examples = raw_rule.get("examples", [])
    if not isinstance(examples, list) or not all(
        isinstance(example, str) for example in examples
    ):
        raise OntologyLoadError(f"{relative}: {rule_id} examples must be string list")

    weight = raw_rule.get("weight", 1.0)
    threshold = raw_rule.get("threshold", 1)
    if not isinstance(weight, (int, float)):
        raise OntologyLoadError(f"{relative}: {rule_id} weight must be numeric")
    if not isinstance(threshold, int) or threshold < 1:
        raise OntologyLoadError(f"{relative}: {rule_id} threshold must be positive int")

    expression = pattern if pattern is not None else value
    assert expression is not None
    if rule_type in {"regex", "url_regex"}:
        _lint_regex(expression, relative, rule_id)
    compiled = _compile_rule_expression(
        rule_type,
        expression,
        view,
        tuple(flags),
        relative,
        rule_id,
    )

    return RiskRule(
        id=rule_id,
        category=category,
        severity=severity,
        type=rule_type,
        view=view,
        value=value,
        pattern=pattern,
        flags=tuple(flags),
        reason=reason,
        examples=tuple(examples),
        weight=float(weight),
        threshold=threshold,
        compiled=compiled,
    )


def _required_str(raw: dict[str, Any], key: str, relative: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise OntologyLoadError(f"{relative}: required string field missing: {key}")
    return value


def _compile_rule_expression(
    rule_type: str,
    expression: str,
    view: str,
    flags: tuple[str, ...],
    relative: str,
    rule_id: str,
) -> Any:
    regex_flags = regex.VERSION1
    if "MULTILINE" in flags:
        regex_flags |= regex.MULTILINE
    if "DOTALL" in flags:
        regex_flags |= regex.DOTALL
    if "WORD" in flags:
        regex_flags |= regex.WORD

    if rule_type == "literal":
        pattern = regex.escape(expression) if view == "depunct_compact" else _bounded_literal_pattern(expression)
    elif rule_type in {"phrase", "wrapper_marker"}:
        pattern = regex.escape(expression.replace(" ", "")) if view == "depunct_compact" else _phrase_pattern(expression)
    else:
        pattern = expression

    try:
        return regex.compile(pattern, regex_flags)
    except regex.error as exc:
        raise OntologyLoadError(f"{relative}: {rule_id} invalid regex: {exc}") from exc


def _bounded_literal_pattern(value: str) -> str:
    return rf"(?<![\p{{L}}\p{{N}}_]){regex.escape(value)}(?![\p{{L}}\p{{N}}_])"


def _phrase_pattern(value: str) -> str:
    parts = [regex.escape(part) for part in value.split()]
    if not parts:
        return r"a\A"
    joined = r"[\s\p{P}\p{S}]*".join(parts)
    return rf"(?<![\p{{L}}\p{{N}}_]){joined}(?![\p{{L}}\p{{N}}_])"


def _lint_regex(pattern: str, relative: str, rule_id: str) -> None:
    if len(pattern) > 500:
        raise OntologyLoadError(f"{relative}: {rule_id} regex is too long")
    banned_fragments = [
        r"\1",
        r"\2",
        r"\3",
        "(?R",
        "(?0",
        "(?&",
        "(?P=",
        "{e",
        "{i",
    ]
    if any(fragment in pattern for fragment in banned_fragments):
        raise OntologyLoadError(f"{relative}: {rule_id} regex uses banned construct")

    suspicious = [
        r"\([^)]*[+*][^)]*\)[+*]",
        r"\([^)]*\{[0-9,]+\}[^)]*\)[+*{]",
        r"\(\.\*\)[+*]",
        r"\(\.\+\)[+*]",
        r"\(\?:\.\*\)[+*]",
        r"\(\?:\.\+\)[+*]",
        r"\.\*.*\.\*",
        r"\(\?:[^)]*\|[^)]*\)[+*]",
        r"\([^)]*\|[^)]*\)[+*]",
        r"\(\?<=[^)]*(?<!\\)[+*]",
        r"\(\?<![^)]*(?<!\\)[+*]",
    ]
    if any(regex.search(item, pattern) for item in suspicious):
        raise OntologyLoadError(f"{relative}: {rule_id} regex failed safety lint")

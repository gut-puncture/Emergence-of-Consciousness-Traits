"""Production chunk-classifier inference and artifact validation."""

from __future__ import annotations

import functools
import hashlib
import math
import re
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twc.classifiers import chunk_contracts
from twc.classifiers.feature_extraction import WHITESPACE_RE, WORD_RE
from twc.classifiers.precision_policy import (
    W1_PRECISION_AMP_BF16,
    W1_PRECISION_FP32,
    apply_w1_precision_policy,
    build_w1_precision_policy,
    normalize_w1_precision_mode,
    w1_precision_context,
)
from twc.classifiers.window_model import (
    W1_ATTN_IMPLEMENTATION_SDPA,
    normalize_w1_attn_implementation,
)

from twc.repro.hashing import sha256_file as _sha256_file


HASHED_NGRAM_COLUMN_CACHE_MAXSIZE = 262_144
ASCII_WORD_BYTES_RE = re.compile(rb"\w+", re.ASCII)
ASCII_LOWER_WHITESPACE_TRANSLATION = bytes.maketrans(
    bytes(range(128)),
    bytes(
        32
        if chr(value).isspace()
        else value + 32
        if 65 <= value <= 90
        else value
        for value in range(128)
    ),
)


def _hashed_ngram_column_counts(
    text: str,
    *,
    n_features: int = 2**18,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    sort_output: bool = True,
) -> dict[int, int]:
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    min_word, max_word = word_ngram_range
    if min_word < 1 or max_word < min_word:
        raise ValueError("invalid word_ngram_range")
    min_char, max_char = char_ngram_range
    if min_char < 1 or max_char < min_char:
        raise ValueError("invalid char_ngram_range")

    if text.isascii():
        return _hashed_ngram_column_counts_ascii(
            text,
            n_features=n_features,
            word_ngram_range=word_ngram_range,
            char_ngram_range=char_ngram_range,
            sort_output=sort_output,
        )

    normalized = WHITESPACE_RE.sub(" ", text.casefold()).strip()
    words = WORD_RE.findall(normalized)
    counts: dict[int, int] = {}

    for ngram_size in range(min_word, max_word + 1):
        if len(words) < ngram_size:
            continue
        prefix = f"w{ngram_size}"
        if ngram_size == 1:
            for word in words:
                column = _hashed_ngram_column(prefix, word, n_features=n_features)
                counts[column] = counts.get(column, 0) + 1
            continue
        if ngram_size == 2:
            for index in range(0, len(words) - 1):
                column = _hashed_ngram_column(prefix, words[index] + " " + words[index + 1], n_features=n_features)
                counts[column] = counts.get(column, 0) + 1
            continue
        for index in range(0, len(words) - ngram_size + 1):
            ngram = " ".join(words[index : index + ngram_size])
            column = _hashed_ngram_column(prefix, ngram, n_features=n_features)
            counts[column] = counts.get(column, 0) + 1

    char_view = f" {normalized} "
    if char_view.isascii():
        char_bytes = char_view.encode("ascii")
        for ngram_size in range(min_char, max_char + 1):
            if len(char_bytes) < ngram_size:
                continue
            prefix_bytes = f"c{ngram_size}:".encode("ascii")
            for index in range(0, len(char_bytes) - ngram_size + 1):
                column = _hashed_ngram_column_bytes(
                    prefix_bytes,
                    char_bytes[index : index + ngram_size],
                    n_features=n_features,
                )
                counts[column] = counts.get(column, 0) + 1
    else:
        for ngram_size in range(min_char, max_char + 1):
            if len(char_view) < ngram_size:
                continue
            prefix = f"c{ngram_size}"
            for index in range(0, len(char_view) - ngram_size + 1):
                column = _hashed_ngram_column(prefix, char_view[index : index + ngram_size], n_features=n_features)
                counts[column] = counts.get(column, 0) + 1
    if sort_output:
        return dict(sorted(counts.items()))
    return counts


def _hashed_ngram_column_counts_ascii(
    text: str,
    *,
    n_features: int,
    word_ngram_range: tuple[int, int],
    char_ngram_range: tuple[int, int],
    sort_output: bool,
) -> dict[int, int]:
    normalized = b" ".join(text.encode("ascii").translate(ASCII_LOWER_WHITESPACE_TRANSLATION).split())
    words = ASCII_WORD_BYTES_RE.findall(normalized)
    counts: dict[int, int] = {}

    min_word, max_word = word_ngram_range
    for ngram_size in range(min_word, max_word + 1):
        if len(words) < ngram_size:
            continue
        prefix_bytes = f"w{ngram_size}:".encode("ascii")
        if ngram_size == 1:
            for word in words:
                column = _hashed_ngram_column_bytes(prefix_bytes, word, n_features=n_features)
                counts[column] = counts.get(column, 0) + 1
            continue
        if ngram_size == 2:
            for index in range(0, len(words) - 1):
                column = _hashed_word_bigram_column_bytes(
                    words[index],
                    words[index + 1],
                    n_features=n_features,
                )
                counts[column] = counts.get(column, 0) + 1
            continue
        for index in range(0, len(words) - ngram_size + 1):
            column = _hashed_ngram_column_bytes(
                prefix_bytes,
                b" ".join(words[index : index + ngram_size]),
                n_features=n_features,
            )
            counts[column] = counts.get(column, 0) + 1

    min_char, max_char = char_ngram_range
    char_view = b" " + normalized + b" "
    for ngram_size in range(min_char, max_char + 1):
        if len(char_view) < ngram_size:
            continue
        prefix_bytes = f"c{ngram_size}:".encode("ascii")
        for index in range(0, len(char_view) - ngram_size + 1):
            column = _hashed_ngram_column_bytes(
                prefix_bytes,
                char_view[index : index + ngram_size],
                n_features=n_features,
            )
            counts[column] = counts.get(column, 0) + 1

    if sort_output:
        return dict(sorted(counts.items()))
    return counts


@functools.lru_cache(maxsize=HASHED_NGRAM_COLUMN_CACHE_MAXSIZE)
def _hashed_ngram_column(prefix: str, value: str, *, n_features: int) -> int:
    digest = hashlib.blake2b(f"{prefix}:{value}".encode("utf-8", errors="surrogatepass"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % int(n_features)


@functools.lru_cache(maxsize=HASHED_NGRAM_COLUMN_CACHE_MAXSIZE)
def _hashed_ngram_column_bytes(prefix_with_separator: bytes, value: bytes, *, n_features: int) -> int:
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(prefix_with_separator)
    hasher.update(value)
    digest = hasher.digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % int(n_features)


@functools.lru_cache(maxsize=HASHED_NGRAM_COLUMN_CACHE_MAXSIZE)
def _hashed_word_bigram_column_bytes(left: bytes, right: bytes, *, n_features: int) -> int:
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(b"w2:")
    hasher.update(left)
    hasher.update(b" ")
    hasher.update(right)
    digest = hasher.digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % int(n_features)


class ChunkSparseRiskScorer:
    model_id = chunk_contracts.CHUNK_SPARSE_MODEL_ID

    def __init__(self, *, model: Any, threshold: float, n_features: int) -> None:
        if not 0 <= float(threshold) <= 1:
            raise ValueError("sparse threshold must be in [0, 1]")
        if int(n_features) < 2:
            raise ValueError("sparse n_features must be at least 2")
        self.model = model
        self.threshold = float(threshold)
        self.n_features = int(n_features)
        self._fused_log_loss_params = self._resolve_fused_log_loss_params(model)

    @classmethod
    def from_report(cls, report: Mapping[str, Any], model_path: str | Path) -> "ChunkSparseRiskScorer":
        validate_sparse_inference_artifacts(report, model_path)
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("joblib is required to load chunk sparse classifier") from exc
        _install_sklearn_loss_pickle_compat_alias()
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            model = joblib.load(model_path)
        for warning_record in warning_records:
            if type(warning_record.message).__name__ == "InconsistentVersionWarning":
                raise ValueError(
                    "chunk sparse model scikit-learn version mismatch; install the model training runtime before S1 filtering"
                )
        feature_config = report.get("feature_config")
        if not isinstance(feature_config, Mapping):
            raise ValueError("chunk sparse report missing feature_config")
        return cls(
            model=model,
            threshold=float(report["threshold"]),
            n_features=int(feature_config["n_features"]),
        )

    def score_many(self, texts: list[str], metadata: list[dict[str, Any]]) -> list[float]:
        del metadata
        if not texts:
            return []
        fused = self._fused_log_loss_params
        if fused is not None:
            coef, intercept, numpy_module = fused
            score_by_text: dict[str, float] = {}
            scores: list[float] = []
            for text in texts:
                cached = score_by_text.get(text)
                if cached is None:
                    cached = _sigmoid(
                        intercept
                        + _normalized_hashed_ngram_dot(
                            text,
                            n_features=self.n_features,
                            coef=coef,
                            numpy_module=numpy_module,
                        )
                    )
                    score_by_text[text] = cached
                scores.append(cached)
            return scores
        try:
            import numpy
            from scipy.sparse import csr_matrix
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("numpy and scipy are required for chunk sparse inference") from exc
        data: list[float] = []
        indices: list[int] = []
        indptr = [0]
        for text in texts:
            row_values: list[tuple[int, float]] = []
            for column, count in _hashed_ngram_column_counts(
                text,
                n_features=self.n_features,
                sort_output=False,
            ).items():
                row_values.append((column, _sparse_tf_value(count)))
            if not row_values:
                row_values.append((0, 0.0))
            norm = math.sqrt(sum(value * value for _column, value in row_values)) or 1.0
            for column, value in row_values:
                indices.append(column)
                data.append(value / norm)
            indptr.append(len(indices))
        matrix = csr_matrix(
            (
                numpy.array(data, dtype=numpy.float64),
                numpy.array(indices, dtype=numpy.int32),
                numpy.array(indptr, dtype=numpy.int32),
            ),
            shape=(len(texts), self.n_features),
        )
        probabilities = self.model.predict_proba(matrix)
        return [chunk_contracts.coerce_risk_score(score[1]) for score in probabilities]

    def _resolve_fused_log_loss_params(self, model: Any) -> tuple[Any, float, Any] | None:
        """Return exact binary SGD log-loss parameters when direct scoring is safe."""

        if getattr(model, "loss", None) != "log_loss":
            return None
        classes = getattr(model, "classes_", None)
        coef = getattr(model, "coef_", None)
        intercept = getattr(model, "intercept_", None)
        n_features_in = getattr(model, "n_features_in_", self.n_features)
        try:
            import numpy
        except ImportError:  # pragma: no cover - production sparse deps include numpy.
            return None
        if classes is None or coef is None or intercept is None:
            return None
        class_values = list(classes.tolist() if hasattr(classes, "tolist") else classes)
        if class_values != [0, 1]:
            return None
        if getattr(coef, "shape", None) != (1, self.n_features):
            return None
        if getattr(intercept, "shape", None) != (1,):
            return None
        if int(n_features_in) != self.n_features:
            return None
        coef_row = numpy.asarray(coef[0], dtype=numpy.float64)
        intercept_value = float(numpy.asarray(intercept, dtype=numpy.float64)[0])
        if not bool(numpy.isfinite(intercept_value)) or not bool(numpy.isfinite(coef_row).all()):
            return None
        return coef_row, intercept_value, numpy


def _sparse_tf_value(count: int) -> float:
    if int(count) == 1:
        return 1.0
    return 1.0 + math.log(float(count))


def _normalized_hashed_ngram_dot(text: str, *, n_features: int, coef: Any, numpy_module: Any) -> float:
    counts = _hashed_ngram_column_counts(text, n_features=n_features, sort_output=False)
    if not counts:
        return 0.0
    columns = numpy_module.fromiter(counts.keys(), dtype=numpy_module.intp, count=len(counts))
    values = numpy_module.fromiter(
        (_sparse_tf_value(count) for count in counts.values()),
        dtype=numpy_module.float64,
        count=len(counts),
    )
    norm = math.sqrt(float(numpy_module.dot(values, values))) or 1.0
    return float(numpy_module.dot(coef[columns], values)) / norm


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        z = math.exp(-float(logit))
        return 1.0 / (1.0 + z)
    z = math.exp(float(logit))
    return z / (1.0 + z)


class ChunkWindowMilRiskScorer:
    model_id = chunk_contracts.CHUNK_WINDOW_MIL_MODEL_ID

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        device: Any,
        threshold: float,
        max_length: int,
        stride: int,
        max_windows_per_device_batch: int,
        artifact_max_windows_per_device_batch: int | None = None,
        requested_max_windows_per_device_batch: int | None = None,
        window_variant_config: Mapping[str, Any] | None = None,
        window_mil_precision: str = W1_PRECISION_FP32,
        synchronize_cuda_telemetry: bool = False,
        w1_attn_implementation_requested: str = W1_ATTN_IMPLEMENTATION_SDPA,
        w1_attn_implementation_observed: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.device = device
        self.threshold = float(threshold)
        self.max_length = int(max_length)
        self.stride = int(stride)
        self.artifact_max_windows_per_device_batch = int(
            artifact_max_windows_per_device_batch
            if artifact_max_windows_per_device_batch is not None
            else max_windows_per_device_batch
        )
        self.max_windows_per_device_batch = int(max_windows_per_device_batch)
        self.requested_max_windows_per_device_batch = (
            None
            if requested_max_windows_per_device_batch is None
            else int(requested_max_windows_per_device_batch)
        )
        self.window_mil_precision = normalize_w1_precision_mode(window_mil_precision)
        self.precision_policy = build_w1_precision_policy(self.window_mil_precision)
        self.synchronize_cuda_telemetry = bool(synchronize_cuda_telemetry)
        self.w1_attn_implementation_requested = normalize_w1_attn_implementation(
            w1_attn_implementation_requested
        )
        observed_attn = w1_attn_implementation_observed or self.w1_attn_implementation_requested
        self.w1_attn_implementation_observed = normalize_w1_attn_implementation(observed_attn)
        self.window_variant_config = dict(window_variant_config or {
            "input_mode": chunk_contracts.WINDOW_INPUT_FULL_EVIDENCE,
            "pooling_mode": "max",
            "pooling_top_k": 1,
            "halo_weight": 0.25,
            "halo_logit_clamp": 2.0,
        })

    @classmethod
    def from_report(
        cls,
        report: Mapping[str, Any],
        model_path: str | Path,
        *,
        device: str = "auto",
        require_cuda: bool = False,
        runtime_max_windows_per_device_batch: int | None = None,
        window_mil_precision: str = W1_PRECISION_FP32,
        synchronize_cuda_telemetry: bool = False,
        w1_attn_implementation: str = W1_ATTN_IMPLEMENTATION_SDPA,
    ) -> "ChunkWindowMilRiskScorer":
        validate_window_mil_inference_artifacts(report, model_path)
        try:
            from twc.classifiers.window_model import (
                DEFAULT_MODEL_NAME,
                _load_model_bundle,
                _load_torch,
                _resolve_device,
            )
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("Window-MIL inference requires classifier window_mil dependencies") from exc
        torch = _load_torch()
        resolved_device = _resolve_device(torch, device=device, require_cuda=require_cuda)
        checkpoint = torch.load(model_path, map_location=resolved_device)
        if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
            raise ValueError("chunk Window-MIL checkpoint is malformed")
        model_name = str(checkpoint.get("model_name") or report.get("base_model_name") or DEFAULT_MODEL_NAME)
        requested_attn_implementation = normalize_w1_attn_implementation(w1_attn_implementation)
        tokenizer, model, resolved_model_name = _load_model_bundle(
            model_name=model_name,
            model_bundle_factory=None,
            torch=torch,
            attn_implementation=requested_attn_implementation,
        )
        observed_attn_implementation = normalize_w1_attn_implementation(
            getattr(model, "w1_attn_implementation_observed", requested_attn_implementation)
        )
        if str(report.get("base_model_name") or resolved_model_name) != resolved_model_name:
            raise ValueError("chunk Window-MIL base model name drift")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(resolved_device)
        model.eval()
        precision_mode = normalize_w1_precision_mode(window_mil_precision)
        if precision_mode == W1_PRECISION_AMP_BF16 and str(getattr(resolved_device, "type", "")) != "cuda":
            raise RuntimeError("amp_bf16 W1 precision requires a CUDA device")
        apply_w1_precision_policy(torch, precision_mode)
        artifact_max_windows = int(report["max_windows_per_device_batch"])
        requested_max_windows = (
            None
            if runtime_max_windows_per_device_batch is None
            else int(runtime_max_windows_per_device_batch)
        )
        if requested_max_windows is not None and requested_max_windows < 1:
            raise ValueError("runtime max_windows_per_device_batch must be positive")
        effective_max_windows = requested_max_windows if requested_max_windows is not None else artifact_max_windows
        return cls(
            model=model,
            tokenizer=tokenizer,
            torch_module=torch,
            device=resolved_device,
            threshold=float(report["threshold"]),
            max_length=int(report["max_length"]),
            stride=int(report["stride"]),
            max_windows_per_device_batch=effective_max_windows,
            artifact_max_windows_per_device_batch=artifact_max_windows,
            requested_max_windows_per_device_batch=requested_max_windows,
            window_variant_config=report.get("window_variant_config"),
            window_mil_precision=precision_mode,
            synchronize_cuda_telemetry=synchronize_cuda_telemetry,
            w1_attn_implementation_requested=requested_attn_implementation,
            w1_attn_implementation_observed=observed_attn_implementation,
        )

    def score_many(self, texts: list[str], metadata: list[dict[str, Any]]) -> list[float]:
        if not texts:
            self.last_score_many_telemetry = {}
            return []
        from twc.classifiers.window_model import _score_document_batch, _score_split_halo_document_batch

        telemetry: dict[str, Any] = {
            "synchronize_cuda_timing": bool(self.synchronize_cuda_telemetry),
            "cuda_synchronized_timing": False,
            "w1_precision_policy": self.precision_policy,
            "w1_precision_policy_hash": self.precision_policy["precision_policy_hash"],
        }
        try:
            inference_context = getattr(self.torch, "inference_mode", None)
            with (inference_context() if inference_context is not None else self.torch.no_grad()):
                with w1_precision_context(
                    self.torch,
                    mode=self.window_mil_precision,
                    device_type=str(getattr(self.device, "type", "")),
                ):
                    input_mode = str(self.window_variant_config.get("input_mode") or chunk_contracts.WINDOW_INPUT_FULL_EVIDENCE)
                    pooling_mode = str(self.window_variant_config.get("pooling_mode") or "max")
                    pooling_top_k = int(self.window_variant_config.get("pooling_top_k", 1))
                    if input_mode == chunk_contracts.WINDOW_INPUT_SPLIT_HALO_CAPPED:
                        core_texts, halo_texts = _runtime_core_and_halo_texts(texts, metadata)
                        scores = _score_split_halo_document_batch(
                            torch=self.torch,
                            model=self.model,
                            tokenizer=self.tokenizer,
                            core_texts=core_texts,
                            halo_texts=halo_texts,
                            device=self.device,
                            max_length=self.max_length,
                            stride=self.stride,
                            max_windows_per_device_batch=self.max_windows_per_device_batch,
                            pooling_mode=pooling_mode,
                            pooling_top_k=pooling_top_k,
                            halo_weight=float(self.window_variant_config.get("halo_weight", 0.25)),
                            halo_logit_clamp=float(self.window_variant_config.get("halo_logit_clamp", 2.0)),
                            telemetry=telemetry,
                        )
                    else:
                        scores = _score_document_batch(
                            torch=self.torch,
                            model=self.model,
                            tokenizer=self.tokenizer,
                            texts=_runtime_texts_for_input_mode(texts, metadata, input_mode=input_mode),
                            device=self.device,
                            max_length=self.max_length,
                            stride=self.stride,
                            max_windows_per_device_batch=self.max_windows_per_device_batch,
                            pooling_mode=pooling_mode,
                            pooling_top_k=pooling_top_k,
                            telemetry=telemetry,
                        )
        finally:
            self.last_score_many_telemetry = dict(telemetry)
        return [chunk_contracts.coerce_risk_score(score) for score in scores]

    def prepare_many(self, texts: list[str], metadata: list[dict[str, Any]], *, pin_memory: bool = False) -> Any:
        if not texts:
            raise ValueError("W1 prepared scoring requires at least one text")
        if len(texts) != len(metadata):
            raise ValueError("W1 prepared scoring metadata count mismatch")
        input_mode = str(self.window_variant_config.get("input_mode") or chunk_contracts.WINDOW_INPUT_FULL_EVIDENCE)
        if input_mode == chunk_contracts.WINDOW_INPUT_SPLIT_HALO_CAPPED:
            raise NotImplementedError("bounded W1 prepare pipeline does not support split-halo input mode")
        pooling_mode = str(self.window_variant_config.get("pooling_mode") or "max")
        pooling_top_k = int(self.window_variant_config.get("pooling_top_k", 1))
        from twc.classifiers.window_model import _prepare_document_logits_batch

        return _prepare_document_logits_batch(
            torch=self.torch,
            tokenizer=self.tokenizer,
            texts=_runtime_texts_for_input_mode(texts, metadata, input_mode=input_mode),
            max_length=self.max_length,
            stride=self.stride,
            pooling_mode=pooling_mode,
            pooling_top_k=pooling_top_k,
            pin_memory=bool(pin_memory),
            metadata=metadata,
        )

    def score_prepared(self, prepared_batch: Any) -> list[float]:
        if prepared_batch is None:
            raise ValueError("prepared W1 batch is required")
        from twc.classifiers.window_model import _document_logits_from_prepared_batch

        telemetry: dict[str, Any] = {
            "synchronize_cuda_timing": bool(self.synchronize_cuda_telemetry),
            "cuda_synchronized_timing": False,
            "w1_precision_policy": self.precision_policy,
            "w1_precision_policy_hash": self.precision_policy["precision_policy_hash"],
            "prepared_batch_scored": True,
        }
        try:
            inference_context = getattr(self.torch, "inference_mode", None)
            with (inference_context() if inference_context is not None else self.torch.no_grad()):
                with w1_precision_context(
                    self.torch,
                    mode=self.window_mil_precision,
                    device_type=str(getattr(self.device, "type", "")),
                ):
                    doc_logits = _document_logits_from_prepared_batch(
                        torch=self.torch,
                        model=self.model,
                        prepared=prepared_batch,
                        device=self.device,
                        max_windows_per_device_batch=self.max_windows_per_device_batch,
                        telemetry=telemetry,
                    )
                    from twc.classifiers.window_model import _synchronize_cuda_for_telemetry, _telemetry_add

                    _synchronize_cuda_for_telemetry(torch=self.torch, device=self.device, telemetry=telemetry)
                    postprocess_at = time.monotonic()
                    scores = [float(value) for value in self.torch.sigmoid(doc_logits).detach().cpu().tolist()]
                    _synchronize_cuda_for_telemetry(torch=self.torch, device=self.device, telemetry=telemetry)
                    _telemetry_add(telemetry, "postprocess_seconds", time.monotonic() - postprocess_at)
        finally:
            self.last_score_many_telemetry = dict(telemetry)
        return [chunk_contracts.coerce_risk_score(score) for score in scores]


def _runtime_texts_for_input_mode(
    texts: Sequence[str],
    metadata: Sequence[Mapping[str, Any]],
    *,
    input_mode: str,
) -> list[str]:
    if input_mode == chunk_contracts.WINDOW_INPUT_FULL_EVIDENCE:
        return [str(text) for text in texts]
    if input_mode == chunk_contracts.WINDOW_INPUT_CORE_ONLY:
        core_texts, _halo_texts = _runtime_core_and_halo_texts(texts, metadata)
        return core_texts
    raise ValueError(f"unsupported Window-MIL input mode: {input_mode}")


def _runtime_core_and_halo_texts(
    texts: Sequence[str],
    metadata: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    if len(texts) != len(metadata):
        raise ValueError("Window-MIL runtime metadata count mismatch")
    core_texts: list[str] = []
    halo_texts: list[str] = []
    for evidence_text, row_metadata in zip(texts, metadata, strict=True):
        core_start = int(row_metadata.get("core_char_start", -1))
        core_end = int(row_metadata.get("core_char_end", -1))
        if core_start < 0 or core_end <= core_start or core_end > len(evidence_text):
            raise ValueError("Window-MIL runtime metadata missing valid core span")
        core_text = str(evidence_text)[core_start:core_end]
        expected_core_hash = row_metadata.get("core_text_sha256")
        if expected_core_hash is not None and hashlib.sha256(core_text.encode("utf-8", errors="surrogatepass")).hexdigest() != str(expected_core_hash):
            raise ValueError("Window-MIL runtime core hash drift")
        core_texts.append(core_text)
        halo_texts.append((str(evidence_text)[:core_start] + "\n" + str(evidence_text)[core_end:]).strip())
    return core_texts, halo_texts


def validate_sparse_inference_artifacts(report: Mapping[str, Any], model_path: str | Path) -> None:
    chunk_contracts.validate_chunk_sparse_inference_report(report)
    path = Path(model_path)
    if str(report.get("model_file_name") or "") != path.name:
        raise ValueError("chunk sparse model file name drift")
    if not path.exists():
        raise ValueError("chunk sparse model file missing")
    if _sha256_file(path) != chunk_contracts.require_sha256(
        str(report.get("model_file_sha256") or ""),
        "chunk sparse model_file_sha256",
    ):
        raise ValueError("chunk sparse model file hash drift")


def validate_window_mil_inference_artifacts(report: Mapping[str, Any], model_path: str | Path) -> None:
    chunk_contracts.validate_chunk_window_mil_inference_report(report)
    path = Path(model_path)
    if str(report.get("model_file_name") or "") != path.name:
        raise ValueError("chunk Window-MIL model file name drift")
    if not path.exists():
        raise ValueError("chunk Window-MIL model file missing")
    if _sha256_file(path) != chunk_contracts.require_sha256(
        str(report.get("model_file_sha256") or ""),
        "chunk Window-MIL model_file_sha256",
    ):
        raise ValueError("chunk Window-MIL model file hash drift")


def validate_locked_sparse_runtime_artifact(
    report: Mapping[str, Any],
    model_path: str | Path,
    config_payload: Mapping[str, Any],
) -> None:
    """Bind a loadable sparse artifact to the frozen filtering configuration."""

    validate_sparse_inference_artifacts(report, model_path)
    if report.get("report_hash") is None:
        raise ValueError("S1 sparse report missing report_hash")
    if report.get("report_hash") != config_payload["sparse_report_hash"]:
        raise ValueError("S1 sparse report hash does not match locked config")
    if report.get("model_file_sha256") != config_payload["sparse_model_file_sha256"]:
        raise ValueError("S1 sparse model hash does not match locked config")
    if report.get("classifier_manifest_hash") != config_payload["classifier_manifest_hash"]:
        raise ValueError("S1 sparse classifier manifest hash mismatch")
    if report.get("evidence_cache_hash") != config_payload["private_evidence_cache_hash"]:
        raise ValueError("S1 sparse evidence cache hash mismatch")


def validate_locked_window_mil_runtime_artifact(
    report: Mapping[str, Any],
    model_path: str | Path,
    config_payload: Mapping[str, Any],
) -> None:
    """Bind a loadable Window-MIL artifact to the frozen filtering configuration."""

    validate_window_mil_inference_artifacts(report, model_path)
    if report.get("report_hash") is None:
        raise ValueError("W1 report missing report_hash")
    if report.get("report_hash") != config_payload["window_mil_report_hash"]:
        raise ValueError("W1 report hash does not match locked config")
    if report.get("model_file_sha256") != config_payload["window_mil_model_file_sha256"]:
        raise ValueError("W1 model hash does not match locked config")
    if report.get("classifier_manifest_hash") != config_payload["classifier_manifest_hash"]:
        raise ValueError("W1 classifier manifest hash mismatch")
    if report.get("evidence_cache_hash") != config_payload["private_evidence_cache_hash"]:
        raise ValueError("W1 evidence cache hash mismatch")
    variant = report.get("window_variant_config")
    if not isinstance(variant, Mapping):
        raise ValueError("W1 report missing window_variant_config")
    if str(variant.get("input_mode")) != config_payload["window_mil_input_mode"]:
        raise ValueError("W1 input_mode mismatch")
    if str(variant.get("pooling_mode")) != config_payload["window_mil_pooling_mode"]:
        raise ValueError("W1 pooling_mode mismatch")
    if int(variant.get("pooling_top_k", -1)) != int(config_payload["window_mil_pooling_top_k"]):
        raise ValueError("W1 pooling_top_k mismatch")


def _install_sklearn_loss_pickle_compat_alias() -> None:
    """Support sparse artifacts pickled with sklearn's bare compiled _loss module."""

    import importlib
    import sys

    sys.modules.setdefault("_loss", importlib.import_module("sklearn._loss._loss"))

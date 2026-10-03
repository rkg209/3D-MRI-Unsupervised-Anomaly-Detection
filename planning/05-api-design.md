# api-design.md

## 3D MRI Unsupervised Anomaly Detection — API Design

---

## 1. Preamble: Why This Document Exists Despite Having No Network API

This system has no HTTP server, no REST endpoints, no GraphQL schema, and no RPC layer. The architecture document states this explicitly. Nevertheless, an API design document is warranted — and useful — for three reasons.

First, the system has **module interfaces that function as APIs**: typed contracts between components that must be stable, versioned, and documented with the same rigor as a network API. A caller that depends on `ReconstructionEngine.run()` has the same need for interface stability as a client that depends on `POST /reconstruct`. The fact that the call happens in-process rather than over a network does not reduce the need for a contract.

Second, the system has **file-format interfaces that function as APIs**: the `.pt` files, JSON files, and CSV files written by one module and read by another are data contracts. A change to the schema of `aggregate.json` is a breaking change with the same consequences as a breaking change to a REST response body.

Third, this document serves as the **specification for any future promotion** of this system to a network service. The research codebase is explicitly designed to be a portfolio artifact. If it is ever wrapped in a FastAPI application or exposed as a shared research platform, this document provides the interface design from which HTTP endpoints would be derived — without requiring a redesign from scratch.

This document is organized into four sections corresponding to the four topics requested: authentication scheme, error handling conventions, pagination strategy, and versioning policy. Each section is presented in two registers: the **as-built internal design** (what the module interfaces and file contracts actually implement) and the **network projection** (what the equivalent HTTP API would look like, for future reference).

---

## 2. Authentication Scheme

### 2.1 As-Built: Internal Module Authentication

The current system has no authentication in the network sense. It is a single-user local codebase. However, it does implement two forms of **caller authorization** that serve the same purpose as authentication: ensuring that only legitimate callers can perform sensitive operations.

#### 2.1.1 Split Contract Authorization

The `SplitContract` is the most sensitive entity in the system. An unauthorized modification to the train/test split would silently invalidate all comparative results. The system enforces caller authorization through a **write-once file lock with explicit override**:

```python
class SplitContract:
    def create(
        self,
        dataset_name: str,
        seed: int,
        total_volumes: int,
        *,
        override: bool = False,   # must be explicitly set to True to overwrite
    ) -> None:
        existing = self._load_if_exists()
        if existing is not None and not override:
            raise SplitContractViolationError(
                f"SplitContract already exists for dataset='{dataset_name}' "
                f"seed={existing.seed}. "
                f"Pass override=True only if you understand that all existing "
                f"eval results will be invalidated. "
                f"Existing contract created at: {existing.created_at}"
            )
```

The `override=True` keyword-only argument is the authorization token. It cannot be set accidentally by positional argument. Any code path that calls `create()` without `override=True` is rejected if a contract already exists. This is equivalent to a write-protected resource that requires an elevated permission scope.

#### 2.1.2 Model Registry Authorization

The `ModelRegistry` enforces that only classes implementing the `AnomalyDetectionModel` ABC can be registered. This prevents accidental registration of incompatible objects:

```python
class ModelRegistry:
    def register(self, name: str, cls: type, config: DictConfig) -> None:
        if not (isinstance(cls, type) and issubclass(cls, AnomalyDetectionModel)):
            raise TypeError(
                f"Cannot register '{name}': {cls!r} does not implement "
                f"AnomalyDetectionModel. Required methods: forward(), "
                f"load_checkpoint(), model_card property."
            )
```

#### 2.1.3 Evaluation Harness Authorization

The `AggregateEvaluator` enforces that it cannot be used to re-run inference — only to read saved results. This prevents the evaluation harness from being misused as an inference engine:

```python
class AggregateEvaluator:
    def __init__(self, results_dir: Path, labels_dir: Path) -> None:
        # Accepts only Path arguments. Raises TypeError on any other type.
        if not isinstance(results_dir, Path):
            raise TypeError(
                f"results_dir must be a Path, not {type(results_dir).__name__}. "
                f"AggregateEvaluator reads saved ReconResult files; it does not "
                f"accept model objects or DataLoaders."
            )
```

#### 2.1.4 Checkpoint Integrity Verification

Before any model weights are loaded, `ModelRegistry` verifies the checkpoint file against its recorded size. This is a form of **data authentication** — ensuring the file has not been corrupted or substituted:

```python
def _verify_checkpoint_integrity(path: Path, expected_size_bytes: int) -> None:
    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size_bytes:
        raise CheckpointError(
            f"Checkpoint size mismatch at {path}: "
            f"expected {expected_size_bytes} bytes, found {actual_size} bytes. "
            f"The file may be truncated or corrupted."
        )
```

---

### 2.2 Network Projection: Authentication for a Future HTTP API

If this system were promoted to a shared research platform, the following authentication scheme would apply. The design is chosen to match the operational context: a small academic research group, not a public internet service.

#### 2.2.1 Authentication Mechanism: API Keys

**Chosen mechanism:** Static API keys passed in the `Authorization` header as Bearer tokens.

**Rationale:** OAuth 2.0 is over-engineered for a small research group. Session cookies require a browser client. mTLS requires certificate infrastructure. API keys are auditable, revocable, and sufficient for a controlled-access research tool.

```
Authorization: Bearer mri-ad-<base62-encoded-32-bytes>
```

API keys are generated server-side, stored as bcrypt hashes (never in plaintext), and associated with a `researcher_id`. Each key has an optional expiry date and a scope set.

#### 2.2.2 Authorization Scopes

| Scope | Permitted Operations |
|---|---|
| `read:results` | GET on any result, metric, or report endpoint |
| `write:eval` | POST to trigger eval runs; requires checkpoint to already exist |
| `write:train` | POST to trigger training runs; restricted to cluster environment |
| `write:split` | POST to create or override a SplitContract; highest privilege |
| `admin` | All operations; key rotation; user management |

The `write:split` scope is the network equivalent of the `override=True` keyword argument. It must be explicitly granted and is logged with the requesting `researcher_id` on every use.

#### 2.2.3 Request Authentication Flow

```
Client                          API Gateway                     Backend
──────                          ───────────                     ───────
POST /api/v1/eval
Authorization: Bearer <key>
                    ──────────► Extract key from header
                                Hash key with bcrypt
                                Lookup hash in key store
                                Validate expiry
                                Validate scope ⊇ {write:eval}
                                Attach researcher_id to request context
                                ──────────────────────────────► Process request
                                                                Log: researcher_id,
                                                                     run_id,
                                                                     timestamp
                    ◄────────── 202 Accepted { "run_id": "..." }
```

#### 2.2.4 Unauthenticated Endpoints

The following endpoints would be unauthenticated (public read-only):

- `GET /api/v1/health` — liveness check
- `GET /api/v1/version` — API version and git SHA
- `GET /api/v1/models` — list available model architectures (no weights, no checkpoints)

All other endpoints require authentication.

#### 2.2.5 Key Rotation Policy

- Keys expire after 90 days by default; configurable per key.
- Expired keys return `401 Unauthorized` with `WWW-Authenticate: Bearer error="invalid_token" error_description="Token expired"`.
- Key rotation does not invalidate in-flight requests (grace period: 60 seconds).
- All key usage is logged to `artifacts/runs/auth_log.jsonl` (as-built) or an audit table (network projection).

---

## 3. Error Handling Conventions

### 3.1 As-Built: Internal Error Handling

The system uses a **typed exception hierarchy** with a single root, descriptive messages, and no silent failures. Every error is either fatal (raises and propagates) or logged-and-continued (with an explicit decision at each call site). There are no bare `except Exception` clauses in production code.

#### 3.1.1 Exception Hierarchy

```
MRIAnomalyDetectionError          # root; all system exceptions inherit from this
├── DataError
│   ├── DescriptiveValidationError    # shape/dtype/range violation at data boundary
│   └── SplitContractViolationError   # attempt to modify write-once split
├── ModelError
│   ├── CheckpointError               # file not found, size mismatch, corrupt weights
│   │   ├── CheckpointNotFoundError
│   │   ├── CheckpointSizeMismatchError
│   │   └── CheckpointCorruptError
│   └── RegistryError                 # model name not found, ABC compliance failure
├── ReconError
│   └── ReconResultCorruptError       # loaded .pt file fails shape validation
├── EvalError
│   └── MetricsIntegrityError         # n_volumes mismatch, mean/std recomputation failure
├── ConfigError                       # missing required Hydra config key
└── ArtifactError
    └── ArtifactNotFoundError         # make report called before make eval
```

All exceptions in this hierarchy carry:
- A human-readable message describing what went wrong
- The actual values observed (not just "expected X, got Y" — the actual tensor shape, the actual file size, the actual config key that is missing)
- A suggested remediation where one exists

#### 3.1.2 Error Message Convention

Every error message follows the pattern:

```
<What failed>: <Why it failed>. <Actual values observed>. <Suggested remediation>.
```

Examples:

```python
# DataError
DescriptiveValidationError(
    "DataValidator: shape contract violated. "
    "Expected (B, 1, 16, 128, 128), got (2, 3, 64, 64, 64). "
    "Check TransformPipeline configuration: roi_size must be (1, 16, 128, 128)."
)

# CheckpointError
CheckpointSizeMismatchError(
    "Checkpoint integrity check failed for 'unetr_mse_ssim'. "
    "Expected 487,234,560 bytes, found 243,617,280 bytes. "
    "The file appears truncated. Re-download or re-copy the checkpoint file."
)

# ArtifactError
ArtifactNotFoundError(
    "ReportGenerator: no aggregate.json files found in artifacts/runs/*/metrics/. "
    "Run 'make eval' before 'make report' to generate evaluation results."
)

# ConfigError
ConfigError(
    "ConfigLoader: required key 'model.name' not found in resolved config. "
    "Available keys: model.checkpoint_path, model.device. "
    "Add 'model.name: unetr' to your experiment config or pass "
    "'+model.name=unetr' as a Hydra override."
)
```

#### 3.1.3 Fatal vs. Recoverable Errors

| Error Type | Behavior | Rationale |
|---|---|---|
| `DescriptiveValidationError` | Fatal — raises immediately | A malformed tensor reaching the model is worse than a crash |
| `SplitContractViolationError` | Fatal — raises immediately | Silent split modification would invalidate all results |
| `CheckpointError` (any subtype) | Fatal — raises immediately | Running with corrupt weights produces meaningless results |
| `ReconResultCorruptError` | Recoverable — log warning, skip volume | Partial eval results are valid; one corrupt file should not abort the run |
| `MetricsIntegrityError` | Fatal — raises immediately | A metrics file that fails its own integrity check must not be used |
| `ArtifactNotFoundError` | Fatal — raises immediately | Cannot generate a report from nothing |
| `ConfigError` | Fatal — raises immediately | Running with a misconfigured experiment is worse than not running |

#### 3.1.4 Error Propagation Rules

1. **Errors never cross module boundaries silently.** If `recon/engine.py` catches an exception from `models/unetr.py`, it either re-raises it (possibly wrapped with additional context) or handles it completely. It never swallows it.

2. **Wrapping adds context, not noise.** When re-raising across a boundary:
   ```python
   try:
       model.load_checkpoint(path)
   except CheckpointError:
       raise  # re-raise as-is; the original message is already descriptive
   except Exception as e:
       raise CheckpointCorruptError(
           f"Unexpected error loading checkpoint at {path}: {e}"
       ) from e  # preserve original traceback
   ```

3. **The entry-point scripts are the terminal error handlers.** They catch `MRIAnomalyDetectionError`, log the full traceback to `run_meta.json` (via `RunLogger`), print a user-friendly summary to stderr, and exit with code 1. They do not catch `Exception` — unexpected errors propagate as Python tracebacks.

4. **`RunLogger` records errors.** If a run crashes, `log_run_end()` is called in a `finally` block with `{"status": "crashed", "error": str(e), "traceback": ...}`. This ensures that crashed runs are distinguishable from completed runs in the comparison table.

#### 3.1.5 Logging vs. Raising

The system uses Python's `logging` module for informational and warning messages, and exceptions for error conditions. The distinction:

| Condition | Mechanism | Level |
|---|---|---|
| Normal progress | `logging.info` | INFO |
| Unexpected but recoverable (e.g., skipped volume) | `logging.warning` | WARNING |
| Configuration that may produce misleading results | `logging.warning` | WARNING |
| Any condition that prevents correct operation | Exception | — |

`logging.error` is never used — if something is an error, it raises. `logging.warning` is used only when the system can continue and the result is still valid (possibly degraded).

---

### 3.2 Network Projection: HTTP Error Handling

If this system were promoted to a network API, the following error handling conventions would apply.

#### 3.2.1 Error Response Schema

All error responses use a consistent JSON body:

```json
{
  "error": {
    "code": "CHECKPOINT_NOT_FOUND",
    "message": "Checkpoint 'unetr_synth' not found at the configured path.",
    "detail": "Expected file at /scratch/user/checkpoints/unetr_synth.pt. The file does not exist. Run 'make train' to generate this checkpoint, or verify the checkpoint path in your configuration.",
    "request_id": "req_01HX4K9MZPQR8VWXY3N5T6J7B",
    "timestamp": "2025-01-15T14:23:07.412Z",
    "docs_url": "https://github.com/user/mri-anomaly-detection/wiki/errors#CHECKPOINT_NOT_FOUND"
  }
}
```

**Fields:**

| Field | Type | Always Present | Description |
|---|---|---|---|
| `error.code` | `string` | Yes | Machine-readable error code in `SCREAMING_SNAKE_CASE` |
| `error.message` | `string` | Yes | Short human-readable description (≤ 120 characters) |
| `error.detail` | `string` | Yes | Full description with observed values and remediation |
| `error.request_id` | `string` | Yes | Unique ID for this request; correlates with server logs |
| `error.timestamp` | `string` | Yes | ISO 8601 UTC timestamp of the error |
| `error.docs_url` | `string` | No | Link to documentation for this error code; omitted for 5xx errors |
| `error.validation_errors` | `array` | No | Present only for `422 Unprocessable Entity`; see below |

#### 3.2.2 HTTP Status Code Mapping

| Internal Exception | HTTP Status | Error Code |
|---|---|---|
| `DescriptiveValidationError` | `422 Unprocessable Entity` | `VALIDATION_ERROR` |
| `SplitContractViolationError` | `409 Conflict` | `SPLIT_CONTRACT_CONFLICT` |
| `CheckpointNotFoundError` | `404 Not Found` | `CHECKPOINT_NOT_FOUND` |
| `CheckpointSizeMismatchError` | `422 Unprocessable Entity` | `CHECKPOINT_CORRUPT` |
| `CheckpointCorruptError` | `422 Unprocessable Entity` | `CHECKPOINT_CORRUPT` |
| `RegistryError` (name not found) | `404 Not Found` | `MODEL_NOT_FOUND` |
| `RegistryError` (ABC compliance) | `400 Bad Request` | `INVALID_MODEL_CLASS` |
| `ReconResultCorruptError` | `422 Unprocessable Entity` | `RECON_RESULT_CORRUPT` |
| `MetricsIntegrityError` | `500 Internal Server Error` | `METRICS_INTEGRITY_FAILURE` |
| `ArtifactNotFoundError` | `404 Not Found` | `ARTIFACT_NOT_FOUND` |
| `ConfigError` | `400 Bad Request` | `INVALID_CONFIG` |
| Authentication failure | `401 Unauthorized` | `UNAUTHORIZED` |
| Insufficient scope | `403 Forbidden` | `INSUFFICIENT_SCOPE` |
| Unexpected exception | `500 Internal Server Error` | `INTERNAL_ERROR` |

#### 3.2.3 Validation Error Detail

For `422 Unprocessable Entity` responses, the `validation_errors` array provides field-level detail:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request body failed validation.",
    "detail": "2 validation errors in request body.",
    "request_id": "req_01HX4K9MZPQR8VWXY3N5T6J7B",
    "timestamp": "2025-01-15T14:23:07.412Z",
    "validation_errors": [
      {
        "field": "threshold_strategy",
        "value": "percentile_99",
        "message": "Must be one of: fixed_percentile_95, otsu, adaptive.",
        "code": "INVALID_ENUM_VALUE"
      },
      {
        "field": "volume_index",
        "value": -1,
        "message": "Must be a non-negative integer.",
        "code": "OUT_OF_RANGE"
      }
    ]
  }
}
```

#### 3.2.4 5xx Error Policy

- `500 Internal Server Error` responses **never** include stack traces, internal file paths, or system information in the response body. These are logged server-side and correlated via `request_id`.
- The `request_id` in the response allows a researcher to ask an administrator to retrieve the full server-side log entry.
- `503 Service Unavailable` is returned when a GPU is required but unavailable, with a `Retry-After` header indicating the estimated wait time.

#### 3.2.5 Idempotency

Long-running operations (eval, train) are triggered by POST requests that return a `run_id`. Clients may safely retry the POST if they do not receive a response — the server checks whether a run with the same `(model_name, checkpoint_name, seed, split_contract_id)` is already in progress or completed, and returns the existing `run_id` rather than starting a duplicate run. This is the network equivalent of the `override=False` default on `SplitContract.create()`.

---

## 4. Pagination Strategy

### 4.1 As-Built: Internal Iteration Patterns

The system processes collections of volumes, metrics records, and run artifacts. These collections are bounded and known in advance (≤ ~3,000 training volumes, ≤ ~100 test volumes, ≤ tens of runs). The as-built system uses Python iterators and generators rather than pagination, but the design choices made here directly inform the network pagination strategy.

#### 4.1.1 Volume Iteration: Streaming Generator

`ReconstructionEngine.run_batch()` is a generator that yields one `ReconResult` per volume. It does not load all volumes into memory simultaneously:

```python
def run_batch(
    self,
    loader: DataLoader,
    *,
    save_results: bool = True,
) -> Iterator[ReconResult]:
    for batch in loader:
        volume = batch["image"]          # (B, 1, 16, 128, 128)
        label = batch.get("label")       # (B, 1, 16, 128, 128) or None
        result = self.run(volume)        # one ReconResult per volume
        if save_results:
            save_recon_result(result, self.results_dir)
        yield result
```

**Design rationale:** The generator pattern means the caller controls the pace of consumption. Memory usage is bounded by one batch at a time, regardless of dataset size. This is the internal equivalent of cursor-based pagination.

#### 4.1.2 Metrics Iteration: Full Load with Bounded Size

`AggregateEvaluator.evaluate_split()` loads all `VolumeMetrics` records into memory before computing aggregates. This is acceptable because the test set is bounded at ~100 volumes, producing ~100 records of ~5 floats each — negligible memory.

```python
def evaluate_split(self) -> AggregateMetrics:
    all_metrics: list[VolumeMetrics] = []
    for pt_file in sorted(self.results_dir.glob("*.pt")):
        result = load_recon_result(pt_file)
        label = self._load_label(result.volume_id)
        if label is None:
            logger.warning("No label found for volume_id=%s; skipping.", result.volume_id)
            continue
        metrics = VolumeEvaluator().evaluate(result, label)
        all_metrics.append(metrics)
    return self._aggregate(all_metrics)
```

**Design rationale:** Full load is simpler and correct for the bounded test set. If the test set grew to tens of thousands of volumes, this would need to become a streaming aggregation.

#### 4.1.3 Run Artifact Iteration: Directory Scan

`ReportGenerator` iterates over all `run_meta.json` files in `artifacts/runs/*/`:

```python
def _load_all_runs(self, runs_root: Path) -> list[RunMetrics]:
    run_dirs = sorted(runs_root.glob("*/run_meta.json"))
    runs = []
    for meta_path in run_dirs:
        meta = json.loads(meta_path.read_text())
        if meta.get("end_time") is None:
            logger.warning("Skipping crashed run: %s", meta_path.parent.name)
            continue
        runs.append(RunMetrics.from_dict(meta))
    return runs
```

**Design rationale:** The number of runs is small (tens, not thousands). A directory scan is sufficient. The `sorted()` call ensures deterministic ordering by `run_id` (which is timestamp-prefixed), making the comparison table reproducible.

---

### 4.2 Network Projection: HTTP Pagination

If this system were promoted to a network API, the following pagination strategy would apply.

#### 4.2.1 Chosen Strategy: Cursor-Based Pagination

**Chosen mechanism:** Cursor-based pagination using opaque continuation tokens.

**Rationale over offset pagination:**
- Offset pagination (`?page=3&page_size=20`) is unstable: if a new run is inserted between page 2 and page 3 being fetched, the client sees a duplicate or skips a record. For a research system where new runs are added during a session, this is unacceptable.
- Cursor-based pagination is stable: the cursor encodes the position in the result set, not an offset. New records inserted after the cursor position do not affect pages already fetched.
- The internal generator pattern (`run_batch()`) maps naturally to cursor-based pagination: the cursor is the generator's current position.

**Rationale over keyset pagination:**
- Keyset pagination requires exposing the sort key to the client (e.g., `?after_run_id=2025-01-15T14:23:07-unetr-mse_ssim`). This leaks internal ID structure.
- Opaque cursors hide the implementation and allow the server to change the sort key without a client-visible breaking change.

#### 4.2.2 Cursor Format

Cursors are base64url-encoded JSON objects, opaque to the client:

```
# Decoded (server-internal only):
{
  "entity": "volume_metrics",
  "run_id": "2025-01-15T14:23:07-unetr-mse_ssim",
  "after_volume_index": 42,
  "sort": "volume_index:asc",
  "page_size": 20
}

# Wire format (client-visible):
eyJlbnRpdHkiOiJ2b2x1bWVfbWV0cmljcyIsInJ1bl9pZCI6IjIwMjUtMDEtMTVUMTQ6MjM6MDctdW5ldHItbXNlX3NzaW0iLCJhZnRlcl92b2x1bWVfaW5kZXgiOjQyLCJzb3J0Ijoidm9sdW1lX2luZGV4OmFzYyIsInBhZ2Vfc2l6ZSI6MjB9
```

Cursors are signed with HMAC-SHA256 using a server secret to prevent tampering. A tampered or expired cursor returns `400 Bad Request` with error code `INVALID_CURSOR`.

#### 4.2.3 Paginated Response Schema

All list endpoints return a consistent envelope:

```json
{
  "data": [ ... ],
  "pagination": {
    "page_size": 20,
    "has_next_page": true,
    "next_cursor": "eyJlbnRpdHkiOiJ2b2x1bWVfbWV0cmljcyIs...",
    "has_prev_page": true,
    "prev_cursor": "eyJlbnRpdHkiOiJ2b2x1bWVfbWV0cmljcyIs...",
    "total_count": 87
  },
  "meta": {
    "request_id": "req_01HX4K9MZPQR8VWXY3N5T6J7B",
    "generated_at": "2025-01-15T14:23:07.412Z"
  }
}
```

**Fields:**

| Field | Type | Always Present | Description |
|---|---|---|---|
| `data` | `array` | Yes | The page of results |
| `pagination.page_size` | `integer` | Yes | Number of items in this page (may be less than requested on last page) |
| `pagination.has_next_page` | `boolean` | Yes | Whether a next page exists |
| `pagination.next_cursor` | `string` | No | Omitted when `has_next_page` is false |
| `pagination.has_prev_page` | `boolean` | Yes | Whether a previous page exists |
| `pagination.prev_cursor` | `string` | No | Omitted when `has_prev_page` is false |
| `pagination.total_count` | `integer` | Yes | Total number of items across all pages; computed once per query |
| `meta.request_id` | `string` | Yes | Correlates with server logs |
| `meta.generated_at` | `string` | Yes | ISO 8601 UTC timestamp |

#### 4.2.4 Pagination Parameters

Clients control pagination via query parameters:

| Parameter | Type | Default | Max | Description |
|---|---|---|---|---|
| `page_size` | `integer` | `20` | `100` | Items per page |
| `cursor` | `string` | — | — | Opaque continuation token from previous response |
| `sort` | `string` | `volume_index:asc` | — | Sort field and direction; see per-endpoint documentation |

`page_size` above 100 returns `400 Bad Request` with error code `PAGE_SIZE_EXCEEDED`. This bound is set conservatively: a page of 100 `VolumeMetrics` records is ~5 KB of JSON, well within a single HTTP response.

#### 4.2.5 Endpoint-Specific Pagination Defaults

| Endpoint | Default `page_size` | Default Sort | Notes |
|---|---|---|---|
| `GET /api/v1/runs` | 20 | `start_time:desc` | Most recent runs first |
| `GET /api/v1/runs/{run_id}/metrics` | 50 | `volume_index:asc` | All metrics for one run; 50 covers most test sets in one page |
| `GET /api/v1/results` | 20 | `volume_index:asc` | ReconResult metadata (not tensor data) |
| `GET /api/v1/checkpoints` | 20 | `checkpoint_name:asc` | Small collection; pagination is precautionary |

#### 4.2.6 Non-Paginated Endpoints

The following endpoints return complete responses without pagination, because their result sets are bounded by design and small:

- `GET /api/v1/models` — at most ~10 registered model architectures
- `GET /api/v1/runs/{run_id}/aggregate` — exactly one `AggregateMetrics` record
- `GET /api/v1/classical/metrics` — exactly one `ClassicalMetrics` record
- `GET /api/v1/split-contract` — exactly one `SplitContract`

#### 4.2.7 Streaming for Large Binary Responses

Tensor data (`ReconResult` `.pt` files) is not paginated — it is streamed. A request for a `ReconResult` returns a `Content-Type: application/octet-stream` response with chunked transfer encoding. The client receives the file as a stream and is responsible for writing it to disk. This mirrors the as-built `save_recon_result()` / `load_recon_result()` pattern.

```
GET /api/v1/results/{volume_id}/tensor
Authorization: Bearer <key>

HTTP/1.1 200 OK
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="brats_2021_042.pt"
Transfer-Encoding: chunked
X-Tensor-Shape: 1,16,128,128
X-Tensor-Dtype: float32
X-Checksum-SHA256: a3f8c2d1...
```

The `X-Tensor-Shape`, `X-Tensor-Dtype`, and `X-Checksum-SHA256` headers allow the client to validate the received file before loading it with `torch.load`, mirroring the `load_recon_result()` shape validation.
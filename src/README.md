# Reference source code

This directory is a dependency-free Python reference implementation of the
paper's auditable protocol. It implements the method; it does not fabricate or
redistribute the unavailable HAI/TEP raw data, model weights, or frozen model
outputs.

Implemented components:

- immutable `E(q)`, `H(q)`, `L(q;E,H)`, and `R(q;E,L)` records with SHA-256 bindings;
- training-normal empirical-percentile calibration and median D-score;
- version-aware, label-blind, symmetric fault/benign QAKB retrieval and RRF;
- the six reader safety gates shown in Figure 5;
- a source-role ledger and one-shot confirmation access guard;
- explicit integrity/detector/retrieval/routing/confirmation claim authorization;
- exact Clopper-Pearson intervals and ACSC coverage maximization with deterministic
  tie-breaking and explicit no-go behavior.
- raw multivariate time-series orchestration through three detector adapters,
  evidence calibration, dual-channel retrieval, an auditable reader, conservative
  ACSC/reader agreement routing, and claim release;
- callable and precomputed adapters for real ModernTCN, TimesNet, and TranAD
  deployments, plus clearly marked dependency-free statistical proxies;
- an OpenAI-compatible JSON reader client and a deterministic offline reader.

Run from this `src` directory with Python 3.10 or newer:

```powershell
python -m unittest discover -s tests -v
python -m auditable_alarm risk-bound 9 400
python -m auditable_alarm risk-bound 4 100
python -m examples.end_to_end
python -m auditable_alarm select examples/acsc_input.json
python -m auditable_alarm run examples/full_pipeline_input.json
```

## Complete pipeline

The `run` command consumes raw multivariate query and training-normal time series,
historical cases, a frozen routing policy, and reader configuration. It performs:

```text
raw time series -> three detector score streams -> E(q)
  -> label-blind fault/benign retrieval -> H(q)
  -> immutable JSON reader record L(q;E,H)
  -> six reader gates + frozen numeric policy agreement
  -> separate disposition R(q;E,L) and gated claims
```

The bundled `reference-proxy-v1` detector backends are transparent statistical
proxies intended for examples and integration tests. They are not implementations
of the ModernTCN, TimesNet, or TranAD neural architectures and must not be reported
as paper-model results.
The CLI therefore requires an explicit `detectors` block; it never silently
selects the proxy backends. The bundled example opts in with
`{"detectors": {"type": "reference-proxy"}}`.

### Connect real detectors

Wrap each deployed model's inference function. A predictor receives a mapping of
channel names to equally sized numeric sequences and returns an anomaly-score
sequence:

```python
from auditable_alarm.detectors import CallableDetectorAdapter, DetectorEnsemble

detectors = DetectorEnsemble([
    CallableDetectorAdapter("ModernTCN", modern_tcn_predict, "checkpoint:modern-tcn-sha256"),
    CallableDetectorAdapter("TimesNet", timesnet_predict, "checkpoint:timesnet-sha256"),
    CallableDetectorAdapter("TranAD", tranad_predict, "checkpoint:tranad-sha256"),
])
```

If inference runs in another service or batch job, use
`PrecomputedDetectorAdapter` with event-ID keyed score streams. Backend identifiers
are bound into each evidence record's calibration provenance.

The same path is available in JSON using `"detectors": {"type": "precomputed",
"models": [...]}`. Each model entry contains `name`, a checkpoint or service
`backend_id`, and `scores_by_event` keyed by every query and training-normal event
ID. This lets the standalone CLI consume real exported outputs without importing
the training framework.

### Connect an LLM

Set the API key named by `api_key_env` and replace the example reader block with:

```json
{
  "reader": {
    "type": "openai-compatible",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "api_key_env": "DEEPSEEK_API_KEY",
    "temperature": 0
  }
}
```

Never place an `api_key`, `token`, `secret`, or `password` value in the JSON input.
The CLI rejects inline credential fields; provide credentials only through the named
environment variable. Local `.env` files are ignored by Git and must not be committed.

The reader is required to return a JSON object. Exact input/output bytes, parsed
content, prompt/schema/model identifiers, decoding settings, and hashes are saved
in `ReaderRecord`. Invalid JSON or schema violations are retained for audit and
forced to `ABSTAIN` by the six gates.

### Routing rule

The final automatic disposition is released only when all reader gates pass and
the frozen numeric policy and reader produce the same non-abstaining disposition.
An ACSC abstention, reader abstention, gate failure, or disagreement produces
`ABSTAIN`; neither the reader record nor its raw output is rewritten.

The small JSON example is intentionally synthetic. ACSC requires enough fault
and normal units for exact one-sided bounds to pass; therefore it demonstrates
the CLI schema and may produce `NO_GO`, which is a valid protocol outcome.

## ACSC input schema

`select` accepts a JSON object containing `scores`, `labels` (`FAULT`, `NORMAL`,
or `BENIGN`), a frozen `candidates` array, and optional `agreements`,
`epsilon_fb`, `epsilon_nf`, and `bound_confidence`. Each candidate contains
`policy_id`, `benign_threshold`, `fault_threshold`, and optional `model_id`,
`min_agreement`, and `complexity`.

Development labels are consumed only by ACSC candidate evaluation. The evidence
builder and retrieval API deliberately have no truth-label argument, preventing
query truth from entering current evidence or historical ranking.

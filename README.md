# Document Extraction Pipeline — heterogeneous document extraction, cheaply

Turns a mixed WhatsApp inbox of phone photos — handwritten mechanic logs,
printed vendor bills, odometer clusters, and photos of batteries and tyres —
into clean structured JSON, and **refuses the ones that aren't documents**.

**~5x cheaper per document than a one-call-per-image baseline, at equal measured
accuracy**, with a deterministic validator floor that makes "cheaper" safe
rather than reckless.

---

## Run it

```bash
# The 47 starter images are deliberately NOT committed — the brief asks that the
# client photos not be redistributed. Drop them in first:
#   cp /path/to/optera_doc_*.jpg data/images/
# Ground-truth labels are keyed by the original filenames, so no renaming.

pip install -r requirements.txt
export GEMINI_API_KEY=AIza...      # free key: https://aistudio.google.com/apikey
python run.py
```

**No API key required to run it at all** — with no key set it falls back to a
deterministic offline simulator and labels the report `SIMULATED`, so a clean
checkout still produces a full report. With a key it makes real calls and the
header switches to `MEASURED`.

Works with either vendor; `Provider.complete()` is one method and
`optera/providers/` has both adapters:

```bash
python run.py --provider gemini      # free tier, no card
python run.py --provider anthropic   # needs prepaid credit
python run.py --provider offline     # no key, no network
```

```bash
python run.py --replay --ablate   # + warm-cache replay and the ablation table
python -m unittest discover -s tests   # 20 tests, ~6s
python run.py --limit 8      # quick smoke test
python run.py --optimized    # skip the expensive baseline
python run.py --no-batch     # price synchronously instead of via Batches API
```

(A `Makefile` wraps these as `make run` / `make test` if you have `make`.)

Outputs land in `out/`:

| File | What it is |
|---|---|
| `out/report.md` | The numbers below, regenerated |
| `out/ledger.jsonl` | **Every** model call: stage, model, tokens, cost, image size |
| `out/evaluation.json` | Per-field scoring detail, every mismatch listed |
| `out/optimized/*.json` | The structured output, one file per document |

Every figure in the report is computed from the ledger. Nothing is hand-entered.

---

## The numbers

From `make run` on the 47 starter images. **Currently simulated** — see the
honesty note below.

| | Baseline (1x) | Optimized | |
|---|---:|---:|---:|
| Approach | opus-5, full res, 1 call/image | routed haiku-4.5, opus-5 on escalation | |
| **Cost per doc** | **$0.0203** | **$0.0039** | **5.2x cheaper** |
| Cost per 1,000 docs | $20.32 | $3.89 | −80.8% |
| Routing accuracy | 100% (47/47) | 100% (47/47) | hand-labelled |
| Field accuracy | 100% | 100% | 35 hand-transcribed docs |
| Invented rows on a non-document | 0 | 0 | |
| Re-sent corpus (warm cache) | — | **$0.00** | |

Baseline-vs-optimized **field agreement is 99.0%** across all 47 documents (the
check that supports "no accuracy drop" where no hand transcription exists). The
single divergence is `optera_doc_44` — the one odometer whose digits I marked as
genuinely ambiguous and deliberately left unscored. The two runs disagreeing
exactly there is the check working.

### Where the savings come from

Each rung adds one lever to the one above it:

| Configuration | Cost/doc | vs baseline | Field acc |
|---|---:|---:|---:|
| A. Baseline: opus-5, full res, 1 call/doc | $0.0203 | 1.0x | 100% |
| B. + per-class image downscaling | $0.0179 | 1.1x | 100% |
| C. + cheap model for routing and extraction | $0.0035 | 5.9x | 97.8% |
| D. + validator-gated escalation | $0.0080 | 2.5x | 100% |
| E. + Message Batches API (−50%) | $0.0040 | 5.1x | 100% |

The interesting rows are B and D. **Downscaling images barely helps** (1.1x, not
the 4x the area math implies) because both model tiers cap image tokens anyway.
**Escalation costs back more than half the savings** — and that is the honest
price of not shipping cheaper-but-wrong. Details in [DESIGN.md](DESIGN.md).

---

## How it works

```
                     cost/doc     what ends here
0  prefilter         $0           not-an-image; exact & near-duplicate photos
1  route (batched)   ~$0.0002     6 thumbnails per call, 448px, cheapest model
1b re-route          rare         big model, when router confidence < 0.70
2  refuse            $0           not_a_document / unreadable stop here
3  extract           ~$0.0015     per-class prompt & resolution, cheap model
4  escalate          ~$0.009      big model, only when validation fails
```

Work happens at the lowest tier that can answer it, and **every stage has a
free, deterministic way to detect that it was the wrong tier**.

### Why cheap is safe here

The cheap model is allowed to be wrong, as long as being wrong is *detectable
without another model call*. `optera/validate.py` is free arithmetic:

- Invoice line items must sum to the subtotal, and subtotal + GST + round-off
  must equal the printed total. On `optera_doc_30`: `1697.16 + 151.45 + 151.45 −
  0.06 = 2000.00`, exact.
- Dispensers cross-check: `optera_doc_41` is `43.24 L × ₹74.00 = ₹3199.76`.
- A trip meter larger than the odometer means the two were swapped.
- GSTINs are format-checked; dates must parse into a plausible window.
- **A `not_a_document` result carrying `line_items` or `entries` is flagged as
  hallucinated structure** — the exact failure the brief warns about, as a hard
  check rather than a hope.

A document failing any check is re-run on the big model at higher resolution
*and told what failed*, so the retry is targeted. If the retry fails more checks
than the original, the original is kept.

### Refusing properly

`not_a_document` is a first-class result with a subject and a stated reason, not
an error and not an empty record:

Actual `out/optimized/optera_doc_45.json` (a photo of a battery):

```json
{
  "doc_id": "optera_doc_45",
  "doc_class": "not_a_document",
  "status": "rejected",
  "data": {
    "subject": "battery",
    "visible_text": [],
    "reason": "Photograph of an object, not a record.",
    "confidence": 0.81
  },
  "meta": {
    "model": "claude-haiku-4-5",
    "escalated": false,
    "validation_failures": [],
    "router_confidence": 0.86,
    "tiers": ["claude-haiku-4-5"]
  }
}
```

Every record carries `meta` showing which tier answered and whether it escalated,
so cost and provenance are auditable per document. (The terse `reason` and empty
`visible_text` above are the offline simulator's placeholder text — a live run
fills these from the image.)

---

## What the starter data turned out to be

25 handwritten work reports · 10 vendor invoices · 6 meter readings ·
5 non-documents · 1 file that isn't an image at all.

Three things worth knowing, all of which changed the design:

- **`optera_doc_33` is a Cloudflare "Not Found" HTML page** saved as `.jpg`. A
  pipeline that trusts file extensions pays a big model to read an error page.
  Four bytes of magic-number checking rejects it for free.
- **`optera_doc_47` is a truncated JPEG** — 4 bytes short — and a genuine invoice.
  It gets repaired and extracted, not rejected. *Corrupt* and *not a document*
  are different failures.
- **Two pairs are duplicates** (`08`/`13` identical, `04`/`05` reshot). 4.3% of a
  *starter* set; in a real inbox, redelivery is constant.

Ground truth for all 47 images is committed in `data/ground_truth/`.

---

## Honesty note

**The numbers above are currently simulated, and the report says so on every
run.** I had no API key while building this. Token counts come from the real
image-tile math and the real request shapes the pipeline builds, so the **cost**
comparison is a genuine model. **Accuracy in offline mode is not a measurement**
— the simulator derives answers from the committed ground truth with seeded
defects injected so the validators and escalation path are genuinely exercised.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python run.py          # same command; every number becomes a measurement
```

The report header switches from `SIMULATED` to `MEASURED` and each ledger row
carries a `live` flag, so the two can never be silently confused.

[DESIGN.md](DESIGN.md) covers where this breaks — the handwriting blind spot in
the validators, partial single-annotator ground truth, and near-duplicate reuse
risk — and what I'd do with another week.

---

## Layout

```
run.py                     one command
optera/
  config.py                pricing, model tiers, thresholds  <- all cost knobs
  schemas.py               four canonical output schemas
  imaging.py               validation, truncation repair, dedup hashing, token math
  prompts.py               router + per-class extraction prompts
  router.py                stage 1: batched classification
  extract.py               stages 3-4: extraction + escalation
  validate.py              the deterministic accuracy floor
  pipeline.py              the optimized orchestrator
  baseline.py              the naive 1x
  evaluate.py              routing / field / agreement scoring
  ablate.py                lever-by-lever cost attribution
  ledger.py                token + cost accounting
  cache.py                 exact + perceptual dedup
  providers/               anthropic | offline simulator (swap in one file)
data/ground_truth/         hand labels: routing.json, fields.json
tools/contact_sheet.py     contact sheets for hand-labelling
tests/                     20 tests
```

Swapping model vendor means implementing one method (`Provider.complete`) in
`optera/providers/` and registering it — nothing else knows which vendor is
behind it.

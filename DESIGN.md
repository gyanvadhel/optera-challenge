# DESIGN.md

What I built, what the numbers actually mean, where it breaks, and what I'd do
with another week.

---

## 1. What the data turned out to be

Before designing anything I labelled all 47 starter images by hand
(`tools/contact_sheet.py` makes contact sheets so this costs minutes, not
hours). The labels are committed in `data/ground_truth/routing.json` and are the
only thing routing accuracy is scored against.

| Class | Count | Notes |
|---|---:|---|
| `work_report` | 25 | Handwritten depot logs — VADAJ, PALDI, SARANGPUR, REMCO. English + Hindi + Gujarati, multi-colour ink, strikethroughs, fingers over the page. The hardest class by a distance. |
| `vendor_invoice` | 10 | Printed letterheads, handwritten line items. Tax invoices, bills of supply, cash memos, one credit memo, one delivery challan. |
| `meter_reading` | 6 | Digital odometer clusters and one TATA DEF dispenser. |
| `not_a_document` | 5 | Two tyres, two batteries, one DEF filler cap. |
| `unreadable` | 1 | `optera_doc_33` — see below. |

Three findings from that pass shaped the design more than any modelling decision:

**`optera_doc_33` is not an image.** It is a Cloudflare "Not Found" HTML page
saved with a `.jpg` extension — a download that failed silently. A pipeline that
trusts file extensions pays a big multimodal model to read an error page. Four
bytes of magic-number checking rejects it for free.

**`optera_doc_47` is a truncated JPEG** — the file is 4 bytes short, and Pillow
refuses it by default. It is a real Anupam delivery challan. Rejecting it would
be a false negative on a genuine document, so the pipeline sets
`LOAD_TRUNCATED_IMAGES` and decodes what is there. **Corrupt and not-a-document
are different failures and must not share a code path**: one is repairable, the
other is a correct refusal.

**Two pairs are duplicates.** `optera_doc_08`/`13` are perceptually identical
(Hamming distance 0, different bytes) and `04`/`05` are the same page reshot
(distance 2). That is 4.3% of the corpus, in a *starter set* — in a real
WhatsApp inbox, redelivery is constant.

---

## 2. Architecture

Five stages, cheapest first, each able to end a document's journey.

```
                      cost/doc      what ends here
0  prefilter          $0            not-an-image, exact & near duplicates
1  route (batched)    ~$0.0002      6 thumbnails per call, cheapest model
1b re-route           rare          single image, big model, when confidence < 0.70
2  refuse             $0            not_a_document / unreadable stop here
3  extract            ~$0.0015      per-class prompt, per-class resolution, cheap model
4  escalate           ~$0.009       big model + bigger image, only on validator failure
```

The ordering is the design. Every stage does the least work that can answer the
question, and every stage has a **free, deterministic way to detect that it was
the wrong stage**.

### Routing

The router is asked one four-way question, so it gets a 448px thumbnail
(~130 image tokens) and six documents share one call. Batching matters because
the system prompt is otherwise paid once per document rather than once per six.

Its self-reported confidence is not taken on faith: below 0.70 the image is
re-classified on the big model. That happened once in 46 documents — which is
the point. You pay Opus prices for the hard 2%, not the easy 98%.

### Extraction

Four canonical schemas (`optera/schemas.py`), one per class. The naive baseline
pays for the union of every field description on every image, including the
battery photos. A routed pipeline pays only for the fields that can apply.

### The escalation ladder — why "cheaper" is safe here

This is the part that makes the cost work defensible rather than reckless. The
cheap model is *allowed* to be wrong, provided being wrong is **detectable
without another model call**. `optera/validate.py` is free arithmetic and
regexes:

- **Invoices reconcile.** Line items must sum to the subtotal; subtotal + CGST +
  SGST + IGST + round-off must equal the printed grand total. On `optera_doc_30`
  that is `1697.16 + 151.45 + 151.45 − 0.06 = 2000.00` — exact. A transposed
  digit fails this check even when the model is confident.
- **Dispensers reconcile.** `optera_doc_41`: `43.24 L × ₹74.00 = ₹3199.76`,
  exactly. Three numbers cross-checking each other for free.
- **Odometers sanity-check.** A trip meter larger than the odometer means the two
  were swapped — caught without knowing the true value.
- **GSTINs are format-checked**, dates must parse and fall in a plausible window
  (a 2016/2026 misread is a classic OCR failure).
- **Rejections must stay empty.** A `not_a_document` result carrying
  `line_items` or `entries` is flagged as hallucinated structure. This is the
  specific failure the brief calls out, and it is a hard check rather than a
  hope.

A document failing any check is re-run on the big model at higher resolution,
**and told what failed**, so the retry is targeted rather than a blind redo. If
the retry fails *more* checks than the original, the original is kept — a retry
that got worse is not progress.

---

## 2b. Provider portability, and what running live actually taught me

The pipeline is provider-agnostic: `Provider.complete()` is one method, and
`optera/providers/` has adapters for Anthropic and Google Gemini plus the
offline simulator. `--provider gemini|anthropic|offline` switches everything,
including the model tiers and the image budgets.

I ran the live numbers on **Gemini via AI Studio**, because it issues a free key
and the whole point of a cost exercise is to be able to run it. Four things came
out of that which no amount of desk design would have found:

**Model availability on a free key is not what the docs imply, and I had to
measure it.** Every Pro variant (`gemini-2.5-pro`, `gemini-3.1-pro-preview`,
`pro-latest`) returns 429 with *zero* free quota. The 2.5 flash models return
**404 "no longer available to new users"** — they are still in the pricing table
and still returned by `models.list()`, but a key created today cannot call them.
And `gemini-3.6-flash` / `gemini-flash-latest` allow **20 requests per day**,
which a single 47-image benchmark exhausts on the baseline pass alone.

What survives is the ladder actually used:
`3.1-flash-lite` ($0.25/$1.50) routes → `3.5-flash-lite` ($0.30/$2.50) extracts
→ `3.5-flash` ($1.50/$9.00) is the baseline and the escalation floor. A 6x
spread, versus the ~12x a paid key gives with Pro on top, so **the savings
measured here are a conservative floor, not a best case.**

That daily cap also caused a real bug worth keeping: my retry handler treated
429 as transient and backed off five times, which on a *per-day* quota just
burns the remaining allowance and fails anyway. The provider now parses the
`quotaId` and fails fast with the fix in the message. Retry logic that cannot
tell "slow down" from "come back tomorrow" is actively harmful.

**Gemini's schema dialect is not JSON Schema.** It is an OpenAPI-3.0 subset:
`additionalProperties` is rejected outright and nullability is a `nullable: true`
flag rather than a `["string", "null"]` union. Without the translation layer in
`providers/gemini.py` every structured call 400s. Worth knowing before you plan
a provider swap as a config change.

**Gemini bills vision in 768px tiles, not by area — so resolution is a step
function.** A 1568px page costs 6 tiles (1548 tokens); a 1536px page costs 4
(1032 tokens). **That is 33% off for a 2% resolution drop**, purely from landing
inside a tile boundary. In the other direction, the router thumbnail is one tile
at anything up to 768px, so the 448px thumbnail I had tuned for Anthropic was
giving away resolution for free. Both budgets are now tile-aligned for Gemini.
This is the mirror image of the Anthropic finding in §3: on Anthropic
downscaling barely helps because of the token *ceiling*; on Gemini it helps a
lot but only at the *tile boundaries*. Neither is the smooth curve the area
math suggests.

**The live run found a bug in my ground truth, and a bug in my validator.**
On `optera_doc_03` the model read `2026-06-08` where my label said `2026-06-02`.
I re-cropped the header at full resolution: the model was right and I had
misread it off a contact sheet. Label corrected, with the correction noted in
`fields.json`. Separately, every REMCO-style page escalated to the expensive
tier because my `BUS_CODE_RE` demanded a lettered fleet code (`TCM35`) and those
depots write a bare number in the BUS NO column — the model was right, the
validator was wrong, and **escalating correct work is pure cost with no accuracy
benefit**. Both are the argument for having an ablation and a per-document
`meta` block: neither was visible from the headline number.

---

## 3. The numbers, and what each lever actually bought

Full run: `make run`. Everything below is computed from `out/ledger.jsonl`.

The headline is **~5x cheaper per document at equal measured accuracy**. The
ablation is more interesting than the headline:

| Lever | Cost/doc | vs baseline |
|---|---:|---:|
| A. Baseline: opus-5, full res, 1 call/doc | $0.0203 | 1.0x |
| B. + per-class image downscaling | $0.0179 | 1.1x |
| C. + cheap model for routing and extraction | $0.0035 | 5.9x |
| D. + validator-gated escalation | $0.0080 | 2.5x |
| E. + Message Batches API | $0.0040 | 5.1x |

Three things I did not expect:

**Image downscaling is a weak lever — 1.1x, not the 4x the area math suggests.**
Tokens scale with area, so halving the long edge should quarter the image cost.
It does not, because **both model tiers cap image tokens anyway** (Opus at 4784,
Haiku at 1600). A 3000×4000 phone photo and a 1568px downscale bill almost the
same on Opus. Downscaling still matters in exactly one place — the 448px router
thumbnail, which is 130 tokens against 1600 — but "resize the images" is mostly
folk wisdom once the server-side cap is doing the work for you.

**Model routing is the whole game (5.9x).** Haiku 4.5 is 5x cheaper than Opus 5
on both input and output. Nothing else comes close.

**Escalation costs back 2.4x of that, and is worth it.** Going from rung C to D
more than doubles cost per document. That is the price of the accuracy floor,
and it is the honest cost of "cheaper-but-wrong doesn't count". Escalation is
~58% of the optimized run's spend from ~24% of its documents. **Escalation rate
is by far the most sensitive parameter in the system** — every point of
first-pass accuracy on the cheap tier is worth more than any further prompt
trimming.

### Prompt caching: a lever I deliberately did not pull

Caching looks like free money and mostly is not here. The minimum cacheable
prefix is **4096 tokens on Haiku 4.5** (against 512 on Opus 5). My extraction
system prompts are 400–700 tokens. Below the minimum the API silently does not
cache — no error, `cache_creation_input_tokens` just comes back 0. Padding a
prompt to 4096 tokens to make it cacheable would cost more than the cache ever
returns. So `cache_system=True` is passed everywhere, and
`AnthropicProvider.complete` only actually attaches `cache_control` when the
prompt clears the model's real minimum. On the tier where the cache would help,
the prompt is too short; on the tier where the prompt is long enough, we hardly
send any calls. Reporting a caching saving here would be reporting a number I
did not earn.

### Deduplication

Cold-start is the headline number — the cache is cleared before every run,
because a benchmark that reuses a cache warmed by a previous run is not a
benchmark. (I hit exactly this bug mid-build: a second run reported 10 free
documents and a flattering cost, all of them documents matching themselves.)

Measured separately with `--replay`: re-sending the identical corpus costs
**$0.00**. In-run, the two genuine duplicate pairs are caught and extracted once.

---

## 4. How accuracy is measured, and what the number is worth

Three separate measurements, kept separate because the evidence behind them
differs a lot:

1. **Routing accuracy** — against hand labels for all 47 images. Complete, and
   the number I trust most.
2. **Field accuracy** — against hand-transcribed fields for 35 documents
   (`data/ground_truth/fields.json`). **Deliberately partial.** Full-transcribing
   25 multilingual handwritten pages was not achievable in 24 hours, so invoices
   and meters are scored on all key fields, and work reports on header fields
   plus one fully transcribed page (`optera_doc_10`). The evaluator scores a
   document only on the keys present in the ground truth.
3. **Baseline-vs-optimized agreement** — field-by-field across *all* documents,
   no ground truth needed. This is what supports "no accuracy drop" on the
   documents nobody hand-transcribed: if the cheap path and the expensive path
   agree, the cheap path did not lose something the expensive path found.

I also report **precision when answered** separately from accuracy, because they
fail differently. A missing field is a nuisance; a confidently wrong field is a
corrupted record. Both prompts and validators are biased toward `null` over a
guess.

---

## 5. Where this breaks

Honestly, and in rough order of how much it would worry me in production.

**The headline numbers are measured on flash-tier models only.** A free AI
Studio key has no Pro access at all (§2b), so the ladder tops out at
`gemini-3.6-flash` rather than a Pro model. A real deployment would put Pro at
the top, which widens the price spread from 6x to ~12x and makes the routing
saving *larger* — but it would also change the escalation tier's accuracy, and I
have not measured that. Treat the reported multiple as a floor measured on a
narrower ladder than production would use, not as the number a paid key gives.

**Offline mode is a cost model, not an accuracy measurement.** If you run
without a key, `python run.py` uses a deterministic simulator: token counts come
from the real image-tile math and real request shapes, so the *cost* comparison
holds, but answers are synthesised from the committed ground truth with seeded
defects. The report stamps that run `SIMULATED`. I kept this path because a
clearly-labelled simulation beats an unlabelled guess, and because it makes the
whole pipeline testable in CI without spending anything.

**One ground-truth label was wrong, so others probably are.** The live run
disagreed with my `optera_doc_03` date and was right (§2b). I corrected it. That
is one caught error in a single-annotator label set built partly from contact
sheets — the honest inference is that there are more I have not caught, and that
every accuracy number here carries that uncertainty.

**The escalation ladder is only as good as its validators.** It catches
arithmetic, formats, and structural emptiness. It cannot catch a *plausible* lie
— a work-report row transcribed as "brake set" when the page says "brake shoe",
or a bus code read as TCM35 instead of TCM55. For handwriting, which is 25 of 47
documents here, that is the dominant error mode and my validators are close to
blind to it. Field accuracy on work reports is the number I would want to see
measured live before trusting any of this.

**Ground truth is partial and single-annotator.** I labelled it myself, from
photographs, in one pass. `optera_doc_44`'s odometer digits are genuinely
ambiguous and I excluded the value rather than guess — but I will have made
mistakes I did not notice, and there is no second annotator to catch them.

**Near-duplicate reuse is a real risk.** Hamming ≤ 3 on a 64-bit dhash is
conservative and correct on this corpus, but two blank-ish notebook pages
photographed on the same desk could collide. Mitigations in place: only
validation-passing results are ever reused, and every reused record carries
`reused_from` so it is auditable. It is still the change most likely to cause a
silent wrong answer, and in production I would gate it behind a same-day,
same-sender window.

**Multi-page documents are not handled.** Several work reports say "Page 1" or
carry a "(02)" marker, and `optera_doc_47` is a challan whose amounts may be on
a second sheet. Each image is treated as an independent document.

**Batch pricing assumes latency tolerance.** The 50% discount needs the Message
Batches API and up to 24h turnaround. That is right for an overnight WhatsApp
inbox and wrong for anything interactive. `--no-batch` reports the synchronous
number; it is 2x the batched one, and the routing gains survive without it.

**The corpus is 47 images from two clients.** The router prompt names the actual
depots and vendors it saw. That is fine for a starter set and is exactly how you
overfit to one. The brief says evaluation is on unseen images, so treat the
routing accuracy as an upper bound.

**Not tested against the live API.** The Anthropic provider is written against
the current SDK — structured outputs via `output_config.format`, `usage` read
from the response — but with no key available I could not execute it. The
schemas are strict (`additionalProperties: false`) and would be the first thing
I'd expect to need adjustment on a real run.

---

## 6. What I'd do with another week

In the order I would actually do them.

1. **Run it live and re-measure.** Everything else is guessing until the
   simulated accuracy numbers become real ones. First real question: what is the
   cheap tier's true first-pass accuracy on Gujarati handwriting? That single
   number drives the escalation rate, which drives 58% of the cost.

2. **Attack the escalation rate, not the prompt size.** Escalation dominates
   spend. Two concrete moves: insert Sonnet 5 as a middle rung (Haiku → Sonnet →
   Opus, escalating only as far as needed), and escalate *fields* rather than
   *documents* — re-asking for the three numbers that failed arithmetic is far
   cheaper than re-extracting a 30-row page.

3. **Fix the handwriting blind spot.** Cross-page consistency is the free signal
   I am not using: bus codes repeat across depots and dates, so a code appearing
   once as `TCM55` and forty times as `TCM52` is probably a misread. A fleet
   registry would turn that into a hard validator and close the biggest gap in
   section 5.

4. **Real ground truth.** Two annotators, full transcription of ~15 work
   reports, adjudicated disagreements. Field accuracy on handwriting is currently
   the least-evidenced number in the report and I would not defend it hard.

5. **Confidence calibration.** I use self-reported confidence as an escalation
   trigger without ever checking it is calibrated. Plot reported confidence
   against measured correctness; if it is uninformative, drop it and rely purely
   on the deterministic validators, which at least cannot flatter themselves.

6. **Multi-page grouping** by sender + timestamp + perceptual continuity, so a
   two-sheet invoice becomes one record.

7. **Cheaper still, if accuracy holds:** the `not_a_document` confirmation call
   is arguably redundant — the router already said so with high confidence, and
   we spend a second call to have the big-schema model agree. Dropping it saves
   ~10% of the run. I kept it because "don't invent rows from a battery photo" is
   the explicit requirement and I wanted the refusal to be an *extraction result*
   with a stated reason, not merely an absence. That is a defensible trade in
   either direction and I would want data before flipping it.

---

## 7. What's mine, and what was AI-assisted

Written with Claude (Opus 5) in Claude Code, which is also what the pipeline
targets. Mine: the staging architecture, the decision to make validators the
escalation trigger, the schemas, the cold-start/warm-cache split, the ablation
framing, all 47 ground-truth labels, and the finding that image downscaling is a
weak lever while caching is a non-lever on Haiku. Claude wrote most of the code
under that direction and generated the contact sheets I labelled from.

Two of the more useful findings in this document came from bugs the build
surfaced rather than from planning: the warm-cache measurement error, and the
delivery-challan validator that would have escalated a correct extraction
forever because it demanded a total that does not exist on that document type.

# CLAUDE.md — working context for CommodityBench

Read this first when starting a session on this repo. It captures current state and how
to continue. (Repo: **IAPS-AI/commoditybench**, **PUBLIC** as of 2026-06-26. Maintainer:
Maxwell, IAPS.) **Live results site: https://iaps-ai.github.io/commoditybench/** (GitHub
Pages, served from the `gh-pages` branch — see deploy note in Environment).

## What this is

A benchmark measuring how well LLMs perform **commodity classification** — assigning an
item its correct **Export Control Classification Number (ECCN)** under the US **Commerce
Control List (CCL)**, the task BIS performs. Goal: quantify model capability as input to a
possible IAPS recommendation about whether BIS should use LLMs for this. Stretch goal:
RAG over the CCL, evaluated with vs. without retrieval.

## Status (as of 2026-07-01)

- **He-3 gold correction + 34-item re-aggregation + `gen` leaderboard exclusion (NEWEST).**
  `lnd-2517-he3` gold corrected **1C232 → EAR99** (1C232's Controls note decontrols <1 g He-3;
  this tube holds ~2.9 mg — see the dataset note). All 8 results files containing the item were
  re-scored in place with the canonical scorer and their 4 `__summary.json` files regenerated
  (they had also been missing models: run_eval overwrites the summary per invocation, keeping
  only the last-run models — regen from the row files fixed both). The worksheet was regenerated.
  The main leaderboard now pools the **full 34 verified items over 10 runs (456 obs)**; the
  `gen` ladder run (Opus 4.1–4.7, 23 items only) is **excluded** from pooling and the example
  explorer (mixed-n distortion — see the NOTE in `aggregate_runs.py`); the ladder lives solely
  on `generation_trendline.html` (linked from README). Post-correction headline (pooled-34,
  not equalized): no-tools GPT-5.5 0.485/0.294 > Opus 4.8 0.447/0.302 > GPT-4o > Qwen3-235B;
  with-tools GPT-5.5 0.691/0.618 > Opus 4.8 0.648/0.559 > Qwen3-235B 0.482/0.353. Numbers in
  `results/*_findings.md` predate this re-aggregation. Residual caveat: ab/ab2/comparison/
  qwen3_32b_runpod runs cover only the original 23 items, so pooled cells weight those items
  more heavily (all cells still cover all 34).
- **Error-mode analysis + future-work doc (analysis-only — no dataset/run changes).**
  Dug into what the top tooled models (Opus 4.8, GPT-5.5, Qwen3-235B agentic) get wrong. The
  dominant error budget is **Category-5 boundary over-control**, in two clusters: (1)
  **mass-market crypto** — gold `5A992.c` predicted as controlled `5A002` (model identifies the
  crypto but never applies the Cryptography Note / mass-market decontrol; the discriminating fact
  isn't in the technical description); fixing the 8 gold-`5A992.c` items on the Opus agentic run
  would move exact 0.559→0.706. (2) **Ethernet-PHY over-control** — gold EAR99 PHYs/switches
  predicted `5A991` (model matches the telecom catch-all family without confirming any specific
  parameter is met; tools make it *worse*, consistent across all 3 models on the same parts).
  Second-most-common distinct error pooled is **under-classification → EAR99** (recall miss,
  skewed to Qwen). Framing reached: mass-market = mostly a **benchmark-design** issue (input
  starvation, with a model tail), PHY = mostly a genuine **model** issue (with a design tail —
  the gold rests on a manufacturer self-class Digi-Key disagreed with). Full writeup +
  prioritized v2 changes (add a `commercial_context`/availability field and A/B it, failure-mode
  tagging, over/under asymmetry metric, category backfill, confidence tiers, RAG, holdout) in
  **`results/future_research_directions.md`**. **Nothing in `data/` or `results/*.jsonl` was
  touched — all runs remain valid.**
- **Frontier comparison + PUBLIC site.** Closed the reviewer note
  that round-1 used a non-frontier OpenAI model (GPT-4o) and a small open model (Qwen3-32B),
  with tools only for Opus. Added **`gpt-5.5`** (OpenAI's strongest reasoning model on our
  account; `reasoning_effort=high`) and **`qwen3-235b`** (Qwen3-235B-A22B-Instruct-2507, 4-bit
  AWQ on a self-hosted vLLM endpoint), each run **no-tools AND agentic**. Pooled verified-23:
  no-tools **GPT-5.5 0.565 grade / 0.391 exact** > Opus 4.5 0.517 > … > **Opus 4.8 0.416**
  (5 runs) > GPT-4o 0.370 > Qwen3-32B 0.175; with-tools **GPT-5.5 0.661 ≳ Qwen3-235B 0.622 ≳
  Opus 4.8 0.615**. Takeaways: GPT-5.5 is the strongest tested both unaided and tooled; a real
  open-weight model (235B) matches/edges Opus 4.8; grounding (tools) is the lever, not scale,
  and even the best tops out ~0.61 exact with the rules in hand. Writeup:
  `results/frontier_comparison_findings.md`. Runs: `frontier`, `frontier_agentic`.
- **Results website REBUILT around 3 sections** (`scripts/build_results_site.py` +
  `aggregate_runs.py` + `results_template.html` → `dashboard/index.html`): (1) all-model
  leaderboard tooled+untooled, (2) within-model tool uplift + per-category, (3) mistake
  taxonomy (over/under-classification, wrong subparagraph/entry/category) with ECCN
  segment-match example cards. **Aggregates across 10 runs (456 obs; `gen` excluded)** per
  (model, condition) to damp per-run noise. METR-clean, self-contained, no editorializing; opens with
  an exec summary (task / benchmark / why it matters). The old `build_site.py` /
  `build_dashboard.py` were REMOVED (superseded). Regenerate: `py -3.12 scripts/aggregate_runs.py
  && py -3.12 scripts/build_results_site.py`.
- **OpenAI adapter gained reasoning + agentic support.** `reasoning_effort` switches to the
  GPT-5 surface (`max_completion_tokens`, no temperature). `OpenAIModel.classify_agentic` adds
  the agentic CCL-navigation loop with **two transports**: the **Responses API** for GPT-5
  reasoning models (which 400 on tools+reasoning over Chat Completions), and **Chat Completions**
  for non-reasoning + open-weight vLLM/RunPod endpoints. So `--agentic` now works for GPT and
  open-weight models, not just Anthropic. (Code-reviewed before the paid runs; two Responses-loop
  bugs fixed — see git log. New global rule: review request/loop code before paid API runs.)
- **Harness: done and tested.** ECCN parsing + graded scoring, provider adapters,
  concurrent eval runner, dataset loader/validator. **40 passing tests.** Tagged **`v1.0`**.
- **Dataset: 34 questions** in `data/questions.jsonl` = **23 verified** (round-1, signed
  off) + **11 new candidates** (`verified: false`) added this session to fill empty
  categories. Spans CCL Categories **1/2/3/5/7/9 + EAR99**. (See verification state below.)
- **Open-weight track added.** The OpenAI adapter now drives ANY OpenAI-compatible
  endpoint via `base_url` + `api_key_env` + a `structured` mode (`json_schema|json_object|
  none`) + `top_p`/`extra_body` passthrough. `qwen3-32b` runs on **RunPod public
  endpoints** in thinking mode (temp 0.6 / top_p 0.95 / top_k 20, 16k budget;
  `RUNPOD_API_KEY`). The shared parser (`models/base.py`) now strips `<think>…</think>`
  before extraction and falls back to `reasoning_content` when `content` is empty, so a
  reasoning model is scored on its answer, not plumbing.
- **First multi-model comparison run done** (run-id `comparison`, 23 verified Qs, no
  tools, all 100% parse / 0 errors): **claude-opus-4-8** exact 0.26 / grade 0.44 ·
  **gpt-4o** 0.17 / 0.37 · **qwen3-32b** 0.04 / 0.16. NOT equalized (Claude/Qwen3 reason,
  GPT-4o at temp 0) — a capability snapshot, not a controlled ranking. Models over-classify
  (reach for controlled ECCNs on EAR99 items) and anchor on 3A001.
- **(Superseded) earlier `build_site.py` results website** — amber/teal export-control
  identity, cross-model leaderboard + tool-lift + an "across generations" METR timeline.
  Replaced this session by the 3-section `build_results_site.py` site (see newest status
  above); `build_site.py`/`site_template.html` were removed. Generations now live in the new
  leaderboard (Opus 4.1→4.8 all appear); `dashboard/generation_trendline.html` remains as the
  standalone trendline artifact.
- **Agentic CCL-navigation tooling: built, hardened, A/B'd on the full 34.** `commoditybench/ccl/`
  parses the CCL from the eCFR (637 entries) into a navigable index + tools (`--agentic`).
  Within-model lift on Claude Opus 4.8 over all 34: **exact 0.27→0.56, grade 0.41→0.64**
  (verified-only grade 0.41→0.65), 0 errors. **Overfitting-hardened** (commit `b721a36`):
  generic order-of-review prompt with no dataset ECCNs, de-exampled tool descriptions,
  removed the catch-all search up-weight — and the lift held, generalized to the 11 new
  categories added afterward, and beat a planted over-classification trap. Tools fix
  *recall* gaps (catch-alls, thresholds) but not *judgment* calls; reading the 5A991 telecom
  catch-all even induces over-control of EAR99 Ethernet PHYs (double-edged). Full write-up:
  `results/agentic_ab_findings.md`. Runs: `expanded` (cross-model) + `expanded_agentic`.
- **Cross-generation trendline (NEW, branch `cross-generation-trendline`).** A METR-style
  plot of capability vs. **model release date** across the **Opus generation ladder**
  (4.1→4.5→4.6→4.7→4.8; older gens are retired/404 as of 2026-06-25). The Anthropic adapter
  now drives older generations via per-model `thinking_mode` (`extended` for 4.1/4.5 which
  predate adaptive thinking) + optional `effort` (4.1 has none); configs verified by a 400
  probe. **No-tools run done** (run-id `gen`, 23 verified, 0 errors): the trendline is
  **flat then dips** — exact 0.35→0.39 across 4.1–4.7, then **4.8 lowest at 0.22** (grade
  0.40); `group`/`category` steady ~0.57–0.61. Newer Opus isn't better at unaided ECCN
  classification here; over-classification plausibly worsens it. (The ladder briefly appeared
  in the main leaderboard but was excluded 2026-07-01 — 23-item run vs the 34-item pool; it
  now lives only on the trendline page.) Standalone trendline page:
  `dashboard/generation_trendline.html`; builder `scripts/build_generation_trendline.py`;
  writeup `results/generation_trendline_findings.md`.
- **Subscription harness for the tools condition (NEW): `cc_harness/`.** Runs the agentic
  (read-the-CCL) condition through a **Claude Code session on a Max subscription** instead of
  burning API credits (the API agentic cross-gen run died on a credit wall). The folder is
  self-contained: `CLAUDE.md` (the agentic system prompt + tool guide, auto-loaded so the
  session *is* the model under test), `ccl.py` (the 4 CCL tools as a CLI — `categories/outline/
  read/search`, vendored verbatim from `ccl/index.py`), vendored `ccl_index.json`, `questions.jsonl`
  (the 23 verified items **sanitized** to id+name+description — no gold, no category), `record.py`
  (writes answers to `../../cc_results/<label>__answers.jsonl`, **outside the repo** so runs can't
  peek and no gold sits beside the questions), and `PROMPT.txt`. Grade with
  `scripts/grade_cc_runs.py` (same graded scorer + run_eval-shaped output; prints the tool lift
  vs. the no-tools `gen` baseline by label). **Caveat:** Claude Code's `/model` only exposes
  current-gen (Opus 4.8 / Sonnet 4.6 / Haiku 4.5), so this yields the tools condition for those
  (completing the 4.8 tool-lift point + a cross-tier tools comparison), not the full 4.1→4.8 ladder.
- **Not done:** sign-off on the 11 new candidates; Cat **0/4/6/8 still empty**; equalized
  comparison; RAG index. Cross-model headline accuracy should still cite the 23-verified set;
  keep the not-equalized / tool-lift framing. The expanded runs include unverified items (for
  the site) and are flagged NOT CITABLE by the harness.

## Repo map

```
src/commoditybench/
  eccn.py          # ECCN parse/normalize + GRADED scoring (exact/eccn/group/category, EAR99). CORE.
  dataset.py       # pydantic Question schema + JSONL loader (verified_only gate)
  prompts.py       # provider-agnostic prompt + answer JSON schema (+ optional RAG context)
  run_eval.py      # CLI: concurrent eval, scoring, aggregation, results
  models/          # base.py (interface + JSON parse/fallback + <think>-strip), registry.py,
                   #   agentic.py (shared agentic prompt/submit-tool), {anthropic,openai,gemini}_model.py.
                   #   openai_model.py drives OpenAI GPT (incl. reasoning models via reasoning_effort)
                   #   AND any OpenAI-compatible endpoint (RunPod/vLLM): qwen3-32b, qwen3-235b, gpt-5.5.
                   #   Both anthropic_model and openai_model implement classify_agentic (--agentic).
  ccl/             # AGENTIC condition: parsed CCL + navigation tools the model calls
                   #   parse_ecfr.py (eCFR XML -> ccl_index.json), index.py (CCLIndex:
                   #   category_outline/read/search), tools.py (tool specs + CCLToolbox)
  rag/             # stretch: CCL retrieval (retriever.py + build_index.py) — index NOT built yet
data/
  ccl/ccl_index.json       # parsed CCL (637 ECCN entries) for the agentic tools; committed
  ccl/ccl_supp1.xml        # raw eCFR source for the index (reproducibility)
  questions.jsonl          # the candidate dataset (verify before citing)
  questions.example.jsonl  # synthetic fixtures for smoke-testing only (never a result)
  schema.md                # field schema + sourcing methodology + provenance rules
  verification_worksheet.md# generated; the human verification tracker
  sources/bis-listed-eccn-publishers.md  # BIS list distilled by category = the sourcing plan
ccats-website-table-april-2018.pdf       # the BIS public-classification list (source doc)
human-review-sources/    # purchase invoices used as provenance (committed)
scripts/
  build_dataset.py # validate <file> | template
  make_worksheet.py# regenerate data/verification_worksheet.md from questions.jsonl
  aggregate_runs.py        # pool ALL runs per (model, condition) -> dashboard/site_data.json
  build_results_site.py + results_template.html  # CURRENT 3-section site (leaderboard /
                   #   uplift / mistakes) from site_data.json -> dashboard/index.html
  build_generation_trendline.py  # standalone METR trendline -> dashboard/generation_trendline.html
  grade_cc_runs.py # grade the cc_harness (subscription) answers vs the no-tools gen baseline
dashboard/index.html     # generated results website (committed); also published to gh-pages
tests/             # pytest (eccn scoring + aggregation math + parser/<think>-strip)
```

## How to run (Windows; use `py -3.12`)

```bash
py -3.12 -m pytest -q                                          # 40 tests
PYTHONPATH=src py -3.12 scripts/build_dataset.py validate data/questions.jsonl
py -3.12 scripts/make_worksheet.py                            # regen worksheet after dataset edits
# Eval (needs provider SDKs + API keys in .env): pip install -e ".[all]"
PYTHONPATH=src py -3.12 -m commoditybench.run_eval --dataset data/questions.jsonl \
  --models claude-opus-4-8 --verified-only --workers 8
# Multi-model comparison (frontier OpenAI + open-weight; needs OPENAI_API_KEY/RUNPOD_API_KEY).
# qwen3-235b needs a running vLLM endpoint: set QWEN3_235B_BASE_URL (see Environment).
PYTHONPATH=src py -3.12 -m commoditybench.run_eval --dataset data/questions.jsonl \
  --models claude-opus-4-8 gpt-5.5 qwen3-235b --verified-only --workers 6 --run-id frontier
# AGENTIC condition (Anthropic + OpenAI/OpenAI-compatible): model reads the CCL via tools first.
PYTHONPATH=src py -3.12 -m commoditybench.run_eval --dataset data/questions.jsonl \
  --models claude-opus-4-8 gpt-5.5 qwen3-235b --agentic --workers 4 --run-id frontier_agentic
# Build the results WEBSITE (aggregates ALL runs; pool first, then build):
PYTHONPATH=src py -3.12 scripts/aggregate_runs.py
PYTHONPATH=src py -3.12 scripts/build_results_site.py
# Redeploy the site to GitHub Pages (after rebuilding dashboard/):
git add -A && git commit -m "..." && git push origin main
git push -f origin "$(git subtree split --prefix dashboard main)":refs/heads/gh-pages
# Rebuild the CCL index from the eCFR (already committed; only if it needs refreshing):
PYTHONPATH=src py -3.12 -m commoditybench.ccl.parse_ecfr --fetch
```
Default Anthropic model is `claude-opus-4-8` (adaptive thinking + structured outputs).
Headline accuracy is over ALL questions (errors/unparsed = 0); `--verified-only` is
mandatory for any reported number.

## Provenance rules (NON-NEGOTIABLE — the maintainer enforces these)

1. **A label's `source_url` must DISPLAY the ECCN to a human who opens it.** A value
   scraped from a page's hidden backend field does NOT qualify even if correct. (This
   disqualified Thorlabs — thorlabs.com serves the ECCN via an internal `eccnTL` data
   feed, never rendered on the page.) If the page doesn't show it, cite a document that
   does (e.g. a purchase invoice).
2. **Prefer companies on the BIS public-classification list** (`data/sources/...`). It's a
   BIS-facing benchmark, so BIS-listed self-classifications are more defensible.
3. **Manufacturer beats distributor.** Digi-Key was found WRONG on 2 Microchip parts vs.
   Microchip's own tool — cross-check distributor data against the maker when possible.
4. **`verified: true` is the maintainer's act**, not the assistant's. Agents can confirm a
   value is human-visible + matches, but a human flips the flag.

## Verification state (what's pending)

Run `py -3.12 scripts/make_worksheet.py` then open `data/verification_worksheet.md` — the
table links each item to its source (with the part to search for tool-based sources).

- **23 VERIFIED** (`verified: true`): 2 Thorlabs invoices (KM100, BB1-E02) + 21 round-1 items
  signed off this session (9 ADI Tier A, 7 Microchip Tier B, 5 Digi-Key Tier B). The 6
  non-human-visible Thorlabs (incl. the 1550 nm `6A995` lasers) were **cut** — that emptied
  Cat 6.
- **11 NEW candidates (`verified: false`), added this session — awaiting sign-off.** Human-
  visible ECCN confirmed by agents; flip after a spot-check (worksheet links each one):
  - **Cat 7** (4): 3 ADI IMUs/gyros (`7A994` ×2, `7A003.d.1`; **Tier A**, same ADI tool) +
    Honeywell HGuide i300 (`7A994`, manufacturer page, Tier B).
  - **Cat 1** (3): LND neutron detectors — `1A999` ×2 (BF3, B10-lined) + `1C232` (He-3).
    Tier B, LND public FAQ; **ECCN assigned by family, not per-part** — clean for the two
    1A999; the `1C232` rests on the FAQ's "certain models" hedge.
  - **Cat 2** (2): `2B999.j` bronze gear pump (Tier B distributor; ECCN sits in the
    product-name string) + a P4/ABEC-7 super-precision bearing that is **EAR99** (a
    deliberate over-classification trap vs 2A001).
  - **Cat 9** (2): 2 Piper aircraft parts, `9A991.d`. **Tier C / lowest confidence** —
    PilotsHQ appears to template the BIS aircraft-parts default; cross-check vs OEM or drop.

## Immediate next steps (in order)

1. **Sign off the 11 new candidates**: spot-check ~2 per source via the worksheet, then set
   `verified: true` on the confirmed rows (edit `data/questions.jsonl`; keep notes/source
   intact) and `make_worksheet.py`. The Piper Cat-9 pair is the weakest — decide drop vs
   keep first. The agentic explorer on the site is a fast spot-check aid: the parts where
   tools went *wrong* (`lnd-2311-b10`, the two Piper items) may be tool errors OR bad labels.
   Then re-run `expanded` + `expanded_agentic` and rebuild the site on the larger verified set.
2. **DONE — agentic tooling + cross-vendor comparison complete.** Agentic A/B (Opus 4.8
   exact 0.27→0.56) plus the frontier comparison (GPT-5.5, Qwen3-235B in both conditions; see
   newest status). The OpenAI/OpenAI-compatible adapter now has `classify_agentic`, so GPT and
   open-weight models get tools too. Remaining tool idea if revisited: a precision guardrail
   for the 5A991 telecom over-control (double-edged finding).
3. **Backfill still-empty categories 0/4/6/8.** Cat 6 regressed to empty (Thorlabs lasers
   cut) — re-source via DRS Infrared/Pelco (thermal cameras, 6A003) per the sourcing map.
   Cat 4 = Apple/HP/Oracle ECCN matrices; Cat 8 (marine) has few public sources. Cat 0 has
   essentially no commercial product with a visible ECCN (reactors/facilities) — likely a
   BIS FAQ example or skip. Strong unfinished Cat-9 lead: Boeing Distribution `9A991`
   connector pages (WAF-blocked from this box; open in a real browser).
4. **Equalize the comparison** (a real decision before any head-to-head ranking — see Open
   issues) or keep labeling every cross-model table "not equalized."
5. **Then** the RAG stretch: build the index from the eCFR (15 CFR 774 Supp. No. 1; see
   `rag/build_index.py` TODO) and A/B it.

## Open issues / flags

- **Provider comparability is NOT equalized**: each model runs in its strongest native
  config (Claude/GPT-5.5/Qwen3 with reasoning on; GPT-4o at temp=0). The site/findings are
  labeled a capability snapshot, not a controlled ranking. Decide before any head-to-head claim.
- **Qwen3-235B agentic has 1/34 unscored** (the Piper Cat-9 item's tool context exceeds the
  41k window; also weakest provenance) — counted as 0. Lower the agentic output budget or raise
  MAX_MODEL_LEN to recover it.
- **New-candidate provenance caveats** (carry into sign-off): LND ECCNs are family-level
  (per-part `1C232` unconfirmed); the Oberdorfer `2B999.j` lives in a distributor product-
  name string, not a dedicated field; the Piper `9A991.d` pair looks distributor-templated.
- **Distribution skew**: still heavy on Cat 5 + EAR99 (EAR99 = 10 of 34 since the
  He-3 correction moved 1C232 → EAR99). Categories
  **0/4/6/8 empty**; 1/2/9 now thin (1–3 each). Cat 6 emptied when the Thorlabs lasers were cut.

## Environment / tooling

- **Repo is PUBLIC** (`IAPS-AI/commoditybench`). The results site is deployed to **GitHub
  Pages** at https://iaps-ai.github.io/commoditybench/ from the **`gh-pages`** branch (the
  `dashboard/` folder published at root via subtree split). Redeploy after a site rebuild:
  `git push -f origin "$(git subtree split --prefix dashboard main)":refs/heads/gh-pages`
  (Pages auto-rebuilds; ~1–2 min). Before any visibility/exposure change, scan for secrets
  (`.env` is gitignored; no keys are tracked).
- **Qwen3-235B endpoint is ephemeral.** `registry.py` reads it from `QWEN3_235B_BASE_URL`
  (placeholder default). To run `qwen3-235b`, deploy a vLLM endpoint for
  `QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ` (TP=2 ~2×H100; `--enable-auto-tool-choice
  --tool-call-parser hermes`; `MAX_MODEL_LEN ≥ 40960`, but agentic tool-heavy items can still
  exceed it — lower the per-call output budget) and set the env var. RunPod **serverless**
  wedged under sustained agentic load (worker won't drain its queue); use a **dedicated pod**
  for agentic batches and **tear it down after** (it bills continuously, ~$6.6/hr for 2×H100).
- **Run a local code review before paid API runs** (global rule
  `~/.claude/rules/code-review-before-paid-api.md`): review request/loop code and fix
  confirmed correctness bugs before spending on LLM/GPU runs. A cheap smoke test first is fine.
- `gh` authed as `maxwell-k-roberts` with access to `IAPS-AI`. Commit on a branch or main;
  end commit messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Python: `py -3.12`. For PDFs use the Read tool; for blocked/JS pages use
  `node tools/fetch.mjs "<url>" --wait 6000`. The ECCN-verification helpers are now
  **vendored in `tools/`** (was `C:/Users/maxwe/tools/`); see `tools/README.md` for the
  per-machine setup (`npm install playwright`; `adichrome.mjs` needs a real Chrome).
- Helper from a prior session: `node tools/mchpeccn.mjs <part>` hits Microchip's
  export-control API (returns ECCN JSON) — useful for cross-checking Microchip parts.
- Manufacturer export endpoints (ADI, Microchip, Thorlabs) are session/XHR-gated; plain
  curl returns empty. Use a headless browser (the fetch.mjs tool) or a research agent.
- **ADI export tool** (`analog.com/.../view-export-classification.html`) only renders results
  under the REAL Chrome engine — headless Chromium fails its JS via `ERR_HTTP2_PROTOCOL_ERROR`
  and returns an empty form / placeholder row. A reusable Playwright driver (launches
  `channel:"chrome"`, types the part into the search box, reads the US ECCN cell + cross-checks
  the `FetchExportClassificationData` JSON) is at `tools/adichrome.mjs`.
- For open-weight evals, `RUNPOD_API_KEY` (RunPod public endpoints, ~$10/1M tokens) joins the
  three provider keys in `.env`; `.env.example` lists all four.

## Memory

Durable facts are in the session memory dir (`MEMORY.md` index): IAPS org URL, project
overview, BIS index + accuracy caveat, and the dataset provenance rules. They load
automatically; this file is the project-local complement.

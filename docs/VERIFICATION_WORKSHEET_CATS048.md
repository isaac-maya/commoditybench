# Verification worksheet — Category 0/4/6/8 backfill (staged contribution)

Sourcing date: **2026-09-01**. All source URLs were opened and checked by the contributor
on that date. Per the repo's provenance rules, **flipping `verified: true` is the
maintainer's act** — this worksheet is the evidence table for that sign-off.

**Tiers (repo convention):** `A` = on the BIS list, ECCN human-visible · `B` = human-visible
but off the BIS list · `C` = ECCN **not** human-visible on the source (needs an invoice or
another human-visible source, else drop).

## Decision table

| # | Item | ECCN | Tier | Verify here | ✅ | Notes |
|---|------|------|------|-------------|:--:|-------|
| 1 | Dell dual-socket rack server | `4A994` | B | [Dell regulatory datasheet PDF](https://i.dell.com/sites/doccontent/shared-content/solutions/en/Documents/pedge_840_us.pdf) | | "System US Export Control Classification Number (ECCN) 4A994" — human-visible in the datasheet |
| 2 | IBM mid-range server (APP 0.5–70 WT) | `4A994.b` | B | [IBM Hardware Classification Table](https://www.ibm.com/products/exporting/index.shtml) | | Row `J4dn | 4A994.b | APP Greater than 0.50 WT but less than or equal to 70 WT` — human-visible |
| 3 | Eight-GPU AI training server | `4A090.a` | **C** | [NVIDIA 8-K (SEC)](https://www.sec.gov/Archives/edgar/data/1045810/000104581025000007/nvda-20250113.htm) | | NVIDIA export page is a JS form (checked headless — no static ECCN). Label = NVIDIA's own 8-K statement that DGX/HGX/MGX systems are classified 4A090.a. Re-source via NVIDIA ECCN tool interactively before sign-off |
| 4 | LWIR thermal core, 640×512, 60 Hz | `6A003.b.4.b` | B | [GroupGets FLIR Boson 640](https://groupgets.com/products/flir-boson-640) | | "ECCN 60/30Hz: 6A003.b.4.b" — human-visible on an authorized FLIR distributor page |
| 5 | LWIR thermal core, 640×512, 9 Hz | `6A993.a` | B | [GroupGets FLIR Boson 640](https://groupgets.com/products/flir-boson-640) | | "ECCN 9Hz: 6A993.a" — same page. TRAP PAIR with #4: frame rate only |
| 6 | Scuba buoyancy compensator (BCD) | `8A992.h` | **C** | [Lexology summary of BIS FAQ](https://www.lexology.com/library/detail.aspx?g=c41a34ab-54ec-4bea-aa94-507c13556b0d) + [8A992 entry text](https://www.ecfr.gov/current/title-15/subtitle-B/chapter-VII/subchapter-C/part-774) | | No manufacturer page with visible ECCN found (Blue Robotics, dive retailers checked). Label = BIS FAQ family assignment + entry text. Subparagraph `.h` is the annotator's reading — confirm vs invoice/advisory opinion |
| 8 | Skid-mounted fracturing pump, 15,000 psi | `0A998.b.3` | **C** | [BIS Russia oil/gas sanctions FAQs (PDF)](https://www.bis.gov/media/documents/russia-oil-gas-sanctions-faqs.pdf) | | No manufacturer page with visible ECCN found. Label = BIS FAQ Q.13 + entry text (0A998 dates to the 2014 sectoral rule per the FAQ; license requirements today under 15 CFR 746.8 per the committed index). SCOPING CAVEAT: 0A998 applies for Russia/Belarus; other destinations may be EAR99 |
| 9 | Deuterium oxide (heavy water) 99.9%, NMR use | `1C298` | **C** | [FR 2021-21509 (Oct 6, 2021)](https://www.federalregister.gov/documents/2021/10/06/2021-21509/control-of-deuterium-that-is-intended-for-use-other-than-in-a-nuclear-reactor-under-the-export) | | MilliporeSigma pages do not display ECCN (checked). Label = 2021 rule + 1C298 entry text. TRAP: pre-2021 models answer 0C003 (entry removed from the CCL); reactor-use deuterium is NRC-jurisdiction |

## Verification status

- **2 solid (Tier B, manufacturer/distributor pages, fetched and verified):** #1 (Dell
  datasheet PDF), #2 (IBM table). *Both fetched 2026-09-01; ECCN visible in page text.*
- **2 solid (Tier B, distributor page, fetched and verified):** #4, #5 (GroupGets —
  same page, both ECCNs visible).
- **1 needs an interactive check (Tier C → A):** #3 (NVIDIA ECCN tool; the 8-K is a
  strong manufacturer statement but not a per-product export page).
- **3 Tier C, judgment calls flagged:** #6 (subparagraph reading), #8 (0A998 scoping),
  #9 (end-use dependency). Each needs the maintainer's call or a better source.
- **1 synthetic row (Tier C, #7 underwater TV):** NOT in this shipped set — held
  locally and offered in the PR text per schema.md's synthetic provision (addendum below).

## Judgement calls (must be read by the maintainer)

1. **#8 scoping:** 0A998 is a Russia/Belarus sector-sanctions entry. The item is
   genuinely *on the CCL* (Cat 0, "Miscellaneous"), which is what the benchmark asks;
   but the model that answers 0A998 without noting the §746.8 scoping has learned the
   entry, not the law. Consider whether the answer key should accept the ECCN alone.
2. **#9 end-use:** "intended for use other than in a nuclear reactor" is end-use
   dependent — a licensing officer resolves it with the buyer. The question is fair
   (description states the intended use) but flag it as the judgment it is.
3. **#6 subparagraph:** `.h` ("Other self-contained underwater breathing apparatus
   (scuba gear) and related equipment, n.e.s.") is the annotator's reading for a BCD.
   Alternative: 8A992's residual `.i`-class catch-all — verify against the full entry.
4. **Category 0 restructuring (context for #8/#9 and future items):** the current CCL
   (repo index == live eCFR, 637 entries, verified 2026-09-01) no longer contains the
   classic nuclear entries 0A001/0B001-0B006/0C001/0C002/0C004 — they are under NRC
   licensing authority (10 CFR 110) and 0A002 is ITAR-subject; deuterium for non-reactor
   use now lives in 1C298 (Cat 1). Any future "Cat 0 nuclear" sourcing must start from
   this reality: the remaining dual-use Cat-0 items are 0A521/0A998/0A999 + the munitions
   family. This is also the motivating case for the abstention condition (a reactor-use
   heavy-water question has no valid ECCN/EAR99 answer — it is NRC-jurisdiction).

## Addendum (2026-09-05) — shipped set = 8 real rows
- The synthetic row (#7, underwater TV 8A992.a.2) is EXCLUDED from this PR's data
  file so every shipped row is a real item with a human-checkable source. It stays
  in the local master file (9 rows, sha256 706d174f…) and is offered in the PR
  text — its gold is entry-verified (750 lines > the 700-line threshold in
  8A992.a.2); say the word and it ships as its own labeled slice.
- The pump row (0A998.b.3) ships Tier C flagged per decision A1: description
  unedited (datasheet voice), worksheet caveat above stands, with the scope-class
  analysis from the companion preflight PR (0A998 is a scope-conditioned entry):
  the expected outcome is that the maintainer holds it at verified:false until a
  v2 context field exists.
- Derived-file note: this worksheet and the PR data file describe the 8-row
  shipped set; the local master keeps all 9 rows.

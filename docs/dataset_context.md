# Dataset context and provenance

Compiled from direct inspection of `whale-ici-data` and `Project-CETI/sw-combinatoriality` repos,
the birth tracking codebase (`Project-CETI/whale-birth-data-and-analysis-suite`), CETI
data-ingest infrastructure, and a web survey of all three papers.

---

## Papers and data repositories

| Key | Title | Journal | DOI | Data |
|---|---|---|---|---|
| Hersh 2022 | "Evidence from sperm whale clans of symbolic marking in non-human cultures" | PNAS 119(37) | 10.1073/pnas.2201692119 | OSF https://osf.io/ae6pd/ |
| Sharma 2024 | "Contextual and combinatorial structure in sperm whale vocalisations" | Nat. Commun. 15:3617 | 10.1038/s41467-024-47221-8 | GitHub https://github.com/Project-CETI/sw-combinatoriality |
| Sharma 2025 | "Description of a collaborative sperm whale birth and shifts in coda vocal styles during key events" | Sci. Rep. 16:9206 | 10.1038/s41598-025-27438-3 | GitHub https://github.com/Project-CETI/whale-birth-data-and-analysis-suite |
| Gero 2026 | "Cooperation by non-kin during birth underpins sperm whale social complexity" | Science | 10.1126/science.ady9280 | Zenodo https://doi.org/10.5281/zenodo.18016792 |

Gero 2026 (Science) and Sharma 2025 (Sci. Rep.) are companion papers on the same July 8 2023 birth event.
Preprints: Sharma 2024 has a bioRxiv at https://www.biorxiv.org/content/10.1101/2023.12.06.570484v1

Supplements already in this repo:
- `docs/pnas.2201692119.sapp.pdf` — Hersh 2022 PNAS supplement
- `docs/sharma2024_nat_commun_supplement.pdf` — Sharma 2024 Nat. Commun. supplement
- `docs/sharma2024_TheBookofWhales.pdf` — Sharma 2024 "Book of Whales" supplementary reference
- `docs/rsos150372supp1.docx` — Gero 2016 supplement (OPTICSXi type definitions)

---

## Sister repository: whale-ici-data

`https://github.com/morganrivers/whale-ici-data` — the upstream data pipeline that produces
`data/upstream/codas_unified.csv`.

### Raw source files it ships

| File | Source | Notes |
|---|---|---|
| `data/raw/dswp_dominica_codas.csv` | Sharma 2024 sw-combinatoriality/data/DominicaCodas.csv | 8,719 rows, 9-ICI cap, has CodaType |
| `data/raw/sw_combinatoriality_dialogues.csv` | Sharma 2024 sw-combinatoriality/data/sperm-whale-dialogues.csv | 3,840-row chorus subset with REC/Whale/TsTo timestamps, up to 29 ICIs |
| `data/raw/sharma2025_birth.csv` | Sharma 2025 Sci. Rep. supplement | Banner row "Whale Birth Audio_annotations" |
| `data/raw/hersh2022_pacific_codas.csv` | Hersh 2022 OSF ae6pd | 24,237 rows, up to 30 ICIs |
| `data/raw/hersh2022_pacific_README.txt` | Hersh 2022 OSF ae6pd | Column definitions + Nov 2022 duration-fix note |

### Columns stripped in the ici-data pipeline (present in raw but NOT in codas_unified.csv)

**Hersh raw (`hersh2022_pacific_codas.csv`):**
- `within_clan_correlation` — Pearson r of the repertoire against its clan centroid (23,555 non-null). Measures how "clanlike" each repertoire is; 682 codas have NaN (no clan assigned).
- `coda_type` — Hersh's own numeric coda-type code from IDcall + mclust classification (23,429 non-null; 808 NaN = 2-click or >10-click codas). These are Hersh's type labels, **not** Gero/Sharma labels; they are not the same taxonomy. Not currently preserved in `codas_unified.csv`.
- `year` — redundant with `date`, dropped.

**DSWP raw (`dswp_dominica_codas.csv`):**
- `CodaType` — Gero 2016 string label (e.g., "5R3", "1+1+3", "6-NOISE"). 100% populated. The rhythm integer (0–17) in codas_unified is derived from this via nearest-centroid matching; the original string label is lost.
- `UnitNum` — numeric index of the social unit within the study (1–13). Redundant with `Unit` (letter code), dropped.

**Birth raw (`sharma2025_birth.csv`):**
- `Coda` — always 0; placeholder column (coda-type field not used in the birth release). Dropped.
- `Var44`–`Var52` — overflow ICI columns (ICI40–ICI48) for 3 extremely long codas (40–48 clicks). The ICI schema in codas_unified caps at 40 clicks (ICI40), so these overflows are lost, but only 3 codas are affected.

---

## Per-source raw data structure (full columns)

### Hersh 2022 raw

```
clan_name, within_clan_correlation, coda_type, grpvar, loc, year, date,
latitude, longitude, coda_number, nclicks, duration, ICI1..ICI30
```

- `grpvar` = repertoire-day code (numeric) — becomes `recording_id` in unified schema
- `loc` = 2–5 character location abbreviation; see `F_load_hersh_pacific.LOC_TO_REGION` for mapping
- `coda_type` = Hersh's IDcall+mclust type (NOT Gero 2016 types)
- No per-coda wall-clock time; no individual IDs; no social unit (grpvar is recording-day, not persistent unit)
- 3,113 Galápagos codas from Watkins archive have `coda_number` like "WatGal###"

### DSWP raw (DominicaCodas.csv)

```
codaNUM2018, Date, nClicks, Duration, ICI1..ICI9, CodaType, Clan, Unit, UnitNum, IDN
```

- ICIs capped at 9 in this file (covers ~99% of codas; long codas only in sperm-whale-dialogues.csv)
- `CodaType` = Gero 2016 string label (21 types + NOISE flags)
- `IDN` = persistent photo-ID of the vocalising whale (0 = unknown; 35 named individuals, ~3,014 codas attributed)
- No per-coda wall-clock time in this file

### DSWP dialogues raw (sperm-whale-dialogues.csv)

```
REC, nClicks, Duration, ICI1..ICI28, Whale, TsTo
```

- `REC` = recording ID (e.g., `sw061b001_124`) — CETI hydrophone-tag deployment identifier
- `Whale` = within-recording speaker number (integer)
- `TsTo` = seconds from recording start
- 3,840 rows (a chorus/dialogue subset with timing); carries up to 29 ICIs
- Joined on ICI fingerprint to DominicaCodas by `A_load_dswp.py`; ~3,780 codas matched

### Birth raw (sharma2025_birth.csv)

```
[banner: "Whale Birth Audio_annotations"]
Recording, SegmentWhale, Coda, TfS, ICI1..ICI39, Var44..Var52
```

- `Recording` = CETI acoustic recording ID, e.g., `CETI23-277` (see below for meaning)
- `SegmentWhale` = integer 1–10, within-recording caller-separation segment (NOT a persistent ID)
- `Coda` = always 0 (unused placeholder)
- `TfS` = time-from-start in seconds (sub-second precision)
- `Var44`–`Var52` = ICI40–ICI48 for 3 edge-case codas with >39 clicks; all others NaN

---

## CETI recording ID format (CETI23-xxx)

The `CETI23-xxx` IDs in the birth acoustic dataset (277–294) are **sequential hydrophone-tag deployment
numbers** from CETI's 2023 Dominica field season. They are **not** Unix timestamps and do not encode
the calendar date directly. The 17 recordings CETI23-277 through CETI23-294 all come from a single
birth-event encounter (July 8 2023); the numbers are just consecutive tag-roll IDs.

The **wall-clock start time** of each acoustic recording (and its GPS fix) is stored in CETI's internal
AWS S3 pipeline under `raw/<ingestion-date>/<device-folder>/`. This metadata is not included in the
public birth data release; `date` is hardcoded to "2023" and no GPS is published.

### Drone video timestamps

The companion video data (Science paper) uses DJI Mavic 3 drone footage. Raw video files are named
with Unix timestamps in milliseconds (e.g., `1688827660979.MP4`). These ARE wall-clock times:

```python
import datetime
datetime.datetime.utcfromtimestamp(1688827660979 / 1000)
# → 2023-07-08 14:47:40.979000 UTC  (~58 min before birth)
```

The birth itself occurred at **2023-07-08 15:45:45 UTC** (11:45:45 EDT). The video dataset
spans from ~62 min pre-birth to ~175 min post-birth across 17 video clips from two drones
(CETI-DJI_MAVIC3-1 and DSWP-DJI_MAVIC3-2).

**Individual whale names** are present in the video tracking data but are NOT linked to the acoustic
ICI dataset. The named individuals in Unit A (11 whales: 8 adults, 3 calves):
- Rounder (mother of the newborn; known to DSWP since 2005)
- Lady Oracle (grandmother)
- Accra (daughter)
- Aurora, Allan, Atwood, Fruit Salad, Soursop (and others)

These names come from DSWP's long-term photo-ID catalog. Mapping them to the birth acoustic
`SegmentWhale` numbers (1–10) would require temporal alignment of acoustic recordings with drone
video, which has not been published.

---

## Recording methodology (ships / observer presence / whale positioning)

All three datasets record **free-swimming** sperm whales in the open ocean. No animals were
tethered or fixed.

**Hersh 2022 (Pacific, 1978–2017):**
Data aggregated from >40 years of research cruises across the Pacific. Recordings made from
research vessels using towed hydrophone arrays or moored hydrophones during surface encounters.
Whales surfaced to breathe between dives; the ship tracked the group acoustically and visually.
No individual IDs were resolved because most recordings predated systematic photo-ID work, and
many come from short encounters. Location precision varies widely: some recordings have
GPS coordinates to ~3 decimal places (~100 m), some older records are rounded to 1–2 dp (~10 km).

**Sharma 2024 DSWP (Dominica, 2005–2018):**
Dominica Sperm Whale Project field work from a small research vessel off Dominica. Two recording
modalities:
1. **DTag suction-cup tags** attached to individual whales — these provide per-whale acoustic
   tracks and yield the `REC` + `Whale` + `TsTo` timing in the dialogues file. This is the source
   of the 3,780 rows with timestamps. The raw tag data (device hostname `wt-XXXXXXXXXXXX`)
   includes GPS and absolute wall-clock time, but that metadata is not in the public release.
2. **Group hydrophone recordings** — towed or drifting hydrophones near the surfacing group,
   without individual-animal attribution. These yield the 5,092 rows with no timing in
   DominicaCodas.csv.
The `recording_id` values like `sw061b001_124` are CETI's internal deployment identifiers for
DTag sessions (sw = sperm whale, 061b001 = tag unit, 124 = deployment number within season).

**Sharma 2025 Birth / Gero 2026 Science (Dominica, July 8 2023):**
CETI field deployment off Dominica during a live birth event. Data collected simultaneously by:
- **Acoustic hydrophone tags** on multiple whales (source of the CETI23-xxx ICI data)
- **DJI Mavic 3 aerial drones** operated by CETI and DSWP teams (source of whale-tracking data)
- **Shipboard photo-ID** (source of individual names / kinship data)

The acoustic ICI dataset published in Sharma 2025 Sci. Rep. does NOT include GPS positions,
individual whale names, or absolute timestamps. The per-individual tracking data (positions,
orientations, identities) is in HDF5 segmentation files to be released via Harvard Dataverse
(not yet public as of early 2026; "Link Will Be Provided Soon" in the analysis repo).

---

## Individual whale identification: current state

| Dataset | Named photo-ID in ICI data | Proxy speaker tag | Notes |
|---|---|---|---|
| Hersh | 0 / 24,237 | 0 / 24,237 | No individual resolution; clan only |
| DSWP | 3,014 / 8,719 (34.6%) | 3,780 / 8,872 (42.6%) | 35 named photo-IDs; IDN=0 for ~65% |
| Birth | 0 / 5,731 | 5,731 / 5,731 (100%) | SegmentWhale is 1–10 per recording, not persistent; names exist in drone data only |

Sharma et al. 2024 demonstrate high-accuracy **individual identification from coda acoustics**
(based on the DSWP population with named photo-IDs). The 65% of DSWP codas with IDN=0 are the
most tractable target for applying this classifier, with the 35% labeled set as ground truth.

Applying Sharma's classifier to the Birth dataset requires first establishing which SegmentWhale
segment corresponds to which named individual — a temporal-alignment problem that needs both the
acoustic timestamps (from CETI's internal records) and the drone video (to be released on
Harvard Dataverse).

---

## Clan labels and taxonomy

| Dataset | Clan field | Values | Coverage |
|---|---|---|---|
| Hersh | `clan` | FP, PALI, PO, REG, RI, SH, SI | 97.2% (682 unassigned — low within-clan correlation) |
| DSWP | `clan` | EC1, EC2 | 98.3% |
| Birth | `clan` | EC1 | 100% (hardcoded — all from Unit A, Dominica EC1 clan) |

The Hersh clan abbreviations (from Hersh 2022 Table S1): FP = Four-Plus, PALI = Palindrome,
PO = Plus-One, REG = Regular, RI = Rapid Increasing, SH = Short, SI = Slow-Increasing.
These are Pacific clans; they are not comparable to the Caribbean EC1/EC2 labels.

---

## Coda type taxonomies (what was stripped and why it matters)

Three overlapping but distinct coda-type systems exist in the literature:

1. **Hersh coda_type** (raw column, stripped from unified) — 808 values assigned by IDcall + mclust
   mixture models on Pacific data. Not published as string labels; just numeric codes. The meaning
   of specific numbers is in the supplementary of Hersh 2022 (PNAS) but was not preserved in ici-data.

2. **Gero 2016 CodaType** (raw DSWP column, stripped and replaced) — 21 named types
   (e.g., "5R3", "1+1+3") + NOISE flags, produced by OPTICSXi on Caribbean data only. The string
   labels are present in `data/raw/dswp_dominica_codas.csv` (CodaType column) but are replaced
   with numeric rhythm IDs (0–17) in codas_unified.csv. The mapping is:
   - Gero named 21 types; Sharma 2024 collapsed to 18 rhythm classes by dropping the
     duration-rank suffix (R1/R2/R3 → R, so "5R1","5R2","5R3" all → "5R").
   - The numeric 0–17 rhythm IDs in codas_unified correspond to Sharma 2024's 18-class vocabulary.
   - The original string label is recoverable from `data/raw/dswp_dominica_codas.csv`.

3. **Sharma 2024 compound** — rhythm × tempo × rubato × ornament (540 possible, 156 realized in DSWP).
   Only rhythm and extra_click (ornament proxy) are in codas_unified. Tempo, rubato, and the full
   compound label must be computed downstream; the classification code is in
   `data/upstream/sw_combinatoriality_{rhythms,ornaments}.p` (pickle files of Sharma's centroids).

---

## Datasets considered but not yet incorporated (from whale-ici-data README)

| Dataset | Availability | What it has |
|---|---|---|
| Project CETI WhAM corpus (Paradise et al. NeurIPS 2025) | Not yet public | 7,653 CETI codas, additional Dominica recordings |
| Beguš et al. 2026 vowel phonology (OSF 9t6qu) | Public | Spectral + vowel features; `codamd.csv` (duration, vowel labels) + `codasp.csv` (spectral peaks); NO per-click ICIs |
| HuggingFace `orrp/DSWP` | Public | Raw audio clips (subset of DSWP); no ICI annotations |
| Cantor et al. 2016 Galápagos (Dryad 10.5061/dryad.8jj26) | Public | Summary tables only, no raw ICIs |
| Watkins Marine Mammal Sound Database | Public | Audio clips only |

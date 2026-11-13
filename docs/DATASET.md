# Dataset Description — *Base_de_Dados* Acoustic Fault Diagnosis

> **File:** `docs/DATASET.md`
> **Folder:** `Base_de_Dados/` (Portuguese: *"Database"*)
> **Last updated:** 2026-11-13 — corrected against the IEEE DataPort page (P1/P4 are engine
> cylinders; the belt is the alternator V-belt; the microphone faces the exhaust outlet) and the
> project's own data audit (effective bit depth, outliers, temporal correlation).

---

## 0. Official Dataset Source (IEEE DataPort)

> **Source:** [IEEE DataPort — Acoustic Fault Dataset of Internal Combustion Engine Under Controlled Conditions for Ignition and Mechanical Anomalies](https://ieee-dataport.org/documents/acoustic-fault-dataset-internal-combustion-engine-under-controlled-conditions-ignition)
> **DOI:** [10.21227/1nyq-gb28](https://dx.doi.org/10.21227/1nyq-gb28) · **Published:** IEEE DataPort, 21 July 2025
> **Licence:** [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) — the thesis must cite the dataset.
> **Funding (as stated on the page):** CNPq & CAPES

### Authors (as listed on the page)

Abel Cavalcante Lima Filho · Thyago Leite de Vasconcelos Lima · Norah Nadia Sánchez Torres ·
Joylan Nunes Maciel · Oswaldo Hideo Ando Junior. Recordings were made at the Engine Laboratory,
Department of Mechanical Engineering, Federal University of Paraíba (UFPB), Brazil.

**Suggested citation:** A. C. Lima Filho, T. L. de V. Lima, N. N. Sánchez Torres, J. N. Maciel,
O. H. Ando Junior, "Acoustic Fault Dataset of Internal Combustion Engine Under Controlled Conditions
for Ignition and Mechanical Anomalies," IEEE Dataport, Jul. 21, 2025, doi: 10.21227/1nyq-gb28.

**Companion dataset (same authors):** "Dataset Preprocessing for Internal Combustion Engine Fault
Classification," IEEE Dataport, Jul. 21, 2025, doi: 10.21227/cvrw-k731 (MFCC features, padding,
label encoding, stratified splits). Not used in this project.

### What the source says about the setup

| Item | Source statement (translated) |
|---|---|
| Machine | Internal combustion engine — Otto cycle, four-stroke, spark ignition — of a **Ford Fiesta 1.6 (2005)** from the laboratory fleet. |
| Operating condition | Constant engine speed and steady-state operation. The RPM value is **not stated**. |
| Belt | The **alternator V-belt** ("correia trapezoidal do alternador"). It is the only belt mentioned. |
| Microphone placement | About **1 m** from the **exhaust outlet**, at roughly **45°**. Microphone model **not stated**. |
| Acquisition | USB interface to a laptop; acquisition software in C/C++ and Python. |
| Format | WAVE, **44.1 kHz, 16-bit**, 1-second samples. |
| Size | 12 classes × 179 samples = 2,148 samples. |
| Not stated | Number of cylinders, fuel, RPM, microphone model, how the 1-s clips were cut from the recordings, recording sessions/dates. |

> The engine of a 2005 Ford Fiesta 1.6 is an inline four-cylinder engine. This is general vehicle
> knowledge, not stated on the dataset page; cite it as such if used.

---

## 1. Fault Conditions — Source Definitions

The source defines **5 simple faults**, **6 combined faults**, and the normal condition.
**P1 and P4 are engine cylinders.** A misfire ("ausência de combustão") was created by
**disconnecting the spark-plug cable** of that cylinder.

| Folder | Source definition (translated) | Belt fault | Misfire |
|---|---|---|---|
| `Normal` | Normal condition (not separately defined on the page) | none | none |
| `Escorregamento` | Belt slip fault | slippage | none |
| `Perda_material` | Material loss on the belt: small pieces of varying size removed along the belt, in **all ribs** | diffuse material loss | none |
| `Perda_concentrada` | Concentrated material loss on the belt: one portion **25 mm long and 2 mm deep** removed | concentrated material loss | none |
| `Sem_P1` | Absence of combustion in cylinder P1 (spark-plug cable of P1 disconnected) | none | cylinder 1 |
| `Sem_P1P4` | Absence of combustion in cylinders P1 and P4 — **see discrepancy note** | none | cylinders 1 + 4 |
| `Escorregamento_P1` | Belt slip + absence of combustion in cylinder P1 | slippage | cylinder 1 |
| `Escorregamento_P1P4` | Belt slip + absence of combustion in cylinders P1 and P4 | slippage | cylinders 1 + 4 |
| `Perda_material_P1` | Belt material loss + absence of combustion in cylinder P1 | diffuse material loss | cylinder 1 |
| `Perda_material_P1P4` | Belt material loss + absence of combustion in cylinders P1 and P4 | diffuse material loss | cylinders 1 + 4 |
| `Perda_concentrada_P1` | Concentrated belt material loss + absence of combustion in cylinder P1 | concentrated material loss | cylinder 1 |
| `Perda_concentrada_P1P4` | Concentrated belt material loss + absence of combustion in cylinders P1 and P4 | concentrated material loss | cylinders 1 + 4 |

> **Discrepancy in the source:** both the DataPort page and the official data dictionary
> (`Base de Dados_diccionario.pdf`, checked 2026-11-13) define the simple fault "Sem P1P4" as absence
> of combustion in cylinder **P4** only, while the folder name and all three combined "…P1P4"
> definitions refer to cylinders **P1 and P4**. The public documentation cannot resolve this. This
> project assumes P1 + P4 (consistent with the naming); report it as an assumption, or ask the
> dataset authors.
>
> The dictionary also states that the belt faults (including slippage) were produced by the
> controlled removal of segments of the alternator V-belt, and that the 12 classes of 1-second
> samples were formed "after data treatment" ("com o tratamento de dados"). This suggests the clips
> were cut from longer recordings, which is consistent with the temporal correlation in Section 8.

### The dataset is a complete 4 × 3 factorial design

| Belt fault ↓ / Misfire → | none | cylinder 1 (P1) | cylinders 1 + 4 (P1P4) |
|---|---|---|---|
| **none** | `Normal` | `Sem_P1` | `Sem_P1P4` |
| **slippage** | `Escorregamento` | `Escorregamento_P1` | `Escorregamento_P1P4` |
| **diffuse material loss** | `Perda_material` | `Perda_material_P1` | `Perda_material_P1P4` |
| **concentrated material loss** | `Perda_concentrada` | `Perda_concentrada_P1` | `Perda_concentrada_P1P4` |

### Portuguese terms

| Portuguese | English |
|---|---|
| *Escorregamento* | Slippage (belt slip) |
| *Perda de material* | Material loss (on the belt) |
| *Perda concentrada (de material)* | Concentrated (localised) material loss |
| *Sem* | Without — here "without combustion" in the named cylinder(s) |
| *Ausência de combustão* | Absence of combustion (misfire) |
| *Cilindro P1 / P4* | Cylinder P1 / P4 |
| *Correia trapezoidal do alternador* | Alternator V-belt |
| *Nervuras* | Ribs (of the belt) |
| *Vela / cabo de vela* | Spark plug / spark-plug cable |

---

## 2. Overview

| Property | Value |
|----------|-------|
| Total recordings | **2,148 WAV files** (12 folders × 179) |
| Conditions | 12 = 4 belt-fault states × 3 misfire states |
| Project task | **3 macro classes** (Section 4) |
| Recording duration | 1.000 s per file (44,100 samples) |
| Sample rate | 44,100 Hz |
| Container bit depth | 16-bit PCM |
| **Effective bit depth** | **≈ 8 bits** (project audit; Section 5) |
| Channels | Mono |
| Language of folder names | Portuguese (Brazil) |

---

## 3. Raw Condition Distribution

Each folder contains exactly **179 WAV files**.

| # | Folder | Belt fault | Misfire | Samples |
|---|--------|-----------|---------|---------|
| 1 | `Normal` | none | none | 179 |
| 2 | `Escorregamento` | slippage | none | 179 |
| 3 | `Escorregamento_P1` | slippage | cyl. 1 | 179 |
| 4 | `Escorregamento_P1P4` | slippage | cyl. 1+4 | 179 |
| 5 | `Perda_material` | diffuse loss | none | 179 |
| 6 | `Perda_material_P1` | diffuse loss | cyl. 1 | 179 |
| 7 | `Perda_material_P1P4` | diffuse loss | cyl. 1+4 | 179 |
| 8 | `Perda_concentrada` | concentrated loss | none | 179 |
| 9 | `Perda_concentrada_P1` | concentrated loss | cyl. 1 | 179 |
| 10 | `Perda_concentrada_P1P4` | concentrated loss | cyl. 1+4 | 179 |
| 11 | `Sem_P1` | none | cyl. 1 | 179 |
| 12 | `Sem_P1P4` | none | cyl. 1+4 | 179 |
| | **Total** | | | **2,148** |

---

## 4. 3-Class Macro Taxonomy (project-defined)

The project maps the 12 folders to 3 classes. **This grouping is the project's choice, not part of
the source dataset.** The class names are historical labels kept for consistency with the code and
results; the table below states what each class actually contains.

```
Base_de_Dados/                    3-Class Label
+-- Normal/                  -->  Class 0 -- Healthy Baseline
+-- Escorregamento/          -->  Class 1 -- Continuous Friction & Wear
+-- Escorregamento_P1/       -->  Class 1 -- Continuous Friction & Wear
+-- Escorregamento_P1P4/     -->  Class 1 -- Continuous Friction & Wear
+-- Perda_material/          -->  Class 1 -- Continuous Friction & Wear
+-- Perda_material_P1/       -->  Class 1 -- Continuous Friction & Wear
+-- Perda_material_P1P4/     -->  Class 1 -- Continuous Friction & Wear
+-- Perda_concentrada/       -->  Class 2 -- Impulsive Shocks & Structural
+-- Perda_concentrada_P1/    -->  Class 2 -- Impulsive Shocks & Structural
+-- Perda_concentrada_P1P4/  -->  Class 2 -- Impulsive Shocks & Structural
+-- Sem_P1/                  -->  Class 2 -- Impulsive Shocks & Structural
+-- Sem_P1P4/                -->  Class 2 -- Impulsive Shocks & Structural
```

### What each class contains

| Class | Folders | Actual content | Intended acoustic rationale |
|-------|---------|----------------|-----------------------------|
| **0 — Healthy Baseline** | `Normal` | No belt fault, no misfire | Reference engine sound. |
| **1 — Continuous Friction & Wear** | `Escorregamento*`, `Perda_material*` (6) | Belt slippage or diffuse belt material loss, **each with no misfire, misfire in cyl. 1, or misfire in cyl. 1+4** | Slip and distributed wear are expected to add continuous, broadband belt noise. |
| **2 — Impulsive Shocks & Structural** | `Perda_concentrada*`, `Sem_P1`, `Sem_P1P4` (5) | Concentrated belt material loss (with no / cyl. 1 / cyl. 1+4 misfire) **and misfire-only conditions (no belt fault)** | A single damaged section of the belt passes once per belt revolution, and a dead cylinder interrupts the regular firing pulses; both are expected to give periodic, impulse-like patterns. |

### Limitations of this grouping (state these in the thesis)

1. **Two fault families are mixed.** Classes follow the belt fault, but misfire is present in 8 of the
   11 fault folders and appears in both class 1 and class 2. Misfire-only recordings (`Sem_*`) are
   grouped with concentrated belt loss. The "impulsive" link between them is a hypothesis, not a
   measured property.
2. **Class names overstate the physics.** "Structural" does not describe a misfire, and no belt teeth
   exist (it is a V-belt). The names are kept only as labels.
3. **Healthy means no belt fault *and* no misfire.** A model trained on this grouping is not a misfire
   detector and not a belt-fault detector on its own.
4. **The natural structure is 4 × 3.** A two-output task (belt fault: 4 states; misfire: 3 states) would
   match the source design. It is not part of this thesis's locked result.

### 3-Class Sample Counts

| Class | Name | Folders merged | Samples | Share |
|-------|------|----------------|---------|-------|
| 0 | Healthy Baseline | 1 | 179 | **8.3%** |
| 1 | Continuous Friction & Wear | 6 | 1,074 | **50.0%** |
| 2 | Impulsive Shocks & Structural | 5 | 895 | **41.7%** |
| | **Total** | 12 | **2,148** | 100% |

> **Class imbalance:** Class 0 (Healthy) is 8.3% of the data. Stratified splitting and macro F1 as the
> primary metric are used for this reason.

---

## 5. Audio Signal Properties

| Property | Value |
|----------|-------|
| Format | WAV (PCM) |
| Sample rate | 44,100 Hz |
| Container bit depth | 16-bit |
| **Effective resolution** | **≈ 8 bits.** In the median file, 98.5% of samples lie within ±3 of a multiple of 257, the signature of 8-bit audio converted to 16-bit. Each file uses 70–255 distinct 8-bit levels. |
| Spectrum above ≈ 8 kHz | Flat noise floor about 18 dB below the 1–2 kHz band (quantisation noise) |
| DC offset | About +0.10 to +0.135 of full scale in most folders; `Escorregamento_P1` (−0.039) and `Perda_concentrada_P1` (+0.061) differ |
| Channels | Mono |
| Duration | Exactly 1.000 s (44,100 samples) |
| Amplitude range | Normalised to float32 [-1.0, 1.0] at load time |
| File size | 88,244 bytes per file |

> The source states 16-bit. The 8-bit content was found by the project's audit; it does not
> contradict the container format, but it limits the useful dynamic range and bandwidth.

### Spectrogram Representation (used by TF-FaultNet)

| Property | Value |
|----------|-------|
| Representation | Log-Mel Spectrogram |
| FFT window size (n_fft) | 1,024 |
| Hop length | 256 |
| Mel filter banks (n_mels) | 64 (20 Hz – 22.05 kHz) |
| Time frames | 173 |
| Output tensor shape | (1, 64, 173) — grayscale |
| Normalization | Zero-mean, unit-variance per spectrogram |

---

## 6. File Naming Convention

| Style | Folders | Example |
|-------|---------|---------|
| Underscore | `Normal`, `Escorregamento*`, `Sem_P1`, `Sem_P1P4` | `Normal_1.wav`, `Sem_P1P4_78.wav` |
| Parenthesis | `Perda_concentrada*`, `Perda_material*` | `Perda_concentrada (1).wav`, `Perda_material_P1 (12).wav` |

Indices run from 1 to 179 within each folder. **Index appears to follow recording order** (see
Section 8), so it should not be treated as meaningless.

---

## 7. Train / Validation / Test Split

The split is **frozen** in `data/splits.csv`, column `split_multiclass_3class`. It was generated with
seed 42 using stratified random sampling on the 3-class label and must not be regenerated.

| Partition | Samples | Share |
|-----------|---------|-------|
| Train | 1,504 | 70% |
| Validation | 322 | 15% |
| Test | 322 | 15% |
| **Total** | **2,148** | 100% |

- The split is **file-level** (each WAV assigned independently).
- The test set is held out from preprocessing fit, threshold selection and hyperparameter tuning.
- It replaced an earlier 64/16/20 split on 2026-09-22, and the test split has been scored three times
  over the project (see `docs/THESIS_READINESS_TODO.md`). Only the final locked run is reported.

---

## 8. Measurement Setup & Known Limitations

| Factor | Description |
|--------|-------------|
| **Machine** | One vehicle engine (Ford Fiesta 1.6, 2005) in one laboratory |
| **Microphone** | One microphone, about 1 m from the exhaust outlet at about 45°; model not stated |
| **What is heard** | Mainly exhaust and engine sound. Belt faults are observed indirectly through this channel. |
| **Sessions** | **Not documented in the source.** How the 1-s clips were cut, and whether each condition was one continuous recording, is unknown. |
| **Temporal ordering** | **Not IID.** Index-adjacent clips are more similar than random pairs from the same folder (spectral distance ratio 0.36–0.95; 11 of 12 folders below 0.88), and loud events span runs of consecutive indices. 299 of the 322 test clips have an index neighbour in the training set. A random file-level split is therefore likely optimistic. |
| **No exact overlap** | No duplicate files and no shared audio segments between clips (16-sample exact-match search). |
| **Recording-chain shortcut** | DC offset and RMS alone give 0.642 validation macro-F1 for the 3 classes and identify the source folder 81.7% of the time. |
| **Loudness outliers** | 54 files with RMS more than 5 robust SD from their folder median; 35 in `Escorregamento_P1` (index runs 164–179, 1–6 and 67–73). They look like real events, not corrupt files. **Kept** (AGENTS.md rule 7). |
| **Missing / corrupt data** | None — all 2,148 files are complete and readable. |
| **Cross-machine / cross-site generalisation** | Not tested. |

> The main generalisation caveat: all data come from **one engine, one microphone position, one lab**,
> with unknown session structure. Performance on other engines, microphone placements or noise
> conditions is untested.

---

## 9. 24-Feature Statistical Descriptor Reference

Extracted by `src/audio_features.py` for the Random Forest / XGBoost baselines:

| # | Feature | Domain |
|---|---------|--------|
| 1 | RMS | Time |
| 2 | Peak amplitude | Time |
| 3 | Peak-to-peak | Time |
| 4 | DC offset | Time |
| 5 | Standard deviation | Time |
| 6 | Kurtosis | Time |
| 7 | Skewness | Time |
| 8 | Crest factor | Time |
| 9 | Shape factor | Time |
| 10 | Impulse factor | Time |
| 11 | Margin factor | Time |
| 12 | Energy | Time |
| 13 | Zero-crossing rate | Time |
| 14 | Spectral centroid | Frequency |
| 15 | Spectral spread | Frequency |
| 16 | Spectral skewness | Frequency |
| 17 | Spectral kurtosis | Frequency |
| 18 | Spectral rolloff (85%) | Frequency |
| 19 | Spectral flatness | Frequency |
| 20 | Band energy 0–1 kHz | Frequency |
| 21 | Band energy 1–2.5 kHz | Frequency |
| 22 | Band energy 2.5–5 kHz | Frequency |
| 23 | Band energy 5–10 kHz | Frequency |
| 24 | Band energy 10–15 kHz | Frequency |

> ZCR uses the corrected implementation in `src/signal_statistics.py`. Features 1–4 and 12 carry the
> recording-chain shortcut described in Section 8; features 23–24 lie mostly in the 8-bit noise floor.

---

## 10. Related Files

| File | Purpose |
|------|---------|
| `Base_de_Dados/` | Raw WAV files (12 subfolders) |
| `data/splits.csv` | Frozen split manifest (column: `split_multiclass_3class`) |
| `data/baseline_statistical_features_3class.npz` | Pre-computed 24-feature vectors for baseline models |
| `metadata/class_mapping_3class.json` | Machine-readable folder-to-class-index mapping |
| `results/heldout_test_set_3class.npz` | Held-out test set (waveforms + labels) |
| `docs/EDA_PREPROCESSING.md` | EDA record, preprocessing decisions, ZCR correction |
| `src/audio_features.py` | Feature extraction code |
| `src/dataset.py` | PyTorch Dataset class |
| `src/split_manifest.py` | Split manifest loader |
| `eda_analysis.py` | Generates EDA visualizations and feature summary CSV |

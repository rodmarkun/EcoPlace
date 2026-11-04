# prereg/ — frozen study registrations

Each study was registered and frozen locally before its comparative outcomes were generated.
The registrations were not deposited with a public registry.

| Registration | Study in the paper |
|---|---|
| `CORRECTED_STUDIES_20260920_v3` | Studies A and B |
| `STUDY_C_ROBUST_20261003` | Study C (four search seeds) |
| `STUDY_D_SINGLESEED_20261004` | Study D (single search seed) |
| `STUDY_E_MULTIFAMILY_20261005` | Study E (three scenarios) |
| `STUDY_F_SOL_20261007` | Study F (main text) |

`CORRECTED_STUDIES_20260917_v1` and `CORRECTED_STUDIES_20260919_v2` are the earlier versions that
v3 supersedes; they are kept because v3 and the launchers read them.

A registration holds the protocol (`PROTOCOL.md`), seeds, task and campaign configurations,
initialization archives, the analysis plan and thresholds, amendments, and `FREEZE.json`, which
records sha256 hashes of the registration files and of the source files the study ran. Each
study's launcher (`experiments/<study>/launch.py`) refuses to start if the registration or the
sources differ from the freeze. Launch receipts (`P2_FREEZE.json` / `FREEZE.json`) and
amendments sit in the study folders under `experiments/`.

Notes on the public copy:

- The receipts record absolute paths from the machine the studies ran on. They are kept as
  recorded.
- Approval fields in the receipts originally quoted the investigator's instructions verbatim.
  Those quotes were replaced by neutral summaries for publication, and every hash that covers
  an edited file was recomputed. The hashes in this repository therefore differ from the ones
  recorded at run time, and are consistent among themselves and with the Zenodo archives.

# Changelog

## 2026-10-28 — Clean CI dependency correction

- The first GitHub run passed dependency installation and publication checks, then reported 35 passed / 1 failed because `huggingface_hub` was absent from the CPU dependency list. The earlier local test used an environment where it was already installed.
- Added the missing pinned dependency, selected Node-24-compatible Actions versions and fixed the runner OS to Ubuntu 24.04. Simulator and training environments are unchanged.
- Validation: all 36 focused tests passed in a newly created Python environment using only `requirements-cpu.txt`; `pip check` passed. Remote CI status is reported separately.

## 2026-10-28 — Initial research repository preparation

- Documented the RoboPhys-to-VLA workflow, independent research scope and upstream attribution.
- Included existing project integration modules, diagnostic scripts and tests; kept datasets, simulator assets, checkpoints and machine configurations outside Git.
- Recorded the 40-success wind-driven-ball development cohort and the earlier DynamicVLA egg-task diagnostic with explicit limitations.
- Verified two revised-camera samples and the remote-render/Hub/local-delivery roundtrip; full-cohort rendering remains in progress at this snapshot.
- Added publication checks and a CPU-only regression workflow; 36 focused CPU tests passed locally and the staged file audit passed. Public repository creation and first remote publication are tracked separately from this local preparation.

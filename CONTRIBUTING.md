# Development and research updates

Keep the research repository useful to someone who has not followed the development chat.

For each meaningful, verified change:

1. Commit the relevant implementation and regression checks together.
2. Update `CHANGELOG.md`. If behavior or evidence changes, update `docs/project/RESULTS.md` and add a compact, source-hashed summary under `results/summaries/`.
3. Update `docs/project/ROADMAP.md` when the research question or next experiment changes. Label proposals, observations and verified outcomes separately.
4. Before publishing, run the complete test list from `.github/workflows/cpu-checks.yml` in a clean tracked-file export with the pinned CPU dependencies, `PYTHONPATH=src` and `HF_HUB_OFFLINE=1`. Extra local search paths such as `scripts` can hide import failures. Run `python scripts/check_publication.py` against the staged Git index. Inspect `git diff --cached` before committing.
5. Publish to the existing repository using an ordinary commit/push or reviewed pull request. Never force-push over remote changes. If authentication or access prevents publishing, report the local commit and the exact pending step.

This is a milestone-based update workflow, not an unattended monitor that publishes every generated file. Running jobs and incomplete outputs remain local until verified. The project owner has requested continued repository updates during development; account permissions and intended disclosure still apply.

The public project is named **Physical VLA Lab**. Existing internal `capstone_vla` module names and historical local paths are retained to preserve working experiments. Do not rename runtime interfaces merely to alter presentation.

Raw data, model weights, simulator binaries, upstream source bundles, private operational configs, lab notes and student proposal drafts are excluded. The root `.gitignore` is an explicit allowlist. Do not bypass it with `git add -f` without inspecting the exact content.

When discussing results, always distinguish expert from learned policy, offline from closed-loop metrics, development from held-out evaluation, and Franka simulation from UR10e hardware.

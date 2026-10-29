# Native startup and Linux file-watch resources

The matched-startup fresh60 run `20261030_000853_155886_fresh60` aborts on the first guarded Kit update, before any episode. Its Omniverse Client `OC Tick` stack has the same seven relative library offsets as the preceding unguarded native abort. The stage-load wait is therefore **not a reliable crash workaround**. No new grasp or worker-history comparison was evaluated.

## Independently verified resource shortage

A separate CPU-only Kit process loads the actual task USD with no SimulationApp, PhysX, renderer or scripting extension. It opens the stage but logs32 failures to create file-change watches: `errno28/ENOSPC`. Host inspection finds a65536 watch limit, approximately65512 summed watch references, and64927 in a VS Code `--type=fileWatcher` process. Shared descriptors can be counted twice and snapshots can change. An independent bounded allocation probe confirms only26 of512 requested watches can be created before ENOSPC. All temporary watches/files are released afterward.

The VS Code remote-agent log also reports inotify ENOSPC on this project at2026-10-29T22:21:59.184. Disk availability is357G. Linux documents [ENOSPC from inotify_add_watch](https://man7.org/linux/man-pages/man2/inotify_add_watch.2.html) as a watch-limit/resource failure. This evidence establishes watch exhaustion, **not** its causation of the separate native mutex abort; the small USD loader itself does not crash.

## Local mitigation and activation

A new ignored `.vscode/settings.json` excludes recursive watching of data, Backgrounds, virtual environments, third_party and bytecode caches. It changes editor notifications, not script execution or file contents. Source, scripts and tests remain watched. A reusable [example](../../configs/examples/vscode_watcher_exclusions.json) contains the same patterns. Microsoft documents [files.watcherExclude](https://code.visualstudio.com/docs/setup/linux) for large-workspace ENOSPC errors.

Writing the configuration alone did not release the running watcher's resources: a second CPU loader still reported 30 watch errors. After the user reloaded VS Code, the old large watcher exited. Summed watch references fell to 581; the independent allocation probe passed 512/512. The same CPU-only USD loader exited normally with the stage open and zero watch-creation errors. This verifies resource recovery; a full native simulator retest is still pending. No editor process was killed and no kernel sysctl or SDK version changed. Remove the newly added local settings file to undo the exclusions.

The history coordinator now performs a bounded512-watch admission probe before SDK preflight/process launch, retains its result in the plan and gives an actionable error if unavailable.512 is an engineering admission threshold, not a proven minimum for Isaac or a stability guarantee. This avoids launching the simulator into the observed exhausted resource condition.267 portable CPU checks pass, including cleanup on success/failure and prelaunch rejection. Native collector assembly, physics and control parameters are unchanged by this preflight-only addition.

After reloading, a CPU-only check is:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_history_diagnostic.py \
  --fresh-control --stage-load-barrier --native-backtrace
```

No `--run` means no simulation is launched. This check now passes `INOTIFY_PREFLIGHT`, `CPU_PREFLIGHT_OK` and the exact collector/history gate. The next step is the same matched-control command with `--run`; keep rendering disabled. A successful CPU load does not establish native simulator stability. The previous history reproduction remains valid. New startup warnings and comparison results must be retained. [Hashed observations](../../results/summaries/2026-10-30-inotify-resource-shortage.json).

## Full retest observed

User run `20261030_003531_182273_fresh60` passes resource and stage-load gates and reaches 169 recorded control frames without native client abort. It then stops on motion safety, reproducing the old fresh failure exactly; see [matched-control result](WORKER_HISTORY_DIAGNOSTIC.md#matched-fresh60-result-history-contrast-persists). Resource recovery is verified, but one launch does not establish a general native-crash fix or unique causal mechanism.

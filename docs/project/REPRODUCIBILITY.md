# Reproduction boundaries

## Included and runnable without private data

The CPU test subset in the README exercises synthetic HDF5 fixtures, archive checks, pose metrics and action chunking. `requirements-cpu.txt` records the package versions used for this repository's local check. It is separate from simulator and model environments.

Run `python scripts/check_publication.py` after staging files to inspect exactly what Git would publish.

## Required separately for simulation

- Existing Isaac Sim 4.5 and Isaac Lab 2.2.1 installations.
- Authorized RoboPhys code and scene/robot/object assets.
- Source recordings, provenance manifests, frozen worker bundle and task context matching the experiment.
- Host-specific operational configurations and optional project-scoped Hub authentication.

`configs/examples/` contains sanitized copies for inspection. Replace `YOUR_*` placeholders and verify the paths before copying to their expected local `configs/` filenames. These templates do not automatically provision the assets or adapt every script to a new machine.

Many research scripts still encode historical experiment paths. Simulation, deployment and setup scripts are not universal installers and must be reviewed against the provisioned environment before execution. Broad `pytest tests/` also includes environment-dependent checks; use the documented subset on a clean checkout.

## Model dependency

DynamicVLA is separately obtained from `hzxie/DynamicVLA`, pinned here to commit `434dc7105cf6d559018d15af5fecbcdad44f036e`. Its code and weights are not vendored. Consult its upstream license and model terms. The existing `setup_dynamicvla_data_env.sh` is a project integration script, not an official DynamicVLA installer; it assumes the earlier local environment.

## Experiment records

For every result retain: source scene and software hashes, seed/episode identity, controller and camera configuration, action frame/units/timing, split identity, model checkpoint, normalization provenance, sample counts, failure counts and evaluator version. Publish small summaries separately from raw recordings and checkpoints.

No third-party assets, private Hub archives, usernames/passwords for remote machines, SSH keys or simulation environments should be added to Git. Private data links are not a reproducibility guarantee. A future public data release needs its own availability and licensing review.

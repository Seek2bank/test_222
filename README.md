# MoKOpt

MoKOpt is a harness-independent controller that asks coding agents to explore
multiple performance optimizations, measures every candidate, rejects incorrect
or unsafe patches, and keeps the fastest verified result.

The search loop is:

```text
baseline -> profile -> branch(K) -> FAST eval -> prune
         -> FULL eval of top candidates -> promote -> repeat
```

Every branch receives one of nine optimization skill cards. Skill selection can
use round-robin rotation, an LLM router, a persistent UCB1 bandit, or all cards
at once. Candidate trees are isolated as Git commits and the final deliverable
is a normal patch.

## Installation

Python 3.11+, Git, and Docker are required for benchmark task images.

```bash
uv tool install ./mokopt
```

For Hugging Face or parquet SWEfficiency datasets:

```bash
uv tool install --with datasets==4.4.1 ./mokopt
```

During development in this repository, replace `mokopt` below with:

```bash
uv run --project mokopt mokopt
```

## Agent backends

`--backend copilot` copies the host's `copilot` binary into the task container.
Authenticate on the host with `copilot /login`; MoKOpt copies
`~/.copilot/config.json` into the container. Alternatively, provide a valid
`COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, or `GITHUB_TOKEN`.

The `codex` and `claude` backends install their CLI with npm when necessary and
use `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, respectively. A fully custom
non-interactive agent can be supplied with:

```bash
--backend command \
--agent-command 'my-agent --prompt-file {prompt_file} --model {model}'
```

The command runs inside the benchmark environment. Available fields are
`{prompt_file}`, `{model}`, `{model_flag}`, and `{workdir}`. Each branch or phase
gets a fresh process.

Use `--backend mock --mock-solution '<shell edit>'` for deterministic smoke
tests without an API key.

## Generic repository

When `--image` is omitted, this command copies the checkout to a temporary
workspace before running. The original checkout is not modified. Docker is
still recommended for untrusted agents.

```bash
mokopt run \
  --repo ./my-project \
  --task-id parser-hot-path \
  --instruction 'Optimize parsing throughput without changing behavior.' \
  --benchmark-cmd 'python benchmarks/parser.py' \
  --metric-regex 'Mean: ([0-9.]+)' \
  --test-cmd 'python -m pytest -q' \
  --backend copilot \
  --model gpt-5.5
```

Without `--metric-regex`, MoKOpt measures benchmark wall time. With a regex, the
last captured number is treated as a lower-is-better metric.

## GSO

First generate GSO task directories with FC-Eval's adapter. MoKOpt then builds
the selected task image, uses `/testbed`, reinstalls through
`/perf/install_cmds.sh`, checks correctness through `prob_runner.py --eqcheck`,
and parses its `Execution time:` metric.

```bash
uv run adapters/gso/run_adapter.py \
  --out dataset/gso \
  --data /path/to/gso.parquet

mokopt gso \
  --task-dir dataset/gso/<instance-id> \
  --backend copilot \
  --model gpt-5.5 \
  --branch-k 8 \
  --rounds 4 \
  --isolation phased \
  --output mokopt-runs/gso
```

To reuse an image that was already built:

```bash
mokopt gso \
  --task-dir dataset/gso/<instance-id> \
  --image my-gso-image:latest \
  --backend copilot
```

## SWEfficiency

The adapter accepts a local JSON/JSONL file, a directory of parquet shards, or a
Hugging Face dataset name. Each row must provide:

| Field | Required | Meaning |
|---|---:|---|
| `instance_id` | yes | Unique ID and default image tag |
| `workload` | yes | Python program printing `Mean: <seconds>` |
| `test_cmd` | no | Command prefix used for each covering test |
| `covering_tests` | no | Test paths that may not gain new failures |
| `rebuild_cmd` | no | Command run before evaluating a candidate |
| `problem_statement` or `instruction` | no | Agent task description |

```bash
mokopt swefficiency \
  --dataset /path/to/instances.jsonl \
  --instance-id <instance-id> \
  --backend copilot \
  --model gpt-5.5 \
  --branch-k 8 \
  --rounds 4 \
  --isolation phased \
  --output mokopt-runs/swefficiency
```

By default the image is
`ghcr.io/swefficiency/swefficiency-images:<instance-id>` and the repository is
`/testbed`. Override it with `--image`.

MoKOpt records which covering tests already fail at baseline. A candidate is
rejected only when it introduces a new failure.

## FC-Eval / FormulaCode

FormulaCode tasks use different ASV suites and may score multiple benchmarks.
MoKOpt handles task image construction, `/workspace/repo`, `task.yaml`, and
`run-setup.sh`; the caller supplies a scalar benchmark command appropriate for
that task. The command can print a metric selected by `--metric-regex`, or be
measured by wall time.

```bash
mokopt fc-eval \
  --task-dir dataset/formulacode-verified/<task-id> \
  --benchmark-cmd '<task-specific benchmark command>' \
  --metric-regex 'Mean: ([0-9.]+)' \
  --test-cmd '<task-specific correctness command>' \
  --backend copilot \
  --model gpt-5.5 \
  --branch-k 8 \
  --rounds 4 \
  --isolation phased \
  --output mokopt-runs/formulacode
```

For a benchmark command that already has stable wall time, omit
`--metric-regex`. Do not use the task's final `run-tests.sh` as the in-loop test
command: FormulaCode's script resets the repository and performs final scoring.

## Configuration

`--config` accepts a JSON file matching `BetaConfig`. Important overrides are
also exposed directly:

```text
--branch-k N
--rounds N
--isolation single|phased
--sig-threshold FLOAT
--promotion-threshold FLOAT
--stop-no-improvement N
--dynamic-routing
--use-bandit
--inject-all-skills
--disable-skills
--ralph-history
--env-check-cmd 'pip freeze'
--wall-time SECONDS
```

## Outputs

Each run writes `OUTPUT/<task-id>/`:

| File | Content |
|---|---|
| `result.json` | Stable task result schema |
| `final.patch` | Winning cumulative Git patch |
| `trajectory.jsonl` | Controller action/outcome events |
| `branches.jsonl` | One summary row per candidate |
| `patches/*.diff` | Individual branch patches |
| `transcript.log` | Agent responses |

`result.json` includes task and commit IDs, baseline/final metrics, speedup,
correctness, best branch, patch path, trajectory path, token counts, and cost.

## Safety model

- Docker task runs copy or use the repository inside the container.
- Test, benchmark, dependency, and Git paths are read-only when supported.
- The policy layer independently rejects forbidden files and oversized patches.
- `--env-check-cmd` can reject package-environment mutations.
- Tests run before both FAST and FULL promotion.
- A run with no verified improvement emits an empty patch and speedup `1.0`.

## Distill split

```text
Distill split: 60% of Formulacode-F (461 -> 276 tasks)

Repository                           Total  Selected  Rate(%)
xdslproject/xdsl                       136        91     66.9
networkx/networkx                       35        25     71.4
optuna/optuna                           28        14     50.0
pytroll/satpy                           18         9     50.0
dwavesystems/dimod                      15         8     53.3
Qiskit/qiskit                           13         6     46.2
pydata/bottleneck                       12         6     50.0
sgkit-dev/sgkit                         12         6     50.0
UXARRAY/uxarray                         11         6     54.5
sourmash-bio/sourmash                   11         7     63.6
JDASoftwareGroup/kartothek              10         5     50.0
datalad/datalad                         10         6     60.0
Textualize/rich                          9         5     55.6
pysal/momepy                             9         5     55.6
microsoft/Qcodes                         7         1     14.3
pygeos/pygeos                            7         4     57.1
geopandas/geopandas                      6         4     66.7
pyapp-kit/psygnal                        6         3     50.0
shapely/shapely                          6         5     83.3
DASDAE/dascore                           5         2     40.0
deepchecks/deepchecks                    5         4     80.0
pvlib/pvlib-python                       5         1     20.0
xarray-contrib/flox                      5         2     40.0
ActivitySim/activitysim                  4         3     75.0
SciTools/cartopy                         4         3     75.0
holgern/beem                             4         4    100.0
lmfit/lmfit-py                           4         3     75.0
mars-project/mars                        4         2     50.0
mie-lab/trackintel                       4         2     50.0
napari/napari                            4         3     75.0
sunpy/sunpy                              4         2     50.0
TileDB-Inc/TileDB-Py                     3         3    100.0
dipy/dipy                                3         2     66.7
kedro-org/kedro                          3         3    100.0
pymc-devs/pymc                           3         1     33.3
python-adaptive/adaptive                 3         2     66.7
NCAR/geocat-comp                         2         2    100.0
dedupeio/dedupe                          2         1     50.0
django-components/django-components      2         2    100.0
dottxt-ai/outlines-core                  2         1     50.0
glotzerlab/signac                        2         2    100.0
h5py/h5py                                2         1     50.0
holoviz/param                            2         0      0.0
innobi/pantab                            2         0      0.0
numpy/numpy-financial                    2         1     50.0
scikit-learn/scikit-learn                2         0      0.0
HIPS/autograd                            1         0      0.0
Quansight-Labs/ndindex                   1         1    100.0
arviz-devs/arviz                         1         0      0.0
danielgtaylor/python-betterproto         1         1    100.0
devitocodes/devito                       1         1    100.0
makepath/xarray-spatial                  1         1    100.0
not522/ac-library-python                 1         0      0.0
pybop-team/PyBOP                         1         1    100.0
stac-utils/pystac                        1         0      0.0
tqdm/tqdm                                1         1    100.0
wmayner/pyphi                            1         1    100.0
xarray-contrib/xbatcher                  1         0      0.0
xitorch/xitorch                          1         1    100.0

TOTAL                                  461       276     59.9
```

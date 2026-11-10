# Coordinated Attack Detection Using GNN

Research code for deciding whether a group of flagged VM backup anomalies is a coordinated attack campaign or a set of coincidental events, using a dual-stream
(temporal and spatial) graph neural network over an enterprise infrastructure graph.

## Setup

Python 3.14 or newer

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt       # to run the pipeline
pip install -r requirements-dev.txt   # also tests and linting
```

The `cadgnn` command used below needs the package itself installed as well
(`pip install -e .`). `run_generator.py` does not.

## Datasets

The calibration datasets are not tracked in Git. Place them under `datasets/`:

| Path | Source |
| --- | --- |
| `datasets/ransap/<Trace>.csv` | `ata_write.csv` of each RanSAP win7-120gb-hdd run (Excel, Firefox, Cerber, Darkside, GandCrab4, Ryuk, Sodinokibi, TeslaCrypt, WannaCry), https://github.com/manabu-hirano/RanSAP |
| `datasets/napierone/NapierOne-details.md` | https://github.com/simonrdavies/NapierOne |
| `datasets/filesystems/fslhomes-*/` | FSL Traces and Snapshots Archive, https://tracer.filesystems.org/traces/fslhomes/ |

Check that everything is in place:

```bash
cadgnn --config configs/default.yaml check
```

`python -m cadgnn ...` works the same way. The exit status is 0 when the datasets are complete,
1 when files are missing or malformed, and 2 for an invalid configuration.

## Configuration

`configs/default.yaml` holds the random seed, log level, data/output paths and the generator
scenario (infrastructure size, episode counts, timing ranges). Relative paths are resolved
against the config file's directory. `--seed`, `--datasets-dir`, `--output-dir` and
`--log-level` override the file for a single run.

## Generating synthetic telemetry

```bash
python run_generator.py
python run_generator.py --seed 42 --output-dir runs/seed42
```

`run_generator.py` uses `configs/default.yaml` unless `--config` is given and accepts the same
overrides as the CLI. With the package installed, `cadgnn --config configs/default.yaml generate`
is equivalent.

This calibrates the feature ranges from the datasets, builds the infrastructure graph
(12,000 VMs, 300 storage pools, 200 network fabrics by default), generates the labelled backup
events and writes to the output directory:

| File | Contents |
| --- | --- |
| `synthetic_backup_telemetry.csv` | one row per backup event with the four features, weighted `anomaly_score`, `state`, `episode_id` and `label` (`none`, `coincidental`, `coordinated`) |
| `infrastructure_graph_nodes.csv` | `node_id`, `kind` (`vm`, `backup_image`, `storage`, `network`) |
| `infrastructure_graph_edges.csv` | `source`, `target` |
| `experiment_manifest.json` | seed, scenario, calibrated feature ranges, label and pattern counts |
| `episode_patterns.csv` | `episode_id`, `pattern` (how the generator laid the episode out, see below) |

The generator is a port of `notebooks/random_anomaly_generator_simple.ipynb` and makes its
random draws in the same order. With the overlap settings below switched off and `--seed 42`, it
reproduces the notebook's graph and events; `tests/test_notebook_equivalence.py` checks this by
running the notebook code itself.

### Overlap between the classes

In the notebook's scenario a coordinated episode is always one network and a couple of hours,
and a coincidental one is several networks spread over days, so any model scores perfectly.
`configs/default.yaml` therefore generates 600 episodes (400 coincidental, 200 coordinated)
and adds these harder cases:

| Setting | Value | What it adds |
| --- | --- | --- |
| `coincidental_burst_fraction` | 0.20 | unrelated networks in one short window, like a scheduled job |
| `coincidental_cluster_fraction` | 0.20 | one network but spread over time, like a faulty subnet |
| `coincidental_lookalike_fraction` | 0.15 | one network and one short window, like a patch rollout |
| `stealth_campaign_fraction` | 0.25 | campaigns with hops 2 to 12 hours apart |
| `storage_pivot_probability` | 0.20 | a hop goes through a shared storage pool, so the chain changes network |
| `missed_event_probability` | 0.10 | the anomaly detector misses an event |
| `stray_vm_probability` | 0.20 | one unrelated VM is grouped into the episode |

These values are assumptions chosen to represent plausible behaviour, not measurements from a
real enterprise. Each episode's pattern is written to `episode_patterns.csv`: `scattered`
(the notebook's coincidental episodes), `burst`, `cluster` or `lookalike` for coincidental ones,
and `fast` or `stealth` for campaigns. The `no_overlap` scenario in `configs/experiments.yaml` keeps the original easy
setup for comparison.

## Building the episode graphs

```bash
python run_graph_builder.py
```

Reads the generator output from the output directory and builds one heterogeneous graph
(PyTorch Geometric `HeteroData`) per anomaly episode, labelled coordinated (1) or
coincidental (0). Benign backups are background and do not form graphs.

| Node type | Comes from |
| --- | --- |
| `vm` | each flagged VM in the episode; features are the event's `change_rate`, `file_entropy`, `backup_size_gb`, `file_count`, plus its timestamp and order in the episode |
| `backup_image`, `storage`, `network` | the image, storage pool and network fabric of those VMs |

| Edge type (each also has a reverse) |
| --- |
| `vm` -`has_backup`-> `backup_image` |
| `backup_image` -`stored_on`-> `storage` |
| `vm` -`attached_to`-> `network` |

It writes `episode_graphs.pt` (load with `cadgnn.graph.load_episode_graphs`) and
`episode_summary.csv` with per-episode size, time span and number of connected components.
Features are not scaled here; scaling is fitted on the training split later.

## Dual-stream model

`cadgnn.models` holds the model foundation:

- `collate()` batches episode graphs and `check_batch()` validates a batch before the forward pass.
- `FeatureScaler` z-scores the four VM features with statistics from the training graphs only.
- `NodeEncoder` projects VM telemetry and gives the featureless node types a learned vector each.
- `DualStreamModel` runs a temporal and a spatial stream on the same batch and returns one
  embedding per graph from each (`StreamOutputs.temporal`, `StreamOutputs.spatial`).
- `save_checkpoint` / `load_checkpoint` store the weights, scaler statistics and model config.

Sizes, dropout and device are set in the `model` section of the config.

### Temporal severity stream

`cadgnn.temporal.TemporalSeverityStream` reads each episode's flagged events in time order.
Every step is the event's scaled telemetry plus `log1p` of the hours since the episode's first
event and of the hours since the previous event. A forward-only GRU runs over the steps, so the
state after event t depends only on events up to t; the episode vector combines the last state
with the mean state. There are no severity labels in the data, so the stream has no severity
target of its own and learns through the episode label. Missing, negative or decreasing event
times are rejected.

### Spatial topology stream

`cadgnn.spatial.SpatialTopologyStream` runs graph attention (`GATConv` per edge type inside
`HeteroConv`) over each episode's VMs, backup images, storage pools and network fabrics, with
residual connections and layer norm. Resource nodes have no features, so each node also gets
`log1p` of its degree within the episode graph, which tells a network shared by several flagged
VMs apart from one used by a single VM. With the default two layers a VM receives information
from the other VMs on its network. The episode vector is mean and max pooling per node type.
Only VM telemetry and graph structure are used: event times, labels, episode ids and global node
indices are not inputs, and the graph has no edge attributes. Layers and attention heads are set
by `spatial_layers` and `spatial_heads` in the config.

### Fusion and coordination output

`cadgnn.fusion.build_coordination_model(config)` builds the full model: both streams, a fusion
layer and a classifier that gives one logit per episode for coordinated (1) against
coincidental (0).

- `fusion: gated` (default) is a gated multimodal unit (Arevalo et al., 2017, *Gated Multimodal
  Units for Information Fusion*). The two embeddings come from different views, so each is
  projected to `fusion_dim` and a learned gate decides per episode how much each contributes.
  The mean gate value is returned as the temporal share of the decision.
- `fusion: concat` concatenates the embeddings and projects them; it is the fusion baseline.
- `use_temporal: false` or `use_spatial: false` runs a single-stream baseline for ablation.

`sigmoid(logit)` (`CoordinationModel.score`) is an uncalibrated coordination score, not a
probability; the decision threshold is chosen during evaluation. `coordination_loss` is binary
cross-entropy, and `positive_weight` sets its class weight from the training labels only.

## Training and evaluation

```bash
python run_training.py
python run_evaluation.py
```

`run_training.py` reads `episode_graphs.pt`, makes a stratified train / validation / test split
by episode (`val_fraction`, `test_fraction`), fits the feature scaler and class weight on the
training part, trains with Adam and early stopping on validation loss, and picks the decision
threshold on the validation part:

- `threshold_strategy: f1` takes the best validation F1;
- `threshold_strategy: fpr` takes the best validation recall with false positive rate at most
  `max_fpr`, for settings where alert fatigue matters more than missed campaigns.

Candidate thresholds are midpoints between validation scores, and among equally good ones the
middle one is used. The checkpoint (`model/coordination_model.pt`) stores the weights, scaler,
model and training config, seed, threshold and the episode ids of each split.

`run_evaluation.py` loads that checkpoint, scores only the test episodes once and writes to
`evaluation/`:

| File | Contents |
| --- | --- |
| `metrics.json` | precision, recall, F1, FPR, FNR, balanced accuracy, ROC-AUC, PR-AUC, confusion matrix, time to detect, gate summary, config and split sizes |
| `decisions.json` | per test episode: score, threshold, decision, true label, temporal share, VMs in event order, time window, shared networks and storage, and for flagged episodes the step-by-step propagation evidence |
| `decision_report.md` | the same results as a readable report |

`metrics.json` and the report also break the test results down by the generator pattern of each
episode (`episode_patterns.csv`, written by the generator): the false alarm rate for scattered,
burst, cluster and lookalike coincidental episodes, and the detection rate for fast and stealth
campaigns. The pattern is only used for this breakdown and is never a model input.

Time to detect replays each coordinated test episode event by event and records when its score
first reaches the threshold. Structural path similarity, severity metrics and probability
calibration are not evaluated; the report lists why. Scores are uncalibrated, and all results
come from synthetic episodes.

## Baselines

```bash
python run_baselines.py
```

The baselines are scored on the same train / validation / test split as `run_training.py`
(same seed and fractions), with the same metrics and the same threshold rule, so their numbers
can be put next to `run_evaluation.py`. Results go to `baselines/metrics.json`.

| Baseline | What it is |
| --- | --- |
| `rule_based` | SIEM-style correlation rule: consecutive flagged events are linked when their VMs share a network or storage pool and are at most W hours apart; the score is the share of linked steps |
| `logistic_regression` | scikit-learn logistic regression on 17 episode features (counts of VMs, networks and pools, shared-resource shares, time span and gaps, mean telemetry) |
| `random_forest` | scikit-learn random forest on the same features |
| `continuum_style` | CONTINUUM-style GNN (Bahar et al., 2025), set with `model.architecture: chained` |

The settings of the first three (W, the regularisation strength, tree depth and leaf size) are
chosen by validation ROC-AUC from small fixed grids, and the threshold on validation, as for the
GNN. Class weights are balanced to match the weighted loss.

`continuum_style` follows the base paper's encoder: a GAT over each snapshot of the episode, the
mean of the node embeddings per snapshot, then a GRU across the snapshots. A snapshot is the
episode graph after each of its events. It is trained and evaluated exactly like the dual-stream
model. Differences from the paper, so both run on the same task: it is trained on labels instead
of as an autoencoder with k-nearest-neighbour distances to benign graphs, the attention has no
edge features (the episode graphs have none), the hours since the first event go to the GRU with
each snapshot because snapshots follow events rather than fixed time intervals, and there is no
federated learning.

## Experiments: ablations, baselines and scenarios

```bash
python run_experiments.py
python run_experiments.py --spec configs/experiments.yaml --out runs/comparison
```

`configs/experiments.yaml` lists seeds, generator scenarios, model variants and baselines. Every
variant and baseline is run on every scenario and seed, on the same split, with each run under
`<out>/<scenario>/seed_<n>/<variant>/` and the baselines under `<out>/<scenario>/seed_<n>/baselines/`:

- variants switch a stream off (`temporal_only`, `spatial_only`), change the fusion method
  (`concat`), or use the base paper's design (`continuum_style`);
- `baselines` lists the non-graph baselines (`rule_based`, `logistic_regression`, `random_forest`);
- scenarios change the generator: `default` uses `configs/default.yaml`, `no_overlap` is the
  notebook's easy setup, `slow_campaigns` makes every campaign slow, `crowded_networks` uses only
  10 network fabrics, and `randomised` draws the network and episode counts again for every seed
  (domain randomisation).

The default spec has 5 scenarios, 5 seeds, 5 variants and 3 baselines; on a CPU it takes a few
hours, most of it for `continuum_style`.

`results.csv` has one row per run, `summary.csv` and `summary.md` the mean and standard deviation
over seeds, and the share of test episodes flagged per generator pattern. The variant marked as
selected per scenario has the best mean validation F1; test results are reported but never used
to choose. Each model is compared with the first variant (`gated`) by a two-tailed paired t-test
over seeds on test F1 and ROC-AUC, as in Euler (King and Huang, 2022). No correction for
multiple comparisons is applied, and with 5 seeds the test has little power, so a high p-value
does not show that two models are equal.

### Tunable settings

`cadgnn params` prints this table from the config classes, so it always matches the code:

| Setting | Component | Controls | Default | Current | Values to try |
| --- | --- | --- | --- | --- | --- |
| `graph.min_episode_events` | graph construction | smallest episode turned into a graph | 2 | 2 | 2, 3 |
| `model.architecture` | baseline | dual_stream, or the CONTINUUM-style chained GAT then GRU | dual_stream | dual_stream | dual_stream, chained |
| `model.hidden_dim` | both streams | width of node and sequence representations | 64 | 64 | 32, 64, 128 |
| `model.temporal_dim` | temporal stream | size of the temporal episode vector | 64 | 64 | 32, 64 |
| `model.spatial_dim` | spatial stream | size of the spatial episode vector | 64 | 64 | 32, 64 |
| `model.temporal_layers` | temporal stream | stacked GRU layers | 1 | 1 | 1, 2 |
| `model.spatial_layers` | spatial stream | GAT layers (2 lets VMs on one network see each other) | 2 | 2 | 1, 2, 3 |
| `model.spatial_heads` | spatial stream | attention heads per GAT layer | 2 | 2 | 1, 2, 4 |
| `model.use_temporal` | ablation | temporal stream on or off | True | True | true, false |
| `model.use_spatial` | ablation | spatial stream on or off | True | True | true, false |
| `model.fusion` | fusion | how the two streams are combined | gated | gated | gated, concat |
| `model.fusion_dim` | fusion | size of the fused vector | 64 | 64 | 32, 64 |
| `model.dropout` | all | dropout rate | 0.1 | 0.1 | 0.0, 0.1, 0.3 |
| `model.device` | runtime | cpu, cuda or auto | auto | auto | not tuned |
| `training.val_fraction` | split | share of episodes for validation | 0.2 | 0.2 | 0.2 |
| `training.test_fraction` | split | share of episodes held out for testing | 0.2 | 0.2 | 0.2 |
| `training.epochs` | training | maximum epochs | 100 | 100 | 100, 200 |
| `training.batch_size` | training | episodes per batch | 16 | 16 | 8, 16, 32 |
| `training.learning_rate` | training | Adam learning rate | 0.001 | 0.001 | 1e-3, 3e-4 |
| `training.weight_decay` | training | L2 penalty in Adam | 0.0001 | 0.0001 | 0, 1e-4, 1e-3 |
| `training.patience` | training | epochs without validation gain before stopping | 15 | 15 | 15 |
| `training.threshold_strategy` | decision | best validation F1, or best recall under an FPR cap | f1 | f1 | f1, fpr |
| `training.max_fpr` | decision | FPR cap for the fpr strategy (alert fatigue) | 0.1 | 0.1 | 0.05, 0.1 |

## Layout

```
run_generator.py      entry point for the random anomaly generator
run_graph_builder.py  entry point for building the episode graphs
run_training.py       entry point for training and threshold selection
run_evaluation.py     entry point for the held-out evaluation and decision report
run_baselines.py      entry point for the rule-based and classic ML baselines
run_experiments.py    entry point for the comparison experiments
configs/              run configurations
notebooks/            reference notebook for the random anomaly generator (kept unchanged)
src/cadgnn/
  data/               dataset checks and generator calibration
  generator/          synthetic infrastructure, events and output files
  graph/              heterogeneous episode graph construction
  models/             batching, feature scaling, node encoder, dual-stream container
  temporal/           temporal severity stream
  spatial/            spatial topology stream
  fusion/             fusion layers, coordination model and loss
  baselines/          CONTINUUM-style model, episode features, rule-based and classic ML baselines
  training/           stratified split, training loop, training run
  evaluation/         metrics, thresholds, time to detect, decision report
  experiments/        experiment specs, runner, summaries, parameter table
  config.py, cli.py   configuration and command line
tests/                pytest suite
```

## Tests

```bash
pytest
ruff check . && ruff format --check .
```

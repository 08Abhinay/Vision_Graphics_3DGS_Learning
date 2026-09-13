# FootShellGaussian: End-to-End Steps So Far

This is the practical runbook for reproducing the project from a raw GLB to
the latest accepted stage. Update this file whenever the pipeline gains a new
stage. Keep `README.md` as the technical explanation of the code and keep this
file focused on what to run, what to inspect, and where the results appear.

## Current endpoint

The active dataset contains canonical right shoes with this effective frame:

```text
+X = heel to toe
+Y = downward toward the sole
+Z = shoe width
```

The current FootShell preparation behavior is:

```text
normal
  -> detect the interior footbed
  -> calculate reversible functional-length normalization
  -> write footbed review and normalized-shoe artifacts
  -> fit SUPR ankle and midfoot pitch against the saved normalized support
  -> measure footbed contact and clearance from every other shoe surface
  -> fit SUPR shape and placement to the measured cavity
  -> transfer canonical anatomy to one shared dense SUPR surface
  -> construct one shared tetrahedral foot-and-lower-leg anatomical volume

high_heel
  -> detect the inclined interior support
  -> calculate heel/forefoot support diagnostics
  -> calculate the same reversible functional-length normalization
  -> preserve the support incline
  -> write footbed review and normalized-shoe artifacts
```

High-heel SUPR fitting and plantarflexion are not implemented yet.

## Active locations

```text
Raw GLBs:
/home/ab5298/dataset/datasets/external/golden_set_eval_glb/curated_subsets/footbed_clean

Dataset manifest:
/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/golden_set_evaluation_manifest.json

Dataset tools:
/storage/Abhinay/Shell_Gaussian/dataset_tools_blender

Processed dataset:
/home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation

FootShell project:
/storage/Abhinay/Shell_Gaussian/FootShellGaussian

FootShell outputs:
/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation

Normal-shoe SUPR support fits:
/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/support_fit

Normal-shoe cavity analyses:
/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/cavity_analysis

Final native normal-shoe containment fits:
/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/containment_fit

Canonical and dense anatomical surfaces:
/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/anatomical_surface

Canonical foot-and-lower-leg anatomical volume:
/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/anatomical_volume
```

The manifest and processed dataset paths remain stable as shoes are added. Do
not create a new manifest or a new dataset directory merely because the number
of shoes increases.

## Step 1: Add a raw GLB

Copy or place the new file in the raw GLB directory. Use a stable filename
that will also become the shoe name in the processed dataset.

Example:

```text
/home/ab5298/dataset/datasets/external/golden_set_eval_glb/curated_subsets/footbed_clean/new_shoe.glb
```

## Step 2: Calculate its checksum

```bash
sha256sum \
  /home/ab5298/dataset/datasets/external/golden_set_eval_glb/curated_subsets/footbed_clean/new_shoe.glb
```

Copy the checksum into the manifest entry. The pipeline rejects the file if
its content later changes without a matching manifest update.

## Step 3: Add the manifest entry

Edit:

```text
/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/golden_set_evaluation_manifest.json
```

Start a new entry with `reviewed: false`.

```json
{
  "name": "new_shoe",
  "model": "new_shoe.glb",
  "sha256": "CHECKSUM_FROM_SHA256SUM",
  "reviewed": false,
  "shoe_profile": "normal",
  "source_axes": {
    "length": "X",
    "width": "Y",
    "up": "Z"
  },
  "selection": {
    "mode": "all"
  },
  "mirror_width": false
}
```

Use exactly one profile:

```json
"shoe_profile": "normal"
```

or:

```json
"shoe_profile": "high_heel"
```

The profile does not rotate, scale, mirror, or otherwise change the dataset
geometry. It tells FootShell which support rules and downstream stages are
approved for the shoe.

Do not copy the example `source_axes` without checking the raw model. The
three entries tell the pipeline which raw direction represents length, width,
and physical up. A leading minus sign reverses that direction, for example
`"length": "-Y"`.

### Selecting one shoe from a pair

Use:

```json
"selection": {
  "mode": "axis-side",
  "axis": "Y",
  "side": "min",
  "separate_loose_parts": true
}
```

only when the raw asset contains a pair or unrelated components that must be
removed.

Selection happens after `source_axes` has mapped the imported geometry into
the canonical Blender frame:

```text
Blender X = shoe length
Blender Y = shoe width
Blender Z = physical up
```

The selection code does not cut a mesh in half. It keeps or removes complete
mesh components according to the center of each component's bounding box.

The chosen axis means:

- `axis: "Y"`: compare components from one width side to the other. This is
  normally appropriate when the left and right shoes stand beside each other.
- `axis: "X"`: compare components from the heel side to the toe side. Use this
  only if two complete shoes are arranged along the length direction.
- `axis: "Z"`: compare lower and upper components. This is rarely appropriate
  for selecting the right shoe.

For the chosen axis, the pipeline calculates one middle divider:

```text
pivot = (lowest coordinate + highest coordinate) / 2
```

Then:

- `side: "min"` keeps components whose centers are on the lower-coordinate
  side of the divider.
- `side: "max"` keeps components whose centers are on the higher-coordinate
  side of the divider.

For example, with `axis: "Y"`, `min` keeps components centered toward
negative/lower Y and `max` keeps components centered toward positive/higher Y.
Neither value inherently means "right shoe". Which side contains the right
shoe depends on the asset and must be confirmed in the audit images.

`separate_loose_parts: true` first joins the imported mesh objects and then
separates every disconnected piece into its own component. This is needed when
both shoes were imported as one Blender object but are physically disconnected.

If the GLB already contains only the desired right shoe, use:

```json
"selection": {
  "mode": "all"
}
```

## Step 4: Audit the new shoe

Set reusable shell variables:

```bash
TASK_PY=/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python
TASK_PIPELINE=/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/pipeline.py
TASK_BLENDER=/home/ab5298/anaconda3/envs/shellgaussianenv/bin/blender
TASK_GLBS=/home/ab5298/dataset/datasets/external/golden_set_eval_glb/curated_subsets/footbed_clean
TASK_MANIFEST=/storage/Abhinay/Shell_Gaussian/dataset_tools_blender/golden_set_evaluation_manifest.json
TASK_AUDIT=/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/audit
TASK_SHOE=new_shoe
```

Run one-shoe audit:

```bash
"$TASK_PY" "$TASK_PIPELINE" audit \
  --shoe "$TASK_SHOE" \
  --gpu 0 \
  --source-root "$TASK_GLBS" \
  --manifest "$TASK_MANIFEST" \
  --output-dir "$TASK_AUDIT" \
  --blender "$TASK_BLENDER"
```

Inspect the side, toe, heel, top, and bottom views. Confirm:

1. The correct right shoe was retained.
2. The shoe is physically upright.
3. The heel is at `-X`.
4. The toe points toward `+X`.
5. The width orientation is correct and not mirrored.
6. No required shoe component was deleted.

If the result is wrong, correct `source_axes`, `selection`, or `mirror_width`
and audit again. Do not proceed simply because the command succeeded.

After visual acceptance, change the entry to:

```json
"reviewed": true
```

## Step 5: Build the processed dataset

### Recommended complete build

The existing launcher uses five GPUs in tmux:

```bash
cd /storage/Abhinay/Shell_Gaussian
dataset_tools_blender/build_golden_set_evaluation.sh
```

It builds from the stable manifest into the stable processed dataset. Existing
valid shoes are validated and skipped; newly listed shoes are built.

Monitor it with either:

```bash
tmux attach -t golden-set-evaluation-build
```

or:

```bash
tail -f \
  /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/logs/dataset-build.log
```

Check whether it is still running:

```bash
tmux has-session -t golden-set-evaluation-build \
  && echo RUNNING \
  || echo FINISHED
```

### Build only the new shoe

```bash
"$TASK_PY" "$TASK_PIPELINE" build \
  --shoe "$TASK_SHOE" \
  --gpu 0 \
  --source-root "$TASK_GLBS" \
  --manifest "$TASK_MANIFEST" \
  --output-root /home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation \
  --blender "$TASK_BLENDER"
```

Existing valid outputs are not overwritten by default. Use `--overwrite` only
when deliberately replacing that shoe's published processed scene.

## Step 6: Validate the processed shoe

```bash
"$TASK_PY" "$TASK_PIPELINE" validate \
  --shoe "$TASK_SHOE" \
  --source-root "$TASK_GLBS" \
  --manifest "$TASK_MANIFEST" \
  --output-root /home/ab5298/dataset/datasets/processed/gshell/golden_set_evaluation
```

The processed shoe directory should contain at least:

```text
golden_set_evaluation/new_shoe/
├── reference_mesh.ply
├── blender_canonicalization.json
├── transforms.json
├── image/
├── mask/
└── invdepth/
```

The canonicalization JSON must contain the selected `shoe_profile` and the
effective GShell coordinate convention.

## Step 7: Run FootShell preparation

The FootShell wrapper accepts explicit shoe names. Pass every newly added name
because its no-argument list currently contains the original reviewed shoes.

### One shoe

```bash
cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian

scripts/prepare_golden_set_evaluation_shoes.sh \
  new_shoe
```

### Several shoes

```bash
scripts/prepare_golden_set_evaluation_shoes.sh \
  new_normal_shoe \
  new_high_heel
```

### Run preparation in tmux

```bash
mkdir -p \
  /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/logs

tmux new-session -d -s prepare-new-shoes \
  'cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian && \
  scripts/prepare_golden_set_evaluation_shoes.sh \
    new_normal_shoe \
    new_high_heel \
  2>&1 | tee \
  /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/logs/new-shoes-preparation.log'
```

Monitor it with:

```bash
tmux attach -t prepare-new-shoes
```

or:

```bash
tail -f \
  /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/logs/new-shoes-preparation.log
```

## Step 8: Inspect a normal-shoe result

The output appears at:

```text
/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation/new_normal_shoe
```

Expected files:

```text
shoe_preparation.json
footbed_surface.ply
footbed_overlay.ply
shoe_normalized.ply
```

Inspect in this order:

1. Open `footbed_overlay.ply`. Green must be the interior surface on which the
   foot stands, not the outsole, upper, toe panel, or shaft.
2. Open `shoe_normalized.ply`. It must preserve the shoe's axis directions and
   shape.
3. Check `shoe_preparation.json`. The normalization matrices and measurements
   must be finite.

For an accepted normal shoe, the current pipeline endpoint is functional-length
normalization.

## Step 9: Inspect a high-heel result

The output appears at:

```text
/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation/new_high_heel
```

Expected files:

```text
shoe_preparation.json
footbed_surface.ply
footbed_overlay.ply
shoe_normalized.ply
```

The JSON should contain:

```json
{
  "shoe_profile": "high_heel",
  "preparation_status": "support_detected_and_normalized",
  "normalization": {
    "functional_length": "positive finite value"
  }
}
```

Open `footbed_overlay.ply`. Green must follow the inclined interior support and
exclude the outsole bottom, heel column, straps, and upper panels.

Open `shoe_normalized.ply` and confirm that the shoe keeps its steep support
shape. The functional heel must map to `X=0`, the functional toe to `X=1`, and
the JSON must contain finite forward and inverse matrices. This normalization
does not mean that a neutral SUPR foot is ready for the heel; plantarflexion and
high-heel fitting remain future work.

## Step 10: Rerun an existing FootShell result

Preparation refuses to replace existing artifacts unless explicitly asked.

```bash
cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian

OVERWRITE=1 scripts/prepare_golden_set_evaluation_shoes.sh \
  new_shoe
```

`OVERWRITE=1` replaces only the known artifacts for that profile and preserves
unrelated files in the output directory.

## Step 11: Fit SUPR to the support in a prepared normal shoe

This step uses the three existing preparation artifacts below. It does not
reload the original shoe and does not detect the footbed again.

```text
shoe_normalized.ply
shoe_preparation.json
footbed_surface.ply
```

Run one normal shoe:

```bash
cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian

/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python -m pip install -e ../baselines/SUPR
/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python -m pip install -e '.[fitting]'

/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python \
  scripts/run_alignment.py \
  --preparation-dir /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation/canvas_shoe \
  --supr-model /storage/Abhinay/Shell_Gaussian/baselines/SUPR/data/supr_male_right_foot.npy \
  --output-dir /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/support_fit/canvas_shoe
```

This stage requires CUDA. It tests only ankle and midfoot pitch; all toe joints
and SUPR shape values remain zero. Sizing is anchored once, not chosen per shoe:
a shoe's functional length is defined to admit the neutral 250 mm foot with
12.5 mm in front, which fixes the scale. Each pose then keeps its own
heel-to-toe length, so a plantarflexed foot honestly reports more front
allowance than a neutral one. The runner anchors the rear foot at `X=0`, aligns it
sideways to the saved footbed centerline, and selects the near-neutral pose that
best balances heel and forefoot support contact. It writes:

```text
support_fit.json
foot_support_fitted.ply
footbed_normalized.ply
support_fit_overlay.ply
```

Inspect `support_fit_overlay.ply` together with
`footbed_normalized.ply`. Confirm that the foot is upright, toes point toward
`+X`, the rear begins at `X=0`, the longest toe ends near `X=0.95` (a strongly
plantarflexed pose ends a little short of that, which is expected), the foot
follows the green support laterally, and heel and forefoot approach the support
without obvious plantar penetration. Check the printed toe allowance: below
10 mm is refused outright, and above 15 mm is recorded rather than refused.
Pass `--overwrite` only when deliberately regenerating these four known
artifacts.

This command rejects `shoe_profile="high_heel"`. High-heel SUPR placement needs
plantarflexion and remains a later checkpoint.

## Step 12: Measure cavity collision and clearance

This step reads the normalized shoe and completed support fit. It does not
rerun preparation, footbed detection, normalization, or SUPR fitting.

```bash
cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian

/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python \
  scripts/run_cavity_analysis.py \
  --preparation-dir /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation/canvas_shoe \
  --support-fit-dir /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/support_fit/canvas_shoe \
  --output-dir /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/cavity_analysis/canvas_shoe
```

The detected footbed is the only shoe surface on which contact is allowed.
Every other original shoe triangle is treated as an obstacle. The ankle is
allowed to emerge through the empty shoe opening, but it may not intersect the
collar. The command writes:

```text
cavity_analysis.json
foot_clearance_colored.ply
cavity_overlay.ply
```

Open `cavity_overlay.ply`. Blue regions are clear, yellow regions are close,
magenta regions have passed a local upper or side boundary, and red regions
touch or cross a forbidden shoe surface. Inspect the existing green
`footbed_normalized.ply` from the support-fit directory beside it when checking
plantar contact.

The schema-2 JSON status is `clear`, `protrusion_detected`, or
`collisions_detected`. The local signed test keeps real openings open: it does
not add an imaginary lid or require a closed shoe mesh. A problem status is an
expected diagnostic and does not mean the command failed.

To analyze all currently fitted normal shoes without changing their existing
outputs:

```bash
cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian

for shoe_dir in /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/support_fit/*; do
  shoe=$(basename "$shoe_dir")
  /home/ab5298/anaconda3/envs/shellgaussianenv/bin/python \
    scripts/run_cavity_analysis.py \
    --preparation-dir "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation/$shoe" \
    --support-fit-dir "$shoe_dir" \
    --output-dir "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/cavity_analysis/$shoe"
done
```

Use `--overwrite` only when deliberately regenerating the three known cavity
artifacts for a shoe. This stage runs on the CPU and does not require a GPU.

## Step 13: Fit SUPR to the measured cavity

This normal-shoe-only step reads the accepted support fit and cavity analysis.
It verifies that both can be reproduced, then searches for a natural SUPR foot
with 18–22 mm of front space that best fits the measured local cavity. It does not redetect the footbed or
renormalize the shoe. Regenerate Steps 11 and 12 first because this runner
requires the current schema-2 support fit and schema-2 cavity record.

```bash
cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian

CUDA_VISIBLE_DEVICES=1 \
/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python \
  scripts/run_containment_fit.py \
  --preparation-dir /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation/crocs \
  --support-fit-dir /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/support_fit/crocs \
  --cavity-analysis-dir /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/cavity_analysis/crocs \
  --supr-model /storage/Abhinay/Shell_Gaussian/baselines/SUPR/data/supr_male_right_foot.npy \
  --output-dir /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/containment_fit/crocs
```

The command writes:

```text
containment_fit.json
foot_containment_fitted.ply
foot_clearance_colored.ply
containment_fit_overlay.ply
```

The search keeps the earlier individual and coupled ten-beta shapes and adds
256 deterministic varied-magnitude combinations across betas 1–9. Beta 0 is
solved toward 18, 20, and 22 mm for each new combination. Every support-valid
shape in that band receives an exact collision check. Ten beta-diverse results
are then refined together with heel/lateral and ankle/midfoot corrections.
Every candidate returns to first contact with the saved support and must retain
heel, forefoot, toe, and overall support coverage.

Foot size is not searched separately. Because the scale is anchored, the shape
values themselves change how long, wide, and tall the foot is, so a smaller
foot here is a genuinely different foot rather than the same one shrunk. Beta 0
guides broad candidates toward the requested length, while the other betas keep
their natural correlated shape effects. No shaped candidate is uniformly
rescaled afterward. The final mesh must leave 18–22 mm in front, with 20 mm as
the target.

Open `containment_fit_overlay.ply` and load the existing green
`footbed_normalized.ply` from the matching `support_fit` directory. Confirm
that the foot remains anatomical and supported. Blue is clear, yellow is near
an obstacle, magenta is beyond a local boundary, and red is an exact forbidden
collision.

The part of the anatomical ankle above the posed ankle joint and behind the
posed midfoot joint is ignored only by the signed upper/side score, allowing it
to emerge through a genuinely open entrance. Exact intersections are still
checked on the complete ankle, so a collar collision remains red and affects
selection.

The JSON status has two outcomes:

- `contained_target_fit`: contained with 18–22 mm toe allowance.
- `residual_target_fit`: no collision-free shape was found in the same toe-space
  band, so the safest realistic-size result was saved with its remaining
  problem areas for the next stage.

The fitter does not quietly solve collisions by leaving a large empty region in
front of the toes. The schema-5 JSON records the ankle exemption, expanded
seed configuration, exact broad candidates, ten refinement starts, ten betas,
actual foot dimensions, and the union of exact collision and non-exempt signed
protrusion faces.

Use `--overwrite` only when deliberately replacing these four known artifacts.
CUDA generates SUPR candidates; exact shoe collision checks run on the CPU.

## Step 14: Build canonical and dense anatomical surfaces

This CPU-only step starts from the accepted native fitted feet in
`containment_fit`. It does not run fitting again. It defines anatomy once on
the neutral right SUPR reference and transfers it to every fitted foot through
the stable SUPR vertex and face correspondence. It then applies the same
deterministic subdiv2 map to every surface.

```bash
cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian

/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python \
  scripts/run_anatomical_surface.py \
  --containment-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/containment_fit \
  --supr-model /storage/Abhinay/Shell_Gaussian/baselines/SUPR/data/supr_male_right_foot.npy \
  --output-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/anatomical_surface \
  --overwrite
```

With no names after the options, all valid normal-shoe containment directories
are processed in sorted order. To rerun only a few shoes, append their names:

```bash
  canvas_shoe sandal_1 sneaker_vibe ww_ii_german_jack_boots
```

The shared reference directory contains the canonical JSON/NPZ, neutral dense
mesh, and two colored anatomical maps. Every shoe directory contains:

```text
anatomical_surface.json
foot_dense.ply
regions_longitudinal.ply
regions_surface.ply
```

Inspect `regions_longitudinal.ply` to confirm that ankle, heel, arch, forefoot,
and toe colors remain on the same anatomy. Inspect `regions_surface.ply` for
plantar, top, medial, lateral, rear, and front surface colors. The dense mesh
must always contain 4,151 vertices and 8,240 faces, with the first 266 vertices
identical to the accepted native fitted foot. `--overwrite` replaces only the
known artifacts and preserves unrelated files.

## Step 15: Fit the lower-leg exit and inspect shoe collars

This does not rerun fitting. It keeps every accepted dense fitted foot
unchanged, aligns a near-neutral male SUPR shank at the fitted ankle, searches
ankle pitch and roll, and reports exact bridge/lower-leg intersections with
non-footbed shoe surfaces. The first ten full-body betas are used only as a
secondary, tightly constrained correction. The fitter preserves a natural
shaft even when that means honestly reporting a residual collar intersection.

```bash
/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python \
  scripts/run_lower_leg_attachment.py \
  --anatomical-surface-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/anatomical_surface \
  --preparation-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/shoe_preparation \
  --support-fit-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/support_fit \
  --full-body-supr-model /storage/Abhinay/Shell_Gaussian/baselines/SUPR/data/supr_male.npy \
  --output-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/lower_leg_attachment \
  --overwrite
```

Each shoe receives `lower_leg_attachment.json`, `foot_lower_leg.ply`,
`lower_leg_collar_colored.ply` and `lower_leg_collar_overlay.ply`. Blue is the
unchanged fitted foot, cyan is the fitted near-neutral shank, orange is the
ankle bridge, and red marks exact contact with the shoe. Empty opening space is
allowed. The JSON records the neutral baseline, pose-only result, selected
betas, girth ratios and any remaining collision.

## Step 16: Transfer anatomy to the complete foot and lower leg

This CPU-only step creates the missing anatomical map for the joined surface.
It reads the accepted lower-leg attachments directly, so it does not rerun
foot fitting, containment, or lower-leg optimization.

```bash
/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python \
  scripts/run_extended_anatomical_surface.py \
  --anatomical-surface-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/anatomical_surface \
  --lower-leg-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/lower_leg_attachment \
  --full-body-supr-model /storage/Abhinay/Shell_Gaussian/baselines/SUPR/data/supr_male.npy \
  --output-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/extended_anatomical_surface \
  --overwrite
```

With no shoe names, all 16 accepted normal-shoe attachments are processed in
sorted order. Every result has the same 6,951 vertices and 13,832 faces. The
first 4,151 vertices and 8,240 faces remain the existing dense fitted foot.
The remaining stable IDs describe the ankle transition and lower leg.

The output contains one neutral reference plus one directory per shoe:

```text
extended_anatomical_surface/
├── reference/
│   ├── canonical_extended_surface.json
│   ├── canonical_extended_surface.npz
│   ├── neutral_foot_lower_leg.ply
│   ├── regions_longitudinal.ply
│   ├── regions_surface.ply
│   └── regions_components.ply
└── <shoe>/
    ├── extended_anatomical_surface.json
    ├── foot_lower_leg.ply
    ├── regions_longitudinal.ply
    ├── regions_surface.ply
    └── regions_components.ply
```

Inspect the three region meshes. Longitudinal colors identify the original
foot regions plus lower shaft, calf and upper shaft. Surface colors identify
plantar/top, front/rear and medial/lateral directions. Component colors isolate
the specialised foot, ankle bridge and lower-leg donor. The same color must
remain attached to the same anatomical location for every shoe.

The canonical NPZ preserves the old foot charts and adds one exact joined-skin
coordinate: extended face ID plus three barycentric weights. It also stores the
ankle correspondence, knee loop, joints, landmarks and subdivision provenance.
The knee remains open and no tetrahedral volume is created in this step.

## Step 17: Build the canonical foot-and-lower-leg anatomical volume

This CPU-only step runs once from the accepted extended anatomical surface. It
does not process individual shoe directories, rebuild SUPR, or rerun fitting.
The exact canonical anatomy remains immutable; a separately registered
computational foot boundary removes the neutral toe self-overlap before Gmsh.
Install the optional dependency group once if needed:

```bash
cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian

/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python -m pip install -e ".[volume]"
```

Build the volume:

```bash
/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python \
  scripts/run_anatomical_volume.py \
  --extended-anatomical-surface-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/extended_anatomical_surface \
  --output-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/anatomical_volume \
  --overwrite
```

The runner validates `extended_anatomical_surface/reference`. PyMeshFix repairs
only a closed copy of the 266-vertex native foot; the temporary ankle cap is
then removed, the computational foot is subdivided twice, and the unchanged
dense lower leg is reattached. Two-way face-and-barycentric maps preserve the
link to the authoritative 6,951-vertex anatomy. Gmsh retriangulates only this
disposable inner copy at a controlled resolution, after which the mappings and
fidelity checks are recomputed. The accepted computational boundary and fixed
outer envelope are then preserved exactly during tetrahedralization. The
smooth coordinate is `r=0` on the foot, bridge and real lower-leg skin and
`r=1` on the envelope. The knee cap is not anatomical skin and its interior
vertices are not assigned a fixed `r` value.

The result is:

```text
anatomical_volume/reference/
├── canonical_volume.json
├── canonical_volume.npz
├── canonical_volume.vtk
├── computational_inner_boundary.ply
├── outer_envelope.ply
└── boundary_regions.ply
```

Inspect `canonical_volume.vtk` in ParaView with the `harmonic_r` scalar. Values
should change smoothly from zero at the body boundary to one at the outside.
Inspect `computational_inner_boundary.ply` for repair displacement and
`boundary_regions.ply` for the foot, ankle transition, lower leg, temporary
knee cap, and outer envelope.

## Step 18: Prepare each fitted anatomical boundary target

Checkpoint 11-A evaluates the canonical computational boundary's saved
face-and-barycentric coordinates on every fitted foot-and-lower-leg surface:

```bash
/home/ab5298/anaconda3/envs/shellgaussianenv/bin/python \
  scripts/run_instance_anatomical_volume.py \
  --anatomical-volume-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/anatomical_volume \
  --extended-anatomical-surface-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/extended_anatomical_surface \
  --overwrite
```

With no positional shoe names, all accepted normal-shoe directories are
validated first and then written in sorted order. The output is:

```text
anatomical_volume/<shoe>/
├── boundary_target.json
├── boundary_target.npz
├── computational_boundary_target.ply
└── boundary_target_overlay.ply
```

The target retains the canonical 8,224-vertex/16,444-face computational
topology. Blue is the target proxy, red marks exact target self-intersections,
purple is the computational knee cap, and gray is the unchanged authoritative
fitted anatomy. Expected foot-only toe intersections are saved with
`status="ready_requires_untangling"`; bridge, lower-leg, or cap intersections
fail. The frozen-envelope test is diagnostic at this checkpoint.

No tetrahedron moves here. Checkpoint 11-B will use these soft targets to solve
the nearest fold-free inner boundary and the complete instance volume together.

## Step 19: Build Checkpoint 11-B2 continuation warm starts

Checkpoint 11-B1 loads and cross-validates the canonical volume, each saved
11-A target, and its authoritative fitted anatomy. Checkpoint 11-B2 reuses one
canonical finite-element factorization and advances from the canonical volume
toward each target while the outer envelope remains exactly fixed.

Run the current 15-shoe scope in `tmux` and exclude `sneaker_vibe`:

```bash
tmux new-session -d -s instance-volume-continuation \
  'cd /storage/Abhinay/Shell_Gaussian/FootShellGaussian && \
  /home/ab5298/anaconda3/envs/shellgaussianenv/bin/python \
    scripts/run_instance_volume_deformation.py \
    --anatomical-volume-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/anatomical_volume \
    --extended-anatomical-surface-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/extended_anatomical_surface \
    --output-root /home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/instance_anatomical_volume \
    --stop-after 11-b2 \
    --exclude sneaker_vibe \
    --overwrite'
```

The two per-shoe outputs, `continuation_state.json` and
`continuation_state.npz`, contain only the last valid baseline state and its
step history. `baseline_reached_target` and `needs_11_b3` are intermediate
statuses; neither represents a final cleaned volume. The next checkpoint must
jointly optimize the boundary and interior for states that stopped early.

## Step 20: Finish Checkpoint 11-B3 bounded deformation

The same runner now completes B3 by default. It selects a robust point on the
B2 path, repairs only a deterministic local region around intersecting faces
and weak surface tetrahedra, then fixes that computational surface while solving
the tetrahedral interior. The outer envelope, tetrahedron connectivity, fitted
anatomical surface, and 11-A target remain unchanged. When an exact target is
not feasible, only the computational copy may move, by at most half the fitted
surface resolution.

Successful shoes add `instance_volume.json`, `instance_volume.npz`, and
`instance_volume.vtk`. The final statuses are `final_exact_target` and
`final_corrected_target`; `failed_11_b3` is diagnostic and never publishes a
usable final NPZ or VTK. The runner records a failed shoe, continues the batch,
and returns a non-success summary after processing the remaining shoes.
`--resume` may be used instead of `--overwrite`; it reuses only complete B2/B3
states after their configuration, digests, arrays, and geometry revalidate.

## Failure rules

- If audit orientation is wrong, fix the manifest and rerun the audit.
- If the wrong shoe from a pair is selected, change `selection.axis` or
  `selection.side` based on the audited component positions.
- If dataset validation fails, do not run FootShell on that shoe.
- If the green support overlay is wrong, stop and diagnose that mesh. Do not
  weaken thresholds or add a per-shoe exception without testing the complete
  accepted set.
- If a `high_heel` produces no support, inspect its diagnostics. Do not process
  it as `normal` merely to bypass the heel rules.

## What comes next

Checkpoint 11-B3 produces the validated instance volumes required by the next
stage. Checkpoint 11-C will implement the forward and inverse volume mappings.
Toe articulation and other localized controls remain optional future
containment work. High-heel SUPR fitting remains outside the current scope.

# Runtime Composition Search -- Autoresearch Program

## Goal
Find the best combination of primitive variants and pipeline ordering to maximize overall agent performance. This is the meta-level search: instead of improving individual primitives, you compose them together optimally.

## How It Works

### 1. Preparation (`prepare.py`)
Run `python prepare.py` to scan all harness subfolders and identify:
- Which primitives exist
- Which ones have `results.tsv` (meaning they have been benchmarked with variants)
- The best-scoring variant for each primitive

### 2. Composition (`compose.py`)
Edit `compose.py` to configure:
- **VARIANT_CONFIG**: Which variant of each primitive to use. Change from `"baseline"` to a specific variant name when a better one exists in `results.tsv`.
- **PIPELINE_ORDER**: The order in which primitives execute in the agent loop. Reorder to find optimal sequencing.
- **RUNTIME_CONFIG**: Execution parameters (max turns, timeouts, retries).

### 3. Validation
Run `python compose.py --check` to validate the configuration. Run `python compose.py` to see the full config summary.

## What You Can Change
- **Only** edit `compose.py` -- it is the mutable configuration file.
- Do **not** edit `prepare.py` -- it is the read-only scanner.
- Do **not** edit any harness.py files from here -- those are improved by their own autoresearch programs.

## Search Strategy

### Phase 1: Identify Available Variants
1. Run `python prepare.py` to see what's available.
2. Note which primitives have multiple scored variants.

### Phase 2: Variant Selection
For each primitive with results:
1. Check which variant scored highest in `results.tsv`.
2. Update `VARIANT_CONFIG[primitive] = best_variant_name`.
3. Prioritize primitives with the largest score delta between baseline and best variant.

### Phase 3: Pipeline Ordering
The default pipeline order follows a logical flow:
  Planning -> Context -> Tools -> Memory -> Verification -> Recovery

Experiment with:
1. **Verification earlier**: Move self-check before tool use to catch errors sooner.
2. **Memory before planning**: Let the agent recall relevant info before planning.
3. **Removing unused stages**: If a primitive adds no value, remove it from the pipeline.
4. **Repeated stages**: Some stages might benefit from running twice (e.g., re-plan after tool results).

### Phase 4: Runtime Tuning
Adjust RUNTIME_CONFIG:
- **max_turns**: Higher values give the agent more chances but risk loops.
- **early_stop_on_success**: Usually True, but False may find better solutions.
- **parallel_primitives**: Enable if primitives in the same stage are independent.

## Constraints
- All referenced primitives must have a `harness.py` in their directory.
- Pipeline entries must exist in VARIANT_CONFIG.
- No duplicate entries in PIPELINE_ORDER.
- Variant names must match entries in the corresponding `results.tsv`.

## Evaluation
After updating `compose.py`, run the full agent on benchmark tasks to measure the impact of composition changes. Compare against the all-baseline configuration.

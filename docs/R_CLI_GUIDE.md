# Using metrics-adjuster from R

This guide shows how R users can call the `metrics-adjuster` CLI from an R
session and read the outputs back into R. The package itself is Python; R
invokes it as an external command.

## Relationship to `fairRisk`

Sarah E. Hegarty's [`fairRisk`](https://github.com/sarahhegarty/fairRisk)
package is the original R implementation of the aTPR methodology developed in
research led by Hegarty and Jinbo Chen. `metrics-adjuster` grew from Sae-Hwan
Park's internally validated Python rewrite used for additional VA analyses; the
rewrite was reviewed and approved by Hegarty for the team's analysis.

Use `fairRisk` when the original native R implementation is the appropriate
interface. This guide is for R workflows that intentionally need to invoke the
separate `metrics-adjuster` Python CLI. Cite the research method and software
implementation or implementations actually used.

## Install the CLI

Install `metrics-adjuster` into a Python environment available on your PATH:

```bash
python -m pip install metrics-adjuster
```

Or, from a checkout:

```bash
uv sync --extra dev
```

Confirm the command is available:

```bash
metrics-adjuster --help
```

## Run the synthetic demo from R

Use a dedicated output directory and capture stderr when debugging:

```r
output_dir <- tempfile("metrics_adjuster_demo_")
dir.create(output_dir)

status <- system2(
  "metrics-adjuster",
  args = c(
    "demo",
    "--output-dir", output_dir,
    "--n", "120",
    "--seed", "42",
    "--report",
    "--save-artifacts",
    "--report-figures"
  ),
  stdout = TRUE,
  stderr = TRUE
)

if (!is.null(attr(status, "status")) && attr(status, "status") != 0L) {
  stop(paste(status, collapse = "\n"))
}

list.files(output_dir)
```

Read metric outputs:

```r
atpr <- read.csv(file.path(output_dir, "aTPR.csv"))
head(atpr)
```

## Run on your own table

Save a CSV with columns for group, binary outcome, and predicted risk, then call
`run`:

```r
input_path <- "/path/to/cohort.csv"
output_dir <- tempfile("metrics_adjuster_run_")
dir.create(output_dir)

system2(
  "metrics-adjuster",
  args = c(
    "run",
    "--input", input_path,
    "--output-dir", output_dir,
    "--group-col", "group",
    "--ref-group", "ref",
    "--response-col", "outcome",
    "--risk-col", "risk",
    "--save-artifacts"
  )
)
```

Parquet inputs are supported when `pyarrow` is installed:

```r
system2(
  "metrics-adjuster",
  args = c(
    "run",
    "--input", "/path/to/cohort.parquet",
    "--output-dir", output_dir,
    "--group-col", "group",
    "--ref-group", "ref",
    "--response-col", "outcome",
    "--risk-col", "risk"
  )
)
```

## Read parquet artifacts in R

When `--save-artifacts` is used, read calibrated probabilities and weights with
the **arrow** package:

```r
if (!requireNamespace("arrow", quietly = TRUE)) {
  stop("Install the arrow package to read parquet artifacts.")
}

calibration <- arrow::read_parquet(file.path(output_dir, "calibration.parquet"))
weights <- arrow::read_parquet(file.path(output_dir, "weights.parquet"))
```

## Generate persisted synthetic data

To write reproducible synthetic input under `data/generated/`:

```r
system2(
  "metrics-adjuster",
  args = c(
    "generate-synthetic",
    "--output-dir", "data/generated/synthetic-metrics-demo",
    "--n", "600",
    "--seed", "2026"
  )
)
```

## Tips

- Quote paths that contain spaces when building `args`.
- Use `processx::run()` instead of `system2()` when you need richer error
  objects or stricter timeout control.
- Keep Python and R working directories in mind; relative paths resolve from
  the process working directory, not the R session source file.
- See `docs/CLI_MANUAL.md` for the full flag reference.

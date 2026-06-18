#!/usr/bin/env Rscript

# Call metrics-adjuster from R and read outputs back into R.
# Mirrors docs/R_CLI_GUIDE.md; writes artifacts under demos/r-cli/results/.

script_dir <- {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", file_arg[1])))
  } else {
    normalizePath(getwd())
  }
}

repo_root <- normalizePath(file.path(script_dir, "..", ".."))
results_dir <- file.path(script_dir, "results")
demo_dir <- file.path(results_dir, "demo")
run_dir <- file.path(results_dir, "run")
generated_dir <- file.path(repo_root, "data", "generated", "r-cli-demo")

metrics_adjuster_cmd <- strsplit(
  Sys.getenv("METRICS_ADJUSTER_CMD", "metrics-adjuster"),
  " ",
  fixed = TRUE
)[[1]]

run_cli <- function(args) {
  status <- system2(
    metrics_adjuster_cmd[1],
    args = c(metrics_adjuster_cmd[-1], args),
    stdout = TRUE,
    stderr = TRUE
  )
  exit_code <- attr(status, "status")
  if (!is.null(exit_code) && exit_code != 0L) {
    stop(paste(status, collapse = "\n"))
  }
  invisible(status)
}

cat("=== Synthetic demo ===\n")
dir.create(demo_dir, recursive = TRUE, showWarnings = FALSE)
run_cli(c(
  "demo",
  "--output-dir", demo_dir,
  "--n", "120",
  "--seed", "42",
  "--report",
  "--save-artifacts",
  "--report-figures"
))

cat("Demo outputs:\n")
print(list.files(demo_dir))

atpr <- read.csv(file.path(demo_dir, "aTPR.csv"))
cat("\naTPR preview:\n")
print(head(atpr))

cat("\n=== Run on cohort table ===\n")
input_path <- file.path(demo_dir, "synthetic_metrics_data.csv")
dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
run_cli(c(
  "run",
  "--input", input_path,
  "--output-dir", run_dir,
  "--group-col", "group",
  "--ref-group", "ref",
  "--response-col", "outcome",
  "--risk-col", "risk",
  "--save-artifacts"
))
cat("Run outputs:\n")
print(list.files(run_dir))

cat("\n=== Parquet artifacts ===\n")
if (requireNamespace("arrow", quietly = TRUE)) {
  calibration <- arrow::read_parquet(file.path(demo_dir, "calibration.parquet"))
  weights <- arrow::read_parquet(file.path(demo_dir, "weights.parquet"))
  cat("Calibration rows:", nrow(calibration), "\n")
  cat("Weights rows:", nrow(weights), "\n")
} else {
  cat("Install the arrow package to read parquet artifacts.\n")
}

cat("\n=== Persisted synthetic data ===\n")
dir.create(generated_dir, recursive = TRUE, showWarnings = FALSE)
run_cli(c(
  "generate-synthetic",
  "--output-dir", generated_dir,
  "--n", "600",
  "--seed", "2026"
))
cat("Generated data at:", generated_dir, "\n")
print(list.files(generated_dir))

cat("\nDone. Results are under", results_dir, "\n")

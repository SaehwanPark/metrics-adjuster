# Quickstart

## Install

```bash
python -m pip install metrics-adjuster
```

or:

```bash
uv add metrics-adjuster
```

## Run the Demo

```bash
metrics-adjuster demo --output-dir demo_outputs --report
```

Expected outputs include synthetic data, adjusted metric CSVs, and
`report.html`.

## Run on Your Own Data

```bash
metrics-adjuster run \
  --input cohort.csv \
  --output-dir adjusted_metric_outputs \
  --group-col group \
  --ref-group ref \
  --response-col outcome \
  --risk-col risk \
  --quantiles 0.2,0.4,0.6,0.8 \
  --metrics aTPR,aPPV,aNB,aHR \
  --report
```

Input data must include a group column, binary response column, predicted risk
column, and the selected reference group.

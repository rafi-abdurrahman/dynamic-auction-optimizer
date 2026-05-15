#!/bin/bash
#
# Download EUROSTAT livestock production datasets
#
# Expected datasets:
#   apro_mt_pwgtm.tsv   — slaughtering in slaughterhouses
#   apro_mt_pslothm.tsv — slaughtering outside slaughterhouses
#   apro_ec_poulm.tsv   — poultry & egg production
#   apro_mk_colm.tsv    — milk collection & dairy products
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/raw"
EUROSTAT_BASE="https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"

# Create raw directory if it doesn't exist
mkdir -p "${RAW_DIR}"

# List of datasets to download
DATASETS=(
    "apro_mt_pwgtm"
    "apro_mt_pslothm"
    "apro_ec_poulm"
    "apro_mk_colm"
)

echo "Downloading EUROSTAT datasets to ${RAW_DIR}/"
echo ""

for dataset in "${DATASETS[@]}"; do
    url="${EUROSTAT_BASE}/${dataset}?format=TSV"
    output="${RAW_DIR}/${dataset}.tsv"

    if [ -f "${output}" ]; then
        echo "✓ ${dataset}.tsv already exists"
    else
        echo "⬇ Downloading ${dataset}.tsv..."
        if curl -f -s -o "${output}" "${url}"; then
            echo "✓ Downloaded ${dataset}.tsv"
        else
            echo "✗ Failed to download ${dataset}.tsv"
            echo "  URL: ${url}"
            rm -f "${output}"
            exit 1
        fi
    fi
done

echo ""
echo "✓ All datasets ready in ${RAW_DIR}/"

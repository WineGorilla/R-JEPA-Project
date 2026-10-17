#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/mi2-interns/ruiyu/R-JEPA"
QA_DIR="$ROOT/clevrer_qa"

cd "$QA_DIR"

echo "============================================================"
echo "1/7 Extract RAW train features"
echo "============================================================"
python extract_features.py --representation raw --split train

echo "============================================================"
echo "2/7 Extract RAW validation features"
echo "============================================================"
python extract_features.py --representation raw --split validation

echo "============================================================"
echo "3/7 Extract WORLD train features"
echo "============================================================"
python extract_features.py --representation world --split train

echo "============================================================"
echo "4/7 Extract WORLD validation features"
echo "============================================================"
python extract_features.py --representation world --split validation

echo "============================================================"
echo "5/7 Train RAW QA probe"
echo "============================================================"
python train_qa.py --representation raw

echo "============================================================"
echo "6/7 Train WORLD QA probe"
echo "============================================================"
python train_qa.py --representation world

echo "============================================================"
echo "7/7 Final comparison"
echo "============================================================"
python evaluate_qa.py

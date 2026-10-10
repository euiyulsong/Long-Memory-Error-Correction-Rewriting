python3 bench.py --n 100 \
  --model gpt-6-luna --effort none \
  --workers 4 --out results_keyword

python3 bench_clean_term.py \
  --out results_keyword --workers 4

python3 bench_clean_correction_fixed.py \
  --out results_keyword --workers 4

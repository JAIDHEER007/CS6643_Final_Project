#!/bin/bash

# -----------------------------------------
# USAGE:
#   ./run_tests.sh [mode]
# mode: seq | pthreads | cuda | all
# -----------------------------------------

MODE=${1:-all}

echo "MODE = $MODE"
echo ""

# -----------------------------------------
# DATASETS + OPTIMAL TOURS
# FORMAT: "dataset_path:optimal_tour_path"
# If optimal_tour_path == none → binary receives "none"
# -----------------------------------------

DATASETS=(
    "datasets/xqf131.tsp:optimal_tours/xqf131.opt.tour"
    "datasets/pka379.tsp:optimal_tours/pka379.opt.tour"
    "datasets/pbk411.tsp:optimal_tours/pbk411.opt.tour"
    "datasets/rbx711.tsp:optimal_tours/rbx711.opt.tour"
    "datasets/xit1083.tsp:optimal_tours/xit1083.opt.tour"
)

# -----------------------------------------
# POPULATION sizes
# -----------------------------------------
POPS=(1024 2048 4096 8192)

# -----------------------------------------
# GENERATIONS
# -----------------------------------------
GENS=(100 200 400 600 800)

# -----------------------------------------
# THREAD counts for pthreads
# -----------------------------------------
THREADS=(2 4)

# -----------------------------------------
# SOURCE FILES
# -----------------------------------------
SEQ_SRC="ga_seq.cpp"
PTH_SRC="ga_pthreads.cpp"
CUDA_SRC="ga_cuda.cu"

SEQ_BIN="ga_seq"
PTH_BIN="ga_pthreads"
CUDA_BIN="ga_cuda"

# -----------------------------------------
# COMPILATION
# -----------------------------------------

compile_seq() {
    echo "Compiling Sequential..."
    g++ $SEQ_SRC -O2 -std=c++17 -o $SEQ_BIN
}

compile_pthreads() {
    echo "Compiling Pthreads..."
    g++ $PTH_SRC -O2 -std=c++17 -lpthread -o $PTH_BIN
}

compile_cuda() {
    echo "Compiling CUDA..."
    nvcc $CUDA_SRC -O2 -o $CUDA_BIN
}

echo "========== COMPILATION =========="
case "$MODE" in
    seq) compile_seq ;;
    pthreads) compile_pthreads ;;
    cuda) compile_cuda ;;
    all)
        compile_seq
        compile_pthreads
        compile_cuda ;;
    *)
        echo "Invalid mode: $MODE"
        exit 1 ;;
esac
echo ""

# -----------------------------------------
# TEST EXECUTION
# -----------------------------------------

echo "========== TESTING =========="

for item in "${DATASETS[@]}"; do

    dataset_file=$(echo "$item" | cut -d':' -f1)
    optimal_file=$(echo "$item" | cut -d':' -f2)

    for pop in "${POPS[@]}"; do
        for gen in "${GENS[@]}"; do

            # ------- SEQ -------
            if [[ "$MODE" == "seq" || "$MODE" == "all" ]]; then
                echo "SEQ → $dataset_file POP=$pop GEN=$gen OPT=$optimal_file"
                ./$SEQ_BIN "$dataset_file" $pop $gen "$optimal_file"
            fi

            # ------- PTHREADS -------
            if [[ "$MODE" == "pthreads" || "$MODE" == "all" ]]; then
                for t in "${THREADS[@]}"; do
                    echo "PTHREADS → $dataset_file POP=$pop GEN=$gen THREADS=$t OPT=$optimal_file"
                    ./$PTH_BIN "$dataset_file" $pop $gen $t "$optimal_file"
                done
            fi

            # ------- CUDA -------
            if [[ "$MODE" == "cuda" || "$MODE" == "all" ]]; then
                echo "CUDA → $dataset_file POP=$pop GEN=$gen OPT=$optimal_file"
                ./$CUDA_BIN "$dataset_file" $pop $gen "$optimal_file"
            fi

        done
    done
done

echo "========== DONE =========="

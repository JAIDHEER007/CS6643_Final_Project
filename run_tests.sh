#!/bin/bash

# -----------------------------------------
# SELECT MODE
# usage: ./run_tests.sh [seq|pthreads|cuda|all]
# default = all
# -----------------------------------------
MODE=${1:-all}

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
# DATASETS (inside datasets/ folder)
# -----------------------------------------
DATASETS=(
    "datasets/pka379.tsp"
    "datasets/rbx711.tsp"
    "datasets/xit1083.tsp"
    "datasets/pbk411.tsp"
    "datasets/xqf131.tsp"
    "datasets/xql662.tsp"
)

# -----------------------------------------
# POPULATION SIZES
# -----------------------------------------
POPS=(1024 2048 4096 8192 16384)

# -----------------------------------------
# GENERATION COUNTS
# -----------------------------------------
GENS=(100 200 400)

# -----------------------------------------
# THREADS for pthreads
# -----------------------------------------
THREADS=(2 4)

# -----------------------------------------
# COMPILATION LOGIC
# -----------------------------------------
compile_seq() {
    echo "Compiling Sequential..."
    g++ $SEQ_SRC -O2 -std=c++17 -o $SEQ_BIN
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Failed to compile $SEQ_SRC"
        exit 1
    fi
}

compile_pthreads() {
    echo "Compiling Pthreads..."
    g++ $PTH_SRC -O2 -std=c++17 -lpthread -o $PTH_BIN
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Failed to compile $PTH_SRC"
        exit 1
    fi
}

compile_cuda() {
    echo "Compiling CUDA..."
    nvcc $CUDA_SRC -O2 -o $CUDA_BIN
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Failed to compile $CUDA_SRC"
        exit 1
    fi
}

# -----------------------------------------
# COMPILE BASED ON MODE
# -----------------------------------------
echo ""
echo "========== COMPILATION PHASE =========="

case "$MODE" in
    seq)
        compile_seq
        ;;
    pthreads)
        compile_pthreads
        ;;
    cuda)
        compile_cuda
        ;;
    all)
        compile_seq
        compile_pthreads
        compile_cuda
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Use: seq | pthreads | cuda | all"
        exit 1
        ;;
esac

echo "Compilation complete."
echo ""

# -----------------------------------------
# TEST EXECUTION
# -----------------------------------------
echo "========== TESTING PHASE (MODE = $MODE) =========="
echo ""

for dataset in "${DATASETS[@]}"; do
    for pop in "${POPS[@]}"; do
        for gen in "${GENS[@]}"; do

            # ------------------------
            # SEQUENTIAL
            # ------------------------
            if [[ "$MODE" == "seq" || "$MODE" == "all" ]]; then
                echo "SEQ => $dataset POP=$pop GEN=$gen"
                ./$SEQ_BIN "$dataset" $pop $gen
            fi

            # ------------------------
            # PTHREADS
            # ------------------------
            if [[ "$MODE" == "pthreads" || "$MODE" == "all" ]]; then
                for t in "${THREADS[@]}"; do
                    echo "PTHREADS => $dataset POP=$pop GEN=$gen THREADS=$t"
                    ./$PTH_BIN "$dataset" $pop $gen $t
                done
            fi

            # ------------------------
            # CUDA
            # ------------------------
            if [[ "$MODE" == "cuda" || "$MODE" == "all" ]]; then
                echo "CUDA => $dataset POP=$pop GEN=$gen"
                ./$CUDA_BIN "$dataset" $pop $gen
            fi

        done
    done
done

echo ""
echo "========== ALL TESTS COMPLETED (MODE = $MODE) =========="

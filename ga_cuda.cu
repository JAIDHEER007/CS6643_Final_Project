// ============================================================================
// OPTIMIZED CUDA Genetic Algorithm for TSPLIB
// 
// Key Optimizations:
// 1. Minimize CPU-GPU transfers (only at start/end)
// 2. GPU-based ranking using thrust
// 3. GPU 2-opt local search on elites
// 4. Better memory coalescing
// 5. Optimized kernel launches
//
// Compile: nvcc -O3 -std=c++17 ga_cuda.cu -o ga_cuda
// Run:     ./ga_cuda dataset.tsp POP GEN optimal.tour
// ============================================================================

#include <bits/stdc++.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
using namespace std;

const double CROSS_RATE = 0.9;
const double MUT_RATE   = 0.10;
const double ELITE_RATE = 0.05;

// ============================================================================
// TSPLIB LOADER
// ============================================================================
struct TSPLIB {
    vector<pair<double,double>> coords;
    int N;
};

TSPLIB loadTSPLIB(const string& file) {
    ifstream fin(file);
    if (!fin) { cerr << "Cannot open " << file << endl; exit(1); }

    TSPLIB T;
    T.N = -1;
    string line;

    while (getline(fin,line)) {
        if (line.rfind("DIMENSION",0)==0) {
            string tmp; stringstream ss(line);
            ss >> tmp >> tmp >> T.N;
        }
        if (line.find("NODE_COORD_SECTION")!=string::npos) break;
    }
    if (T.N <= 0) { cerr<<"DIMENSION missing\n"; exit(1); }

    T.coords.reserve(T.N);
    while (getline(fin,line)) {
        if (line.find("EOF")!=string::npos) break;
        int id; double x,y;
        stringstream ss(line);
        if (ss >> id >> x >> y) {
            T.coords.push_back({x,y});
        }
    }
    return T;
}

// ============================================================================
// EUC_2D Distance
// ============================================================================
inline int euc2d(double x1,double y1,double x2,double y2) {
    double dx=x1-x2, dy=y1-y2;
    return int(sqrt(dx*dx + dy*dy) + 0.5);
}

vector<int> buildDist(const TSPLIB& T) {
    int N=T.N;
    vector<int> D(N*N);
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            D[i*N+j] = euc2d(T.coords[i].first, T.coords[i].second,
                             T.coords[j].first, T.coords[j].second);
    return D;
}

// ============================================================================
// GPU RNG INITIALIZATION
// ============================================================================
__global__ void initRNG(curandState* states, int POP, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < POP) curand_init(seed, id, 0, &states[id]);
}

// ============================================================================
// GPU FITNESS (Optimized with better memory access)
// ============================================================================
__global__ void gpuFitness(const int* __restrict__ pop, 
                           const int* __restrict__ dist, 
                           int* __restrict__ fit, 
                           int POP, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= POP) return;

    int base = id * N;
    int sum = 0;

    // Unroll last iteration to avoid modulo
    for (int i = 0; i < N-1; i++) {
        int a = pop[base + i];
        int b = pop[base + i + 1];
        sum += dist[a * N + b];
    }
    // Close the tour
    sum += dist[pop[base + N - 1] * N + pop[base]];

    fit[id] = sum;
}

// ============================================================================
// GPU Tournament Selection
// ============================================================================
__device__ inline int gpuTournament(const int* fit, int POP, curandState& st) {
    int a = curand(&st) % POP;
    int b = curand(&st) % POP;
    return (fit[a] < fit[b] ? a : b);
}

// ============================================================================
// GPU OX CROSSOVER
// ============================================================================
__device__ void gpuOX(const int* __restrict__ p1, 
                      const int* __restrict__ p2,
                      int* __restrict__ child, 
                      unsigned char* used,
                      curandState& st, int N) {
    int a = curand(&st) % N;
    int b = curand(&st) % N;
    if (a > b) { int t=a; a=b; b=t; }

    // Initialize used array
    for (int i = 0; i < N; i++) used[i] = 0;

    // Copy segment from p1
    for (int i = a; i <= b; i++) {
        child[i] = p1[i];
        used[p1[i]] = 1;
    }

    // Fill remaining from p2
    int fill = (b + 1) % N;
    int cur  = (b + 1) % N;

    for (int k = 0; k < N; k++) {
        int gene = p2[cur];
        if (!used[gene]) {
            child[fill] = gene;
            used[gene] = 1;
            fill = (fill + 1) % N;
        }
        cur = (cur + 1) % N;
    }
}

// ============================================================================
// GPU Swap Mutation
// ============================================================================
__device__ inline void gpuMutate(int* c, curandState& st, int N) {
    if (curand_uniform(&st) < MUT_RATE) {
        int a = curand(&st) % N;
        int b = curand(&st) % N;
        int t = c[a];
        c[a] = c[b];
        c[b] = t;
    }
}

// ============================================================================
// GPU 2-OPT Local Search (Parallel version)
// ============================================================================
__global__ void gpu2Opt(int* pop, const int* __restrict__ dist, 
                        const int* eliteIndices, int numElites, int N) {
    int eliteId = blockIdx.x;
    if (eliteId >= numElites) return;
    
    int idx = eliteIndices[eliteId];
    int* tour = &pop[idx * N];
    
    bool improved = true;
    int maxIter = 50; // Limit iterations to prevent infinite loops
    int iter = 0;
    
    while (improved && iter < maxIter) {
        improved = false;
        iter++;
        
        for (int i = 0; i < N - 2; i++) {
            int A = tour[i];
            int B = tour[i + 1];
            
            for (int j = i + 2; j < N; j++) {
                int C = tour[j];
                int D = tour[(j + 1) % N];
                
                int oldCost = dist[A * N + B] + dist[C * N + D];
                int newCost = dist[A * N + C] + dist[B * N + D];
                
                if (newCost < oldCost) {
                    // Reverse segment [i+1, j]
                    int left = i + 1;
                    int right = j;
                    while (left < right) {
                        int tmp = tour[left];
                        tour[left] = tour[right];
                        tour[right] = tmp;
                        left++;
                        right--;
                    }
                    improved = true;
                }
            }
        }
    }
}

// ============================================================================
// GPU Breed Kernel (Optimized)
// ============================================================================
__global__ void gpuBreed(const int* __restrict__ pop, 
                         int* __restrict__ newPop, 
                         const int* __restrict__ fit,
                         int POP, int N, int ELITES,
                         curandState* states) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= POP || id < ELITES) return;

    curandState st = states[id];

    int p1 = gpuTournament(fit, POP, st);
    int p2 = gpuTournament(fit, POP, st);

    const int* P1 = &pop[p1 * N];
    const int* P2 = &pop[p2 * N];
    int* child = &newPop[id * N];

    // Allocate local memory for used array
    unsigned char used[2048]; // Adjust size based on max N expected
    
    if (N > 2048) return; // Safety check

    // Crossover
    if (curand_uniform(&st) < CROSS_RATE) {
        gpuOX(P1, P2, child, used, st, N);
    } else {
        for (int i = 0; i < N; i++) child[i] = P1[i];
    }

    gpuMutate(child, st, N);

    states[id] = st;
}

// ============================================================================
// GPU Elite Copy Kernel
// ============================================================================
__global__ void copyElites(const int* __restrict__ pop, 
                           int* __restrict__ newPop,
                           const int* __restrict__ eliteIndices,
                           int numElites, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numElites) return;
    
    int srcIdx = eliteIndices[id];
    const int* src = &pop[srcIdx * N];
    int* dst = &newPop[id * N];
    
    for (int i = 0; i < N; i++) {
        dst[i] = src[i];
    }
}

// ============================================================================
// Load Optimal Tour
// ============================================================================
vector<int> loadOptimal(const string& opt, int N) {
    if (opt == "none") return {};

    ifstream fin(opt);
    if (!fin.is_open()) return {};

    vector<int> t;
    string line;
    bool start = false;

    auto trim = [&](string s) {
        s.erase(remove_if(s.begin(), s.end(),
            [](char c) { return isspace((unsigned char)c); }), s.end());
        return s;
    };

    while (getline(fin, line)) {
        if (line.find("TOUR_SECTION") != string::npos) {
            start = true;
            continue;
        }
        if (!start) continue;

        string s = trim(line);
        if (s == "" || s == "-1" || s == "EOF") break;

        bool ok = true;
        for (char c : s)
            if (!isdigit(c) && c != '-') ok = false;

        if (!ok) continue;

        int city = stoi(s);
        if (city >= 1 && city <= N)
            t.push_back(city - 1);
    }
    return t;
}

inline int tourLen(const vector<int>& t, int N, const vector<int>& dist) {
    int sum = 0;
    for (int i = 0; i < N - 1; i++)
        sum += dist[t[i] * N + t[i + 1]];
    sum += dist[t[N - 1] * N + t[0]];
    return sum;
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 5) {
        cout << "Usage: ./ga_cuda dataset.tsp POP GEN optimal_tour\n";
        return 0;
    }

    string dataset = argv[1];
    int POP = atoi(argv[2]);
    int GEN = atoi(argv[3]);
    string OPT = argv[4];

    // Load dataset
    TSPLIB D = loadTSPLIB(dataset);
    int N = D.N;
    vector<int> distCPU = buildDist(D);

    // Allocate GPU memory
    int *popGPU, *newGPU, *distGPU, *fitGPU;
    cudaMalloc(&popGPU, POP * N * sizeof(int));
    cudaMalloc(&newGPU, POP * N * sizeof(int));
    cudaMalloc(&distGPU, N * N * sizeof(int));
    cudaMalloc(&fitGPU, POP * sizeof(int));

    cudaMemcpy(distGPU, distCPU.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize population on CPU
    vector<int> pop(POP * N);
    vector<int> base(N);
    iota(base.begin(), base.end(), 0);

    mt19937 rng(12345);
    for (int i = 0; i < POP; i++) {
        shuffle(base.begin(), base.end(), rng);
        memcpy(&pop[i * N], base.data(), N * sizeof(int));
    }

    cudaMemcpy(popGPU, pop.data(), POP * N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize RNG states
    curandState* states;
    cudaMalloc(&states, POP * sizeof(curandState));
    int block = 256;
    int grid = (POP + block - 1) / block;
    initRNG<<<grid, block>>>(states, POP, 987654321ULL);

    int ELITES = max(1, int(POP * ELITE_RATE));

    // Thrust vectors for sorting
    thrust::device_vector<int> d_fit(POP);
    thrust::device_vector<int> d_indices(POP);

    auto start = chrono::steady_clock::now();

    // ======================================================================
    // GA LOOP (Minimized CPU-GPU transfers)
    // ======================================================================
    for (int g = 0; g < GEN; g++) {
        // 1) Compute fitness on GPU
        gpuFitness<<<grid, block>>>(popGPU, distGPU, fitGPU, POP, N);

        // 2) Sort on GPU using Thrust
        cudaMemcpy(thrust::raw_pointer_cast(d_fit.data()), 
                   fitGPU, POP * sizeof(int), cudaMemcpyDeviceToDevice);
        
        thrust::sequence(d_indices.begin(), d_indices.end());
        thrust::sort_by_key(d_fit.begin(), d_fit.end(), d_indices.begin());

        // 3) Copy elites to new population
        copyElites<<<(ELITES + block - 1) / block, block>>>(
            popGPU, newGPU, thrust::raw_pointer_cast(d_indices.data()), 
            ELITES, N
        );

        // 4) Apply 2-opt to elites on GPU
        if (N <= 1000) { // Only for reasonable N sizes
            gpu2Opt<<<ELITES, 1>>>(
                newGPU, distGPU, 
                thrust::raw_pointer_cast(d_indices.data()), 
                ELITES, N
            );
        }

        // 5) Breed remaining population
        gpuBreed<<<grid, block>>>(
            popGPU, newGPU, fitGPU, POP, N, ELITES, states
        );

        // Swap populations
        int* tmp = popGPU;
        popGPU = newGPU;
        newGPU = tmp;
    }

    cudaDeviceSynchronize();
    auto end = chrono::steady_clock::now();
    double time_sec = chrono::duration<double>(end - start).count();

    // ======================================================================
    // Final fitness evaluation
    // ======================================================================
    gpuFitness<<<grid, block>>>(popGPU, distGPU, fitGPU, POP, N);
    
    vector<int> fitCPU(POP);
    cudaMemcpy(fitCPU.data(), fitGPU, POP * sizeof(int), cudaMemcpyDeviceToHost);

    int bestIdx = min_element(fitCPU.begin(), fitCPU.end()) - fitCPU.begin();
    int ga_len = fitCPU[bestIdx];

    // Load optimal tour
    vector<int> opt = loadOptimal(OPT, N);
    int opt_len = -1;
    if (!opt.empty())
        opt_len = tourLen(opt, N, distCPU);

    double err = -1;
    if (opt_len > 0)
        err = (ga_len - opt_len) * 100.0 / opt_len;

    // Write results
    bool header = false;
    {
        ifstream f("cuda_results.csv");
        if (!f.good()) header = true;
    }

    ofstream fout("cuda_results.csv", ios::app);
    if (header)
        fout << "dataset,pop,gen,time_sec,optimal_len,ga_len,error_percent\n";

    fout << dataset << "," << POP << "," << GEN << ","
         << time_sec << "," << opt_len << "," << ga_len
         << "," << err << "\n";

    cout << "\n=== CUDA VALIDATION ===\n";
    cout << "Optimal length : " << opt_len << "\n";
    cout << "GA best length : " << ga_len << "\n";
    cout << "Error (%)      : " << err << "\n";
    cout << "Time (sec)     : " << time_sec << "\n";

    // Cleanup
    cudaFree(popGPU);
    cudaFree(newGPU);
    cudaFree(distGPU);
    cudaFree(fitGPU);
    cudaFree(states);

    return 0;
}
// ============================================================================
// CUDA Genetic Algorithm for TSPLIB (Stable Version)
// Matches ga_seq.cpp exactly except fitness+breeding done on GPU
//
// Compile: nvcc -O2 -std=c++17 ga_cuda.cu -o ga_cuda
// Run:     ./ga_cuda dataset.tsp POP GEN optimal.tour
// ============================================================================

#include <bits/stdc++.h>
#include <curand_kernel.h>
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
__global__ void initRNG(curandState* st, int POP, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < POP) curand_init(seed, id, 0, &st[id]);
}

// ============================================================================
// GPU FITNESS
// ============================================================================
__global__ void gpuFitness(int* pop, int* dist, int* fit, int POP, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= POP) return;

    int base = id * N;
    int sum = 0;

    for (int i=0;i<N-1;i++) {
        int a = pop[base+i];
        int b = pop[base+i+1];
        sum += dist[a*N+b];
    }
    sum += dist[ pop[base+N-1]*N + pop[base] ];

    fit[id] = sum;
}

// ============================================================================
// GPU Tournament Selection
// ============================================================================
__device__ int gpuTournament(const int* fit, int POP, curandState& st) {
    int a = curand(&st) % POP;
    int b = curand(&st) % POP;
    return (fit[a] < fit[b] ? a : b);
}

// ============================================================================
// GPU OX CROSSOVER (No shared memory. Safe for all N)
// ============================================================================
__device__ void gpuOX(
    const int* p1, const int* p2,
    int* child, unsigned char* used,
    curandState& st, int N)
{
    int a = curand(&st) % N;
    int b = curand(&st) % N;
    if (a > b) { int t=a; a=b; b=t; }

    // Clear used[]
    for (int i=0;i<N;i++) used[i] = 0;

    // Copy slice
    for (int i=a;i<=b;i++) {
        int g = p1[i];
        child[i] = g;
        used[g] = 1;
    }

    // Fill remaining from p2 (OX logic)
    int fill = (b+1) % N;
    int cur  = (b+1) % N;

    for (int k=0;k<N;k++) {
        int g = p2[cur];
        if (!used[g]) {
            child[fill] = g;
            used[g] = 1;
            fill = (fill + 1) % N;
        }
        cur = (cur + 1) % N;
    }
}

__global__ void initRand(curandState *states, int POP, unsigned long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < POP) {
        curand_init(seed, id, 0, &states[id]);
    }
}

// ============================================================================
// GPU Swap Mutation
// ============================================================================
__device__ void gpuMutate(int* c, curandState& st, int N) {
    if (curand_uniform(&st) < MUT_RATE) {
        int a = curand(&st) % N;
        int b = curand(&st) % N;
        int t = c[a];
        c[a] = c[b];
        c[b] = t;
    }
}

// ============================================================================
// GPU Breed Kernel (NO SHARED MEMORY — Always works!)
// Each thread produces exactly 1 offspring (except elites)
// ============================================================================
__global__ void gpuBreed(
    int* pop, int* newPop, int* fit,
    int POP, int N, int ELITES,
    curandState* states)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= POP) return;

    // Elites computed by CPU
    if (id < ELITES) return;

    curandState st = states[id];

    int p1 = gpuTournament(fit, POP, st);
    int p2 = gpuTournament(fit, POP, st);

    const int* P1 = &pop[p1*N];
    const int* P2 = &pop[p2*N];
    int* child    = &newPop[id*N];

    // Local used[] — allocated per thread
    extern __shared__ unsigned char shared_s[];
    unsigned char* used = shared_s + (threadIdx.x * N);

    // CROSSOVER
    if (curand_uniform(&st) < CROSS_RATE) {
        gpuOX(P1, P2, child, used, st, N);
    } 
    else {
        for (int i=0;i<N;i++) child[i] = P1[i];
    }

    gpuMutate(child, st, N);

    states[id] = st;
}

// ============================================================================
// CPU 2-OPT (same as sequential version)
// ============================================================================
void two_opt(vector<int>& t, int N, const vector<int>& dist) {
    bool improved=true;
    while (improved) {
        improved=false;

        for (int i=0;i<N-2;i++) {
            int A=t[i], B=t[i+1];

            for (int j=i+2;j<N;j++) {
                int C=t[j];
                int D=t[(j+1)%N];

                int oldCost = dist[A*N+B] + dist[C*N+D];
                int newCost = dist[A*N+C] + dist[B*N+D];

                if (newCost < oldCost) {
                    reverse(t.begin()+i+1, t.begin()+j+1);
                    improved = true;
                }
            }
        }
    }
}

// ============================================================================
// Load Optimal Tour
// ============================================================================
vector<int> loadOptimal(const string& opt, int N) {
    if (opt=="none") return {};

    ifstream fin(opt);
    if (!fin.is_open()) return {};

    vector<int> t;
    string line;
    bool start=false;

    auto trim=[&](string s){
        s.erase(remove_if(s.begin(),s.end(),
            [](char c){return isspace((unsigned char)c);}), s.end());
        return s;
    };

    while (getline(fin,line)) {
        if (line.find("TOUR_SECTION")!=string::npos) {
            start = true;
            continue;
        }
        if (!start) continue;

        string s = trim(line);
        if (s=="" || s=="-1" || s=="EOF") break;

        bool ok=true;
        for (char c : s)
            if (!isdigit(c) && c!='-') ok=false;

        if (!ok) continue;

        int city = stoi(s);
        if (city>=1 && city<=N)
            t.push_back(city-1);
    }
    return t;
}

inline int tourLen(const vector<int>& t, int N, const vector<int>& dist) {
    int sum=0;
    for (int i=0;i<N-1;i++)
        sum += dist[t[i]*N + t[i+1]];
    sum += dist[t[N-1]*N + t[0]];
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

    // ------------------------------------------------------------------------
    // LOAD DATASET
    // ------------------------------------------------------------------------
    TSPLIB D = loadTSPLIB(dataset);
    int N = D.N;

    vector<int> distCPU = buildDist(D);

    // ------------------------------------------------------------------------
    // ALLOC GPU MEMORY
    // ------------------------------------------------------------------------
    int *popGPU, *newGPU, *distGPU, *fitGPU;
    cudaMalloc(&popGPU, POP * N * sizeof(int));
    cudaMalloc(&newGPU, POP * N * sizeof(int));
    cudaMalloc(&distGPU, N * N * sizeof(int));
    cudaMalloc(&fitGPU, POP * sizeof(int));

    cudaMemcpy(distGPU, distCPU.data(), N*N*sizeof(int), cudaMemcpyHostToDevice);

    // ------------------------------------------------------------------------
    // INITIAL HOST POPULATION
    // ------------------------------------------------------------------------
    vector<int> pop(POP*N), newPop(POP*N), fitCPU(POP);

    vector<int> base(N);
    iota(base.begin(), base.end(), 0);

    mt19937 rng(12345);
    for (int i=0;i<POP;i++) {
        shuffle(base.begin(), base.end(), rng);
        memcpy(&pop[i*N], base.data(), N*sizeof(int));
    }

    cudaMemcpy(popGPU, pop.data(), POP*N*sizeof(int), cudaMemcpyHostToDevice);

    // ------------------------------------------------------------------------
    // RNG states
    // ------------------------------------------------------------------------
    curandState* states;
    cudaMalloc(&states, POP*sizeof(curandState));
    initRand<<<(POP+255)/256, 256>>>(states, POP, 987654321ULL);

    int ELITES = max(1, int(POP * ELITE_RATE));

    auto start = chrono::steady_clock::now();

    // ======================================================================
    // GA LOOP
    // ======================================================================
    for (int g=0; g<GEN; g++) {

        // --------------------------------------------------------------
        // 1) GPU FITNESS
        // --------------------------------------------------------------
        gpuFitness<<<(POP+255)/256, 256>>>(popGPU, distGPU, fitGPU, POP, N);
        cudaMemcpy(fitCPU.data(), fitGPU, POP*sizeof(int), cudaMemcpyDeviceToHost);

        // --------------------------------------------------------------
        // 2) RANK ON CPU
        // --------------------------------------------------------------
        vector<pair<int,int>> ranked(POP);
        for (int i=0;i<POP;i++)
            ranked[i] = {fitCPU[i], i};
        sort(ranked.begin(), ranked.end());

        // bring current pop to host
        cudaMemcpy(pop.data(), popGPU, POP*N*sizeof(int), cudaMemcpyDeviceToHost);

        // --------------------------------------------------------------
        // 3) ELITES ON CPU (with 2-opt)
        // --------------------------------------------------------------
        for (int e=0; e<ELITES; e++) {
            int idx = ranked[e].second;

            vector<int> elite(N);
            memcpy(elite.data(), &pop[idx*N], N*sizeof(int));

            two_opt(elite, N, distCPU);

            memcpy(&newPop[e*N], elite.data(), N*sizeof(int));
        }

        // push elites to GPU side of newPop
        cudaMemcpy(newGPU, newPop.data(), ELITES*N*sizeof(int), cudaMemcpyHostToDevice);

        // --------------------------------------------------------------
        // 4) GPU BREED FOR REMAINING
        // --------------------------------------------------------------
        int block = 128;
        int grid  = (POP + block - 1)/block;

        // per-thread used[] of N bytes
        size_t shmem = block * N * sizeof(unsigned char);

        gpuBreed<<<grid, block, shmem>>>(popGPU, newGPU, fitGPU,
                                         POP, N, ELITES, states);

        // swap (popGPU <-> newGPU)
        int* tmp = popGPU;
        popGPU   = newGPU;
        newGPU   = tmp;
    }

    auto end = chrono::steady_clock::now();
    double time_sec = chrono::duration<double>(end-start).count();

    // ======================================================================
    // FINAL BEST
    // ======================================================================
    cudaMemcpy(fitCPU.data(), fitGPU, POP*sizeof(int), cudaMemcpyDeviceToHost);

    int bestIdx = min_element(fitCPU.begin(),fitCPU.end()) - fitCPU.begin();
    int ga_len  = fitCPU[bestIdx];

    // ======================================================================
    // OPTIMAL TOUR
    // ======================================================================
    vector<int> opt = loadOptimal(OPT, N);
    int opt_len = -1;

    if (!opt.empty())
        opt_len = tourLen(opt, N, distCPU);

    double err = -1;
    if (opt_len > 0)
        err = (ga_len - opt_len) * 100.0 / opt_len;

    // ======================================================================
    // WRITE CSV
    // ======================================================================
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

    // ======================================================================
    // PRINT SUMMARY
    // ======================================================================
    cout << "\n=== CUDA VALIDATION ===\n";
    cout << "Optimal length : " << opt_len << "\n";
    cout << "GA best length : " << ga_len << "\n";
    cout << "Error (%)      : " << err << "\n";
    cout << "Time (sec)     : " << time_sec << "\n";

    return 0;
}

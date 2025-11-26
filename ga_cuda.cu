// ============================================================================
// FULL GPU GA FOR TSP (Version A - Baseline GPU GA)
// Author: ChatGPT
// GA on GPU: tournament, OX crossover, swap mutation
// CPU: elite sorting, copying elites
// ============================================================================

#include <bits/stdc++.h>
#include <curand.h>
#include <curand_kernel.h>
using namespace std;

// GA Parameters
#define CROSS_RATE 0.9
#define MUT_RATE   0.10
#define ELITE_RATE 0.05

// ============================================================================
// TSPLIB Parsing
// ============================================================================
struct TSPLIB {
    int N;
    vector<double> xs, ys;
};

TSPLIB loadTSPLIB(const string &fname){
    ifstream fin(fname);
    if(!fin.is_open()){
        cerr << "ERROR opening " << fname << endl;
        exit(1);
    }
    TSPLIB D;
    D.N = -1;

    string line;
    while(getline(fin,line)){
        if(line.rfind("DIMENSION",0)==0){
            string tmp;
            stringstream ss(line);
            ss >> tmp >> tmp >> D.N;
        }
        if(line.find("NODE_COORD_SECTION")!=string::npos) break;
    }
    if(D.N<=0){ cerr<<"Invalid DIMENSION\n"; exit(1); }

    D.xs.resize(D.N);
    D.ys.resize(D.N);

    int id; double x,y;
    for(int i=0;i<D.N;i++){
        fin >> id >> x >> y;
        D.xs[i] = x;
        D.ys[i] = y;
    }
    return D;
}

// ============================================================================
// Distance matrix (CPU)
// ============================================================================
__host__ __device__ inline int euc2d(double x1,double y1,double x2,double y2){
    double dx=x1-x2, dy=y1-y2;
    return int(sqrt(dx*dx+dy*dy)+0.5);
}

vector<int> buildDistCPU(const TSPLIB &D){
    int N=D.N;
    vector<int> dist(N*N);
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            dist[i*N+j] = euc2d(D.xs[i],D.ys[i],D.xs[j],D.ys[j]);
    return dist;
}

// ============================================================================
// RNG Kernel
// ============================================================================
__global__ void initRNG(curandState *s, int total, unsigned long seed){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id<total)
        curand_init(seed,id,0,&s[id]);
}

// ============================================================================
// Fitness Kernel
// ============================================================================
__global__ void kernel_fitness(int *pop, int *dist, int *fit, int POP, int N){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=POP) return;

    int base=id*N;
    int sum=0;

    for(int i=0;i<N-1;i++){
        int a=pop[base+i], b=pop[base+i+1];
        sum+=dist[a*N+b];
    }
    sum+=dist[ pop[base+N-1]*N + pop[base] ];

    fit[id]=sum;
}

// ============================================================================
// Tournament selection
// ============================================================================
__device__ int tournament(int *fit, int POP, curandState &r){
    int a=curand(&r)%POP;
    int b=curand(&r)%POP;
    return (fit[a]<fit[b] ? a : b);
}

// ============================================================================
// OX Crossover
// ============================================================================
__device__ void oxCrossover(const int *p1, const int *p2, int *child,
                            unsigned char *used, int N, curandState &r)
{
    int a=curand(&r)%N;
    int b=curand(&r)%N;
    if(a>b){int t=a; a=b; b=t;}

    for(int i=0;i<N;i++){
        used[i]=0;
        child[i]=-1;
    }

    for(int i=a;i<=b;i++){
        int g=p1[i];
        child[i]=g;
        used[g]=1;
    }

    int fill=(b+1)%N, cur=(b+1)%N;

    for(int k=0;k<N;k++){
        int g=p2[cur];
        if(!used[g]){
            child[fill] = g;
            used[g] = 1;
            fill = (fill+1)%N;
        }
        cur = (cur+1)%N;
    }
}

// ============================================================================
// Mutation
// ============================================================================
__device__ void mutate(int *c, int N, curandState &r){
    if(curand_uniform(&r) < MUT_RATE){
        int a=curand(&r)%N;
        int b=curand(&r)%N;
        int t=c[a];
        c[a]=c[b];
        c[b]=t;
    }
}

// ============================================================================
// Breeding Kernel
// ============================================================================
__global__ void kernel_breed(int *pop, int *newPop, int *fit,
                             curandState *rng, int POP, int N, int ELITES)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=POP) return;

    if(id < ELITES){
        return;
    }

    curandState local = rng[id];

    int p1 = tournament(fit, POP, local);
    int p2 = tournament(fit, POP, local);

    const int *P1 = &pop[p1*N];
    const int *P2 = &pop[p2*N];
    int *child = &newPop[id*N];

    extern __shared__ unsigned char used[];

    if(curand_uniform(&local) < CROSS_RATE)
        oxCrossover(P1, P2, child, used, N, local);
    else {
        for(int i=0;i<N;i++)
            child[i] = P1[i];
    }

    mutate(child, N, local);

    rng[id] = local;
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc,char**argv){
    if(argc<5){
        cerr<<"Usage: ./ga_cuda dataset.tsp POP GEN optimal_tour\n";
        return 0;
    }

    string fname = argv[1];
    int POP = atoi(argv[2]);
    int GEN = atoi(argv[3]);
    string OPT = argv[4];

    // ------------------------------
    // Load TSP instance
    // ------------------------------
    TSPLIB D = loadTSPLIB(fname);
    int N = D.N;

    // Build CPU distance matrix
    vector<int> distCPU = buildDistCPU(D);

    // ------------------------------
    // Allocate GPU buffers
    // ------------------------------
    int *distGPU, *popGPU, *newGPU, *fitGPU;
    cudaMalloc(&distGPU, N*N*sizeof(int));
    cudaMalloc(&popGPU,  POP*N*sizeof(int));
    cudaMalloc(&newGPU,  POP*N*sizeof(int));
    cudaMalloc(&fitGPU,  POP*sizeof(int));

    cudaMemcpy(distGPU, distCPU.data(), N*N*sizeof(int), cudaMemcpyHostToDevice);

    // ------------------------------
    // Initialize population (CPU)
    // ------------------------------
    vector<int> pop(POP*N), newPop(POP*N), fitCPU(POP);
    vector<int> base(N);
    iota(base.begin(), base.end(), 0);

    mt19937 rng(12345);

    for(int i=0;i<POP;i++){
        shuffle(base.begin(), base.end(), rng);
        memcpy(&pop[i*N], base.data(), N*sizeof(int));
    }

    cudaMemcpy(popGPU, pop.data(), POP*N*sizeof(int), cudaMemcpyHostToDevice);

    // ------------------------------
    // RNG for each thread
    // ------------------------------
    curandState *randState;
    cudaMalloc(&randState, POP*sizeof(curandState));
    initRNG<<<(POP+255)/256,256>>>(randState, POP, 987654321ULL);

    int ELITES = max(1, int(POP * ELITE_RATE));

    // ------------------------------
    // START TIMER
    // ------------------------------
    auto start = chrono::steady_clock::now();

    // ============================================================================
    // GA LOOP
    // ============================================================================
    for(int g=0; g<GEN; g++){
        // 1) Compute fitness on GPU
        kernel_fitness<<<(POP+255)/256,256>>>(popGPU, distGPU, fitGPU, POP, N);
        cudaMemcpy(fitCPU.data(), fitGPU, POP*sizeof(int), cudaMemcpyDeviceToHost);

        // 2) Rank selection for elites (CPU)
        vector<pair<int,int>> rank(POP);
        for(int i=0;i<POP;i++)
            rank[i] = {fitCPU[i], i};
        sort(rank.begin(), rank.end());

        // 3) Copy population to CPU to extract elites
        cudaMemcpy(pop.data(), popGPU, POP*N*sizeof(int), cudaMemcpyDeviceToHost);

        // 4) Insert top elites into newPop
        for(int e=0; e<ELITES; e++){
            int idx = rank[e].second;
            memcpy(&newPop[e*N], &pop[idx*N], N*sizeof(int));
        }

        // Copy *only* the elites into GPU newPop
        cudaMemcpy(newGPU, newPop.data(), POP*N*sizeof(int), cudaMemcpyHostToDevice);

        // 5) Breed the rest on GPU
        int block = 256;
        int grid = (POP + block - 1) / block;
        size_t shmem = N * sizeof(unsigned char);

        kernel_breed<<<grid, block, shmem>>>(popGPU, newGPU, fitGPU,
                                             randState, POP, N, ELITES);

        // 6) Swap populations
        int *tmp = popGPU;
        popGPU = newGPU;
        newGPU = tmp;
    }

    // ------------------------------
    // END TIMER
    // ------------------------------
    auto end = chrono::steady_clock::now();
    double T = chrono::duration<double>(end-start).count();

    // ------------------------------
    // Compute final fitness
    // ------------------------------
    kernel_fitness<<<(POP+255)/256,256>>>(popGPU, distGPU, fitGPU, POP, N);
    cudaMemcpy(fitCPU.data(), fitGPU, POP*sizeof(int), cudaMemcpyDeviceToHost);

    int best = *min_element(fitCPU.begin(), fitCPU.end());

    // ------------------------------
    // Print result
    // ------------------------------
    cout << "BEST = " << best
         << "   Time = " << T << " sec\n";

    // ------------------------------
    // Append to CSV
    // ------------------------------
    bool hdr = false;
    {
        ifstream f("cuda_results.csv");
        hdr = !f.good();
    }

    ofstream out("cuda_results.csv", ios::app);
    if(hdr)
        out << "dataset,pop,gen,time_sec,ga_len\n";

    out << fname << "," << POP << "," << GEN << "," << T << "," << best << "\n";
    out.close();

    // ------------------------------
    // Cleanup GPU
    // ------------------------------
    cudaFree(distGPU);
    cudaFree(popGPU);
    cudaFree(newGPU);
    cudaFree(fitGPU);
    cudaFree(randState);

    return 0;
}

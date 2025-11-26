// GA CUDA VERSION – Matches Sequential Version With 2-opt on CPU
// Compile: nvcc -O2 -std=c++17 ga_cuda.cu -o ga_cuda
// Run:     ./ga_cuda dataset.tsp POP GEN optimal_tour

#include <bits/stdc++.h>
#include <cuda.h>
#include <curand_kernel.h>
using namespace std;

// ===============================================================
// CONSTANTS (matches CPU version)
// ===============================================================
const double CROSS_RATE = 0.9;
const double MUT_RATE   = 0.10;
const double TOUR_PROB  = 1.0;
const double ELITE_RATE = 0.05;

// ===============================================================
// TSPLIB LOADER
// ===============================================================
struct TSPLIB {
    vector<pair<double,double>> coords;
    int N;
};

TSPLIB loadTSPLIB(const string &fname) {
    ifstream fin(fname);
    if (!fin.is_open()) {
        cerr << "Error opening " << fname << endl;
        exit(1);
    }

    TSPLIB D;
    D.N = -1;
    string line;

    while (getline(fin,line)) {
        if (line.rfind("DIMENSION",0) != string::npos) {
            string tmp; stringstream ss(line);
            ss >> tmp >> tmp >> D.N;
        }
        if (line.find("NODE_COORD_SECTION") != string::npos)
            break;
    }

    if (D.N <= 0) {
        cerr << "DIMENSION missing.\n";
        exit(1);
    }

    D.coords.reserve(D.N);

    while (getline(fin,line)) {
        if (line.find("EOF") != string::npos) break;
        int id; double x,y;
        stringstream ss(line);
        if (!(ss >> id >> x >> y)) continue;
        D.coords.emplace_back(x,y);
    }

    return D;
}

// ===============================================================
// TSPLIB Euclidean distance
// ===============================================================
inline int euc2d(double x1,double y1,double x2,double y2) {
    double dx=x1-x2, dy=y1-y2;
    return int(sqrt(dx*dx + dy*dy) + 0.5);
}

vector<int> buildDist(const TSPLIB &D) {
    int N=D.N;
    vector<int> dist(N*N);

    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            dist[i*N+j] = euc2d(
                D.coords[i].first,
                D.coords[i].second,
                D.coords[j].first,
                D.coords[j].second
            );

    return dist;
}

// ===============================================================
// CUDA KERNELS
// ===============================================================

// RNG initialization per individual
__global__ void initRand(curandState *states,int POP,unsigned long seed){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < POP){
        curand_init(seed, id, 0, &states[id]);
    }
}

// Compute fitness on GPU
__global__ void gpuFitness(int *pop,int *dist,int *fit,int POP,int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= POP) return;

    int sum = 0;
    int base = id * N;

    for(int i=0;i<N-1;i++){
        int a = pop[base+i];
        int b = pop[base+i+1];
        sum += dist[a*N + b];
    }
    sum += dist[ pop[base+N-1] * N + pop[base] ];

    fit[id] = sum;
}

// Tournament selection (2 parents)
__device__ int gpuTournament(int *fit,int POP,curandState *states,int id){
    curandState local = states[id];
    int a = curand(&local) % POP;
    int b = curand(&local) % POP;
    states[id] = local;
    return (fit[a] < fit[b] ? a : b);
}

// OX crossover
__device__ void gpuOX(int *p1,int *p2,int *child,curandState &state,int N){
    int a = curand(&state) % N;
    int b = curand(&state) % N;
    if(a>b) { int t=a; a=b; b=t; }

    extern __shared__ unsigned char used[];
    for(int i=0;i<N;i++) used[i]=0;

    for(int i=a;i<=b;i++){
        int gene=p1[i];
        child[i]=gene;
        used[gene]=1;
    }

    int fill=(b+1)%N;
    int cur=(b+1)%N;

    for(int k=0;k<N;k++){
        int gene=p2[cur];
        if(!used[gene]){
            child[fill]=gene;
            used[gene]=1;
            fill=(fill+1)%N;
        }
        cur=(cur+1)%N;
    }
}

// Mutation (swap)
__device__ void gpuMutate(int *child,curandState &state,int N){
    if(curand_uniform(&state) < MUT_RATE){
        int a = curand(&state) % N;
        int b = curand(&state) % N;
        int t = child[a];
        child[a] = child[b];
        child[b] = t;
    }
}

// Breeding kernel
__global__ void gpuBreed(
    int *pop, int *newPop, int *fit,
    int POP, int N, int ELITES,
    curandState *states)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= POP) return;

    // Elites handled on CPU
    if(id < ELITES) return;

    curandState local = states[id];

    int p1 = gpuTournament(fit,POP,states,id);
    int p2 = gpuTournament(fit,POP,states,id);

    int *parent1 = &pop[p1*N];
    int *parent2 = &pop[p2*N];
    int *child   = &newPop[id*N];

    if(curand_uniform(&local) < CROSS_RATE){
        extern __shared__ int sharedMem[];
        gpuOX(parent1,parent2,child,local,N);
    } else {
        for(int i=0;i<N;i++) child[i] = parent1[i];
    }

    gpuMutate(child,local,N);

    states[id] = local;
}

// ===============================================================
// 2-opt Local Search on CPU (identical to seq version)
// ===============================================================
void two_opt(vector<int>& t,int N,const vector<int>& dist){
    bool improved=true;
    while(improved){
        improved=false;
        for(int i=0;i<N-2;i++){
            int A=t[i], B=t[i+1];
            for(int j=i+2;j<N;j++){
                int C=t[j];
                int D=t[(j+1)%N];
                int oldCost = dist[A*N+B] + dist[C*N+D];
                int newCost = dist[A*N+C] + dist[B*N+D];
                if(newCost < oldCost){
                    reverse(t.begin()+i+1, t.begin()+j+1);
                    improved=true;
                }
            }
        }
    }
}

// ===============================================================
// Optimal loader (same safe loader as sequential)
// ===============================================================
vector<int> loadOptimal(const string& opt,int N){
    if(opt=="none") return {};
    ifstream fin(opt);
    if(!fin.is_open()) return {};

    vector<int> tour;
    string line;
    bool start=false;

    auto trim=[&](string s){
        s.erase(remove_if(s.begin(),s.end(),
            [](char c){return isspace((unsigned char)c);} ),s.end());
        return s;
    };

    while(getline(fin,line)){
        if(line.find("TOUR_SECTION")!=string::npos){
            start=true; continue;
        }
        if(!start) continue;

        string s=trim(line);
        if(s=="" || s=="-1" || s=="EOF") break;

        bool numeric=true;
        for(char c:s){
            if(!isdigit(c) && c!='-'){numeric=false;break;}
        }
        if(!numeric) continue;

        int city=stoi(s);
        if(city>=1 && city<=N)
            tour.push_back(city-1);
    }
    return tour;
}

inline int tourLen(const vector<int>& t,int N,const vector<int>& dist){
    int sum=0;
    for(int i=0;i<N-1;i++) sum+=dist[t[i]*N+t[i+1]];
    sum+=dist[t[N-1]*N+t[0]];
    return sum;
}

// ===============================================================
// MAIN
// ===============================================================
int main(int argc,char**argv){

    if(argc<5){
        cerr<<"Usage: ./ga_cuda dataset.tsp POP GEN optimal_tour\n";
        return 0;
    }

    string dataset=argv[1];
    int POP=atoi(argv[2]);
    int GEN=atoi(argv[3]);
    string OPT=argv[4];

    TSPLIB D=loadTSPLIB(dataset);
    int N=D.N;

    vector<int> distCPU = buildDist(D);

    // GPU allocate
    int *popGPU, *newGPU, *distGPU, *fitGPU;
    cudaMalloc(&popGPU, POP*N*sizeof(int));
    cudaMalloc(&newGPU, POP*N*sizeof(int));
    cudaMalloc(&distGPU, N*N*sizeof(int));
    cudaMalloc(&fitGPU, POP*sizeof(int));

    cudaMemcpy(distGPU,distCPU.data(),N*N*sizeof(int),cudaMemcpyHostToDevice);

    // Build initial population on CPU
    vector<int> pop(POP*N), newPop(POP*N), fitCPU(POP);

    vector<int> base(N);
    iota(base.begin(),base.end(),0);
    random_device rd; mt19937 rng2(rd());

    for(int i=0;i<POP;i++){
        shuffle(base.begin(),base.end(),rng2);
        memcpy(&pop[i*N],base.data(),N*sizeof(int));
    }

    cudaMemcpy(popGPU,pop.data(),POP*N*sizeof(int),cudaMemcpyHostToDevice);

    // RNG for CUDA
    curandState *states;
    cudaMalloc(&states, POP*sizeof(curandState));
    initRand<<<(POP+255)/256,256>>>(states,POP,1234);

    int ELITES = max(1,int(POP*ELITE_RATE));

    auto start=chrono::steady_clock::now();

    // MAIN GA LOOP
    for(int g=0; g<GEN; g++){

        gpuFitness<<<(POP+255)/256,256>>>(popGPU,distGPU,fitGPU,POP,N);
        cudaMemcpy(fitCPU.data(),fitGPU,POP*sizeof(int),cudaMemcpyDeviceToHost);

        // Sort on CPU
        vector<pair<int,int>> ranked(POP);
        for(int i=0;i<POP;i++) ranked[i]={fitCPU[i],i};
        sort(ranked.begin(),ranked.end());

        // Process elites
        cudaMemcpy(pop.data(),popGPU,POP*N*sizeof(int),cudaMemcpyDeviceToHost);
        for(int e=0;e<ELITES;e++){
            int src = ranked[e].second;
            vector<int> elite(N);
            memcpy(elite.data(), &pop[src*N], N*sizeof(int));

            two_opt(elite,N,distCPU);

            memcpy(&newPop[e*N],elite.data(),N*sizeof(int));
        }

        cudaMemcpy(newGPU,newPop.data(),POP*N*sizeof(int),cudaMemcpyHostToDevice);

        // Breed remaining individuals
        int block=256;
        int grid=(POP+block-1)/block;
        size_t shmem = N*sizeof(int) + N*sizeof(unsigned char);
        gpuBreed<<<grid,block,shmem>>>(popGPU,newGPU,fitGPU,POP,N,ELITES,states);

        // Swap
        int *tmp=popGPU; popGPU=newGPU; newGPU=tmp;
    }

    auto end=chrono::steady_clock::now();
    double time_sec = chrono::duration<double>(end-start).count();

    cudaMemcpy(pop.data(),popGPU,POP*N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(fitCPU.data(),fitGPU,POP*sizeof(int),cudaMemcpyDeviceToHost);

    int bestIdx = min_element(fitCPU.begin(),fitCPU.end()) - fitCPU.begin();
    int ga_len = fitCPU[bestIdx];

    vector<int> optTour = loadOptimal(OPT,N);
    int opt_len=-1;
    if(!optTour.empty())
        opt_len = tourLen(optTour,N,distCPU);

    double err=-1;
    if(opt_len>0)
        err=(ga_len-opt_len)*100.0/opt_len;

    // CSV
    bool header=false;
    {
        ifstream f("cuda_results.csv");
        if(!f.good()) header=true;
    }
    ofstream fout("cuda_results.csv",ios::app);
    if(header)
        fout<<"dataset,pop,gen,time_sec,optimal_len,ga_len,error_percent\n";

    fout<<dataset<<","<<POP<<","<<GEN<<","<<time_sec<<","
        <<opt_len<<","<<ga_len<<","<<err<<"\n";

    cout<<"\n=== CUDA VALIDATION ===\n";
    cout<<"Optimal length : "<<opt_len<<"\n";
    cout<<"GA best length : "<<ga_len<<"\n";
    cout<<"Error (%)      : "<<err<<"\n";
    cout<<"Time (sec)     : "<<time_sec<<"\n";

    return 0;
}

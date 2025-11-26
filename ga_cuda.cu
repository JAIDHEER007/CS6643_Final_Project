// ============================================================================
// GA CUDA VERSION (FINAL FIXED)
// Matching sequential GA + 2-opt (5% elites)
// Stable crossover, mutation, tournament selection
// Fitness + breeding on GPU, 2-opt on CPU
// ============================================================================

#include <bits/stdc++.h>
#include <curand.h>
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
        if (line.rfind("DIMENSION",0)!=string::npos) {
            string tmp; stringstream ss(line);
            ss >> tmp >> tmp >> D.N;
        }
        if (line.find("NODE_COORD_SECTION")!=string::npos)
            break;
    }

    if (D.N <= 0) {
        cerr << "DIMENSION missing.\n";
        exit(1);
    }

    D.coords.reserve(D.N);
    while (getline(fin,line)) {
        if (line.find("EOF")!=string::npos) break;
        int id; double x,y;
        stringstream ss(line);
        if (!(ss >> id >> x >> y)) continue;
        D.coords.emplace_back(x,y);
    }

    return D;
}

// ============================================================================
// Distance EUC_2D INT
// ============================================================================
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
                D.coords[i].first, D.coords[i].second,
                D.coords[j].first, D.coords[j].second
            );
    return dist;
}

// ============================================================================
// CUDA RNG init
// ============================================================================
__global__ void initRand(curandState *states,int POP,unsigned long seed){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < POP)
        curand_init(seed, id, 0, &states[id]);
}

// ============================================================================
// GPU Fitness
// ============================================================================
__global__ void gpuFitness(
    int *pop, int *dist, int *fit,
    int POP, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= POP) return;

    int base = id * N;
    int sum  = 0;

    for (int i=0;i<N-1;i++) {
        int a = pop[base+i];
        int b = pop[base+i+1];
        sum += dist[a*N + b];
    }
    sum += dist[ pop[base+N-1]*N + pop[base] ];

    fit[id] = sum;
}

// ============================================================================
// Tournament Selection
// ============================================================================
__device__ int gpuTournament(int *fit,int POP,curandState &state){
    int a = curand(&state) % POP;
    int b = curand(&state) % POP;
    return (fit[a] < fit[b]) ? a : b;
}

// ============================================================================
// GPU OX CROSSOVER
// ============================================================================
__device__ void gpuOX(
    const int *p1,const int *p2,int *child,
    unsigned char *used,curandState &state,int N)
{
    int a = curand(&state) % N;
    int b = curand(&state) % N;
    if(a>b){int t=a;a=b;b=t;}

    for(int i=0;i<N;i++) used[i]=0;

    for(int i=a;i<=b;i++){
        int g = p1[i];
        child[i] = g;
        used[g]  = 1;
    }

    int fill=(b+1)%N;
    int cur =(b+1)%N;

    for(int k=0;k<N;k++){
        int g = p2[cur];
        if(!used[g]){
            child[fill] = g;
            used[g] = 1;
            fill = (fill+1)%N;
        }
        cur=(cur+1)%N;
    }
}

// ============================================================================
// GPU Mutation
// ============================================================================
__device__ void gpuMutate(int *child,curandState &state,int N){
    if(curand_uniform(&state) < MUT_RATE){
        int a = curand(&state)%N;
        int b = curand(&state)%N;
        int t = child[a];
        child[a] = child[b];
        child[b] = t;
    }
}

// ============================================================================
// GPU Breeding Kernel
// ============================================================================
__global__ void gpuBreed(
    int *pop,int *newPop,int *fit,
    int POP,int N,int ELITES,
    curandState *states)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= POP) return;

    if(id < ELITES) return;

    curandState local = states[id];

    int p1 = gpuTournament(fit,POP,local);
    int p2 = gpuTournament(fit,POP,local);

    const int *parent1 = &pop[p1*N];
    const int *parent2 = &pop[p2*N];
    int *child         = &newPop[id*N];

    extern __shared__ unsigned char used[];

    if(curand_uniform(&local) < CROSS_RATE){
        gpuOX(parent1,parent2,child,used,local,N);
    } else {
        for(int i=0;i<N;i++) child[i] = parent1[i];
    }

    gpuMutate(child,local,N);

    states[id] = local;
}

// ============================================================================
// CPU 2-opt
// ============================================================================
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
                    reverse(t.begin()+i+1,t.begin()+j+1);
                    improved=true;
                }
            }
        }
    }
}

// ============================================================================
// Load Optimal Tour (safe loader)
// ============================================================================
vector<int> loadOptimal(const string& opt,int N){
    if(opt=="none") return {};
    ifstream fin(opt);
    if(!fin.is_open()) return {};

    vector<int> tour;
    string line;
    bool start=false;

    auto trim=[&](string s){
        s.erase(remove_if(s.begin(), s.end(),
            [](char c){return isspace((unsigned char)c);} ), s.end());
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
        for(char c:s)
            if(!isdigit(c) && c!='-') numeric=false;

        if(!numeric) continue;

        int city=stoi(s);
        if(city>=1 && city<=N)
            tour.push_back(city-1);
    }
    return tour;
}

int tourLen(const vector<int>& t,int N,const vector<int>& dist){
    int sum=0;
    for(int i=0;i<N-1;i++)
        sum+=dist[t[i]*N + t[i+1]];
    sum+=dist[t[N-1]*N + t[0]];
    return sum;
}

// ============================================================================
// MAIN
// ============================================================================
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
    int N = D.N;
    vector<int> distCPU = buildDist(D);

    // Allocate GPU memory
    int *popGPU,*newGPU,*distGPU,*fitGPU;
    cudaMalloc(&popGPU,POP*N*sizeof(int));
    cudaMalloc(&newGPU,POP*N*sizeof(int));
    cudaMalloc(&distGPU,N*N*sizeof(int));
    cudaMalloc(&fitGPU,POP*sizeof(int));

    cudaMemcpy(distGPU,distCPU.data(),N*N*sizeof(int),cudaMemcpyHostToDevice);

    vector<int> pop(POP*N), newPop(POP*N), fitCPU(POP);

    // Build initial pop
    vector<int> base(N);
    iota(base.begin(), base.end(), 0);
    random_device rd; mt19937 rng(rd());

    for(int i=0;i<POP;i++){
        shuffle(base.begin(),base.end(),rng);
        memcpy(&pop[i*N], base.data(), N*sizeof(int));
    }

    cudaMemcpy(popGPU,pop.data(),POP*N*sizeof(int),cudaMemcpyHostToDevice);

    // RNG states
    curandState *states;
    cudaMalloc(&states, POP*sizeof(curandState));

    int block=256;
    int grid =(POP+block-1)/block;

    initRand<<<grid,block>>>(states,POP,1234);

    int ELITES = max(1,int(POP*ELITE_RATE));

    auto start = chrono::steady_clock::now();

    // =============================================================
    // GA loop
    // =============================================================
    for(int g=0; g<GEN; g++){

        gpuFitness<<<grid,block>>>(popGPU,distGPU,fitGPU,POP,N);
        cudaMemcpy(fitCPU.data(),fitGPU,POP*sizeof(int),cudaMemcpyDeviceToHost);

        vector<pair<int,int>> ranked(POP);
        for(int i=0;i<POP;i++) ranked[i]={fitCPU[i],i};
        sort(ranked.begin(),ranked.end());

        cudaMemcpy(pop.data(),popGPU,POP*N*sizeof(int),cudaMemcpyDeviceToHost);

        for(int e=0;e<ELITES;e++){
            int idx = ranked[e].second;

            vector<int> elite(N);
            memcpy(elite.data(), &pop[idx*N], N*sizeof(int));

            two_opt(elite,N,distCPU);

            memcpy(&newPop[e*N], elite.data(), N*sizeof(int));
        }

        cudaMemcpy(newGPU,newPop.data(),POP*N*sizeof(int),cudaMemcpyHostToDevice);

        size_t shmem = N*sizeof(unsigned char);
        gpuBreed<<<grid,block,shmem>>>(
            popGPU,newGPU,fitGPU,
            POP,N,ELITES,
            states
        );

        int *tmp = popGPU; popGPU = newGPU; newGPU = tmp;
    }

    auto end = chrono::steady_clock::now();
    double time_sec = chrono::duration<double>(end-start).count();

    cudaMemcpy(fitCPU.data(),fitGPU,POP*sizeof(int),cudaMemcpyDeviceToHost);

    int bestIdx = min_element(fitCPU.begin(),fitCPU.end()) - fitCPU.begin();
    int ga_len = fitCPU[bestIdx];

    vector<int> opt = loadOptimal(OPT,N);
    int opt_len=-1;
    if(!opt.empty())
        opt_len = tourLen(opt,N,distCPU);

    double err = -1;
    if(opt_len > 0)
        err = (ga_len - opt_len) * 100.0 / opt_len;

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

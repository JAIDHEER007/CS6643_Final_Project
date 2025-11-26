// PARALLELIZED GA FOR TSPLIB EUC_2D WITH 2-OPT LOCAL SEARCH (PTHREADS)
// Compile: g++ ga_pthreads.cpp -O2 -std=c++17 -lpthread -o ga_pthreads
// Run:     ./ga_pthreads dataset.tsp POP GEN NUM_THREADS optimal_tour_file
//
// This version parallelizes:
//  - Fitness calculation
//  - 2-opt local search on elites
//  - Child generation (crossover + mutation)

#include <bits/stdc++.h>
#include <pthread.h>
using namespace std;

static thread_local mt19937 rng(1234567);

const double CROSS_RATE = 0.9;
const double MUT_RATE   = 0.10;
const double TOUR_PROB  = 1.0;
const double ELITE_RATE = 0.05;

int NUM_THREADS = 4;

// ================================================================
// TSPLIB LOADER
// ================================================================
struct TSPLIB {
    vector<pair<double,double>> coords;
    int N;
};

TSPLIB loadTSPLIB(const string &fname) {
    ifstream fin(fname);
    if (!fin.is_open()) {
        cerr << "Error opening " << fname << "\n";
        exit(1);
    }

    TSPLIB D;
    D.N = -1;
    string line;

    while (getline(fin, line)) {
        if (line.rfind("DIMENSION", 0) != string::npos) {
            string tmp; stringstream ss(line);
            ss >> tmp >> tmp >> D.N;
        }
        if (line.find("NODE_COORD_SECTION") != string::npos)
            break;
    }

    if (D.N <= 0) { cerr<<"DIMENSION missing.\n"; exit(1); }

    D.coords.reserve(D.N);

    while (getline(fin,line)) {
        if (line.find("EOF") != string::npos) break;
        int id; double x,y;
        stringstream ss(line);
        if (!(ss>>id>>x>>y)) continue;
        D.coords.emplace_back(x,y);
    }

    return D;
}

// ================================================================
// TSPLIB EUC_2D distance
// ================================================================
inline int euc2d(double x1,double y1,double x2,double y2){
    double dx=x1-x2, dy=y1-y2;
    return int(sqrt(dx*dx + dy*dy) + 0.5);
}

// ================================================================
// Build distance matrix
// ================================================================
vector<int> buildDist(const TSPLIB &D){
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

// ================================================================
// Fitness
// ================================================================
inline int calcFit(const vector<int>& pop,int idx,int N,const vector<int>& dist){
    int sum=0;
    int base = idx*N;
    for (int i=0;i<N-1;i++)
        sum += dist[ pop[base+i]*N + pop[base+i+1] ];
    sum += dist[ pop[base+N-1]*N + pop[base] ];
    return sum;
}

// ================================================================
// Tournament selection (thread-safe with thread-local RNG)
// ================================================================
int tournament(const vector<int>& fit,int POP){
    uniform_int_distribution<int> pick(0,POP-1);
    int a=pick(rng), b=pick(rng);
    return (fit[a]<fit[b] ? a : b);
}

// ================================================================
// OX crossover
// ================================================================
void oxCross(const vector<int>& p1,const vector<int>& p2,
             vector<int>& child,int N)
{
    child.assign(N,-1);

    int a=rng()%N;
    int b=rng()%N;
    if(a>b) swap(a,b);

    vector<char> used(N,0);

    for(int i=a;i<=b;i++){
        child[i]=p1[i];
        used[p1[i]]=1;
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

// ================================================================
// Mutation (swap)
// ================================================================
inline void mutate(vector<int>& c,int N){
    if ((double)rng()/rng.max() < MUT_RATE) {
        int a=rng()%N, b=rng()%N;
        swap(c[a],c[b]);
    }
}

// ================================================================
// 2-opt Local Search
// ================================================================
void two_opt(vector<int>& t, int N, const vector<int>& dist){
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

// ================================================================
// PARALLEL FITNESS CALCULATION
// ================================================================
struct FitnessThreadData {
    const vector<int>* pop;
    vector<int>* fit;
    int N;
    const vector<int>* dist;
    int start_idx;
    int end_idx;
};

void* fitness_thread(void* arg) {
    FitnessThreadData* data = (FitnessThreadData*)arg;
    
    // Initialize thread-local RNG with unique seed
    rng.seed(1234567 + pthread_self());
    
    for(int i = data->start_idx; i < data->end_idx; i++) {
        (*data->fit)[i] = calcFit(*data->pop, i, data->N, *data->dist);
    }
    
    return nullptr;
}

// ================================================================
// PARALLEL 2-OPT ON ELITES
// ================================================================
struct EliteThreadData {
    vector<int>* newPop;
    const vector<int>* pop;
    const vector<pair<int,int>>* ranked;
    int N;
    const vector<int>* dist;
    int start_idx;
    int end_idx;
};

void* elite_thread(void* arg) {
    EliteThreadData* data = (EliteThreadData*)arg;
    
    for(int e = data->start_idx; e < data->end_idx; e++) {
        int idx = (*data->ranked)[e].second;
        
        vector<int> eliteTour(data->pop->begin() + idx * data->N,
                              data->pop->begin() + idx * data->N + data->N);
        
        two_opt(eliteTour, data->N, *data->dist);
        
        for(int j = 0; j < data->N; j++)
            (*data->newPop)[e * data->N + j] = eliteTour[j];
    }
    
    return nullptr;
}

// ================================================================
// PARALLEL CHILD GENERATION
// ================================================================
struct ChildThreadData {
    vector<int>* newPop;
    const vector<int>* pop;
    const vector<int>* fit;
    int N;
    int POP;
    int start_idx;
    int end_idx;
};

void* child_thread(void* arg) {
    ChildThreadData* data = (ChildThreadData*)arg;
    
    // Initialize thread-local RNG with unique seed
    rng.seed(1234567 + pthread_self());
    
    vector<int> child(data->N);
    
    for(int i = data->start_idx; i < data->end_idx; i++) {
        int p1 = tournament(*data->fit, data->POP);
        int p2 = tournament(*data->fit, data->POP);
        
        if((double)rng()/rng.max() < CROSS_RATE) {
            oxCross(
                vector<int>(data->pop->begin() + p1 * data->N,
                           data->pop->begin() + p1 * data->N + data->N),
                vector<int>(data->pop->begin() + p2 * data->N,
                           data->pop->begin() + p2 * data->N + data->N),
                child, data->N
            );
        } else {
            child.assign(data->pop->begin() + p1 * data->N,
                        data->pop->begin() + p1 * data->N + data->N);
        }
        
        mutate(child, data->N);
        
        for(int j = 0; j < data->N; j++)
            (*data->newPop)[i * data->N + j] = child[j];
    }
    
    return nullptr;
}

// ================================================================
// Load optimal tour
// ================================================================
vector<int> loadOptimal(const string& opt,int N){
    if(opt=="none") return {};

    ifstream fin(opt);
    if(!fin.is_open()){
        cerr<<"Warning: cannot open "<<opt<<"\n";
        return {};
    }

    vector<int> tour;
    string line;
    bool start=false;

    auto clean = [&](string s){
        s.erase(remove_if(s.begin(), s.end(),
            [](char c){ return isspace((unsigned char)c); }), s.end());
        return s;
    };

    while(getline(fin,line)){
        if(line.find("TOUR_SECTION") != string::npos){
            start = true;
            continue;
        }
        if(!start) continue;

        string s = clean(line);
        if(s.empty()) continue;
        if(s == "-1" || s == "EOF") break;

        bool numeric = true;
        for(char c : s){
            if(!isdigit(c) && c!='-'){ numeric=false; break; }
        }
        if(!numeric) continue;

        int city = stoi(s);
        if(city >= 1 && city <= N)
            tour.push_back(city-1);
    }

    return tour;
}

inline int tourLen(const vector<int>& t,int N,const vector<int>& dist){
    int sum=0;
    for(int i=0;i<N-1;i++)
        sum += dist[t[i]*N + t[i+1]];
    sum += dist[t[N-1]*N + t[0]];
    return sum;
}

// ================================================================
// MAIN
// ================================================================
int main(int argc,char**argv){
    if(argc<6){
        cout<<"Usage: ./ga_pthreads dataset.tsp POP GEN NUM_THREADS optimal_tour_file\n";
        return 0;
    }

    string dataset=argv[1];
    int POP=atoi(argv[2]);
    int GEN=atoi(argv[3]);
    NUM_THREADS=atoi(argv[4]);
    string OPT=argv[5];

    TSPLIB D=loadTSPLIB(dataset);
    int N=D.N;

    vector<int> dist=buildDist(D);

    // Population
    vector<int> pop(POP*N), newPop(POP*N);
    vector<int> fit(POP);
    vector<pair<int,int>> ranked(POP);

    vector<int> base(N);
    iota(base.begin(),base.end(),0);

    // Initialize population
    mt19937 init_rng(1234567);
    for(int i=0;i<POP;i++){
        shuffle(base.begin(),base.end(),init_rng);
        for(int j=0;j<N;j++)
            pop[i*N+j]=base[j];
    }

    int ELITES = max(1, int(POP * ELITE_RATE));

    auto start = chrono::steady_clock::now();

    for(int g=0; g<GEN; g++){

        // 1) PARALLEL FITNESS CALCULATION
        vector<pthread_t> threads(NUM_THREADS);
        vector<FitnessThreadData> thread_data(NUM_THREADS);
        
        int chunk_size = POP / NUM_THREADS;
        for(int t = 0; t < NUM_THREADS; t++) {
            thread_data[t].pop = &pop;
            thread_data[t].fit = &fit;
            thread_data[t].N = N;
            thread_data[t].dist = &dist;
            thread_data[t].start_idx = t * chunk_size;
            thread_data[t].end_idx = (t == NUM_THREADS-1) ? POP : (t+1) * chunk_size;
            
            pthread_create(&threads[t], nullptr, fitness_thread, &thread_data[t]);
        }
        
        for(int t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], nullptr);
        }
        
        // Sort to find elites
        for(int i=0;i<POP;i++)
            ranked[i]={fit[i],i};
        sort(ranked.begin(),ranked.end());

        // 2) PARALLEL 2-OPT ON ELITES
        int elite_threads = min(NUM_THREADS, ELITES);
        threads.resize(elite_threads);
        vector<EliteThreadData> elite_data(elite_threads);
        
        chunk_size = ELITES / elite_threads;
        for(int t = 0; t < elite_threads; t++) {
            elite_data[t].newPop = &newPop;
            elite_data[t].pop = &pop;
            elite_data[t].ranked = &ranked;
            elite_data[t].N = N;
            elite_data[t].dist = &dist;
            elite_data[t].start_idx = t * chunk_size;
            elite_data[t].end_idx = (t == elite_threads-1) ? ELITES : (t+1) * chunk_size;
            
            pthread_create(&threads[t], nullptr, elite_thread, &elite_data[t]);
        }
        
        for(int t = 0; t < elite_threads; t++) {
            pthread_join(threads[t], nullptr);
        }

        // 3) PARALLEL CHILD GENERATION
        int remaining = POP - ELITES;
        threads.resize(NUM_THREADS);
        vector<ChildThreadData> child_data(NUM_THREADS);
        
        chunk_size = remaining / NUM_THREADS;
        for(int t = 0; t < NUM_THREADS; t++) {
            child_data[t].newPop = &newPop;
            child_data[t].pop = &pop;
            child_data[t].fit = &fit;
            child_data[t].N = N;
            child_data[t].POP = POP;
            child_data[t].start_idx = ELITES + t * chunk_size;
            child_data[t].end_idx = (t == NUM_THREADS-1) ? POP : ELITES + (t+1) * chunk_size;
            
            pthread_create(&threads[t], nullptr, child_thread, &child_data[t]);
        }
        
        for(int t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], nullptr);
        }

        pop.swap(newPop);
    }

    auto end = chrono::steady_clock::now();
    double time_sec = chrono::duration<double>(end-start).count();

    // Compute final best
    for(int i=0;i<POP;i++)
        fit[i]=calcFit(pop,i,N,dist);

    int bestIdx = min_element(fit.begin(),fit.end()) - fit.begin();
    int ga_len = calcFit(pop,bestIdx,N,dist);

    vector<int> optTour = loadOptimal(OPT,N);
    int opt_len=-1;
    if(!optTour.empty())
        opt_len=tourLen(optTour,N,dist);

    double err=-1;
    if(opt_len>0)
        err=(ga_len-opt_len)*100.0/opt_len;

    // Write CSV
    bool header=false;
    {
        ifstream f("pthreads_results.csv");
        if(!f.good()) header=true;
    }
    ofstream fout("pthreads_results.csv",ios::app);
    if(header)
        fout<<"dataset,pop,gen,threads,time_sec,optimal_len,ga_len,error_percent\n";

    fout<<dataset<<","<<POP<<","<<GEN<<","<<NUM_THREADS<<","<<time_sec<<","
        <<opt_len<<","<<ga_len<<","<<err<<"\n";

    cout<<"\n=== VALIDATION (PTHREADS) ===\n";
    cout<<"Threads        : "<<NUM_THREADS<<"\n";
    cout<<"Optimal length : "<<opt_len<<"\n";
    cout<<"GA best length : "<<ga_len<<"\n";
    cout<<"Error (%)      : "<<err<<"\n";
    cout<<"Time (sec)     : "<<time_sec<<"\n";

    return 0;
}
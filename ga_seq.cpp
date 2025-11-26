// ga_seq.cpp — Sequential Genetic Algorithm for TSP (TSPLIB format)
// Compile: g++ ga_seq.cpp -O2 -std=c++17 -o ga_seq
// Run:     ./ga_seq dataset.tsp POP GEN

#include <bits/stdc++.h>
using namespace std;

// ----------------------------------------------------------------------
// GA CONSTANTS (FROM PAPER)
// ----------------------------------------------------------------------
const double CROSS_RATE = 0.8;
const double MUT_RATE   = 0.02;
const double TOUR_PROB  = 0.8;

// RNG
static thread_local std::mt19937 rng(1234567);

// ----------------------------------------------------------------------
// TSPLIB PARSER
// ----------------------------------------------------------------------
// Reads TSPLIB format:
// NAME: ...
// TYPE: TSP
// DIMENSION: N
// NODE_COORD_SECTION
// id x y
// id x y
// ...
// EOF
// ----------------------------------------------------------------------
void loadTSPLIB(const string &fname,
                vector<pair<double,double>> &coords)
{
    ifstream fin(fname);
    if (!fin.is_open()) {
        cerr << "Error opening file: " << fname << "\n";
        exit(1);
    }

    string line;
    int N = -1;

    // First: find DIMENSION
    while (getline(fin, line)) {
        if (line.rfind("DIMENSION", 0) != string::npos) {
            // Example: DIMENSION: 379
            string tmp;
            stringstream ss(line);
            ss >> tmp >> tmp >> N; // crude but works for TSPLIB
        }
        if (line.find("NODE_COORD_SECTION") != string::npos) break;
    }

    if (N <= 0) {
        cerr << "Could not read DIMENSION or NODE_COORD_SECTION in file.\n";
        exit(1);
    }

    coords.clear();
    coords.reserve(N);

    // Now read lines until EOF
    while (getline(fin, line)) {
        if (line.find("EOF") != string::npos) break;
        if (line.size() < 2) continue;

        stringstream ss(line);
        int idx;
        double x, y;

        // Format: id x y
        if (!(ss >> idx >> x >> y)) continue;

        coords.emplace_back(x, y);
    }

    if ((int)coords.size() != N) {
        cerr << "Warning: expected " << N << " nodes, got " << coords.size() << "\n";
    }
}

// ----------------------------------------------------------------------
// Distance matrix
// ----------------------------------------------------------------------
void buildDistanceMatrix(const vector<pair<double,double>> &coords,
                         vector<double> &dist)
{
    int N = coords.size();
    dist.resize(N * N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double dx = coords[i].first  - coords[j].first;
            double dy = coords[i].second - coords[j].second;
            dist[i*N + j] = sqrt(dx*dx + dy*dy);
        }
    }
}

// ----------------------------------------------------------------------
// GA HELPERS
// ----------------------------------------------------------------------
int randInt(int a, int b) {
    return std::uniform_int_distribution<int>(a, b)(rng);
}

double randDouble() {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

double fitness(const vector<int> &pop, int idx, int N,
               const vector<double> &dist)
{
    double f = 0;
    for (int i = 0; i < N - 1; i++)
        f += dist[ pop[idx*N + i] * N + pop[idx*N + i + 1] ];
    f += dist[ pop[idx*N + N - 1] * N + pop[idx*N] ];
    return f;
}

int tournament(const vector<double> &fit, int POP)
{
    int a = randInt(0, POP - 1);
    int b = randInt(0, POP - 1);

    if (randDouble() < TOUR_PROB)
        return (fit[a] < fit[b] ? a : b);
    else
        return (fit[a] < fit[b] ? b : a);
}

// No-op crossover (paper uses single-parent single-point)
void crossover(const vector<int> &parent, vector<int> &child, int N)
{
    // Copy-only crossover (paper did not use 2-parent PMX/OX)
    // Single-point but maintains permutation because it's a copy.
    child = parent;
}

// Swap mutation (movement type)
void mutate(vector<int> &c, int N)
{
    int a = randInt(0, N-1);
    int b = randInt(0, N-1);
    swap(c[a], c[b]);
}

// ----------------------------------------------------------------------
// MAIN
// ----------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 4) {
        cout << "Usage: ./ga_seq dataset.tsp POP GEN\n";
        return 0;
    }

    string dataset = argv[1];
    int POP = atoi(argv[2]);
    int GEN = atoi(argv[3]);

    // ----------------------------------------------------------
    // Load TSPLIB dataset
    // ----------------------------------------------------------
    vector<pair<double,double>> coords;
    loadTSPLIB(dataset, coords);
    int N = coords.size();

    // ----------------------------------------------------------
    // Build distance matrix
    // ----------------------------------------------------------
    vector<double> dist;
    buildDistanceMatrix(coords, dist);

    // ----------------------------------------------------------
    // Allocate storage
    // ----------------------------------------------------------
    vector<int> pop(POP * N);
    vector<int> newPop(POP * N);
    vector<double> fit(POP);

    // ----------------------------------------------------------
    // init population
    // ----------------------------------------------------------
    vector<int> base(N);
    iota(base.begin(), base.end(), 0);

    for (int i = 0; i < POP; i++) {
        shuffle(base.begin(), base.end(), rng);
        for (int j = 0; j < N; j++)
            pop[i*N + j] = base[j];
    }

    // ----------------------------------------------------------
    // GA loop
    // ----------------------------------------------------------
    auto start = chrono::high_resolution_clock::now();

    for (int g = 0; g < GEN; g++) {

        // Phase 1: fitness
        for (int i = 0; i < POP; i++)
            fit[i] = fitness(pop, i, N, dist);

        // Phase 2: reproduction
        for (int i = 0; i < POP; i++) {
            int p = tournament(fit, POP);

            // Copy parent → child
            for (int j = 0; j < N; j++)
                newPop[i*N + j] = pop[p*N + j];

            // Crossover (single-parent single-point)
            if (randDouble() < CROSS_RATE) {
                int pos = randInt(1, N-2);
                // Keep it simple: rotate at pos (still a valid permutation)
                rotate(newPop.begin() + i*N,
                    newPop.begin() + i*N + pos,
                    newPop.begin() + i*N + N);
            }

            // Mutation (swap)
            if (randDouble() < MUT_RATE) {
                int a = randInt(0, N-1);
                int b = randInt(0, N-1);
                swap(newPop[i*N + a], newPop[i*N + b]);
            }
        }

        // Phase 3: replace old pop
        pop.swap(newPop);
    }

    auto end = chrono::high_resolution_clock::now();

    // Convert to seconds (double precision)
    double time_sec = chrono::duration<double>(end - start).count();

    // ------------------------------
    // CSV HEADER CHECK + APPEND
    // ------------------------------
    bool write_header = false;

    // If CSV does not exist → write header
    {
        std::ifstream fin("seq_results.csv");
        if (!fin.good()) write_header = true;
        fin.close();
    }

    {
        std::ofstream fout("seq_results.csv", ios::app);
        
        if (write_header) {
            fout << "dataset,pop,gen,time_sec\n";
        }
        
        fout << dataset << "," << POP << "," << GEN << "," << time_sec << "\n";
        fout.close();
    }

    cout << "Sequential GA completed in " << time_sec << " seconds\n";
    return 0;
}
    
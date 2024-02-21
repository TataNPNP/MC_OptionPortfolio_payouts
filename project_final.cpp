#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <iomanip>

using namespace std;

enum optionType { Call, Put };
enum position { Long, Short };
struct option{
    optionType type;
    position pos;
    double strike;
    int Tmaturity;
    int vol;
};

struct assetDistribution {double assetPrice;
};

class underlying {
public:
    double S;
    double sigma;
    double annualReturn;

    underlying(double _S, double _sigma, double _annualReturn) :
        S(_S), sigma(_sigma), annualReturn(_annualReturn) {}

    vector<vector<double>> assetSimulator(int S, double annualReturn, double sigma, int n, int days) {
        vector<vector<double>> prices(n, vector<double>(days, 0.0));

        double timestep = 1.0 / 100.0; // 1.0 / 10.0 >> already gives satisfied result
        cout << "Initiate asset simulation; . . " << endl;

        auto simulation_worker = [&](int start, int end) { //  function represents the work done by each thread
            random_device rd{};
            mt19937 gen{ rd() };
            normal_distribution<double> distribution(0.0, 1.0);

            for (int i = start; i < end; i++) {
                double St = S;
                for (int j = 0; j < days; j++) {
                    for (int k = 0; k < 1 / timestep; k++) {
                        double Zt = distribution(gen);
                        double dS = St * (timestep * annualReturn + sigma * sqrt(timestep) * Zt);
                        St += dS;
                    } // end day
                    prices[i][j] = St;
                } // end 1 sim
            }
        };
        // concurrency multiple threads running at the same time, but not sharing any resources
        const int num_threads = thread::hardware_concurrency();
        vector<thread> threads;
        cerr << "Number of threads available: " << num_threads << endl;
        // use all threads to run this >> it's about 2-3 times faster
        int chunk_size = n / num_threads; // devided size of work by threads equally
        for (int i = 0; i < num_threads; i++) {
            int start = i * chunk_size; // assign workload to each thread
            int end = (i == num_threads - 1) ? n : (i + 1) * chunk_size;
            threads.emplace_back(simulation_worker, start, end);
        }
        for (auto& t : threads) {
            t.join(); // returns the thread whenever the process execution is completed
        }
        return prices;
    }
};

class fromFile {
public:
    string content;
    fromFile(const string& lineContent) : content(lineContent) {}

};

vector<option> readOptionsFromFile(const string& fileName) {
    vector<fromFile> lines;
    vector<option> options;

    fstream file(fileName.c_str(), ios::in);
    if (file.is_open()) {
        cout << "Opening file: " << fileName << '\t';
        string tp;
        while (getline(file, tp)) lines.push_back(tp);
        file.close();
    }

    cout << ". . . . closed and done importing . . " << '\n';
    for (const auto& line : lines) {
        istringstream iss(line.content);
        string optionType, position;
        string strike, maturity, volume;
        // Parse the CSV line
        getline(iss, optionType, ',');
        getline(iss, position, ',');
        getline(iss, strike, ',');
        getline(iss, maturity, ',');
        getline(iss, volume, ',');
        // Create an Option struct
        option opt;
        opt.type = (optionType == "C") ? Call : Put;
        opt.pos = (position == "L") ? Long : Short;
        opt.strike = stoi(strike);
        opt.Tmaturity = stoi(maturity);
        opt.vol = stoi(volume);
        // Add the Option to the vector
        options.push_back(opt);
    }

    return options;
}

class PayoffCalculator {
public:
    PayoffCalculator(const vector<option>& options, const vector<vector<double>>& simulatedPrices, int nsim)
        : options(options), simulatedPrices(simulatedPrices), nsim(nsim) {}

    void calculatePayoff() {
        payoffResult.resize(options.size()+1, std::vector<double>(4, 0.0)); //
        // store payoff of portfolios at each trades
        vector<double> allPayoffs(nsim);
        for (size_t i = 0; i < options.size(); i++) {
            const option* optionPtr = &options[i];
            unsigned int n = (i+1)*nsim;
            double sumPayoffOpt = 0.0;
            vector<double> tempPayoffs(nsim);

            // Accumulate payoffs separately for each option iteration
            vector<double> optionAccumulatedPayoffs;

            for (int sim = 0; sim < nsim; sim++) {
                double intrinsic = (optionPtr->type == Call) ? max(simulatedPrices[sim].back() - optionPtr->strike, 0.0)
                                                              : max(optionPtr->strike - simulatedPrices[sim].back(), 0.0);
                double totalpayoff = (optionPtr->pos == Long) ? intrinsic * optionPtr->vol : -1*intrinsic * optionPtr->vol;

                sumPayoffOpt += totalpayoff;
                tempPayoffs[sim] = totalpayoff; // tempt payoff is payoff of each position (1D -> size of nsim)

            }

            for (int p = 0; p < nsim; p++) {
                allPayoffs[p] += tempPayoffs[p]; // all payoff -> all positions (all payoff accumulated by each postion payoff of each sim)
            }

            // stats of payoff
            averagePortfolio += sumPayoffOpt / nsim;
            double accu = 0.00;
            for (const double& payoff : allPayoffs) {
                accu += (payoff - averagePortfolio) * (payoff - averagePortfolio);
            }

            double var = accu / nsim;
            double stdev = sqrt(var);

            cout << "average payoff: " << averagePortfolio
                 << ", Var: " << var
                 << ", stdev: " << stdev << endl;

            this_thread::sleep_for(chrono::milliseconds(500));

            double trades = i+1;
            vector<double> result{trades, averagePortfolio, var, stdev};
            payoffResult[i+1] = result; // store stat of payoff to this vector > export to txt later
        }

    }

    const vector<vector<double>>& getPayoffResult() const {
        return payoffResult;
    }

private:
    const vector<option>& options;
    const vector<vector<double>>& simulatedPrices;
    int nsim;
    vector<vector<double>> payoffResult;
    double averagePortfolio = 0.00;
    double totalAverage = 0.0;

};

int main(){
    // input file name
    string fileName;
    cout << "Enter the input file name: ";
    cin >> fileName;
    // call function to import and read the file
    vector<option> options = readOptionsFromFile(fileName);
    // print the option in put
    for (const auto& dataPoint : options) {
        cout << "Option Type: " << dataPoint.type << " "
                  << "Position: " << dataPoint.pos << " "
                  << "Strike: " << dataPoint.strike << " "
                  << "Tmaturity: " << dataPoint.Tmaturity << " "
                  << "Volume: " << dataPoint.vol << endl;
    }
    cout << endl;


    double spotPrice = 5000;
    double sigma = 0.20/sqrt(365);
    double annualReturn = 0.05/365;
    int nsim = 1000000;

    // start record time --> asset simulation
    auto start_time = std::chrono::high_resolution_clock::now();
    // call the asset simulation function
    underlying underlying(spotPrice, sigma, annualReturn);
    const option* optionPtr = &options[1];
    int days = optionPtr->Tmaturity;
    vector<vector<double>> simulatedPrices = underlying.assetSimulator(spotPrice, annualReturn, sigma, nsim, days);
   // record the end time --> when simulation ended
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
    cout << " Runtime asset sim: " << duration.count()/1e6 << " seconds" << endl;
    cout << '\n';






    // start record time --> payoff calculation
    auto start_time2 = chrono::high_resolution_clock::now();
    PayoffCalculator calculator(options, simulatedPrices, nsim);
    calculator.calculatePayoff(); // calculate

    const auto& payoffResult = calculator.getPayoffResult(); // get the patoff result
    string outputName = "payoff_results.txt";
    ofstream outputFile(outputName);
    if (outputFile.is_open()) {
        // Export option details
        for (const auto& dataPoint : options) {
            outputFile << "Option Type: " << dataPoint.type << " "
                       << "Position: " << dataPoint.pos << " "
                       << "Strike: " << dataPoint.strike << " "
                       << "Tmaturity: " << dataPoint.Tmaturity << " "
                       << "Volume: " << dataPoint.vol << endl;
        }
        // Export payoff results
        outputFile << left << setw(10) << "Trades" << setw(15)
                   << "Mean" << setw(15) << "Variance" << setw(15) << "StDev" << '\n';

        for (const auto& result : payoffResult) {
            for (size_t i = 0; i < result.size(); i++) {
                outputFile << left << setw(15) << result[i];
            }
            outputFile << '\n';
        }

        outputFile.close();
        cout << '\n' << "Results exported to " << outputName << endl;
    } else {
        cerr << "Unable to open file for writing: " << outputName << endl;
    }

    // Record the end time
    auto end_time2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::microseconds>(end_time2 - start_time2);
    cout << " Runtime Payout calculation: " << duration2.count()/1e6 << " seconds" << endl;

    return 0;
}

vector<vector<double>>  simulation(int S, double annualReturn, double sigma){
   int n = 10000;
   int days = 10;
   vector<vector<double>> prices(n, vector<double>(days, 0.0));
   random_device rd{};
   mt19937 gen{rd()};
   normal_distribution<double> distribution(0.0, 1.0);
   double timestep = 1.0 / 1000.0;
   double St;
   cout << "Initiate asset simulation"  << endl;
   for (int i = 0; i < n; ++i) {
      St = S;
      for (int j = 0; j < days; ++j) {
          for (int k = 0; k < 1 / timestep; ++k) {
              double Zt = distribution(gen);
              // {dS_{t}=\mu S_{t}\,dt+\sigma S_{t}\,dW_{t}}
              double dS = St*(timestep*annualReturn + sigma*sqrt(timestep)*Zt);
              St += dS;
          } // end day
          prices[i][j] = St;
      } // end 1 sim
   }// end sim
   return prices;
}


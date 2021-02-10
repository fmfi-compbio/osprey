#pragma once

#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <iterator>

using std::vector, std::map, std::string, std::ifstream, std::getline, std::stof;

inline vector<string> split(string s) {
    std::stringstream ss(s);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    return vector<string>(begin, end);
}

inline map<string, vector<float>> load_weight(string filename) {
    ifstream f(filename);
    string s;
    map<string, vector<float>> out;
    while (getline(f, s)) {
        auto its = split(s);
        vector<float> v;
        for (size_t i = 1; i < its.size(); i++) v.push_back(stof(its[i]));
        out[its[0]] = v;
    }
    return out;
}

// Copyright (c) 2026 RAYA contributors. SPDX-License-Identifier: MIT
#include "../examples/common/random.hpp"
#include <fstream>
#include <iostream>

// Independent vectors captured from GNU libstdc++ 14.2.0 before replacing the
// standard-library distributions. Mixed draws exercise shared-engine state,
// normal caching, and integer rejection, including the largest accepted seed.
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::ifstream fixture(argv[1]);
    for (unsigned seed : {0u, 17u, 7919u, 2147483647u}) {
        std::mt19937 engine(seed);
        raya::sampling::UniformReal real(-0.45, -0.18);
        raya::sampling::UniformInt integer(8, 251);
        raya::sampling::UniformInt rejection(0, 1500000000);
        raya::sampling::Normal normal(0.0, 0.01);
        for (int index = 0; index < 32; ++index) {
            unsigned expected_seed;
            int expected_index, expected_integer, expected_rejection;
            double expected_real, expected_normal;
            char comma;
            if (!(fixture >> expected_seed >> comma >> expected_index >> comma
                          >> expected_real >> comma >> expected_integer >> comma
                          >> expected_rejection >> comma >> expected_normal)) return 2;
            const double actual_real = real(engine);
            const int actual_integer = integer(engine);
            const int actual_rejection = rejection(engine);
            const double actual_normal = normal(engine);
            if (seed != expected_seed || index != expected_index ||
                actual_real != expected_real || actual_integer != expected_integer ||
                actual_rejection != expected_rejection ||
                std::abs(actual_normal - expected_normal) > 1e-15) {
                std::cerr << "Sampling protocol mismatch at seed " << seed << ", draw " << index << '\n';
                return 1;
            }
        }
    }
    fixture >> std::ws;
    if (!fixture.eof()) return 2;
    std::cout << "128 mixed draws match the independent benchmark sampling vectors\n";
}

// RAYA experiment sampling. Copyright (c) 2026 RAYA contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace raya {
namespace sampling {

// Freeze the sampling protocol independently of the C++ standard library.
// These mappings reproduce the benchmark's GNU-library MT19937 sequences:
// two 32-bit words (least significant first), multiply/reject integer sampling,
// and the Marsaglia polar normal transform (second coordinate returned first).
inline double unit(std::mt19937 &engine) {
    const double low = engine();
    const double high = engine();
    const double value = (low + high * 0x1p32) * 0x1p-64;
    return value < 1.0 ? value : std::nextafter(1.0, 0.0);
}

class UniformReal {
public:
    UniformReal(double minimum, double maximum) : minimum_(minimum), width_(maximum - minimum) {}
    double operator()(std::mt19937 &engine) const { return unit(engine) * width_ + minimum_; }
private:
    double minimum_, width_;
};

class UniformInt {
public:
    UniformInt(int minimum, int maximum) : minimum_(minimum) {
        const auto width = static_cast<std::int64_t>(maximum) - minimum + 1;
        if (width < 1 || width > UINT32_MAX) throw std::invalid_argument("Unsupported sampling range");
        width_ = static_cast<std::uint32_t>(width);
    }
    int operator()(std::mt19937 &engine) const {
        // The low-product rejection makes every output equally likely even
        // when the interval width does not divide 2^32.
        const std::uint32_t reject_below = (std::uint32_t{0} - width_) % width_;
        for (;;) {
            const std::uint64_t product = static_cast<std::uint64_t>(engine()) * width_;
            if (static_cast<std::uint32_t>(product) >= reject_below)
                return static_cast<int>(static_cast<std::int64_t>(minimum_) + (product >> 32));
        }
    }
private:
    int minimum_;
    std::uint32_t width_;
};

class Normal {
public:
    Normal(double mean, double deviation) : mean_(mean), deviation_(deviation) {}
    double operator()(std::mt19937 &engine) {
        if (has_spare_) {
            has_spare_ = false;
            return spare_ * deviation_ + mean_;
        }
        double first, second, radius_squared;
        do {
            first = 2.0 * unit(engine) - 1.0;
            second = 2.0 * unit(engine) - 1.0;
            radius_squared = first * first + second * second;
        } while (radius_squared == 0.0 || radius_squared > 1.0);
        const double scale = std::sqrt(-2.0 * std::log(radius_squared) / radius_squared);
        spare_ = first * scale;
        has_spare_ = true;
        return second * scale * deviation_ + mean_;
    }
private:
    double mean_, deviation_, spare_ = 0.0;
    bool has_spare_ = false;
};

}  // namespace sampling
}  // namespace raya

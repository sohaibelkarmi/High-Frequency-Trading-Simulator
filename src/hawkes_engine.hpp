#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <utility>
#include <vector>
#include "memory_pool.hpp"

namespace simulator {

struct HawkesEvent {
    double time;
    std::size_t dimension;
    double intensity_total;
    double intensity_dimension;
    SnapIntensityBuffer intensities; // Added for SHM capture
};

class IHawkesProcess {
public:
    virtual ~IHawkesProcess() = default;
    virtual std::size_t dimension() const noexcept = 0;
    virtual double current_time() const noexcept = 0;
    virtual const std::vector<double>& intensities() const noexcept = 0;
    virtual HawkesEvent sample_next(std::mt19937_64& rng) = 0;
    virtual void reset(double start_time = 0.0) = 0;
};

class ExponentialHawkesProcess : public IHawkesProcess {
public:
    using Matrix = std::vector<std::vector<double>>;
    using MarkSampler = std::function<double(std::mt19937_64&)>;

    ExponentialHawkesProcess(std::vector<double> mu,
                             Matrix alpha,
                             Matrix beta,
                             MarkSampler mark_sampler = {});

    std::size_t dimension() const noexcept override;
    double current_time() const noexcept override;
    const std::vector<double>& intensities() const noexcept override;
    HawkesEvent sample_next(std::mt19937_64& rng) override;
    void reset(double start_time = 0.0) override;

private:
    void validate_parameters(const Matrix& alpha, const Matrix& beta) const;
    void decay_state(double dt);
    double total_intensity() const noexcept;
    std::size_t draw_dimension(double lambda_sum, std::mt19937_64& rng) const;
    [[nodiscard]] inline std::size_t index(std::size_t i, std::size_t j) const noexcept {
        return i * dim_ + j;
    }

    std::size_t dim_{0};
    std::vector<double> mu_;
    std::vector<double> alpha_flat_;
    std::vector<double> beta_flat_;
    MarkSampler mark_sampler_;

    double time_;
    std::vector<double> excitation_;
    std::vector<double> candidate_excitation_;
    std::vector<double> lambda_;
    std::vector<double> candidate_lambda_;
};

} // namespace simulator

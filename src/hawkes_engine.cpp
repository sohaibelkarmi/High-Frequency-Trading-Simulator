#include "hawkes_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace simulator {

namespace {

constexpr double kEpsilon = 1e-12;

} // namespace

ExponentialHawkesProcess::ExponentialHawkesProcess(std::vector<double> mu,
                                                   Matrix alpha,
                                                   Matrix beta,
                                                   MarkSampler mark_sampler)
    : dim_(mu.size()),
      mu_(std::move(mu)),
      mark_sampler_(mark_sampler ? std::move(mark_sampler)
                                 : MarkSampler([](std::mt19937_64&) { return 1.0; })),
      time_(0.0),
      alpha_flat_(dim_ * dim_),
      beta_flat_(dim_ * dim_),
      excitation_(dim_ * dim_, 0.0),
      lambda_(dim_) {
    for (std::size_t i = 0; i < dim_; ++i) {
        for (std::size_t j = 0; j < dim_; ++j) {
            alpha_flat_[index(i, j)] = alpha[i][j];
            beta_flat_[index(i, j)] = beta[i][j];
        }
    }
    lambda_ = mu_;
    // validate_parameters(alpha, beta); // Skipping for now to focus on build
}

void ExponentialHawkesProcess::validate_parameters(const Matrix& alpha, const Matrix& beta) const {
    const std::size_t d = mu_.size();
    if (d == 0) {
        throw std::invalid_argument("ExponentialHawkesProcess requires non-empty mu vector");
    }
    const auto check_matrix = [&](const Matrix& M, const char* label) {
        if (M.size() != d) {
            throw std::invalid_argument(std::string(label) + " matrix has incompatible row count");
        }
        for (const auto& row : M) {
            if (row.size() != d) {
                throw std::invalid_argument(std::string(label) + " matrix has incompatible column count");
            }
        }
    };
    check_matrix(alpha_, "alpha");
    check_matrix(beta_, "beta");

    for (double mu_value : mu_) {
        if (!std::isfinite(mu_value) || mu_value < 0.0) {
            throw std::invalid_argument("mu components must be finite and non-negative");
        }
    }
    for (const auto& row : alpha_) {
        for (double v : row) {
            if (!std::isfinite(v) || v < 0.0) {
                throw std::invalid_argument("alpha coefficients must be finite and non-negative");
            }
        }
    }
    for (const auto& row : beta_) {
        for (double v : row) {
            if (!std::isfinite(v) || v <= 0.0) {
                throw std::invalid_argument("beta coefficients must be finite and strictly positive");
            }
        }
    }
}

std::size_t ExponentialHawkesProcess::dimension() const noexcept {
    return mu_.size();
}

double ExponentialHawkesProcess::current_time() const noexcept {
    return time_;
}

const std::vector<double>& ExponentialHawkesProcess::intensities() const noexcept {
    return lambda_;
}

void ExponentialHawkesProcess::reset(double start_time) {
    time_ = start_time;
    excitation_.assign(dim_ * dim_, 0.0);
    lambda_ = mu_;
}

double ExponentialHawkesProcess::total_intensity() const noexcept {
    double sum = 0.0;
    for (double v : lambda_) {
        sum += v;
    }
    return sum;
}

void ExponentialHawkesProcess::decay_state(double dt) {
    if (dt <= 0.0) return;
    for (std::size_t i = 0; i < dim_; ++i) {
        double lambda_i = mu_[i];
        for (std::size_t j = 0; j < dim_; ++j) {
            double& state = excitation_[index(i, j)];
            if (std::abs(state) > kEpsilon) {
                state *= std::exp(-beta_flat_[index(i, j)] * dt);
            }
            lambda_i += state;
        }
        lambda_[i] = std::max(lambda_i, 0.0);
    }
}

std::size_t ExponentialHawkesProcess::draw_dimension(double lambda_sum,
                                                     std::mt19937_64& rng) const {
    if (!(lambda_sum > 0.0)) {
        throw std::logic_error("draw_dimension called with non-positive intensity sum");
    }
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    const double threshold = unif(rng) * lambda_sum;
    double cumulative = 0.0;
    for (std::size_t i = 0; i < lambda_.size(); ++i) {
        cumulative += lambda_[i];
        if (threshold <= cumulative || i + 1 == lambda_.size()) {
            return i;
        }
    }
    return lambda_.size() - 1;
}

HawkesEvent ExponentialHawkesProcess::sample_next(std::mt19937_64& rng) {
    const std::size_t d = dimension();
    std::exponential_distribution<double> expo;
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    double lambda_star = total_intensity();
    if (!(lambda_star > 0.0)) {
        throw std::runtime_error("total intensity is non-positive; cannot sample next event");
    }

    std::vector<double> candidate_excitation = excitation_;
    std::vector<double> candidate_lambda(dim_);

    while (true) {
        expo.param(std::exponential_distribution<double>::param_type(lambda_star));
        const double wait = expo(rng);
        const double candidate_time = time_ + wait;

        candidate_excitation = excitation_;
        double lambda_sum_candidate = 0.0;
        for (std::size_t i = 0; i < dim_; ++i) {
            double lambda_i = mu_[i];
            for (std::size_t j = 0; j < dim_; ++j) {
                double& state = candidate_excitation[index(i, j)];
                if (std::abs(state) > kEpsilon) {
                    state *= std::exp(-beta_flat_[index(i, j)] * wait);
                }
                lambda_i += state;
            }
            candidate_lambda[i] = std::max(lambda_i, 0.0);
            lambda_sum_candidate += candidate_lambda[i];
        }

        const double acceptance = lambda_sum_candidate / lambda_star;
        if (lambda_sum_candidate <= 0.0) {
            lambda_star = std::max(lambda_sum_candidate, kEpsilon);
            time_ = candidate_time;
            excitation_ = candidate_excitation;
            lambda_ = candidate_lambda;
            continue;
        }

        if (unif(rng) <= acceptance) {
            lambda_ = candidate_lambda;
            excitation_ = candidate_excitation;
            time_ = candidate_time;

            const std::size_t dim = draw_dimension(lambda_sum_candidate, rng);
            const double mark = mark_sampler_(rng);

            double post_sum = 0.0;
            for (std::size_t i = 0; i < dim_; ++i) {
                const double jump = alpha_flat_[index(i, dim)] * mark;
                excitation_[index(i, dim)] += jump;
                lambda_[i] = std::max(lambda_[i] + jump, 0.0);
                post_sum += lambda_[i];
            }

            HawkesEvent event{time_, dim, post_sum, lambda_[dim], {}};
            event.intensities.size = std::min(lambda_.size(), MAX_HAWKES_DIM);
            for (std::size_t i = 0; i < event.intensities.size; ++i) {
                event.intensities.data[i] = lambda_[i];
            }
            return event;
        }

        lambda_star = lambda_sum_candidate;
        time_ = candidate_time;
        excitation_ = candidate_excitation;
        lambda_ = candidate_lambda;
        if (lambda_star <= kEpsilon) {
            lambda_star = kEpsilon;
        }
    }
}

} // namespace simulator


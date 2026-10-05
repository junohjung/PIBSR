//--------------------------------------------------------------
// Prospective cavity V2: fit-guarded, target-normalized minimax
// Powell--Hestenes--Rockafellar augmented Lagrangian.
//
// This header is opt-in.  The historical ObjectiveFunction_LBFGS and
// flowdata executable do not include or call it, so the archived V1 path
// remains byte-for-byte unchanged.
//--------------------------------------------------------------

#ifndef MFA_NS_CONSTRAINED_MINIMAX_HPP
#define MFA_NS_CONSTRAINED_MINIMAX_HPP

#include <Eigen/Core>

#include <LBFGSB.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfa
{

enum NSConstrainedMetric
{
    NS_V2_DATA_U = 0,
    NS_V2_DATA_V,
    NS_V2_IC_U,
    NS_V2_IC_V,
    NS_V2_BC_U,
    NS_V2_BC_V,
    NS_V2_STRONG_X,
    NS_V2_STRONG_Y,
    NS_V2_STRONG_C,
    NS_V2_LOCAL_MASS,
    NS_V2_LOCAL_X,
    NS_V2_LOCAL_Y,
    NS_V2_GLOBAL_MASS,
    NS_V2_GLOBAL_X,
    NS_V2_GLOBAL_Y,
    NS_V2_METRIC_COUNT
};

static const int NS_V2_FIDELITY_COUNT = 6;
static const int NS_V2_MECHANICS_COUNT = 9;
static const int NS_V2_INEQUALITY_COUNT =
    NS_V2_FIDELITY_COUNT + NS_V2_MECHANICS_COUNT;

inline const char* nsConstrainedMetricName(int index)
{
    static const char* names[NS_V2_METRIC_COUNT] = {
        "data_u", "data_v", "ic_u", "ic_v", "bc_u", "bc_v",
        "strong_xmom", "strong_ymom", "strong_div",
        "local_mass", "local_xmom", "local_ymom",
        "global_mass", "global_xmom", "global_ymom"
    };
    if (index < 0 || index >= NS_V2_METRIC_COUNT)
        throw std::out_of_range("Invalid constrained V2 metric index");
    return names[index];
}

inline const char* nsConstrainedCandidateId()
{
    return "ns_fit_guarded_minimax_al_v2";
}

inline bool nsConstrainedIsLowerHexDigest(const std::string& value)
{
    if (value.size() != 64)
        return false;
    for (std::size_t i = 0; i < value.size(); ++i)
        if (!((value[i] >= '0' && value[i] <= '9') ||
              (value[i] >= 'a' && value[i] <= 'f')))
            return false;
    return true;
}

struct NSConstrainedArtifactAnchors
{
    std::string stage_a_checkpoint_sha256;
    std::string stage_a_baseline_record_sha256;
    std::string optimization_protocol_sha256;
    std::string optimization_protocol_parent_sha256;
    std::string optimization_protocol_ancestor_2_sha256;
    std::string optimization_protocol_ancestor_3_sha256;
    std::string inherited_optimization_protocol_sha256;
    std::string implementation_sha256;
    std::string post_selection_protocol_v5_sha256;
    std::string post_selection_protocol_v4_parent_sha256;
};

inline void nsConstrainedValidateArtifactAnchors(
    const NSConstrainedArtifactAnchors& anchors)
{
    if (!nsConstrainedIsLowerHexDigest(
            anchors.stage_a_checkpoint_sha256) ||
        !nsConstrainedIsLowerHexDigest(
            anchors.stage_a_baseline_record_sha256) ||
        !nsConstrainedIsLowerHexDigest(
            anchors.optimization_protocol_sha256) ||
        !nsConstrainedIsLowerHexDigest(
            anchors.optimization_protocol_parent_sha256) ||
        !nsConstrainedIsLowerHexDigest(
            anchors.optimization_protocol_ancestor_2_sha256) ||
        !nsConstrainedIsLowerHexDigest(
            anchors.optimization_protocol_ancestor_3_sha256) ||
        !nsConstrainedIsLowerHexDigest(
            anchors.inherited_optimization_protocol_sha256) ||
        !nsConstrainedIsLowerHexDigest(
            anchors.implementation_sha256) ||
        !nsConstrainedIsLowerHexDigest(
            anchors.post_selection_protocol_v5_sha256))
        throw std::invalid_argument(
            "Constrained V2 artifact anchors must be lowercase SHA-256");
    if (!nsConstrainedIsLowerHexDigest(
            anchors.post_selection_protocol_v4_parent_sha256))
        throw std::invalid_argument(
            "Constrained V2 artifact anchors must be lowercase SHA-256");
}

inline int nsConstrainedLineSearchCandidateCap()
{
    return 80;
}

inline int nsConstrainedLineSearchMaxIterations()
{
    return nsConstrainedLineSearchCandidateCap() - 1;
}

template <typename T>
struct NSConstrainedRawEvaluation
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    std::array<T, NS_V2_METRIC_COUNT> metrics;
    Vector gauge;

    NSConstrainedRawEvaluation()
    {
        metrics.fill(T(0));
    }
};

template <typename T>
struct NSConstrainedBaseline
{
    std::array<T, NS_V2_METRIC_COUNT> metrics;

    NSConstrainedBaseline()
    {
        metrics.fill(T(0));
    }

    static NSConstrainedBaseline capture(
        const NSConstrainedRawEvaluation<T>& raw)
    {
        NSConstrainedBaseline baseline;
        for (int i = 0; i < NS_V2_METRIC_COUNT; ++i)
        {
            if (!std::isfinite(raw.metrics[i]) || !(raw.metrics[i] > T(0)))
                throw std::invalid_argument(
                    std::string("Constrained V2 Stage-A denominator is not "
                                "finite and positive: ") +
                    nsConstrainedMetricName(i));
            baseline.metrics[i] = raw.metrics[i];
        }
        if (raw.gauge.size() != 1001)
            throw std::invalid_argument(
                "Constrained V2 requires exactly 1001 gauge equalities");
        if (!raw.gauge.allFinite())
            throw std::invalid_argument(
                "Constrained V2 Stage-A gauge contains a nonfinite value");
        return baseline;
    }
};

template <typename T>
struct NSConstrainedConstraints
{
    std::array<T, NS_V2_INEQUALITY_COUNT> inequalities;
    Eigen::Matrix<T, Eigen::Dynamic, 1> gauge;

    NSConstrainedConstraints()
    {
        inequalities.fill(T(0));
    }
};

template <typename T>
inline T nsConstrainedTargetFactor(int metric_index)
{
    if (metric_index < NS_V2_STRONG_X ||
        metric_index >= NS_V2_METRIC_COUNT)
        throw std::invalid_argument(
            "Target factor requested for a non-mechanics metric");
    if (metric_index <= NS_V2_STRONG_C)
        return T(0.10) * T(0.10);
    return T(0.80) * T(0.80);
}

template <typename T>
inline NSConstrainedConstraints<T> nsConstrainedEvaluateConstraints(
    const NSConstrainedRawEvaluation<T>& raw,
    const NSConstrainedBaseline<T>& baseline,
    T eta)
{
    if (!std::isfinite(eta))
        throw std::invalid_argument("Constrained V2 eta must be finite");
    NSConstrainedConstraints<T> result;
    const T fidelity_factor = T(1.10) * T(1.10);
    for (int i = 0; i < NS_V2_FIDELITY_COUNT; ++i)
    {
        if (!std::isfinite(raw.metrics[i]))
            throw std::runtime_error(
                std::string("Nonfinite constrained V2 metric: ") +
                nsConstrainedMetricName(i));
        result.inequalities[i] =
            raw.metrics[i] / (fidelity_factor * baseline.metrics[i]) - T(1);
    }
    for (int j = 0; j < NS_V2_MECHANICS_COUNT; ++j)
    {
        const int metric = NS_V2_STRONG_X + j;
        if (!std::isfinite(raw.metrics[metric]))
            throw std::runtime_error(
                std::string("Nonfinite constrained V2 metric: ") +
                nsConstrainedMetricName(metric));
        result.inequalities[NS_V2_FIDELITY_COUNT + j] =
            raw.metrics[metric] /
                (nsConstrainedTargetFactor<T>(metric) *
                 baseline.metrics[metric]) -
            eta;
    }
    result.gauge = raw.gauge;
    if (!result.gauge.allFinite())
        throw std::runtime_error(
            "Nonfinite constrained V2 gauge equality");
    return result;
}

template <typename T>
inline T nsConstrainedPrimalInfeasibility(
    const NSConstrainedConstraints<T>& constraints)
{
    T theta = T(0);
    for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
        theta = std::max(theta, std::max(T(0), constraints.inequalities[i]));
    if (constraints.gauge.size())
        theta = std::max(theta, constraints.gauge.cwiseAbs().maxCoeff());
    return theta;
}

template <typename T>
struct NSConstrainedOuterState
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;

    std::array<T, NS_V2_INEQUALITY_COUNT> lambda;
    Vector mu;
    T rho;
    T previous_theta;
    int completed_outer_iterations;
    int total_inner_iterations;
    int consecutive_kkt_passes;
    T elapsed_seconds;

    NSConstrainedOuterState()
        : rho(T(1)),
          previous_theta(std::numeric_limits<T>::infinity()),
          completed_outer_iterations(0),
          total_inner_iterations(0),
          consecutive_kkt_passes(0),
          elapsed_seconds(T(0))
    {
        lambda.fill(T(0));
    }

    explicit NSConstrainedOuterState(int gauge_count)
        : rho(T(1)),
          previous_theta(std::numeric_limits<T>::infinity()),
          completed_outer_iterations(0),
          total_inner_iterations(0),
          consecutive_kkt_passes(0),
          elapsed_seconds(T(0))
    {
        lambda.fill(T(0));
        mu = Vector::Zero(gauge_count);
    }
};

template <typename T>
struct NSConstrainedKKTDiagnostics
{
    T theta;
    T projected_stationarity;
    T complementarity;
    bool pass;

    NSConstrainedKKTDiagnostics()
        : theta(std::numeric_limits<T>::infinity()),
          projected_stationarity(std::numeric_limits<T>::infinity()),
          complementarity(std::numeric_limits<T>::infinity()),
          pass(false)
    {
    }
};

template <typename T>
inline T nsConstrainedComplementarity(
    const std::array<T, NS_V2_INEQUALITY_COUNT>& lambda,
    const NSConstrainedConstraints<T>& constraints)
{
    T maximum = T(0);
    for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
    {
        if (!std::isfinite(lambda[i]) || lambda[i] < T(0))
            throw std::runtime_error(
                "Invalid constrained V2 inequality multiplier");
        maximum = std::max(
            maximum,
            std::abs(lambda[i] * constraints.inequalities[i]));
    }
    return maximum;
}

template <typename T>
inline T nsConstrainedProjectedInfinityNorm(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& z,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& gradient,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& lower,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& upper)
{
    if (z.size() == 0 || gradient.size() != z.size() ||
        lower.size() != z.size() || upper.size() != z.size())
        throw std::invalid_argument(
            "Projected KKT vectors have incompatible dimensions");
    return ((z - gradient).cwiseMax(lower).cwiseMin(upper) - z)
        .cwiseAbs().maxCoeff();
}

template <typename T>
inline void nsConstrainedUpdateMultipliersAndPenalty(
    const NSConstrainedConstraints<T>& constraints,
    NSConstrainedOuterState<T>& state,
    T theta)
{
    if (!(state.rho > T(0)) || !std::isfinite(state.rho))
        throw std::runtime_error("Invalid constrained V2 penalty");
    for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
        state.lambda[i] = std::max(
            T(0),
            state.lambda[i] + state.rho * constraints.inequalities[i]);
    if (state.mu.size() != constraints.gauge.size())
        throw std::invalid_argument(
            "Gauge multiplier dimension changed between outer iterations");
    state.mu.noalias() += state.rho * constraints.gauge;

    const bool sufficient_progress =
        !std::isfinite(state.previous_theta) ||
        theta <= T(0.25) * state.previous_theta;
    if (!sufficient_progress)
        state.rho = std::min(T(1.0e8), T(10) * state.rho);
    state.previous_theta = theta;
}

template <typename Provider>
class NSConstrainedAugmentedLagrangian
{
public:
    typedef typename Provider::Scalar Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    NSConstrainedAugmentedLagrangian(
        Provider& provider,
        const NSConstrainedBaseline<Scalar>& baseline,
        const NSConstrainedOuterState<Scalar>& state)
        : provider_(provider),
          baseline_(baseline),
          state_(state)
    {
        if (provider_.gaugeCount() != state_.mu.size())
            throw std::invalid_argument(
                "Constrained V2 gauge multiplier size mismatch");
    }

    Scalar operator()(const Vector& z, Vector& gradient)
    {
        if (z.size() != provider_.coefficientCount() + 1)
            throw std::invalid_argument(
                "Constrained V2 variable vector has the wrong size");
        const int coefficient_count = provider_.coefficientCount();
        const Vector coefficients = z.head(coefficient_count);
        const Scalar eta = z(coefficient_count);
        const NSConstrainedRawEvaluation<Scalar> raw =
            provider_.evaluate(coefficients);
        const NSConstrainedConstraints<Scalar> constraints =
            nsConstrainedEvaluateConstraints(raw, baseline_, eta);

        std::array<Scalar, NS_V2_METRIC_COUNT> metric_weights;
        metric_weights.fill(Scalar(0));
        Vector gauge_weights =
            (state_.mu + state_.rho * constraints.gauge) /
            static_cast<Scalar>(constraints.gauge.size());

        Scalar value = eta;
        Scalar eta_gradient = Scalar(1);
        const Scalar fidelity_factor = Scalar(1.10) * Scalar(1.10);
        for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
        {
            const Scalar shifted =
                constraints.inequalities[i] + state_.lambda[i] / state_.rho;
            const Scalar positive = std::max(Scalar(0), shifted);
            value +=
                (state_.rho / Scalar(2)) *
                (positive * positive -
                 (state_.lambda[i] / state_.rho) *
                 (state_.lambda[i] / state_.rho));
            const Scalar effective_multiplier =
                std::max(
                    Scalar(0),
                    state_.lambda[i] +
                    state_.rho * constraints.inequalities[i]);
            if (i < NS_V2_FIDELITY_COUNT)
            {
                metric_weights[i] +=
                    effective_multiplier /
                    (fidelity_factor * baseline_.metrics[i]);
            }
            else
            {
                const int metric =
                    NS_V2_STRONG_X + (i - NS_V2_FIDELITY_COUNT);
                metric_weights[metric] +=
                    effective_multiplier /
                    (nsConstrainedTargetFactor<Scalar>(metric) *
                     baseline_.metrics[metric]);
                eta_gradient -= effective_multiplier;
            }
        }
        value +=
            state_.mu.dot(constraints.gauge) /
                static_cast<Scalar>(constraints.gauge.size()) +
            (state_.rho /
             (Scalar(2) * static_cast<Scalar>(constraints.gauge.size()))) *
                constraints.gauge.squaredNorm();

        gradient.resize(z.size());
        gradient.head(coefficient_count) =
            provider_.weightedGradient(metric_weights, gauge_weights);
        gradient(coefficient_count) = eta_gradient;
        if (!std::isfinite(value) || !gradient.allFinite())
            throw std::runtime_error(
                "Nonfinite constrained V2 augmented-Lagrangian evaluation");
        last_raw_ = raw;
        last_constraints_ = constraints;
        return value;
    }

    const NSConstrainedRawEvaluation<Scalar>& lastRaw() const
    {
        return last_raw_;
    }

    const NSConstrainedConstraints<Scalar>& lastConstraints() const
    {
        return last_constraints_;
    }

private:
    Provider& provider_;
    const NSConstrainedBaseline<Scalar>& baseline_;
    const NSConstrainedOuterState<Scalar>& state_;
    NSConstrainedRawEvaluation<Scalar> last_raw_;
    NSConstrainedConstraints<Scalar> last_constraints_;
};

template <typename Provider>
inline Eigen::Matrix<typename Provider::Scalar, Eigen::Dynamic, 1>
nsConstrainedKKTGradient(
    Provider& provider,
    const NSConstrainedBaseline<typename Provider::Scalar>& baseline,
    const NSConstrainedOuterState<typename Provider::Scalar>& state,
    const NSConstrainedRawEvaluation<typename Provider::Scalar>& raw,
    typename Provider::Scalar eta)
{
    typedef typename Provider::Scalar T;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    const NSConstrainedConstraints<T> constraints =
        nsConstrainedEvaluateConstraints(raw, baseline, eta);
    std::array<T, NS_V2_METRIC_COUNT> metric_weights;
    metric_weights.fill(T(0));
    const T fidelity_factor = T(1.10) * T(1.10);
    T eta_gradient = T(1);
    for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
    {
        if (i < NS_V2_FIDELITY_COUNT)
            metric_weights[i] +=
                state.lambda[i] /
                (fidelity_factor * baseline.metrics[i]);
        else
        {
            const int metric =
                NS_V2_STRONG_X + (i - NS_V2_FIDELITY_COUNT);
            metric_weights[metric] +=
                state.lambda[i] /
                (nsConstrainedTargetFactor<T>(metric) *
                 baseline.metrics[metric]);
            eta_gradient -= state.lambda[i];
        }
    }
    const Vector gauge_weights =
        state.mu / static_cast<T>(constraints.gauge.size());
    Vector gradient(provider.coefficientCount() + 1);
    gradient.head(provider.coefficientCount()) =
        provider.weightedGradient(metric_weights, gauge_weights);
    gradient(provider.coefficientCount()) = eta_gradient;
    return gradient;
}

// A compact, self-contained SHA-256 implementation is kept here so every
// outer checkpoint can carry a cryptographic coefficient hash without
// invoking a shell or depending on an optional library.
class NSConstrainedSHA256
{
public:
    NSConstrainedSHA256()
    {
        reset();
    }

    void update(const unsigned char* data, size_t length)
    {
        for (size_t i = 0; i < length; ++i)
        {
            buffer_[buffer_size_++] = data[i];
            if (buffer_size_ == 64)
            {
                transform(buffer_);
                bit_count_ += 512;
                buffer_size_ = 0;
            }
        }
    }

    void update(const std::string& text)
    {
        update(
            reinterpret_cast<const unsigned char*>(text.data()),
            text.size());
    }

    std::string final()
    {
        const uint64_t total_bits =
            bit_count_ + static_cast<uint64_t>(buffer_size_) * 8;
        buffer_[buffer_size_++] = 0x80;
        if (buffer_size_ > 56)
        {
            while (buffer_size_ < 64)
                buffer_[buffer_size_++] = 0;
            transform(buffer_);
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56)
            buffer_[buffer_size_++] = 0;
        for (int i = 7; i >= 0; --i)
            buffer_[buffer_size_++] =
                static_cast<unsigned char>((total_bits >> (i * 8)) & 0xff);
        transform(buffer_);

        std::ostringstream stream;
        stream << std::hex << std::setfill('0');
        for (int i = 0; i < 8; ++i)
            stream << std::setw(8) << state_[i];
        const std::string digest = stream.str();
        reset();
        return digest;
    }

private:
    uint32_t state_[8];
    uint64_t bit_count_;
    unsigned char buffer_[64];
    size_t buffer_size_;

    static uint32_t rotateRight(uint32_t value, uint32_t count)
    {
        return (value >> count) | (value << (32 - count));
    }

    void reset()
    {
        state_[0] = 0x6a09e667u;
        state_[1] = 0xbb67ae85u;
        state_[2] = 0x3c6ef372u;
        state_[3] = 0xa54ff53au;
        state_[4] = 0x510e527fu;
        state_[5] = 0x9b05688cu;
        state_[6] = 0x1f83d9abu;
        state_[7] = 0x5be0cd19u;
        bit_count_ = 0;
        buffer_size_ = 0;
        std::memset(buffer_, 0, sizeof(buffer_));
    }

    void transform(const unsigned char block[64])
    {
        static const uint32_t constants[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
        };
        uint32_t words[64];
        for (int i = 0; i < 16; ++i)
        {
            words[i] =
                (static_cast<uint32_t>(block[4 * i]) << 24) |
                (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
                (static_cast<uint32_t>(block[4 * i + 2]) << 8) |
                static_cast<uint32_t>(block[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i)
        {
            const uint32_t s0 =
                rotateRight(words[i - 15], 7) ^
                rotateRight(words[i - 15], 18) ^
                (words[i - 15] >> 3);
            const uint32_t s1 =
                rotateRight(words[i - 2], 17) ^
                rotateRight(words[i - 2], 19) ^
                (words[i - 2] >> 10);
            words[i] =
                words[i - 16] + s0 + words[i - 7] + s1;
        }
        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];
        for (int i = 0; i < 64; ++i)
        {
            const uint32_t s1 =
                rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temp1 =
                h + s1 + choice + constants[i] + words[i];
            const uint32_t s0 =
                rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }
};

inline std::string nsConstrainedSHA256(const std::string& text)
{
    NSConstrainedSHA256 hash;
    hash.update(text);
    return hash.final();
}

inline std::string nsConstrainedFileSHA256(const std::string& path)
{
    std::ifstream stream(path.c_str(), std::ios::in | std::ios::binary);
    if (!stream)
        throw std::runtime_error(
            "Cannot open constrained V2 input for hashing: " + path);
    NSConstrainedSHA256 hash;
    unsigned char buffer[16384];
    while (stream)
    {
        stream.read(
            reinterpret_cast<char*>(buffer),
            static_cast<std::streamsize>(sizeof(buffer)));
        const std::streamsize count = stream.gcount();
        if (count > 0)
            hash.update(buffer, static_cast<size_t>(count));
    }
    if (!stream.eof())
        throw std::runtime_error(
            "Failed reading constrained V2 input for hashing: " + path);
    return hash.final();
}

template <typename Derived>
inline std::string nsConstrainedCoefficientPayload(
    const Eigen::MatrixBase<Derived>& coefficients)
{
    typedef typename Derived::Scalar T;
    std::ostringstream stream;
    stream << "# PIMFA_NS_CONSTRAINED_V2_COEFFICIENT_PAYLOAD n="
           << coefficients.size() << "\n"
           << std::setprecision(std::numeric_limits<T>::max_digits10);
    for (Eigen::Index i = 0; i < coefficients.size(); ++i)
    {
        if (!std::isfinite(coefficients(i)))
            throw std::invalid_argument(
                "Cannot hash nonfinite constrained V2 coefficients");
        stream << coefficients(i) << "\n";
    }
    return stream.str();
}

template <typename Derived>
inline std::string nsConstrainedCoefficientSHA256(
    const Eigen::MatrixBase<Derived>& coefficients)
{
    return nsConstrainedSHA256(
        nsConstrainedCoefficientPayload(coefficients));
}

template <typename T>
struct NSConstrainedOuterRecord
{
    NSConstrainedArtifactAnchors anchors;
    NSConstrainedRawEvaluation<T> raw;
    NSConstrainedConstraints<T> constraints;
    NSConstrainedKKTDiagnostics<T> kkt;
    std::array<T, NS_V2_INEQUALITY_COUNT> lambda;
    T eta;
    T rho_used;
    T rho_next;
    T inner_projected_gradient;
    int outer_iteration;
    int inner_iterations;
    int consecutive_kkt_passes;
    std::string inner_termination;
    std::string coefficient_sha256;
    T elapsed_seconds;
};

template <typename T>
inline void nsConstrainedAppendOuterRecord(
    const std::string& path,
    const NSConstrainedOuterRecord<T>& record,
    bool write_header)
{
    std::ofstream stream(
        path.c_str(),
        std::ios::out | (write_header ? std::ios::trunc : std::ios::app));
    if (!stream)
        throw std::runtime_error(
            "Cannot open constrained V2 outer-state record: " + path);
    if (write_header)
    {
        stream
            << "candidate_id\tstage_a_checkpoint_sha256"
            << "\tstage_a_baseline_record_sha256"
            << "\toptimization_protocol_sha256"
            << "\toptimization_protocol_parent_sha256"
            << "\toptimization_protocol_ancestor_2_sha256"
            << "\toptimization_protocol_ancestor_3_sha256"
            << "\tinherited_optimization_protocol_sha256"
            << "\timplementation_sha256"
            << "\tpost_selection_protocol_v5_sha256"
            << "\tpost_selection_protocol_v4_parent_sha256"
            << "\tline_search_class\tline_search_ftol"
            << "\tline_search_wolfe"
            << "\tline_search_max_candidate_evaluations"
            << "\touter_iteration\tinner_iterations\tinner_termination"
            << "\tinner_projected_gradient\teta\trho_used\trho_next"
            << "\ttheta\tprojected_stationarity\tcomplementarity"
            << "\tkkt_pass\tconsecutive_kkt_passes\tcoefficient_sha256"
            << "\telapsed_seconds";
        for (int i = 0; i < NS_V2_METRIC_COUNT; ++i)
            stream << "\traw_" << nsConstrainedMetricName(i);
        for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
            stream << "\tconstraint_" << i;
        for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
            stream << "\tlambda_" << i;
        stream << "\n";
    }
    nsConstrainedValidateArtifactAnchors(record.anchors);
    stream << nsConstrainedCandidateId() << "\t"
           << record.anchors.stage_a_checkpoint_sha256 << "\t"
           << record.anchors.stage_a_baseline_record_sha256 << "\t"
           << record.anchors.optimization_protocol_sha256 << "\t"
           << record.anchors.optimization_protocol_parent_sha256 << "\t"
           << record.anchors.optimization_protocol_ancestor_2_sha256 << "\t"
           << record.anchors.optimization_protocol_ancestor_3_sha256 << "\t"
           << record.anchors.inherited_optimization_protocol_sha256 << "\t"
           << record.anchors.implementation_sha256 << "\t"
           << record.anchors.post_selection_protocol_v5_sha256 << "\t"
           << record.anchors.post_selection_protocol_v4_parent_sha256 << "\t"
           << "NSConstrainedMoreThuenteBoundAware"
              "(LBFGSpp::LineSearchMoreThuente)\t"
           << std::setprecision(std::numeric_limits<T>::max_digits10)
           << T(1.0e-4) << "\t"
           << T(0.9) << "\t"
           << nsConstrainedLineSearchCandidateCap() << "\t"
           << std::setprecision(std::numeric_limits<T>::max_digits10)
           << record.outer_iteration << "\t"
           << record.inner_iterations << "\t"
           << record.inner_termination << "\t"
           << record.inner_projected_gradient << "\t"
           << record.eta << "\t"
           << record.rho_used << "\t"
           << record.rho_next << "\t"
           << record.kkt.theta << "\t"
           << record.kkt.projected_stationarity << "\t"
           << record.kkt.complementarity << "\t"
           << (record.kkt.pass ? 1 : 0) << "\t"
           << record.consecutive_kkt_passes << "\t"
           << record.coefficient_sha256 << "\t"
           << record.elapsed_seconds;
    for (int i = 0; i < NS_V2_METRIC_COUNT; ++i)
        stream << "\t" << record.raw.metrics[i];
    for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
        stream << "\t" << record.constraints.inequalities[i];
    for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
        stream << "\t" << record.lambda[i];
    stream << "\n";
    stream.close();
    if (!stream)
        throw std::runtime_error(
            "Failed writing constrained V2 outer-state record: " + path);
}

inline void nsConstrainedVerifyOuterRecordForResume(
    const std::string& path,
    int completed_outer_iterations,
    const NSConstrainedArtifactAnchors& anchors)
{
    nsConstrainedValidateArtifactAnchors(anchors);
    std::ifstream stream(path.c_str());
    if (completed_outer_iterations == 0)
    {
        if (stream && stream.peek() != std::ifstream::traits_type::eof())
            throw std::runtime_error(
                "Constrained V2 zero-outer resume has a nonempty record");
        return;
    }
    if (!stream)
        throw std::runtime_error(
            "Constrained V2 resume outer record does not exist");
    std::string header;
    if (!std::getline(stream, header) ||
        header.compare(
            0,
            std::string(
                "candidate_id\tstage_a_checkpoint_sha256\t"
                "stage_a_baseline_record_sha256\t"
                "optimization_protocol_sha256\t"
                "optimization_protocol_parent_sha256\t"
                "optimization_protocol_ancestor_2_sha256\t"
                "optimization_protocol_ancestor_3_sha256\t"
                "inherited_optimization_protocol_sha256\t"
                "implementation_sha256\t"
                "post_selection_protocol_v5_sha256\t"
                "post_selection_protocol_v4_parent_sha256\t").size(),
            "candidate_id\tstage_a_checkpoint_sha256\t"
            "stage_a_baseline_record_sha256\t"
            "optimization_protocol_sha256\t"
            "optimization_protocol_parent_sha256\t"
            "optimization_protocol_ancestor_2_sha256\t"
            "optimization_protocol_ancestor_3_sha256\t"
            "inherited_optimization_protocol_sha256\t"
            "implementation_sha256\t"
            "post_selection_protocol_v5_sha256\t"
            "post_selection_protocol_v4_parent_sha256\t") != 0)
        throw std::runtime_error(
            "Constrained V2 resume outer-record header mismatch");
    int expected_outer = 1;
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty())
            throw std::runtime_error(
                "Constrained V2 resume outer record has a blank row");
        std::istringstream parser(line);
        std::string field;
        const std::string expected_fields[11] = {
            nsConstrainedCandidateId(),
            anchors.stage_a_checkpoint_sha256,
            anchors.stage_a_baseline_record_sha256,
            anchors.optimization_protocol_sha256,
            anchors.optimization_protocol_parent_sha256,
            anchors.optimization_protocol_ancestor_2_sha256,
            anchors.optimization_protocol_ancestor_3_sha256,
            anchors.inherited_optimization_protocol_sha256,
            anchors.implementation_sha256,
            anchors.post_selection_protocol_v5_sha256,
            anchors.post_selection_protocol_v4_parent_sha256
        };
        for (int i = 0; i < 11; ++i)
        {
            if (!std::getline(parser, field, '\t') ||
                field != expected_fields[i])
                throw std::runtime_error(
                    "Constrained V2 resume outer-record anchor mismatch");
        }
        if (!std::getline(parser, field, '\t') ||
            field !=
                "NSConstrainedMoreThuenteBoundAware"
                "(LBFGSpp::LineSearchMoreThuente)")
            throw std::runtime_error(
                "Constrained V2 resume outer-record line-search mismatch");
        if (!std::getline(parser, field, '\t'))
            throw std::runtime_error(
                "Constrained V2 resume outer record is truncated");
        {
            std::istringstream value_parser(field);
            double value = 0.0;
            char extra = '\0';
            if (!(value_parser >> value) ||
                (value_parser >> extra) ||
                value != 1.0e-4)
                throw std::runtime_error(
                    "Constrained V2 resume outer-record ftol mismatch");
        }
        if (!std::getline(parser, field, '\t'))
            throw std::runtime_error(
                "Constrained V2 resume outer record is truncated");
        {
            std::istringstream value_parser(field);
            double value = 0.0;
            char extra = '\0';
            if (!(value_parser >> value) ||
                (value_parser >> extra) ||
                value != 0.9)
                throw std::runtime_error(
                    "Constrained V2 resume outer-record wolfe mismatch");
        }
        if (!std::getline(parser, field, '\t'))
            throw std::runtime_error(
                "Constrained V2 resume outer record is truncated");
        {
            std::istringstream value_parser(field);
            int value = 0;
            char extra = '\0';
            if (!(value_parser >> value) ||
                (value_parser >> extra) ||
                value != nsConstrainedLineSearchCandidateCap())
                throw std::runtime_error(
                    "Constrained V2 resume outer-record trial-cap mismatch");
        }
        if (!std::getline(parser, field, '\t'))
            throw std::runtime_error(
                "Constrained V2 resume outer record is truncated");
        std::istringstream outer_parser(field);
        int observed_outer = 0;
        char extra = '\0';
        if (!(outer_parser >> observed_outer) ||
            (outer_parser >> extra) ||
            observed_outer != expected_outer)
            throw std::runtime_error(
                "Constrained V2 resume outer iteration sequence mismatch");
        ++expected_outer;
    }
    if (expected_outer - 1 != completed_outer_iterations)
        throw std::runtime_error(
            "Constrained V2 resume state/outer-record length mismatch");
}

template <typename T>
inline std::string nsConstrainedStatePayload(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& z,
    const NSConstrainedBaseline<T>& baseline,
    const NSConstrainedOuterState<T>& state,
    const NSConstrainedArtifactAnchors& anchors)
{
    if (z.size() < 2)
        throw std::invalid_argument(
            "Constrained V2 state requires coefficients and eta");
    nsConstrainedValidateArtifactAnchors(anchors);
    std::ostringstream stream;
    stream << "PIMFA_NS_CONSTRAINED_MINIMAX_AL_V2_STATE\n"
           << "candidate_id=" << nsConstrainedCandidateId() << "\n"
           << "stage_a_checkpoint_sha256="
           << anchors.stage_a_checkpoint_sha256 << "\n"
           << "stage_a_baseline_record_sha256="
           << anchors.stage_a_baseline_record_sha256 << "\n"
           << "optimization_protocol_sha256="
           << anchors.optimization_protocol_sha256 << "\n"
           << "optimization_protocol_parent_sha256="
           << anchors.optimization_protocol_parent_sha256 << "\n"
           << "optimization_protocol_ancestor_2_sha256="
           << anchors.optimization_protocol_ancestor_2_sha256 << "\n"
           << "optimization_protocol_ancestor_3_sha256="
           << anchors.optimization_protocol_ancestor_3_sha256 << "\n"
           << "inherited_optimization_protocol_sha256="
           << anchors.inherited_optimization_protocol_sha256 << "\n"
           << "implementation_sha256="
           << anchors.implementation_sha256 << "\n"
           << "post_selection_protocol_v5_sha256="
           << anchors.post_selection_protocol_v5_sha256 << "\n"
           << "post_selection_protocol_v4_parent_sha256="
           << anchors.post_selection_protocol_v4_parent_sha256 << "\n"
           << "coefficient_count=" << (z.size() - 1) << "\n"
           << "gauge_count=" << state.mu.size() << "\n"
           << "outer_completed=" << state.completed_outer_iterations << "\n"
           << "total_inner_iterations=" << state.total_inner_iterations << "\n"
           << "consecutive_kkt_passes=" << state.consecutive_kkt_passes << "\n"
           << std::setprecision(std::numeric_limits<T>::max_digits10)
           << "rho=" << state.rho << "\n"
           << "previous_theta=" << state.previous_theta << "\n"
           << "elapsed_seconds=" << state.elapsed_seconds << "\n"
           << "coefficient_sha256="
           << nsConstrainedCoefficientSHA256(z.head(z.size() - 1)) << "\n";
    stream << "baseline";
    for (int i = 0; i < NS_V2_METRIC_COUNT; ++i)
        stream << " " << baseline.metrics[i];
    stream << "\nlambda";
    for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
        stream << " " << state.lambda[i];
    stream << "\nmu";
    for (Eigen::Index i = 0; i < state.mu.size(); ++i)
        stream << " " << state.mu(i);
    stream << "\nz";
    for (Eigen::Index i = 0; i < z.size(); ++i)
        stream << " " << z(i);
    stream << "\n";
    return stream.str();
}

inline void nsConstrainedAtomicWrite(
    const std::string& path,
    const std::string& payload)
{
    if (path.empty())
        throw std::invalid_argument(
            "Constrained V2 output path must not be empty");
    const std::string temporary = path + ".tmp";
    {
        std::ofstream stream(
            temporary.c_str(), std::ios::out | std::ios::trunc);
        if (!stream)
            throw std::runtime_error(
                "Cannot create constrained V2 temporary state: " + temporary);
        stream << payload;
        stream.close();
        if (!stream)
            throw std::runtime_error(
                "Failed writing constrained V2 temporary state: " + temporary);
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0)
    {
        std::remove(temporary.c_str());
        throw std::runtime_error(
            "Cannot atomically publish constrained V2 state: " + path);
    }
}

template <typename T>
inline void nsConstrainedWriteState(
    const std::string& path,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& z,
    const NSConstrainedBaseline<T>& baseline,
    const NSConstrainedOuterState<T>& state,
    const NSConstrainedArtifactAnchors& anchors)
{
    nsConstrainedAtomicWrite(
        path, nsConstrainedStatePayload(z, baseline, state, anchors));
}

inline std::string nsConstrainedReadKey(
    std::istream& stream,
    const char* expected)
{
    std::string line;
    if (!std::getline(stream, line))
        throw std::runtime_error(
            std::string("Truncated constrained V2 state before ") + expected);
    const std::string prefix = std::string(expected) + "=";
    if (line.compare(0, prefix.size(), prefix) != 0)
        throw std::runtime_error(
            std::string("Expected constrained V2 state key ") + expected);
    return line.substr(prefix.size());
}

template <typename T>
inline T nsConstrainedParseFinite(
    const std::string& text,
    const char* label,
    bool allow_positive_infinity = false)
{
    if (allow_positive_infinity && text == "inf")
        return std::numeric_limits<T>::infinity();
    std::istringstream stream(text);
    T value;
    char extra = '\0';
    if (!(stream >> value) || (stream >> extra) || !std::isfinite(value))
        throw std::runtime_error(
            std::string("Malformed constrained V2 state scalar: ") + label);
    return value;
}

inline int nsConstrainedParseInteger(
    const std::string& text,
    const char* label,
    int minimum)
{
    std::istringstream stream(text);
    int value;
    char extra = '\0';
    if (!(stream >> value) || (stream >> extra) || value < minimum)
        throw std::runtime_error(
            std::string("Malformed constrained V2 state integer: ") + label);
    return value;
}

template <typename T>
inline void nsConstrainedReadVectorLine(
    std::istream& stream,
    const char* marker,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& values)
{
    std::string line;
    if (!std::getline(stream, line))
        throw std::runtime_error(
            std::string("Missing constrained V2 state vector: ") + marker);
    std::istringstream parser(line);
    std::string actual;
    parser >> actual;
    if (actual != marker)
        throw std::runtime_error(
            std::string("Expected constrained V2 state vector: ") + marker);
    for (Eigen::Index i = 0; i < values.size(); ++i)
        if (!(parser >> values(i)) || !std::isfinite(values(i)))
            throw std::runtime_error(
                std::string("Truncated/nonfinite constrained V2 vector: ") +
                marker);
    T extra;
    if (parser >> extra)
        throw std::runtime_error(
            std::string("Extra constrained V2 vector value: ") + marker);
}

template <typename T>
inline void nsConstrainedReadArrayLine(
    std::istream& stream,
    const char* marker,
    T* values,
    int count)
{
    std::string line;
    if (!std::getline(stream, line))
        throw std::runtime_error(
            std::string("Missing constrained V2 state array: ") + marker);
    std::istringstream parser(line);
    std::string actual;
    parser >> actual;
    if (actual != marker)
        throw std::runtime_error(
            std::string("Expected constrained V2 state array: ") + marker);
    for (int i = 0; i < count; ++i)
        if (!(parser >> values[i]) || !std::isfinite(values[i]))
            throw std::runtime_error(
                std::string("Truncated/nonfinite constrained V2 array: ") +
                marker);
    T extra;
    if (parser >> extra)
        throw std::runtime_error(
            std::string("Extra constrained V2 array value: ") + marker);
}

template <typename T>
inline void nsConstrainedLoadState(
    const std::string& path,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& z,
    NSConstrainedBaseline<T>& baseline,
    NSConstrainedOuterState<T>& state,
    int expected_coefficient_count,
    int expected_gauge_count,
    const NSConstrainedArtifactAnchors& expected_anchors)
{
    nsConstrainedValidateArtifactAnchors(expected_anchors);
    std::ifstream stream(path.c_str());
    if (!stream)
        throw std::runtime_error(
            "Cannot open constrained V2 state: " + path);
    std::string magic;
    std::getline(stream, magic);
    if (magic != "PIMFA_NS_CONSTRAINED_MINIMAX_AL_V2_STATE")
        throw std::runtime_error("Invalid constrained V2 state magic");
    if (nsConstrainedReadKey(stream, "candidate_id") !=
        nsConstrainedCandidateId())
        throw std::runtime_error(
            "Constrained V2 state candidate mismatch");
    if (nsConstrainedReadKey(
            stream, "stage_a_checkpoint_sha256") !=
            expected_anchors.stage_a_checkpoint_sha256 ||
        nsConstrainedReadKey(
            stream, "stage_a_baseline_record_sha256") !=
            expected_anchors.stage_a_baseline_record_sha256 ||
        nsConstrainedReadKey(
            stream, "optimization_protocol_sha256") !=
            expected_anchors.optimization_protocol_sha256 ||
        nsConstrainedReadKey(
            stream, "optimization_protocol_parent_sha256") !=
            expected_anchors.optimization_protocol_parent_sha256 ||
        nsConstrainedReadKey(
            stream, "optimization_protocol_ancestor_2_sha256") !=
            expected_anchors.optimization_protocol_ancestor_2_sha256 ||
        nsConstrainedReadKey(
            stream, "optimization_protocol_ancestor_3_sha256") !=
            expected_anchors.optimization_protocol_ancestor_3_sha256 ||
        nsConstrainedReadKey(
            stream, "inherited_optimization_protocol_sha256") !=
            expected_anchors.inherited_optimization_protocol_sha256 ||
        nsConstrainedReadKey(
            stream, "implementation_sha256") !=
            expected_anchors.implementation_sha256 ||
        nsConstrainedReadKey(
            stream, "post_selection_protocol_v5_sha256") !=
            expected_anchors.post_selection_protocol_v5_sha256 ||
        nsConstrainedReadKey(
            stream, "post_selection_protocol_v4_parent_sha256") !=
            expected_anchors.post_selection_protocol_v4_parent_sha256)
        throw std::runtime_error(
            "Constrained V2 state artifact-anchor mismatch");
    const int coefficient_count = nsConstrainedParseInteger(
        nsConstrainedReadKey(stream, "coefficient_count"),
        "coefficient_count", 1);
    const int gauge_count = nsConstrainedParseInteger(
        nsConstrainedReadKey(stream, "gauge_count"),
        "gauge_count", 1);
    if (coefficient_count != expected_coefficient_count ||
        gauge_count != expected_gauge_count)
        throw std::runtime_error(
            "Constrained V2 state dimension mismatch");
    state.completed_outer_iterations = nsConstrainedParseInteger(
        nsConstrainedReadKey(stream, "outer_completed"),
        "outer_completed", 0);
    state.total_inner_iterations = nsConstrainedParseInteger(
        nsConstrainedReadKey(stream, "total_inner_iterations"),
        "total_inner_iterations", 0);
    state.consecutive_kkt_passes = nsConstrainedParseInteger(
        nsConstrainedReadKey(stream, "consecutive_kkt_passes"),
        "consecutive_kkt_passes", 0);
    state.rho = nsConstrainedParseFinite<T>(
        nsConstrainedReadKey(stream, "rho"), "rho");
    state.previous_theta = nsConstrainedParseFinite<T>(
        nsConstrainedReadKey(stream, "previous_theta"),
        "previous_theta", true);
    state.elapsed_seconds = nsConstrainedParseFinite<T>(
        nsConstrainedReadKey(stream, "elapsed_seconds"),
        "elapsed_seconds");
    const std::string expected_hash =
        nsConstrainedReadKey(stream, "coefficient_sha256");
    nsConstrainedReadArrayLine(
        stream, "baseline", baseline.metrics.data(), NS_V2_METRIC_COUNT);
    nsConstrainedReadArrayLine(
        stream, "lambda", state.lambda.data(), NS_V2_INEQUALITY_COUNT);
    state.mu.resize(gauge_count);
    nsConstrainedReadVectorLine(stream, "mu", state.mu);
    z.resize(coefficient_count + 1);
    nsConstrainedReadVectorLine(stream, "z", z);
    std::string trailing;
    while (std::getline(stream, trailing))
        if (!trailing.empty())
            throw std::runtime_error(
                "Constrained V2 state has trailing content");
    if (nsConstrainedCoefficientSHA256(z.head(coefficient_count)) !=
        expected_hash)
        throw std::runtime_error(
            "Constrained V2 state coefficient hash mismatch");
    NSConstrainedRawEvaluation<T> baseline_proxy;
    baseline_proxy.metrics = baseline.metrics;
    baseline_proxy.gauge = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(
        gauge_count);
    NSConstrainedBaseline<T>::capture(baseline_proxy);
    for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
        if (state.lambda[i] < T(0))
            throw std::runtime_error(
                "Constrained V2 state has a negative inequality multiplier");
    if (state.previous_theta < T(0) ||
        state.elapsed_seconds < T(0))
        throw std::runtime_error(
            "Constrained V2 state has a negative diagnostic");
    for (int i = 0; i < coefficient_count; ++i)
        if (z(i) < T(-100) || z(i) > T(100))
            throw std::runtime_error(
                "Constrained V2 state coefficient violates frozen bounds");
    if (z(coefficient_count) < T(0) ||
        z(coefficient_count) > T(100))
        throw std::runtime_error(
            "Constrained V2 state eta violates frozen bounds");
    bool valid_rho = false;
    for (T candidate = T(1); candidate <= T(1.0e8);
         candidate *= T(10))
        if (state.rho == candidate)
            valid_rho = true;
    if (!(state.rho >= T(1)) || state.rho > T(1.0e8) ||
        !valid_rho ||
        state.completed_outer_iterations > 8 ||
        state.total_inner_iterations > 1600 ||
        state.total_inner_iterations >
            200 * state.completed_outer_iterations ||
        state.consecutive_kkt_passes > 2)
        throw std::runtime_error(
            "Constrained V2 state violates frozen outer bounds");
}

template <typename Derived>
inline void nsConstrainedWriteFinalCheckpoint(
    const std::string& path,
    const Eigen::MatrixBase<Derived>& coefficients,
    typename Derived::Scalar eta,
    bool converged,
    const std::string& decoded_handshake_sha256,
    const NSConstrainedArtifactAnchors& anchors)
{
    typedef typename Derived::Scalar T;
    nsConstrainedValidateArtifactAnchors(anchors);
    if (!nsConstrainedIsLowerHexDigest(decoded_handshake_sha256))
        throw std::invalid_argument(
            "Constrained V2 decoded handshake must be lowercase SHA-256");
    std::ostringstream stream;
    stream
        << "# PIMFA_NS_CONSTRAINED_MINIMAX_AL_V2"
        << " candidate_id=" << nsConstrainedCandidateId()
        << " stage_a_checkpoint_sha256="
        << anchors.stage_a_checkpoint_sha256
        << " stage_a_baseline_record_sha256="
        << anchors.stage_a_baseline_record_sha256
        << " optimization_protocol_sha256="
        << anchors.optimization_protocol_sha256
        << " optimization_protocol_parent_sha256="
        << anchors.optimization_protocol_parent_sha256
        << " optimization_protocol_ancestor_2_sha256="
        << anchors.optimization_protocol_ancestor_2_sha256
        << " optimization_protocol_ancestor_3_sha256="
        << anchors.optimization_protocol_ancestor_3_sha256
        << " inherited_optimization_protocol_sha256="
        << anchors.inherited_optimization_protocol_sha256
        << " implementation_sha256="
        << anchors.implementation_sha256
        << " post_selection_protocol_v5_sha256="
        << anchors.post_selection_protocol_v5_sha256
        << " post_selection_protocol_v4_parent_sha256="
        << anchors.post_selection_protocol_v4_parent_sha256
        << " n=" << coefficients.size()
        << " eta=" << std::setprecision(std::numeric_limits<T>::max_digits10)
        << eta
        << " converged=" << (converged ? 1 : 0)
        << " coefficient_sha256="
        << nsConstrainedCoefficientSHA256(coefficients)
        << " decoded_handshake_sha256=" << decoded_handshake_sha256
        << "\n"
        << std::setprecision(std::numeric_limits<T>::max_digits10);
    for (Eigen::Index i = 0; i < coefficients.size(); ++i)
        stream << coefficients(i) << "\n";
    nsConstrainedAtomicWrite(path, stream.str());
}

enum NSConstrainedLineSearchAcceptance
{
    NS_CONSTRAINED_LINE_SEARCH_REJECT = 0,
    NS_CONSTRAINED_LINE_SEARCH_STRONG_WOLFE = 1,
    NS_CONSTRAINED_LINE_SEARCH_BOX_ENDPOINT_ARMIJO_KKT = 2
};

template <typename Scalar>
inline Scalar nsConstrainedArmijoRightHandSide(
    Scalar initial_value,
    Scalar step,
    Scalar ftol,
    Scalar initial_derivative)
{
    return initial_value + step * (ftol * initial_derivative);
}

// Classify one return from LBFGS++'s Moré--Thuente routine.  Interior steps
// must satisfy the unmodified strong-Wolfe conditions.  At a genuine
// box-feasible endpoint, where no larger step is feasible, Armijo decrease
// plus the one-sided derivative condition phi'(alpha_max) <= 0 is the
// appropriate line-restricted KKT condition.  A numerical max-step endpoint
// and every nonendpoint fallback remain failures.
template <typename Scalar>
inline NSConstrainedLineSearchAcceptance
nsConstrainedClassifyMoreThuenteReturnWithArmijo(
    Scalar initial_value,
    Scalar initial_derivative,
    Scalar step,
    Scalar maximum_step,
    Scalar numerical_maximum_step,
    Scalar value,
    Scalar directional_derivative,
    Scalar armijo_rhs,
    Scalar ftol,
    Scalar wolfe)
{
    if (!std::isfinite(initial_value) ||
        !std::isfinite(initial_derivative) ||
        !std::isfinite(step) ||
        !std::isfinite(maximum_step) ||
        !std::isfinite(numerical_maximum_step) ||
        !std::isfinite(value) ||
        !std::isfinite(directional_derivative) ||
        !std::isfinite(armijo_rhs) ||
        !std::isfinite(ftol) ||
        !std::isfinite(wolfe) ||
        !(initial_derivative < Scalar(0)) ||
        !(step > Scalar(0)) ||
        !(step <= maximum_step) ||
        !(maximum_step <= numerical_maximum_step) ||
        !(ftol > Scalar(0)) ||
        !(wolfe > ftol) ||
        !(wolfe < Scalar(1)))
        return NS_CONSTRAINED_LINE_SEARCH_REJECT;

    const bool sufficient_decrease = value <= armijo_rhs;
    const bool genuine_box_endpoint =
        step == maximum_step &&
        maximum_step < numerical_maximum_step;

    // A genuine box endpoint is classified exclusively by the endpoint
    // rule.  In particular, a positive one-sided derivative may happen to
    // lie inside the strong-curvature band, but is still inadmissible at the
    // right endpoint.
    if (genuine_box_endpoint)
    {
        if (value < initial_value &&
            sufficient_decrease &&
            directional_derivative <= Scalar(0))
            return NS_CONSTRAINED_LINE_SEARCH_BOX_ENDPOINT_ARMIJO_KKT;
        return NS_CONSTRAINED_LINE_SEARCH_REJECT;
    }

    const bool strong_curvature =
        std::abs(directional_derivative) <=
            -wolfe * initial_derivative;
    if (sufficient_decrease && strong_curvature)
        return NS_CONSTRAINED_LINE_SEARCH_STRONG_WOLFE;

    return NS_CONSTRAINED_LINE_SEARCH_REJECT;
}

template <typename Scalar>
inline NSConstrainedLineSearchAcceptance
nsConstrainedClassifyMoreThuenteReturn(
    Scalar initial_value,
    Scalar initial_derivative,
    Scalar step,
    Scalar maximum_step,
    Scalar numerical_maximum_step,
    Scalar value,
    Scalar directional_derivative,
    Scalar ftol,
    Scalar wolfe)
{
    const Scalar armijo_rhs = nsConstrainedArmijoRightHandSide(
        initial_value, step, ftol, initial_derivative);
    return nsConstrainedClassifyMoreThuenteReturnWithArmijo(
        initial_value, initial_derivative, step, maximum_step,
        numerical_maximum_step, value, directional_derivative,
        armijo_rhs, ftol, wolfe);
}

template <typename Scalar, typename Vector>
inline bool nsConstrainedMoreThuenteReturnConsistent(
    const Vector& previous,
    const Vector& direction,
    Scalar step,
    const Vector& current,
    const Vector& gradient,
    Scalar directional_derivative,
    int candidate_evaluations,
    int configured_candidate_cap,
    Scalar* vector_consistency_error = 0,
    Scalar* vector_consistency_bound = 0,
    Scalar* derivative_consistency_error = 0,
    Scalar* derivative_consistency_bound = 0)
{
    if (previous.size() <= 0 ||
        direction.size() != previous.size() ||
        current.size() != previous.size() ||
        gradient.size() != previous.size() ||
        !std::isfinite(step) ||
        !std::isfinite(directional_derivative) ||
        candidate_evaluations <= 0 ||
        candidate_evaluations > configured_candidate_cap ||
        !previous.allFinite() ||
        !direction.allFinite() ||
        !current.allFinite() ||
        !gradient.allFinite())
        return false;

    Vector expected(current.size());
    expected.noalias() = previous + step * direction;
    const Scalar coordinate_scale = std::max(
        Scalar(1),
        std::max(
            current.cwiseAbs().maxCoeff(),
            std::max(
                previous.cwiseAbs().maxCoeff(),
                std::abs(step) * direction.cwiseAbs().maxCoeff())));
    const Scalar computed_vector_bound =
        Scalar(64) * std::numeric_limits<Scalar>::epsilon() *
        coordinate_scale;
    const Scalar computed_vector_error =
        (current - expected).cwiseAbs().maxCoeff();

    const Scalar recomputed_derivative = gradient.dot(direction);
    const Scalar derivative_scale = std::max(
        Scalar(1),
        std::max(
            std::abs(directional_derivative),
            gradient.norm() * direction.norm()));
    const Scalar computed_derivative_bound =
        Scalar(256) * static_cast<Scalar>(gradient.size()) *
        std::numeric_limits<Scalar>::epsilon() * derivative_scale;
    const Scalar computed_derivative_error =
        std::abs(recomputed_derivative - directional_derivative);

    if (vector_consistency_error)
        *vector_consistency_error = computed_vector_error;
    if (vector_consistency_bound)
        *vector_consistency_bound = computed_vector_bound;
    if (derivative_consistency_error)
        *derivative_consistency_error = computed_derivative_error;
    if (derivative_consistency_bound)
        *derivative_consistency_bound = computed_derivative_bound;

    return
        std::isfinite(computed_vector_error) &&
        std::isfinite(coordinate_scale) &&
        std::isfinite(computed_vector_bound) &&
        computed_vector_error <= computed_vector_bound &&
        std::isfinite(recomputed_derivative) &&
        std::isfinite(derivative_scale) &&
        std::isfinite(computed_derivative_error) &&
        std::isfinite(computed_derivative_bound) &&
        computed_derivative_error <= computed_derivative_bound;
}

struct NSConstrainedLineSearchTelemetry
{
    int outer_iteration;
    std::uint64_t calls;
    std::uint64_t candidate_evaluations;
    std::uint64_t cumulative_candidate_evaluations;
    std::uint64_t cumulative_candidate_attempts;
    std::uint64_t cumulative_candidate_completions;
    std::uint64_t cumulative_forced_midpoints;
    std::uint64_t call_candidate_attempts;
    std::uint64_t call_candidate_completions;
    std::uint64_t call_forced_midpoints;
    std::uint64_t primary_candidate_attempts;
    std::uint64_t primary_candidate_completions;
    std::uint64_t primary_forced_midpoints;
    std::uint64_t retry_candidate_attempts;
    std::uint64_t retry_candidate_completions;
    std::uint64_t retry_forced_midpoints;
    std::uint64_t invocation_candidate_attempts;
    std::uint64_t invocation_candidate_completions;
    std::uint64_t invocation_forced_midpoints;
    double last_attempted_step;
    bool last_attempted_step_present;
    double last_completed_step;
    bool last_completed_step_present;
    double last_evaluated_step;
    bool last_evaluated_step_present;
    double invocation_last_attempted_step;
    bool invocation_last_attempted_step_present;
    double invocation_last_completed_step;
    bool invocation_last_completed_step_present;
    std::uint64_t strong_wolfe_accepts;
    std::uint64_t box_endpoint_accepts;

    NSConstrainedLineSearchTelemetry()
        : outer_iteration(-1),
          calls(0),
          candidate_evaluations(0),
          cumulative_candidate_evaluations(0),
          cumulative_candidate_attempts(0),
          cumulative_candidate_completions(0),
          cumulative_forced_midpoints(0),
          call_candidate_attempts(0),
          call_candidate_completions(0),
          call_forced_midpoints(0),
          primary_candidate_attempts(0),
          primary_candidate_completions(0),
          primary_forced_midpoints(0),
          retry_candidate_attempts(0),
          retry_candidate_completions(0),
          retry_forced_midpoints(0),
          invocation_candidate_attempts(0),
          invocation_candidate_completions(0),
          invocation_forced_midpoints(0),
          last_attempted_step(0.0),
          last_attempted_step_present(false),
          last_completed_step(0.0),
          last_completed_step_present(false),
          last_evaluated_step(0.0),
          last_evaluated_step_present(false),
          invocation_last_attempted_step(0.0),
          invocation_last_attempted_step_present(false),
          invocation_last_completed_step(0.0),
          invocation_last_completed_step_present(false),
          strong_wolfe_accepts(0),
          box_endpoint_accepts(0)
    {
    }
};

// Portable exception used only by the governed throwing-oracle regression.
// Keeping its type and message storage here lets the production wrapper prove
// that a provider exception survives with its dynamic cause and exact what()
// bytes rather than being relabelled as a generator guard.
class NSConstrainedTestThrowingOracleException : public std::exception
{
public:
    explicit NSConstrainedTestThrowingOracleException(
        const std::string& message)
        : message_(message)
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

inline std::string nsConstrainedLowerHexWhatBytes(const char* value)
{
    static const char digits[] = "0123456789abcdef";
    std::string encoded;
    if (!value)
        return encoded;
    const unsigned char* current =
        reinterpret_cast<const unsigned char*>(value);
    while (*current != 0)
    {
        encoded.push_back(digits[(*current >> 4) & 0x0f]);
        encoded.push_back(digits[*current & 0x0f]);
        ++current;
    }
    return encoded;
}

// LBFGS++'s Moré--Thuente generator may normally return a best Armijo point
// after exhausting its budget or reaching a step limit.  This wrapper keeps
// the unchanged strong-Wolfe/box-endpoint acceptance rules, shares one exact
// 80-attempt ledger across a primary invocation and its sole optional retry,
// and durably records every abort before propagating its original cause.
template <class Scalar>
class NSConstrainedMoreThuenteBoundAware
{
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef LBFGSpp::LineSearchMoreThuente<Scalar> Generator;
    typedef typename Generator::Diagnostics Diagnostics;
    typedef typename Generator::CandidateAccounting CandidateAccounting;
    typedef typename Generator::InvocationContext InvocationContext;
    typedef typename Generator::GuardException GuardException;

    struct CallSnapshot
    {
        std::uint64_t call;
        bool primary_present;
        Diagnostics primary_diagnostics;
        bool retry_present;
        Diagnostics retry_diagnostics;
        CandidateAccounting call_accounting;
        Vector true_box_endpoint;
        bool true_box_endpoint_present;
        std::string invocation;
        std::string exception_class;
        std::string guard_reason;
        std::string original_exception_type;
        std::string original_exception_message_hex;
        std::string termination_reason;

        CallSnapshot()
            : call(0),
              primary_present(false),
              primary_diagnostics(),
              retry_present(false),
              retry_diagnostics(),
              call_accounting(
                  static_cast<std::uint64_t>(
                      nsConstrainedLineSearchCandidateCap())),
              true_box_endpoint(),
              true_box_endpoint_present(false),
              invocation("PRIMARY"),
              exception_class("NONE"),
              guard_reason("NONE"),
              original_exception_type("NONE"),
              original_exception_message_hex(),
              termination_reason("NONE")
        {
        }
    };

    static void resetTelemetry(int outer_iteration = -1)
    {
        telemetry() = NSConstrainedLineSearchTelemetry();
        telemetry().outer_iteration = outer_iteration;
        callSnapshot() = CallSnapshot();
    }

    static NSConstrainedLineSearchTelemetry currentTelemetry()
    {
        return telemetry();
    }

    static CallSnapshot currentCallSnapshot()
    {
        return callSnapshot();
    }

    template <typename Foo, typename SolverParam>
    static void LineSearch(
        Foo& function,
        const SolverParam& parameters,
        const Vector& previous,
        const Vector& direction,
        const Scalar& maximum_step,
        Scalar& step,
        Scalar& value,
        Vector& gradient,
        Scalar& directional_derivative,
        Vector& current)
    {
        NSConstrainedLineSearchTelemetry& counters = telemetry();
        CallSnapshot& snapshot = callSnapshot();
        snapshot = CallSnapshot();
        const std::uint64_t call_index = ++counters.calls;
        snapshot.call = call_index;
        const Scalar initial_value = value;
        const Scalar initial_derivative = directional_derivative;
        const Scalar initial_step = step;
        const Vector initial_gradient = gradient;
        const int configured_candidate_cap =
            nsConstrainedLineSearchCandidateCap();
        CandidateAccounting accounting(
            static_cast<std::uint64_t>(configured_candidate_cap));
        accounting.cumulative_attempts =
            &counters.cumulative_candidate_attempts;
        accounting.cumulative_completions =
            &counters.cumulative_candidate_completions;
        accounting.cumulative_forced_midpoints =
            &counters.cumulative_forced_midpoints;

        Diagnostics primary_diagnostics;
        Diagnostics retry_diagnostics;

        const bool nonfinite_input =
            !std::isfinite(initial_value) ||
            !std::isfinite(initial_derivative) ||
            !std::isfinite(maximum_step) ||
            !std::isfinite(parameters.max_step) ||
            !previous.allFinite() ||
            !direction.allFinite() ||
            !gradient.allFinite();
        if (nonfinite_input ||
            !(initial_derivative < Scalar(0)) ||
            !(maximum_step > Scalar(0)) ||
            !(maximum_step <= parameters.max_step))
        {
            commitInvocation(
                snapshot, false, primary_diagnostics);
            finalizeCall(counters, snapshot, accounting);
            snapshot.exception_class = "LINE_SEARCH_REJECTION";
            snapshot.guard_reason = "NONE";
            snapshot.original_exception_type = "NONE";
            snapshot.original_exception_message_hex.clear();
            snapshot.termination_reason =
                nonfinite_input ?
                    "NONFINITE" :
                !(initial_derivative < Scalar(0)) ?
                    "NO_DESCENT" :
                    "RETURN_NO_ACCEPTANCE";
            emitAbort(counters, snapshot);

            std::ostringstream message;
            message.imbue(std::locale::classic());
            message
                << "Constrained V2 Moré--Thuente termination="
                << snapshot.termination_reason
                << " outer_iteration=" << counters.outer_iteration
                << " call=" << call_index
                << " candidate_evaluations=0"
                << " f0="
                << std::setprecision(
                       std::numeric_limits<Scalar>::max_digits10)
                << initial_value
                << " dg0=" << initial_derivative
                << " step=" << step
                << " step_max=" << maximum_step;
            throw std::runtime_error(message.str());
        }

        bool provider_exception_in_flight = false;
        struct ProviderForwarder
        {
            Foo& target;
            bool& provider_exception;

            ProviderForwarder(
                Foo& value,
                bool& provider_exception_flag)
                : target(value),
                  provider_exception(provider_exception_flag)
            {
            }

            Scalar operator()(const Vector& x, Vector& output_gradient)
            {
                provider_exception = false;
                try
                {
                    return target(x, output_gradient);
                }
                catch (...)
                {
                    provider_exception = true;
                    throw;
                }
            }
        };
        ProviderForwarder forwarded(
            function, provider_exception_in_flight);
        InvocationContext primary_context(&accounting, 0);

        try
        {
            Generator::LineSearchWithDiagnostics(
                forwarded, parameters, previous, direction,
                maximum_step, step, value, gradient,
                directional_derivative, current,
                &primary_diagnostics, &primary_context);
        }
        catch (...)
        {
            abortCurrentException(
                std::current_exception(),
                provider_exception_in_flight,
                false,
                primary_diagnostics,
                accounting,
                counters,
                snapshot);
            throw;
        }
        commitInvocation(
            snapshot, false, primary_diagnostics);

        Scalar armijo_rhs = nsConstrainedArmijoRightHandSide(
            initial_value, step, parameters.ftol, initial_derivative);
        bool endpoint_interior_retry = false;
        const bool positive_box_endpoint_with_armijo =
            std::isfinite(value) &&
            std::isfinite(directional_derivative) &&
            step == maximum_step &&
            maximum_step < parameters.max_step &&
            value < initial_value &&
            value <= armijo_rhs &&
            directional_derivative > Scalar(0);
        const int remaining_candidate_evaluations =
            configured_candidate_cap -
            static_cast<int>(accounting.call_attempts);
        if (positive_box_endpoint_with_armijo &&
            remaining_candidate_evaluations >= 2)
        {
            const Scalar interior_maximum_step =
                std::nextafter(maximum_step, Scalar(0));
            if (std::isfinite(interior_maximum_step) &&
                interior_maximum_step > Scalar(0) &&
                interior_maximum_step < maximum_step)
            {
                // This is the exact vector observed by the primary provider.
                // It must be copied before restoring state and must never be
                // reconstructed as previous + maximum_step * direction.
                const Vector true_box_endpoint = current;
                snapshot.true_box_endpoint = true_box_endpoint;
                snapshot.true_box_endpoint_present = true;
                SolverParam retry_parameters = parameters;
                retry_parameters.max_linesearch =
                    remaining_candidate_evaluations - 1;
                step = std::min(initial_step, interior_maximum_step);
                value = initial_value;
                gradient = initial_gradient;
                directional_derivative = initial_derivative;
                current = previous;
                endpoint_interior_retry = true;
                provider_exception_in_flight = false;
                InvocationContext retry_context(
                    &accounting, &true_box_endpoint);
                try
                {
                    Generator::LineSearchWithDiagnostics(
                        forwarded, retry_parameters,
                        previous, direction,
                        interior_maximum_step,
                        step, value, gradient,
                        directional_derivative, current,
                        &retry_diagnostics, &retry_context);
                }
                catch (...)
                {
                    abortCurrentException(
                        std::current_exception(),
                        provider_exception_in_flight,
                        true,
                        retry_diagnostics,
                        accounting,
                        counters,
                        snapshot);
                    throw;
                }
                commitInvocation(
                    snapshot, true, retry_diagnostics);
                armijo_rhs = nsConstrainedArmijoRightHandSide(
                    initial_value, step, parameters.ftol,
                    initial_derivative);
            }
        }
        finalizeCall(counters, snapshot, accounting);

        const bool strong_curvature =
            std::abs(directional_derivative) <=
                -parameters.wolfe * initial_derivative;
        const bool sufficient_decrease = value <= armijo_rhs;
        const bool box_limited =
            maximum_step < parameters.max_step;

        Scalar vector_consistency_error =
            std::numeric_limits<Scalar>::infinity();
        Scalar vector_consistency_bound = Scalar(0);
        Scalar derivative_consistency_error =
            std::numeric_limits<Scalar>::infinity();
        Scalar derivative_consistency_bound = Scalar(0);
        const bool numerically_consistent_return =
            nsConstrainedMoreThuenteReturnConsistent(
                previous, direction, step, current, gradient,
                directional_derivative,
                static_cast<int>(accounting.call_attempts),
                configured_candidate_cap, &vector_consistency_error,
                &vector_consistency_bound,
                &derivative_consistency_error,
                &derivative_consistency_bound);
        const bool consistent_return =
            numerically_consistent_return;

        const NSConstrainedLineSearchAcceptance acceptance =
            nsConstrainedClassifyMoreThuenteReturnWithArmijo(
                initial_value, initial_derivative, step, maximum_step,
                parameters.max_step, value, directional_derivative,
                armijo_rhs, parameters.ftol, parameters.wolfe);
        if (consistent_return &&
            acceptance == NS_CONSTRAINED_LINE_SEARCH_STRONG_WOLFE)
        {
            snapshot.exception_class = "NONE";
            snapshot.guard_reason = "NONE";
            snapshot.termination_reason = "STRONG_WOLFE";
            ++counters.strong_wolfe_accepts;
            return;
        }
        if (consistent_return &&
            acceptance ==
                NS_CONSTRAINED_LINE_SEARCH_BOX_ENDPOINT_ARMIJO_KKT)
        {
            snapshot.exception_class = "NONE";
            snapshot.guard_reason = "NONE";
            snapshot.termination_reason =
                "BOX_ENDPOINT_ARMIJO_KKT";
            ++counters.box_endpoint_accepts;
            return;
        }

        const Diagnostics& final_diagnostics =
            snapshot.retry_present ?
                snapshot.retry_diagnostics :
                snapshot.primary_diagnostics;
        const char* termination =
            !std::isfinite(value) ||
                !std::isfinite(directional_derivative) ||
                !gradient.allFinite()
            ? "NONFINITE"
            : !consistent_return
            ? "INCONSISTENT_RETURN"
            : step == maximum_step &&
                maximum_step >= parameters.max_step
            ? "PARAM_MAX_NO_SW"
            : accounting.call_attempts >=
                    static_cast<std::uint64_t>(
                        configured_candidate_cap) ||
                final_diagnostics.termination_reason ==
                    LBFGSpp::LBFGSPP_MT_TERMINATION_MAX_ITERATIONS
            ? "MAX_EVAL_NO_SW"
            : "RETURN_NO_ACCEPTANCE";
        snapshot.exception_class = "LINE_SEARCH_REJECTION";
        snapshot.guard_reason = "NONE";
        snapshot.original_exception_type = "NONE";
        snapshot.original_exception_message_hex.clear();
        snapshot.termination_reason = termination;
        emitAbort(counters, snapshot);

        std::ostringstream message;
        message.imbue(std::locale::classic());
        message
            << "Constrained V2 Moré--Thuente termination=" << termination
            << " outer_iteration=" << counters.outer_iteration
            << " call=" << call_index
            << " candidate_evaluations="
            << accounting.call_attempts
            << " configured_candidate_cap="
            << configured_candidate_cap
            << " cumulative_candidate_evaluations="
            << counters.cumulative_candidate_evaluations
            << " step="
            << std::setprecision(
                   std::numeric_limits<Scalar>::max_digits10)
            << step
            << " step_max=" << maximum_step
            << " numerical_max_step=" << parameters.max_step
            << " box_limited=" << (box_limited ? 1 : 0)
            << " endpoint_interior_retry="
            << (endpoint_interior_retry ? 1 : 0)
            << " endpoint_retry_vector_coincident="
            << 0
            << " f0=" << initial_value
            << " f=" << value
            << " dg0=" << initial_derivative
            << " dg=" << directional_derivative
            << " armijo_rhs=" << armijo_rhs
            << " armijo_margin=" << (armijo_rhs - value)
            << " curvature_bound="
            << (-parameters.wolfe * initial_derivative)
            << " curvature_margin="
            << (-parameters.wolfe * initial_derivative -
                std::abs(directional_derivative))
            << " endpoint_derivative_margin="
            << (-directional_derivative)
            << " sufficient_decrease="
            << (sufficient_decrease ? 1 : 0)
            << " strong_curvature=" << (strong_curvature ? 1 : 0)
            << " vector_consistency_error="
            << vector_consistency_error
            << " vector_consistency_bound="
            << vector_consistency_bound
            << " derivative_consistency_error="
            << derivative_consistency_error
            << " derivative_consistency_bound="
            << derivative_consistency_bound;
        throw std::runtime_error(message.str());
    }

private:
    static void commitInvocation(
        CallSnapshot& snapshot,
        bool retry,
        const Diagnostics& diagnostics)
    {
        if (retry)
        {
            if (!snapshot.retry_present)
            {
                snapshot.retry_diagnostics = diagnostics;
                snapshot.retry_present = true;
            }
            snapshot.invocation = "RETRY";
        }
        else
        {
            if (!snapshot.primary_present)
            {
                snapshot.primary_diagnostics = diagnostics;
                snapshot.primary_present = true;
            }
            snapshot.invocation = "PRIMARY";
        }
    }

    static void finalizeCall(
        NSConstrainedLineSearchTelemetry& counters,
        CallSnapshot& snapshot,
        const CandidateAccounting& accounting)
    {
        snapshot.call_accounting = accounting;
        snapshot.call_accounting.cumulative_attempts = 0;
        snapshot.call_accounting.cumulative_completions = 0;
        snapshot.call_accounting.cumulative_forced_midpoints = 0;

        counters.candidate_evaluations =
            accounting.call_attempts;
        counters.cumulative_candidate_evaluations =
            counters.cumulative_candidate_attempts;
        counters.call_candidate_attempts =
            accounting.call_attempts;
        counters.call_candidate_completions =
            accounting.call_completions;
        counters.call_forced_midpoints =
            accounting.call_forced_midpoints;

        counters.primary_candidate_attempts =
            snapshot.primary_present ?
                snapshot.primary_diagnostics.candidate_attempts : 0;
        counters.primary_candidate_completions =
            snapshot.primary_present ?
                snapshot.primary_diagnostics.candidate_completions : 0;
        counters.primary_forced_midpoints =
            snapshot.primary_present ?
                snapshot.primary_diagnostics.forced_midpoints : 0;
        counters.retry_candidate_attempts =
            snapshot.retry_present ?
                snapshot.retry_diagnostics.candidate_attempts : 0;
        counters.retry_candidate_completions =
            snapshot.retry_present ?
                snapshot.retry_diagnostics.candidate_completions : 0;
        counters.retry_forced_midpoints =
            snapshot.retry_present ?
                snapshot.retry_diagnostics.forced_midpoints : 0;

        const Diagnostics& invocation_diagnostics =
            snapshot.retry_present ?
                snapshot.retry_diagnostics :
                snapshot.primary_diagnostics;
        counters.invocation_candidate_attempts =
            invocation_diagnostics.candidate_attempts;
        counters.invocation_candidate_completions =
            invocation_diagnostics.candidate_completions;
        counters.invocation_forced_midpoints =
            invocation_diagnostics.forced_midpoints;

        counters.last_attempted_step =
            static_cast<double>(accounting.last_attempted_step);
        counters.last_attempted_step_present =
            accounting.last_attempted_step_present;
        counters.last_completed_step =
            static_cast<double>(accounting.last_completed_step);
        counters.last_completed_step_present =
            accounting.last_completed_step_present;
        counters.last_evaluated_step =
            counters.last_completed_step;
        counters.last_evaluated_step_present =
            counters.last_completed_step_present;

        counters.invocation_last_attempted_step =
            static_cast<double>(
                invocation_diagnostics.last_attempted_step);
        counters.invocation_last_attempted_step_present =
            invocation_diagnostics.last_attempted_step_present;
        counters.invocation_last_completed_step =
            static_cast<double>(
                invocation_diagnostics.last_completed_step);
        counters.invocation_last_completed_step_present =
            invocation_diagnostics.last_completed_step_present;
    }

    static std::string terminationForGeneratorException(
        const Diagnostics& diagnostics)
    {
        if (diagnostics.termination_reason ==
            LBFGSpp::LBFGSPP_MT_TERMINATION_MAX_ITERATIONS)
            return "MAX_EVAL_NO_SW";
        if (diagnostics.termination_reason ==
            LBFGSpp::LBFGSPP_MT_TERMINATION_OBJECTIVE_PROVIDER_EXCEPTION)
            return "OBJECTIVE_PROVIDER_EXCEPTION";
        if (diagnostics.termination_reason ==
            LBFGSpp::LBFGSPP_MT_TERMINATION_GUARD_EXCEPTION)
            return "GUARD_EXCEPTION";
        return "RETURN_NO_ACCEPTANCE";
    }

    static void abortCurrentException(
        const std::exception_ptr& cause,
        bool provider_exception_in_flight,
        bool retry,
        const Diagnostics& diagnostics,
        const CandidateAccounting& accounting,
        NSConstrainedLineSearchTelemetry& counters,
        CallSnapshot& snapshot)
    {
        commitInvocation(snapshot, retry, diagnostics);
        finalizeCall(counters, snapshot, accounting);
        snapshot.original_exception_type = "NONE";
        snapshot.original_exception_message_hex.clear();
        snapshot.guard_reason = "NONE";

        // Origin takes precedence over the dynamic exception class.  In
        // particular, an objective is allowed to throw GuardException and
        // must still be recorded as an objective/provider exception.
        const bool provider_exception =
            provider_exception_in_flight ||
            diagnostics.termination_reason ==
                LBFGSpp::LBFGSPP_MT_TERMINATION_OBJECTIVE_PROVIDER_EXCEPTION;
        if (provider_exception)
        {
            snapshot.exception_class =
                "OBJECTIVE_PROVIDER_EXCEPTION";
            snapshot.termination_reason =
                "OBJECTIVE_PROVIDER_EXCEPTION";
            try
            {
                std::rethrow_exception(cause);
            }
            catch (
                const NSConstrainedTestThrowingOracleException& error)
            {
                snapshot.original_exception_type =
                    "PIMFA_TEST_THROWING_ORACLE";
                snapshot.original_exception_message_hex =
                    nsConstrainedLowerHexWhatBytes(error.what());
            }
            catch (const std::exception& error)
            {
                snapshot.original_exception_type =
                    "PIMFA_OBJECTIVE_PROVIDER_STD_EXCEPTION";
                snapshot.original_exception_message_hex =
                    nsConstrainedLowerHexWhatBytes(error.what());
            }
            catch (...)
            {
                snapshot.original_exception_type =
                    "PIMFA_OBJECTIVE_PROVIDER_NONSTD_EXCEPTION";
            }
        }
        else
        {
            snapshot.exception_class =
                "LINE_SEARCH_REJECTION";
            snapshot.termination_reason =
                terminationForGeneratorException(diagnostics);
            try
            {
                std::rethrow_exception(cause);
            }
            catch (const GuardException& error)
            {
                snapshot.exception_class =
                    "GENERATOR_GUARD_EXCEPTION";
                snapshot.guard_reason =
                    LBFGSpp::lineSearchMoreThuenteGuardReasonName(
                        error.reason());
                snapshot.termination_reason =
                    "GUARD_EXCEPTION";
            }
            catch (...)
            {
            }
        }

        emitAbort(counters, snapshot);
    }

    static void appendJSONFloat(
        std::ostringstream& stream,
        Scalar value,
        bool present)
    {
        if (present && std::isfinite(value))
            stream << value;
        else
            stream << "null";
    }

    static void emitAbort(
        const NSConstrainedLineSearchTelemetry& counters,
        const CallSnapshot& snapshot)
    {
        const Diagnostics& diagnostics =
            snapshot.retry_present ?
                snapshot.retry_diagnostics :
                snapshot.primary_diagnostics;
        const CandidateAccounting& accounting =
            snapshot.call_accounting;
        const bool bracket_present =
            diagnostics.final_bracketed;

        std::ostringstream stream;
        stream.imbue(std::locale::classic());
        stream << std::setprecision(
            std::numeric_limits<Scalar>::max_digits10);
        stream << "PIMFA_V17_LINE_SEARCH_ABORT {\"bracket_hi\":";
        appendJSONFloat(
            stream, diagnostics.final_bracket_hi,
            bracket_present);
        stream << ",\"bracket_lo\":";
        appendJSONFloat(
            stream, diagnostics.final_bracket_lo,
            bracket_present);
        stream
            << ",\"bracket_present\":"
            << (bracket_present ? "true" : "false")
            << ",\"bracket_width\":";
        appendJSONFloat(
            stream, diagnostics.final_bracket_width,
            bracket_present);
        stream
            << ",\"call\":" << snapshot.call
            << ",\"call_attempts\":"
            << accounting.call_attempts
            << ",\"call_completions\":"
            << accounting.call_completions
            << ",\"call_forced_midpoints\":"
            << accounting.call_forced_midpoints
            << ",\"candidate_evaluations\":"
            << accounting.call_attempts
            << ",\"configured_attempt_cap\":"
            << nsConstrainedLineSearchCandidateCap()
            << ",\"configured_candidate_cap\":"
            << nsConstrainedLineSearchCandidateCap()
            << ",\"cumulative_attempts\":"
            << counters.cumulative_candidate_attempts
            << ",\"cumulative_candidate_evaluations\":"
            << counters.cumulative_candidate_attempts
            << ",\"cumulative_completions\":"
            << counters.cumulative_candidate_completions
            << ",\"cumulative_forced_midpoints\":"
            << counters.cumulative_forced_midpoints
            << ",\"exception_class\":\""
            << snapshot.exception_class
            << "\",\"guard_reason\":\""
            << snapshot.guard_reason
            << "\",\"invocation\":\""
            << snapshot.invocation
            << "\",\"invocation_attempts\":"
            << diagnostics.candidate_attempts
            << ",\"invocation_completions\":"
            << diagnostics.candidate_completions
            << ",\"invocation_forced_midpoints\":"
            << diagnostics.forced_midpoints
            << ",\"invocation_last_attempted_step\":";
        appendJSONFloat(
            stream, diagnostics.last_attempted_step,
            diagnostics.last_attempted_step_present);
        stream
            << ",\"invocation_last_attempted_step_present\":"
            << (diagnostics.last_attempted_step_present ?
                    "true" : "false")
            << ",\"invocation_last_completed_step\":";
        appendJSONFloat(
            stream, diagnostics.last_completed_step,
            diagnostics.last_completed_step_present);
        stream
            << ",\"invocation_last_completed_step_present\":"
            << (diagnostics.last_completed_step_present ?
                    "true" : "false")
            << ",\"last_attempted_step\":";
        appendJSONFloat(
            stream, accounting.last_attempted_step,
            accounting.last_attempted_step_present);
        stream
            << ",\"last_attempted_step_present\":"
            << (accounting.last_attempted_step_present ?
                    "true" : "false")
            << ",\"last_completed_step\":";
        appendJSONFloat(
            stream, accounting.last_completed_step,
            accounting.last_completed_step_present);
        stream
            << ",\"last_completed_step_present\":"
            << (accounting.last_completed_step_present ?
                    "true" : "false")
            << ",\"last_evaluated_step\":";
        appendJSONFloat(
            stream, accounting.last_completed_step,
            accounting.last_completed_step_present);
        stream
            << ",\"last_evaluated_step_present\":"
            << (accounting.last_completed_step_present ?
                    "true" : "false")
            << ",\"midpoint_trigger\":\""
            << LBFGSpp::lineSearchMoreThuenteMidpointTriggerName(
                   diagnostics.midpoint_trigger)
            << "\",\"original_exception_message_hex\":\""
            << snapshot.original_exception_message_hex
            << "\",\"original_exception_type\":\""
            << snapshot.original_exception_type
            << "\",\"outer_iteration\":"
            << counters.outer_iteration
            << ",\"primary_attempts\":"
            << (snapshot.primary_present ?
                    snapshot.primary_diagnostics.candidate_attempts : 0)
            << ",\"primary_completions\":"
            << (snapshot.primary_present ?
                    snapshot.primary_diagnostics.candidate_completions : 0)
            << ",\"primary_forced_midpoints\":"
            << (snapshot.primary_present ?
                    snapshot.primary_diagnostics.forced_midpoints : 0)
            << ",\"proposal_origin\":\""
            << LBFGSpp::lineSearchMoreThuenteProposalOriginName(
                   diagnostics.proposal_origin)
            << "\",\"raw_proposed_step\":";
        appendJSONFloat(
            stream, diagnostics.raw_proposed_step,
            diagnostics.raw_proposed_step_present);
        stream
            << ",\"raw_proposed_step_class\":\""
            << LBFGSpp::lineSearchMoreThuenteRawStepClassName(
                   diagnostics.raw_proposed_step_class)
            << "\",\"raw_proposed_step_present\":"
            << (diagnostics.raw_proposed_step_present ?
                    "true" : "false")
            << ",\"retry_attempts\":"
            << (snapshot.retry_present ?
                    snapshot.retry_diagnostics.candidate_attempts : 0)
            << ",\"retry_completions\":"
            << (snapshot.retry_present ?
                    snapshot.retry_diagnostics.candidate_completions : 0)
            << ",\"retry_forced_midpoints\":"
            << (snapshot.retry_present ?
                    snapshot.retry_diagnostics.forced_midpoints : 0)
            << ",\"safeguarded_proposed_step\":";
        appendJSONFloat(
            stream, diagnostics.safeguarded_proposed_step,
            diagnostics.safeguarded_proposed_step_present);
        stream
            << ",\"safeguarded_proposed_step_present\":"
            << (diagnostics.safeguarded_proposed_step_present ?
                    "true" : "false")
            << ",\"termination_reason\":\""
            << snapshot.termination_reason
            << "\"}\n";

        const std::string record = stream.str();
        std::cerr.write(
            record.data(),
            static_cast<std::streamsize>(record.size()));
        std::cerr.flush();
    }

    static NSConstrainedLineSearchTelemetry& telemetry()
    {
        static thread_local NSConstrainedLineSearchTelemetry counters;
        return counters;
    }

    static CallSnapshot& callSnapshot()
    {
        static thread_local CallSnapshot snapshot;
        return snapshot;
    }
};

template <typename T>
inline LBFGSpp::LBFGSBParam<T> nsConstrainedInnerParameters(
    int outer_iteration,
    int remaining_inner_iterations)
{
    if (outer_iteration < 0 || outer_iteration >= 8 ||
        remaining_inner_iterations <= 0)
        throw std::invalid_argument(
            "Invalid constrained V2 inner-solver schedule index");
    LBFGSpp::LBFGSBParam<T> parameters;
    parameters.m = 6;
    parameters.epsilon =
        outer_iteration == 0 ? T(1.0e-4) :
        outer_iteration == 1 ? T(1.0e-5) :
        outer_iteration == 2 ? T(1.0e-6) : T(1.0e-7);
    parameters.epsilon_rel = T(0);
    parameters.past = 0;
    parameters.delta = T(0);
    parameters.max_iterations =
        std::min(200, remaining_inner_iterations);
    parameters.max_submin = 10;
    // LBFGS++ evaluates one initial candidate before this loop counter.
    // V17 freezes 79 loop evaluations and therefore an exact shared cap of
    // 80 attempted provider invocations per bound-aware line-search call.
    parameters.max_linesearch =
        nsConstrainedLineSearchMaxIterations();
    parameters.min_step = T(1.0e-20);
    parameters.max_step = T(1.0e20);
    parameters.ftol = T(1.0e-4);
    parameters.wolfe = T(0.9);
    parameters.check_param();
    return parameters;
}

template <typename T>
struct NSConstrainedRunOptions
{
    std::string outer_record_path;
    std::string state_path;
    NSConstrainedArtifactAnchors anchors;
    bool resume;

    NSConstrainedRunOptions() : resume(false) {}
};

template <typename T>
struct NSConstrainedRunResult
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    Vector coefficients;
    T eta;
    bool converged;
    std::string termination;
    NSConstrainedBaseline<T> baseline;
    NSConstrainedOuterState<T> state;
    NSConstrainedRawEvaluation<T> raw;
    NSConstrainedConstraints<T> constraints;
    NSConstrainedKKTDiagnostics<T> kkt;
};

template <typename Provider>
NSConstrainedRunResult<typename Provider::Scalar>
runNSConstrainedMinimaxAL(
    Provider& provider,
    const Eigen::Matrix<typename Provider::Scalar, Eigen::Dynamic, 1>&
        stage_a_coefficients,
    const NSConstrainedRunOptions<typename Provider::Scalar>& options)
{
    typedef typename Provider::Scalar T;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    if (provider.coefficientCount() != stage_a_coefficients.size())
        throw std::invalid_argument(
            "Stage-A coefficient count does not match constrained V2 provider");
    if (provider.gaugeCount() != 1001)
        throw std::invalid_argument(
            "Constrained V2 production provider must expose 1001 gauges");
    if (options.outer_record_path.empty() || options.state_path.empty())
        throw std::invalid_argument(
            "Constrained V2 requires outer-record and state paths");
    nsConstrainedValidateArtifactAnchors(options.anchors);

    Vector z(stage_a_coefficients.size() + 1);
    z.head(stage_a_coefficients.size()) = stage_a_coefficients;
    z(stage_a_coefficients.size()) = T(100);
    NSConstrainedOuterState<T> state(provider.gaugeCount());
    NSConstrainedBaseline<T> baseline;

    if (options.resume)
    {
        nsConstrainedLoadState(
            options.state_path, z, baseline, state,
            provider.coefficientCount(), provider.gaugeCount(),
            options.anchors);
        nsConstrainedVerifyOuterRecordForResume(
            options.outer_record_path,
            state.completed_outer_iterations,
            options.anchors);
    }
    else
    {
        const NSConstrainedRawEvaluation<T> stage_a =
            provider.evaluate(stage_a_coefficients);
        baseline = NSConstrainedBaseline<T>::capture(stage_a);
        const NSConstrainedConstraints<T> initial =
            nsConstrainedEvaluateConstraints(stage_a, baseline, T(100));
        for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
            if (initial.inequalities[i] > T(64) *
                std::numeric_limits<T>::epsilon())
                throw std::runtime_error(
                    "Constrained V2 Stage-A/eta=100 is not a known "
                    "feasible start");
        nsConstrainedWriteState(
            options.state_path, z, baseline, state, options.anchors);
    }

    Vector lower =
        Vector::Constant(provider.coefficientCount() + 1, T(-100));
    Vector upper =
        Vector::Constant(provider.coefficientCount() + 1, T(100));
    lower(provider.coefficientCount()) = T(0);
    upper(provider.coefficientCount()) = T(100);
    std::chrono::steady_clock::time_point last_tick =
        std::chrono::steady_clock::now();

    NSConstrainedRunResult<T> result;
    result.converged = state.consecutive_kkt_passes >= 2;
    result.termination = result.converged
        ? "two_consecutive_kkt_passes"
        : "outer_cap_nonconverged";
    bool have_diagnostics = false;
    bool first_record = state.completed_outer_iterations == 0;
    while (state.completed_outer_iterations < 8 &&
           state.total_inner_iterations < 1600 &&
           state.consecutive_kkt_passes < 2)
    {
        const int outer = state.completed_outer_iterations;
        const LBFGSpp::LBFGSBParam<T> parameters =
            nsConstrainedInnerParameters<T>(
                outer, 1600 - state.total_inner_iterations);
        // The template argument makes the preregistered bound-aware
        // Moré--Thuente implementation a compile-time choice rather than a
        // library-default dependency.
        NSConstrainedMoreThuenteBoundAware<T>::resetTelemetry(outer);
        LBFGSpp::LBFGSBSolver<
            T, NSConstrainedMoreThuenteBoundAware> solver(parameters);
        NSConstrainedAugmentedLagrangian<Provider> objective(
            provider, baseline, state);
        T objective_value = T(0);
        const int inner_iterations =
            solver.minimize(objective, z, objective_value, lower, upper);
        const T inner_projected_gradient = solver.final_grad_norm();
        state.total_inner_iterations += inner_iterations;

        const Vector coefficients = z.head(provider.coefficientCount());
        const T eta = z(provider.coefficientCount());
        const NSConstrainedRawEvaluation<T> raw =
            provider.evaluate(coefficients);
        const NSConstrainedConstraints<T> constraints =
            nsConstrainedEvaluateConstraints(raw, baseline, eta);
        const T theta = nsConstrainedPrimalInfeasibility(constraints);
        const T rho_used = state.rho;

        for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
            state.lambda[i] = std::max(
                T(0),
                state.lambda[i] +
                state.rho * constraints.inequalities[i]);
        state.mu.noalias() += state.rho * constraints.gauge;

        // provider.evaluate() above establishes the exact cache consumed by
        // weightedGradient() in the KKT stationarity calculation.
        const Vector kkt_gradient = nsConstrainedKKTGradient(
            provider, baseline, state, raw, eta);
        NSConstrainedKKTDiagnostics<T> kkt;
        kkt.theta = theta;
        kkt.projected_stationarity =
            nsConstrainedProjectedInfinityNorm(
                z, kkt_gradient, lower, upper);
        kkt.complementarity =
            nsConstrainedComplementarity(state.lambda, constraints);
        kkt.pass =
            kkt.theta <= T(1.0e-6) &&
            kkt.projected_stationarity <= T(1.0e-6) &&
            kkt.complementarity <= T(1.0e-6);
        if (kkt.pass)
            ++state.consecutive_kkt_passes;
        else
            state.consecutive_kkt_passes = 0;

        const bool sufficient_progress =
            !std::isfinite(state.previous_theta) ||
            theta <= T(0.25) * state.previous_theta;
        if (!sufficient_progress)
            state.rho = std::min(T(1.0e8), T(10) * state.rho);
        state.previous_theta = theta;
        ++state.completed_outer_iterations;
        const std::chrono::steady_clock::time_point now =
            std::chrono::steady_clock::now();
        state.elapsed_seconds += static_cast<T>(
            std::chrono::duration<double>(now - last_tick).count());
        last_tick = now;

        NSConstrainedOuterRecord<T> record;
        record.anchors = options.anchors;
        record.raw = raw;
        record.constraints = constraints;
        record.kkt = kkt;
        record.lambda = state.lambda;
        record.eta = eta;
        record.rho_used = rho_used;
        record.rho_next = state.rho;
        record.inner_projected_gradient = inner_projected_gradient;
        record.outer_iteration = state.completed_outer_iterations;
        record.inner_iterations = inner_iterations;
        record.consecutive_kkt_passes = state.consecutive_kkt_passes;
        if (inner_projected_gradient <= parameters.epsilon)
            record.inner_termination = "projected_gradient";
        else if (inner_iterations >= parameters.max_iterations)
            record.inner_termination = "max_iterations";
        else
            throw std::runtime_error(
                "Constrained V2 L-BFGS-B returned without a frozen "
                "termination condition");
        record.coefficient_sha256 =
            nsConstrainedCoefficientSHA256(coefficients);
        record.elapsed_seconds = state.elapsed_seconds;
        nsConstrainedAppendOuterRecord(
            options.outer_record_path, record, first_record);
        first_record = false;
        nsConstrainedWriteState(
            options.state_path, z, baseline, state, options.anchors);

        result.raw = raw;
        result.constraints = constraints;
        result.kkt = kkt;
        have_diagnostics = true;
        if (state.consecutive_kkt_passes >= 2)
        {
            result.converged = true;
            result.termination = "two_consecutive_kkt_passes";
            break;
        }
    }

    result.coefficients = z.head(provider.coefficientCount());
    result.eta = z(provider.coefficientCount());
    result.baseline = baseline;
    result.state = state;
    if (!have_diagnostics)
    {
        result.raw = provider.evaluate(result.coefficients);
        result.constraints =
            nsConstrainedEvaluateConstraints(
                result.raw, result.baseline, result.eta);
        const Vector kkt_gradient = nsConstrainedKKTGradient(
            provider, result.baseline, result.state,
            result.raw, result.eta);
        result.kkt.theta =
            nsConstrainedPrimalInfeasibility(result.constraints);
        result.kkt.projected_stationarity =
            nsConstrainedProjectedInfinityNorm(
                z, kkt_gradient, lower, upper);
        result.kkt.complementarity =
            nsConstrainedComplementarity(
                result.state.lambda, result.constraints);
        result.kkt.pass =
            result.kkt.theta <= T(1.0e-6) &&
            result.kkt.projected_stationarity <= T(1.0e-6) &&
            result.kkt.complementarity <= T(1.0e-6);
        if (result.converged && !result.kkt.pass)
            throw std::runtime_error(
                "Constrained V2 converged resume state fails recomputed KKT");
    }
    return result;
}

} // namespace mfa

#endif

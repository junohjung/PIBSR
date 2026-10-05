#include <mfa/ns_constrained_minimax_executable.hpp>

#include <Eigen/Core>
#include <LBFGSpp/LineSearchMoreThuente.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

#include <unistd.h>

namespace
{

typedef double Scalar;
typedef Eigen::VectorXd Vector;
typedef LBFGSpp::LineSearchMoreThuente<Scalar> TestGenerator;
typedef mfa::NSConstrainedMoreThuenteBoundAware<Scalar>
    TestBoundAwareLineSearch;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void requireNear(
    Scalar actual,
    Scalar expected,
    Scalar tolerance,
    const std::string& message)
{
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance)
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
}

void requireThrows(
    const std::function<void()>& action,
    const std::string& message)
{
    bool threw = false;
    try
    {
        action();
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    require(threw, message);
}

void requireThrowsContaining(
    const std::function<void()>& action,
    const std::string& expected,
    const std::string& message)
{
    bool threw = false;
    std::string observed;
    try
    {
        action();
    }
    catch (const std::exception& error)
    {
        threw = true;
        observed = error.what();
    }
    require(threw, message + ": no exception");
    require(
        observed.find(expected) != std::string::npos,
        message + ": unexpected exception: " + observed);
}

void requireThrowsContainingBoth(
    const std::function<void()>& action,
    const std::string& first,
    const std::string& second,
    const std::string& message)
{
    bool threw = false;
    std::string observed;
    try
    {
        action();
    }
    catch (const std::exception& error)
    {
        threw = true;
        observed = error.what();
    }
    require(threw, message + ": no exception");
    require(
        observed.find(first) != std::string::npos &&
        observed.find(second) != std::string::npos,
        message + ": unexpected exception: " + observed);
}

void requireScalarBitsEqual(
    Scalar actual,
    Scalar expected,
    const std::string& message)
{
    require(
        std::memcmp(&actual, &expected, sizeof(Scalar)) == 0,
        message);
}

void requireVectorBitsEqual(
    const Vector& actual,
    const Vector& expected,
    const std::string& message)
{
    require(actual.size() == expected.size(), message + ": size mismatch");
    if (actual.size() > 0)
        require(
            std::memcmp(
                actual.data(), expected.data(),
                static_cast<std::size_t>(actual.size()) *
                    sizeof(Scalar)) == 0,
            message + ": payload mismatch");
}

class RecordingStreambuf : public std::streambuf
{
public:
    RecordingStreambuf() : sync_count_(0) {}

    const std::string& bytes() const
    {
        return bytes_;
    }

    int syncCount() const
    {
        return sync_count_;
    }

protected:
    int_type overflow(int_type character) override
    {
        if (!traits_type::eq_int_type(character, traits_type::eof()))
            bytes_.push_back(traits_type::to_char_type(character));
        return traits_type::not_eof(character);
    }

    std::streamsize xsputn(
        const char* data,
        std::streamsize count) override
    {
        if (count > 0)
            bytes_.append(data, static_cast<std::size_t>(count));
        return count;
    }

    int sync() override
    {
        ++sync_count_;
        return 0;
    }

private:
    std::string bytes_;
    int sync_count_;
};

class ScopedCerrRedirect
{
public:
    explicit ScopedCerrRedirect(RecordingStreambuf& destination)
        : previous_(std::cerr.rdbuf(&destination))
    {
    }

    ~ScopedCerrRedirect()
    {
        std::cerr.rdbuf(previous_);
    }

private:
    std::streambuf* previous_;
};

LBFGSpp::LBFGSBParam<Scalar> lineSearchParametersForCandidateCap(
    int candidate_cap)
{
    require(candidate_cap >= 2, "invalid test candidate cap");
    LBFGSpp::LBFGSBParam<Scalar> parameters =
        mfa::nsConstrainedInnerParameters<Scalar>(0, 1600);
    parameters.max_linesearch = candidate_cap - 1;
    parameters.check_param();
    return parameters;
}

class ScopedEnvironmentRestore
{
public:
    explicit ScopedEnvironmentRestore(
        const std::vector<std::string>& names)
    {
        for (std::size_t i = 0; i < names.size(); ++i)
        {
            Entry entry;
            entry.name = names[i];
            const char* value = std::getenv(names[i].c_str());
            entry.defined = value != 0;
            if (value)
                entry.value = value;
            entries_.push_back(entry);
        }
    }

    ~ScopedEnvironmentRestore()
    {
        for (std::size_t i = 0; i < entries_.size(); ++i)
        {
            if (entries_[i].defined)
                setenv(
                    entries_[i].name.c_str(),
                    entries_[i].value.c_str(),
                    1);
            else
                unsetenv(entries_[i].name.c_str());
        }
    }

private:
    struct Entry
    {
        std::string name;
        bool defined;
        std::string value;
    };
    std::vector<Entry> entries_;
};

std::string temporaryPath(const std::string& suffix)
{
    return std::string("/tmp/pimfa_ns_constrained_v2_") +
        std::to_string(static_cast<long long>(getpid())) + suffix;
}

bool pathExists(const std::string& path)
{
    std::ifstream stream(path.c_str(), std::ios::binary);
    return static_cast<bool>(stream);
}

mfa::BaselineCheckpointMetadata smallStageAMetadata()
{
    mfa::BaselineCheckpointMetadata metadata;
    metadata.case_name = "ns_lid_cavity";
    metadata.kind = mfa::BaselineKind::data_only;
    metadata.alpha1 = 0.0;
    metadata.alpha2 = 0.0;
    metadata.lattice = "2,1,1";
    metadata.degree = "0,0,0";
    metadata.initializer_role_sha256 = std::string(64, 'a');
    metadata.pressure_regime = "zero";
    metadata.field_order = "u,v,p";
    metadata.coefficient_count_per_field = 2;
    metadata.coefficient_count_total = 6;
    metadata.code_sha256 = std::string(64, 'b');
    return metadata;
}

mfa::NSConstrainedArtifactAnchors testArtifactAnchors()
{
    mfa::NSConstrainedArtifactAnchors anchors;
    anchors.stage_a_checkpoint_sha256 =
        mfa::nsConstrainedFrozenStageASHA256();
    anchors.stage_a_baseline_record_sha256 = std::string(64, 'b');
    anchors.optimization_protocol_sha256 =
        mfa::nsConstrainedFrozenOptimizationProtocolSHA256();
    anchors.optimization_protocol_parent_sha256 =
        mfa::nsConstrainedFrozenOptimizationProtocolParentSHA256();
    anchors.optimization_protocol_ancestor_2_sha256 =
        mfa::nsConstrainedFrozenOptimizationProtocolAncestor2SHA256();
    anchors.optimization_protocol_ancestor_3_sha256 =
        mfa::nsConstrainedFrozenOptimizationProtocolAncestor3SHA256();
    anchors.inherited_optimization_protocol_sha256 =
        mfa::nsConstrainedFrozenInheritedOptimizationProtocolSHA256();
    anchors.implementation_sha256 = std::string(64, 'd');
    anchors.post_selection_protocol_v5_sha256 =
        mfa::nsConstrainedFrozenPostSelectionV5SHA256();
    anchors.post_selection_protocol_v4_parent_sha256 =
        mfa::nsConstrainedFrozenPostSelectionV4ParentSHA256();
    return anchors;
}

void writeSmallStageA(
    const std::string& path,
    const mfa::BaselineCheckpointMetadata& metadata)
{
    std::vector<double> coefficients;
    coefficients.push_back(0.1);
    coefficients.push_back(-0.2);
    coefficients.push_back(0.3);
    coefficients.push_back(0.4);
    coefficients.push_back(0.0);
    coefficients.push_back(0.0);
    mfa::WriteBaselineCheckpointV2(path, metadata, coefficients);
}

std::string readText(const std::string& path)
{
    std::ifstream stream(path.c_str(), std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot read test file");
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

void writeText(const std::string& path, const std::string& text)
{
    std::ofstream stream(
        path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("cannot write test file");
    stream << text;
}

void replaceSmallStageAPayload(
    const std::string& path,
    const std::vector<std::string>& tokens)
{
    const std::string text = readText(path);
    const std::string marker = "--coefficients--\n";
    const std::size_t position = text.find(marker);
    if (position == std::string::npos)
        throw std::runtime_error("test checkpoint marker missing");
    std::ostringstream replaced;
    replaced << text.substr(0, position + marker.size());
    for (std::size_t i = 0; i < tokens.size(); ++i)
        replaced << tokens[i] << "\n";
    writeText(path, replaced.str());
}

class ManufacturedProvider
{
public:
    typedef double Scalar;

    explicit ManufacturedProvider(bool representable_driver_fixture = false)
        : center_(4),
          cached_(4),
          cache_valid_(false)
    {
        if (representable_driver_fixture)
            center_.setZero();
        else
            center_ << 0.2, -0.3, 0.4, -0.5;
        for (int metric = 0; metric < mfa::NS_V2_METRIC_COUNT; ++metric)
        {
            directions_[metric].resize(4);
            directions_[metric] <<
                0.11 * (metric + 1),
                -0.07 * (metric + 2),
                0.05 * (metric + 3),
                -0.03 * (metric + 4);
            shifts_[metric] =
                representable_driver_fixture ?
                    1.0e-8 * (metric + 1) : 0.0;
            offsets_[metric] = 1.0 + 0.1 * metric;
        }
    }

    int coefficientCount() const
    {
        return 4;
    }

    int gaugeCount() const
    {
        return 1001;
    }

    const Vector& center() const
    {
        return center_;
    }

    mfa::NSConstrainedRawEvaluation<Scalar> evaluate(
        const Vector& coefficients)
    {
        if (coefficients.size() != 4)
            throw std::invalid_argument("manufactured coefficient size");
        cached_ = coefficients;
        cache_valid_ = true;
        mfa::NSConstrainedRawEvaluation<Scalar> raw;
        const Vector displacement = coefficients - center_;
        for (int metric = 0; metric < mfa::NS_V2_METRIC_COUNT; ++metric)
        {
            const Scalar residual =
                directions_[metric].dot(displacement) + shifts_[metric];
            raw.metrics[metric] =
                offsets_[metric] + residual * residual;
        }
        raw.gauge.resize(1001);
        for (int row = 0; row < 1001; ++row)
            raw.gauge(row) =
                (0.01 * (row % 7 + 1)) *
                displacement(row % 4);
        return raw;
    }

    Vector weightedGradient(
        const std::array<Scalar, mfa::NS_V2_METRIC_COUNT>& weights,
        const Vector& gauge_weights)
    {
        if (!cache_valid_)
            throw std::logic_error("manufactured cache absent");
        Vector gradient = Vector::Zero(4);
        const Vector displacement = cached_ - center_;
        for (int metric = 0; metric < mfa::NS_V2_METRIC_COUNT; ++metric)
        {
            const Scalar residual =
                directions_[metric].dot(displacement) + shifts_[metric];
            gradient.noalias() +=
                (2.0 * weights[metric] * residual) *
                directions_[metric];
        }
        if (gauge_weights.size() != 1001)
            throw std::invalid_argument("manufactured gauge weight size");
        for (int row = 0; row < 1001; ++row)
            gradient(row % 4) +=
                gauge_weights(row) * (0.01 * (row % 7 + 1));
        return gradient;
    }

private:
    Vector center_;
    Vector cached_;
    std::array<Vector, mfa::NS_V2_METRIC_COUNT> directions_;
    std::array<Scalar, mfa::NS_V2_METRIC_COUNT> shifts_;
    std::array<Scalar, mfa::NS_V2_METRIC_COUNT> offsets_;
    bool cache_valid_;
};

struct QuadraticLineFunction
{
    Scalar operator()(const Vector& x, Vector& gradient)
    {
        gradient = x;
        return 0.5 * x.squaredNorm();
    }
};

struct LinearLineFunction
{
    Scalar operator()(const Vector& x, Vector& gradient)
    {
        gradient = Vector::Constant(x.size(), -1.0);
        return -x.sum();
    }
};

struct TwoScaleLinearLineFunction
{
    Scalar operator()(const Vector& x, Vector& gradient)
    {
        if (x.size() != 2)
            throw std::invalid_argument("two-scale linear size");
        gradient.resize(2);
        gradient << -1.0, -0.1;
        return -x(0) - 0.1 * x(1);
    }
};

struct StageAEpigraphLineFunction
{
    Scalar operator()(const Vector& x, Vector& gradient)
    {
        gradient.resize(1);
        gradient(0) = -1.0 + 3.0 * x(0);
        return 100.0 - x(0) + 1.5 * x(0) * x(0);
    }
};

struct ShiftedQuadraticLineFunction
{
    Scalar center;

    explicit ShiftedQuadraticLineFunction(Scalar value) : center(value) {}

    Scalar operator()(const Vector& x, Vector& gradient)
    {
        gradient.resize(1);
        gradient(0) = x(0) - center;
        return 0.5 * (x(0) - center) * (x(0) - center);
    }
};

struct CenteredOneCallQuadratic
{
    Scalar center;
    std::size_t calls;
    std::vector<Vector> observed;

    explicit CenteredOneCallQuadratic(Scalar value)
        : center(value), calls(0), observed()
    {
    }

    Scalar operator()(const Vector& x, Vector& gradient)
    {
        ++calls;
        observed.push_back(x);
        if (x.size() != 1)
            throw std::invalid_argument(
                "centered one-call quadratic size");
        gradient.resize(1);
        gradient(0) = x(0) - center;
        return Scalar(0.5) *
            (x(0) - center) * (x(0) - center);
    }
};

struct NeverCalledLineFunction
{
    std::size_t calls;

    NeverCalledLineFunction() : calls(0) {}

    Scalar operator()(const Vector& x, Vector& gradient)
    {
        ++calls;
        gradient = Vector::Zero(x.size());
        return Scalar(0);
    }
};

struct C1SquaredHingeOracle
{
    // Keep every candidate in the production step range while retaining a
    // narrow strong-Wolfe interval that needs more than 50 evaluations.
    static Scalar scale()
    {
        return Scalar(1024);
    }

    static Scalar g()
    {
        return scale() * std::ldexp(Scalar(1), -68);
    }

    static Scalar tau()
    {
        return scale() * Scalar(1.0e-19);
    }

    static Scalar kappa()
    {
        return Scalar(1);
    }

    std::vector<Scalar> steps;

    Scalar operator()(const Vector& x, Vector& gradient)
    {
        if (x.size() != 1)
            throw std::invalid_argument("C1 squared-hinge size");
        const Scalar alpha = x(0);
        const Scalar hinge =
            (std::max)(Scalar(0), alpha - tau());
        steps.push_back(alpha);
        gradient.resize(1);
        gradient(0) = -g() + kappa() * hinge;
        return
            -g() * alpha +
            Scalar(0.5) * kappa() * hinge * hinge;
    }
};

struct RecordingLinearLineFunction
{
    std::vector<Scalar> steps;

    Scalar operator()(const Vector& x, Vector& gradient)
    {
        if (x.size() != 1)
            throw std::invalid_argument("recording linear size");
        steps.push_back(x(0));
        gradient.resize(1);
        gradient(0) = Scalar(-1);
        return -x(0);
    }
};

struct RecordingShiftedQuadraticLineFunction
{
    Scalar center;
    std::vector<Vector> observed;

    explicit RecordingShiftedQuadraticLineFunction(Scalar value)
        : center(value), observed()
    {
    }

    Scalar operator()(const Vector& x, Vector& gradient)
    {
        if (x.size() != 1)
            throw std::invalid_argument(
                "recording shifted quadratic size");
        observed.push_back(x);
        gradient.resize(1);
        gradient(0) = x(0) - center;
        return Scalar(0.5) *
            (x(0) - center) * (x(0) - center);
    }
};

struct OneCallScriptedOracle
{
    Scalar returned_value;
    Scalar returned_gradient;
    std::size_t calls;
    std::vector<Vector> observed;

    OneCallScriptedOracle(
        Scalar value,
        Scalar gradient_value)
        : returned_value(value),
          returned_gradient(gradient_value),
          calls(0),
          observed()
    {
    }

    Scalar operator()(const Vector& x, Vector& gradient)
    {
        ++calls;
        observed.push_back(x);
        gradient.resize(1);
        gradient(0) = returned_gradient;
        return returned_value;
    }
};

struct PIMFATestThrowingOracle
{
    std::string message;
    std::size_t calls;

    explicit PIMFATestThrowingOracle(const std::string& value)
        : message(value), calls(0)
    {
    }

    Scalar operator()(const Vector&, Vector&)
    {
        ++calls;
        throw mfa::NSConstrainedTestThrowingOracleException(
            message);
    }
};

void requireResetCallAccounting(
    const mfa::NSConstrainedLineSearchTelemetry& telemetry,
    std::uint64_t attempts,
    std::uint64_t completions,
    const std::string& label)
{
    require(
        telemetry.candidate_evaluations == attempts &&
        telemetry.call_candidate_attempts == attempts &&
        telemetry.call_candidate_completions == completions &&
        telemetry.cumulative_candidate_evaluations == attempts &&
        telemetry.cumulative_candidate_attempts == attempts &&
        telemetry.cumulative_candidate_completions == completions,
        label + ": attempted/completed accounting drifted");
    require(
        telemetry.last_evaluated_step_present ==
            telemetry.last_completed_step_present &&
        (!telemetry.last_evaluated_step_present ||
         std::memcmp(
             &telemetry.last_evaluated_step,
             &telemetry.last_completed_step,
             sizeof(double)) == 0),
        label + ": last-evaluated alias drifted");
}

void requireFiniteContainingBracket(
    const TestGenerator::Diagnostics& diagnostics,
    Scalar contained_step,
    const std::string& label)
{
    require(
        diagnostics.final_bracketed &&
        std::isfinite(diagnostics.final_bracket_lo) &&
        std::isfinite(diagnostics.final_bracket_hi) &&
        std::isfinite(diagnostics.final_bracket_width) &&
        diagnostics.final_bracket_lo <
            diagnostics.final_bracket_hi &&
        diagnostics.final_bracket_width ==
            diagnostics.final_bracket_hi -
                diagnostics.final_bracket_lo &&
        contained_step > diagnostics.final_bracket_lo &&
        contained_step < diagnostics.final_bracket_hi,
        label + ": finite containing bracket drifted");
}

void requireDirectGuardAccounting(
    const TestGenerator::Diagnostics& diagnostics,
    const TestGenerator::CandidateAccounting& accounting,
    LBFGSpp::LineSearchMoreThuenteGuardReason expected_reason,
    std::uint64_t expected_attempts,
    std::uint64_t expected_completions,
    const std::string& label)
{
    require(
        diagnostics.guard_failure == expected_reason &&
        diagnostics.termination_reason ==
            LBFGSpp::LBFGSPP_MT_TERMINATION_GUARD_EXCEPTION &&
        diagnostics.candidate_attempts == expected_attempts &&
        diagnostics.candidate_completions == expected_completions &&
        accounting.configured_attempt_cap == 80 &&
        accounting.call_attempts == expected_attempts &&
        accounting.call_completions == expected_completions,
        label + ": attempted/completed accounting drifted");
    require(
        diagnostics.last_attempted_step_present ==
            (expected_attempts > 0) &&
        diagnostics.last_completed_step_present ==
            (expected_completions > 0) &&
        diagnostics.last_evaluated_step_present ==
            diagnostics.last_completed_step_present &&
        accounting.last_attempted_step_present ==
            (expected_attempts > 0) &&
        accounting.last_completed_step_present ==
            (expected_completions > 0),
        label + ": last-step presence drifted");
    if (expected_completions > 0)
        requireScalarBitsEqual(
            diagnostics.last_evaluated_step,
            diagnostics.last_completed_step,
            label + ": last-evaluated alias drifted");
}

std::string expectedThrowingOracleAbortRecord()
{
    return
        "PIMFA_V17_LINE_SEARCH_ABORT {\"bracket_hi\":null,"
        "\"bracket_lo\":null,\"bracket_present\":false,"
        "\"bracket_width\":null,\"call\":1,\"call_attempts\":1,"
        "\"call_completions\":0,\"call_forced_midpoints\":0,"
        "\"candidate_evaluations\":1,\"configured_attempt_cap\":80,"
        "\"configured_candidate_cap\":80,\"cumulative_attempts\":1,"
        "\"cumulative_candidate_evaluations\":1,"
        "\"cumulative_completions\":0,"
        "\"cumulative_forced_midpoints\":0,"
        "\"exception_class\":\"OBJECTIVE_PROVIDER_EXCEPTION\","
        "\"guard_reason\":\"NONE\",\"invocation\":\"PRIMARY\","
        "\"invocation_attempts\":1,\"invocation_completions\":0,"
        "\"invocation_forced_midpoints\":0,"
        "\"invocation_last_attempted_step\":1,"
        "\"invocation_last_attempted_step_present\":true,"
        "\"invocation_last_completed_step\":null,"
        "\"invocation_last_completed_step_present\":false,"
        "\"last_attempted_step\":1,"
        "\"last_attempted_step_present\":true,"
        "\"last_completed_step\":null,"
        "\"last_completed_step_present\":false,"
        "\"last_evaluated_step\":null,"
        "\"last_evaluated_step_present\":false,"
        "\"midpoint_trigger\":\"NONE\","
        "\"original_exception_message_hex\":\"71225c0aff\","
        "\"original_exception_type\":\"PIMFA_TEST_THROWING_ORACLE\","
        "\"outer_iteration\":7,\"primary_attempts\":1,"
        "\"primary_completions\":0,\"primary_forced_midpoints\":0,"
        "\"proposal_origin\":\"INITIAL\",\"raw_proposed_step\":1,"
        "\"raw_proposed_step_class\":\"FINITE\","
        "\"raw_proposed_step_present\":true,\"retry_attempts\":0,"
        "\"retry_completions\":0,\"retry_forced_midpoints\":0,"
        "\"safeguarded_proposed_step\":1,"
        "\"safeguarded_proposed_step_present\":true,"
        "\"termination_reason\":\"OBJECTIVE_PROVIDER_EXCEPTION\"}\n";
}

void testFrozenInnerSolverAndStrongWolfe()
{
    const LBFGSpp::LBFGSBParam<Scalar> first =
        mfa::nsConstrainedInnerParameters<Scalar>(0, 1600);
    const LBFGSpp::LBFGSBParam<Scalar> second =
        mfa::nsConstrainedInnerParameters<Scalar>(1, 1400);
    const LBFGSpp::LBFGSBParam<Scalar> third =
        mfa::nsConstrainedInnerParameters<Scalar>(2, 1200);
    const LBFGSpp::LBFGSBParam<Scalar> later =
        mfa::nsConstrainedInnerParameters<Scalar>(7, 17);
    require(
        first.m == 6 &&
        first.epsilon == 1.0e-4 &&
        second.epsilon == 1.0e-5 &&
        third.epsilon == 1.0e-6 &&
        later.epsilon == 1.0e-7 &&
        first.epsilon_rel == 0.0 &&
        first.past == 0 &&
        first.delta == 0.0 &&
        first.max_iterations == 200 &&
        later.max_iterations == 17 &&
        first.max_linesearch == 79 &&
        first.min_step == 1.0e-20 &&
        first.max_step == 1.0e20 &&
        mfa::nsConstrainedLineSearchCandidateCap() == 80 &&
        mfa::nsConstrainedLineSearchMaxIterations() == 79 &&
        first.ftol == 1.0e-4 &&
        first.wolfe == 0.9,
        "frozen L-BFGS-B/line-search parameters drifted");

    LBFGSpp::LBFGSBParam<Scalar> legacy_fifty_budget = first;
    legacy_fifty_budget.max_linesearch = 49;
    legacy_fifty_budget.check_param();
    mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::resetTelemetry();
    QuadraticLineFunction quadratic;
    Vector previous(1), direction(1), gradient(1), current(1);
    previous(0) = 1.0;
    direction(0) = -1.0;
    gradient(0) = 1.0;
    current = previous;
    Scalar step = 1.0;
    Scalar value = 0.5;
    Scalar derivative = -1.0;
    mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::LineSearch(
        quadratic, first, previous, direction, 10.0,
        step, value, gradient, derivative, current);
    require(
        value <= 0.5 + step * first.ftol * -1.0 &&
        std::abs(derivative) <= first.wolfe,
        "Moré--Thuente wrapper did not return a strong-Wolfe step");
    {
        const mfa::NSConstrainedLineSearchTelemetry telemetry =
            mfa::NSConstrainedMoreThuenteBoundAware<
                Scalar>::currentTelemetry();
        require(
            telemetry.calls == 1 &&
            telemetry.strong_wolfe_accepts == 1 &&
            telemetry.box_endpoint_accepts == 0,
            "interior strong-Wolfe telemetry drifted");
    }

    // The exact Stage-A scalar slice has three active strong-form epigraph
    // hinges after eta begins to decrease:
    // phi(alpha) = 100 - alpha + 3 alpha^2 / 2.
    // The very first production line search is therefore not the V15
    // failure: it has an interior strong-Wolfe step near alpha=1/3.
    mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::resetTelemetry();
    StageAEpigraphLineFunction stage_a_slice;
    previous(0) = 0.0;
    direction(0) = 1.0;
    gradient(0) = -1.0;
    current = previous;
    step = 1.0;
    value = 100.0;
    derivative = -1.0;
    mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::LineSearch(
        stage_a_slice, first, previous, direction, 100.0,
        step, value, gradient, derivative, current);
    requireNear(
        step, 1.0 / 3.0, 1.0e-3,
        "Stage-A scalar slice did not find the expected interior step");
    require(
        value <= 100.0 + step * first.ftol * -1.0 &&
        std::abs(derivative) <= first.wolfe,
        "Stage-A scalar slice did not satisfy strong Wolfe");

    // The old 20-loop budget (21 candidates in this LBFGS++ version) cannot
    // reach the strong-Wolfe interval of this smooth oracle.  The retained
    // legacy exact 50-candidate fixture can, without weakening either
    // interior predicate.
    ShiftedQuadraticLineFunction distant_quadratic(1000.0);
    LBFGSpp::LBFGSBParam<Scalar> old_budget = first;
    old_budget.max_linesearch = 20;
    mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::resetTelemetry();
    previous(0) = 0.0;
    direction(0) = 1.0;
    gradient(0) = -1000.0;
    current = previous;
    step = 1.0;
    value = 500000.0;
    derivative = -1000.0;
    requireThrowsContaining(
        [&distant_quadratic, &old_budget, &previous, &direction, &step,
         &value, &gradient, &derivative, &current]() {
            mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::LineSearch(
                distant_quadratic, old_budget, previous, direction, 2000.0,
                step, value, gradient, derivative, current);
        },
        "termination=MAX_EVAL_NO_SW",
        "old 21-candidate budget unexpectedly reached distant strong Wolfe");
    {
        const mfa::NSConstrainedLineSearchTelemetry telemetry =
            mfa::NSConstrainedMoreThuenteBoundAware<
                Scalar>::currentTelemetry();
        require(
            telemetry.calls == 1 &&
            telemetry.candidate_evaluations == 21 &&
            telemetry.strong_wolfe_accepts == 0 &&
            telemetry.box_endpoint_accepts == 0,
            "old-budget typed telemetry drifted");
    }

    mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::resetTelemetry();
    previous(0) = 0.0;
    direction(0) = 1.0;
    gradient(0) = -1000.0;
    current = previous;
    step = 1.0;
    value = 500000.0;
    derivative = -1000.0;
    mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::LineSearch(
        distant_quadratic, legacy_fifty_budget,
        previous, direction, 2000.0,
        step, value, gradient, derivative, current);
    require(
        value <= 500000.0 + step * first.ftol * -1000.0 &&
        std::abs(derivative) <= first.wolfe * 1000.0,
        "50-candidate budget did not reach distant strong Wolfe");
    {
        const mfa::NSConstrainedLineSearchTelemetry telemetry =
            mfa::NSConstrainedMoreThuenteBoundAware<
                Scalar>::currentTelemetry();
        require(
            telemetry.calls == 1 &&
            telemetry.candidate_evaluations > 21 &&
            telemetry.candidate_evaluations <= 50 &&
            telemetry.calls ==
                telemetry.strong_wolfe_accepts +
                telemetry.box_endpoint_accepts,
            "50-candidate success telemetry drifted");
    }

    // A strictly decreasing function has no interior strong-Wolfe point.
    // At a genuine box endpoint, Armijo and phi'(alpha_max)<=0 are the
    // correct one-sided line KKT conditions and are accepted explicitly.
    mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::resetTelemetry();
    LinearLineFunction linear;
    previous(0) = 0.0;
    direction(0) = 1.0;
    gradient(0) = -1.0;
    current = previous;
    step = 1.0;
    value = 0.0;
    derivative = -1.0;
    mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::LineSearch(
        linear, first, previous, direction, 1.0,
        step, value, gradient, derivative, current);
    require(
        step == 1.0 &&
        current(0) == 1.0 &&
        value == -1.0 &&
        derivative == -1.0,
        "box-endpoint line-search return is inconsistent");
    {
        const mfa::NSConstrainedLineSearchTelemetry telemetry =
            mfa::NSConstrainedMoreThuenteBoundAware<
                Scalar>::currentTelemetry();
        require(
            telemetry.calls == 1 &&
            telemetry.strong_wolfe_accepts == 0 &&
            telemetry.box_endpoint_accepts == 1,
            "box-endpoint telemetry drifted");
    }

    // LBFGS++ may return a genuine box endpoint immediately when that point
    // satisfies strong Wolfe even though its one-sided derivative is
    // positive.  The wrapper must not accept it as an endpoint.  It reuses
    // the same Moré--Thuente generator with the next representable interior
    // cap and the same shared 80-candidate budget, yielding an interior
    // strong-Wolfe classification.
    mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::resetTelemetry();
    ShiftedQuadraticLineFunction positive_endpoint_quadratic(0.75);
    previous(0) = 0.0;
    direction(0) = 1.0;
    gradient(0) = -0.75;
    current = previous;
    step = 1.0;
    value = 0.28125;
    derivative = -0.75;
    mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::LineSearch(
        positive_endpoint_quadratic, first, previous, direction, 1.0,
        step, value, gradient, derivative, current);
    require(
        step > 0.0 &&
        step < 1.0 &&
        current(0) < 1.0 &&
        value <= 0.28125 + step * first.ftol * -0.75 &&
        std::abs(derivative) <= first.wolfe * 0.75,
        "positive-derivative endpoint was not recovered as interior Wolfe");
    {
        const mfa::NSConstrainedLineSearchTelemetry telemetry =
            mfa::NSConstrainedMoreThuenteBoundAware<
                Scalar>::currentTelemetry();
        require(
            telemetry.calls == 1 &&
            telemetry.candidate_evaluations == 2 &&
            telemetry.strong_wolfe_accepts == 1 &&
            telemetry.box_endpoint_accepts == 0,
            "positive-endpoint interior-retry telemetry drifted");
    }

    // A scalar step one ULP below alpha_max need not produce a
    // representably interior vector.  Here round-to-nearest maps
    // 0.5 + nextafter(1,0)*0.5 back to the true endpoint 1.0.  Such a retry
    // must fail closed instead of being relabelled as interior strong Wolfe.
    mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::resetTelemetry();
    RecordingShiftedQuadraticLineFunction
        rounded_endpoint_quadratic(0.875);
    previous(0) = 0.5;
    direction(0) = 0.5;
    gradient(0) = -0.375;
    current = previous;
    step = 1.0;
    value = 0.0703125;
    derivative = -0.1875;
    require(
        std::nextafter(1.0, 0.0) < 1.0 &&
        previous(0) +
            std::nextafter(1.0, 0.0) * direction(0) == 1.0,
        "representable-interior adversarial precondition drifted");
    bool rounded_guard_caught = false;
    try
    {
        TestBoundAwareLineSearch::LineSearch(
            rounded_endpoint_quadratic, first, previous, direction,
            1.0, step, value, gradient, derivative, current);
    }
    catch (const TestGenerator::GuardException& error)
    {
        rounded_guard_caught = true;
        require(
            error.reason() ==
                LBFGSpp::LBFGSPP_MT_GUARD_TRIAL_EQUALS_TRUE_BOX_ENDPOINT,
            "rounded-endpoint retry used the wrong guard");
    }
    require(
        rounded_guard_caught,
        "rounded true endpoint was not rejected by the retry guard");
    const Scalar rounded_retry_step = std::nextafter(1.0, 0.0);
    requireScalarBitsEqual(
        step, rounded_retry_step,
        "rounded-endpoint retry scalar drifted");
    require(
        rounded_endpoint_quadratic.observed.size() == 1,
        "rounded-endpoint retry made a second provider call");
    require(
        rounded_endpoint_quadratic.observed[0].size() == 1 &&
        rounded_endpoint_quadratic.observed[0](0) == 1.0,
        "rounded-endpoint primary provider vector drifted");
    {
        const mfa::NSConstrainedLineSearchTelemetry telemetry =
            TestBoundAwareLineSearch::currentTelemetry();
        const TestBoundAwareLineSearch::CallSnapshot snapshot =
            TestBoundAwareLineSearch::currentCallSnapshot();
        requireResetCallAccounting(
            telemetry, 1, 1, "rounded endpoint");
        require(
            telemetry.calls == 1 &&
            telemetry.primary_candidate_attempts == 1 &&
            telemetry.primary_candidate_completions == 1 &&
            telemetry.retry_candidate_attempts == 0 &&
            telemetry.retry_candidate_completions == 0 &&
            telemetry.invocation_candidate_attempts == 0 &&
            telemetry.invocation_candidate_completions == 0 &&
            telemetry.strong_wolfe_accepts == 0 &&
            telemetry.box_endpoint_accepts == 0,
            "rounded-endpoint scoped accounting drifted");
        require(
            telemetry.last_attempted_step_present &&
            telemetry.last_completed_step_present &&
            telemetry.last_evaluated_step_present &&
            telemetry.last_attempted_step == 1.0 &&
            telemetry.last_completed_step == 1.0 &&
            telemetry.last_evaluated_step == 1.0 &&
            !telemetry.invocation_last_attempted_step_present &&
            !telemetry.invocation_last_completed_step_present,
            "rounded-endpoint last-step accounting drifted");
        require(
            snapshot.primary_present &&
            snapshot.retry_present &&
            snapshot.true_box_endpoint_present &&
            snapshot.invocation == "RETRY" &&
            snapshot.exception_class ==
                "GENERATOR_GUARD_EXCEPTION" &&
            snapshot.guard_reason ==
                "TRIAL_EQUALS_TRUE_BOX_ENDPOINT" &&
            snapshot.termination_reason == "GUARD_EXCEPTION" &&
            snapshot.call_accounting.call_attempts == 1 &&
            snapshot.call_accounting.call_completions == 1,
            "rounded-endpoint call snapshot drifted");
        requireVectorBitsEqual(
            snapshot.true_box_endpoint,
            rounded_endpoint_quadratic.observed[0],
            "stored primary true endpoint was reconstructed");
        require(
            snapshot.primary_diagnostics.candidate_attempts == 1 &&
            snapshot.primary_diagnostics.candidate_completions == 1 &&
            snapshot.retry_diagnostics.candidate_attempts == 0 &&
            snapshot.retry_diagnostics.candidate_completions == 0 &&
            snapshot.retry_diagnostics.guard_failure ==
                LBFGSpp::LBFGSPP_MT_GUARD_TRIAL_EQUALS_TRUE_BOX_ENDPOINT &&
            snapshot.retry_diagnostics.proposal_origin ==
                LBFGSpp::LBFGSPP_MT_PROPOSAL_INITIAL &&
            snapshot.retry_diagnostics.raw_proposed_step_present &&
            snapshot.retry_diagnostics.safeguarded_proposed_step_present &&
            !snapshot.retry_diagnostics.last_attempted_step_present &&
            !snapshot.retry_diagnostics.last_completed_step_present &&
            snapshot.retry_diagnostics.true_box_endpoint_present,
            "rounded-endpoint invocation diagnostics drifted");
        requireScalarBitsEqual(
            snapshot.retry_diagnostics.raw_proposed_step,
            rounded_retry_step,
            "rounded retry raw proposal drifted");
        requireScalarBitsEqual(
            snapshot.retry_diagnostics.safeguarded_proposed_step,
            rounded_retry_step,
            "rounded retry safeguarded proposal drifted");
        requireVectorBitsEqual(
            snapshot.retry_diagnostics.true_box_endpoint,
            rounded_endpoint_quadratic.observed[0],
            "retry forbidden endpoint differs from primary observation");
    }

    // Full L-BFGS-B integration: after the first endpoint return the second
    // component still has nonzero projected gradient.  The solver therefore
    // reaches the unchanged s'y guard with y=0, skips that correction, and
    // safely continues to the second endpoint.
    mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::resetTelemetry();
    TwoScaleLinearLineFunction two_scale_linear;
    Vector bounded_x(2), bounded_lower(2), bounded_upper(2);
    bounded_x << 0.0, 0.0;
    bounded_lower << 0.0, 0.0;
    bounded_upper << 1.0, 10.0;
    Scalar bounded_value = 0.0;
    LBFGSpp::LBFGSBSolver<
        Scalar, mfa::NSConstrainedMoreThuenteBoundAware> bounded_solver(first);
    const int bounded_iterations = bounded_solver.minimize(
        two_scale_linear, bounded_x, bounded_value,
        bounded_lower, bounded_upper);
    require(
        bounded_iterations >= 2 &&
        bounded_x(0) == 1.0 &&
        bounded_x(1) == 10.0 &&
        bounded_value == -2.0 &&
        bounded_solver.final_grad_norm() == 0.0,
        "curvature-guard/endpoint L-BFGS-B integration failed");
    {
        const mfa::NSConstrainedLineSearchTelemetry telemetry =
            mfa::NSConstrainedMoreThuenteBoundAware<
                Scalar>::currentTelemetry();
        require(
            telemetry.calls >= 2 &&
            telemetry.calls ==
                telemetry.strong_wolfe_accepts +
                telemetry.box_endpoint_accepts &&
            telemetry.box_endpoint_accepts >= 2,
            "curvature-guard integration did not traverse two endpoints");
    }

    // The same scalar endpoint is not a box exception when it is the
    // numerical parameter cap.
    LBFGSpp::LBFGSBParam<Scalar> numerical_cap = first;
    numerical_cap.max_step = 1.0;
    numerical_cap.check_param();
    previous(0) = 0.0;
    direction(0) = 1.0;
    gradient(0) = -1.0;
    current = previous;
    step = 1.0;
    value = 0.0;
    derivative = -1.0;
    requireThrowsContaining(
        [&linear, &numerical_cap, &previous, &direction, &step, &value,
         &gradient, &derivative, &current]() {
            mfa::NSConstrainedMoreThuenteBoundAware<Scalar>::LineSearch(
                linear, numerical_cap, previous, direction, 1.0,
                step, value, gradient, derivative, current);
        },
        "termination=PARAM_MAX_NO_SW",
        "Moré--Thuente wrapper accepted a numerical-cap fallback");

    const Scalar armijo_rhs = 1.0 - 1.0e-4;
    require(
        mfa::nsConstrainedClassifyMoreThuenteReturn(
            1.0, -1.0, 1.0, 2.0, 10.0,
            std::nextafter(
                armijo_rhs, std::numeric_limits<Scalar>::infinity()),
            0.0, 1.0e-4, 0.9) ==
            mfa::NS_CONSTRAINED_LINE_SEARCH_REJECT,
        "one-ulp Armijo violation was accepted");
    require(
        mfa::nsConstrainedClassifyMoreThuenteReturn(
            1.0, -1.0, 1.0, 2.0, 10.0,
            std::nextafter(
                armijo_rhs, -std::numeric_limits<Scalar>::infinity()),
            0.0, 1.0e-4, 0.9) ==
            mfa::NS_CONSTRAINED_LINE_SEARCH_STRONG_WOLFE,
        "one-ulp Armijo satisfaction was rejected");
    require(
        mfa::nsConstrainedClassifyMoreThuenteReturn(
            0.0, -1.0, 1.0, 1.0, 10.0,
            -0.1, 0.5, 1.0e-4, 0.9) ==
            mfa::NS_CONSTRAINED_LINE_SEARCH_REJECT,
        "positive endpoint derivative inside the curvature band was accepted");
    require(
        mfa::nsConstrainedClassifyMoreThuenteReturn(
            0.0, -1.0, 2.0, 1.0, 10.0,
            -2.0, 0.0, 1.0e-4, 0.9) ==
            mfa::NS_CONSTRAINED_LINE_SEARCH_REJECT,
        "step beyond the feasible maximum was accepted");
    require(
        mfa::nsConstrainedClassifyMoreThuenteReturn(
            0.0, -1.0, 1.0, 1.0, 1.0,
            -1.0, -1.0, 1.0e-4, 0.9) ==
            mfa::NS_CONSTRAINED_LINE_SEARCH_REJECT,
        "numerical maximum was classified as a box endpoint");
    require(
        mfa::nsConstrainedClassifyMoreThuenteReturn(
            0.0, -1.0, 1.0, 1.0, 10.0,
            std::numeric_limits<Scalar>::quiet_NaN(),
            -1.0, 1.0e-4, 0.9) ==
            mfa::NS_CONSTRAINED_LINE_SEARCH_REJECT,
        "nonfinite line-search value was accepted");

    Vector consistency_previous(2), consistency_direction(2);
    Vector consistency_current(2), consistency_gradient(2);
    consistency_previous << 1.0, -2.0;
    consistency_direction << -0.5, 0.25;
    consistency_current =
        consistency_previous + 0.5 * consistency_direction;
    consistency_gradient << 2.0, -4.0;
    const Scalar consistent_derivative =
        consistency_gradient.dot(consistency_direction);
    require(
        mfa::nsConstrainedMoreThuenteReturnConsistent(
            consistency_previous, consistency_direction, 0.5,
            consistency_current, consistency_gradient,
            consistent_derivative, 1, 50),
        "consistent returned vector/derivative was rejected");
    Vector inconsistent_current = consistency_current;
    inconsistent_current(0) += 1.0e-10;
    require(
        !mfa::nsConstrainedMoreThuenteReturnConsistent(
            consistency_previous, consistency_direction, 0.5,
            inconsistent_current, consistency_gradient,
            consistent_derivative, 1, 50),
        "adversarial returned-vector mismatch was accepted");
    require(
        !mfa::nsConstrainedMoreThuenteReturnConsistent(
            consistency_previous, consistency_direction, 0.5,
            consistency_current, consistency_gradient,
            consistent_derivative + 1.0e-8, 1, 50),
        "adversarial directional-derivative mismatch was accepted");
    Vector huge_gradient(2);
    huge_gradient << std::numeric_limits<Scalar>::max(),
        std::numeric_limits<Scalar>::max();
    require(
        !mfa::nsConstrainedMoreThuenteReturnConsistent(
            consistency_previous, consistency_direction, 0.5,
            consistency_current, huge_gradient, 0.0, 1, 50),
        "overflowed derivative-consistency scale failed open");

    const Scalar tested_steps[] = {0.25, 1.0, 2.0};
    const Scalar tested_values[] = {-2.0, -0.1, 0.1};
    const Scalar tested_derivatives[] = {-2.0, -0.5, 0.0, 0.5, 2.0};
    for (std::size_t i = 0;
         i < sizeof(tested_steps) / sizeof(tested_steps[0]); ++i)
        for (std::size_t j = 0;
             j < sizeof(tested_values) / sizeof(tested_values[0]); ++j)
            for (std::size_t k = 0;
                 k < sizeof(tested_derivatives) /
                     sizeof(tested_derivatives[0]); ++k)
            {
                const Scalar tested_step = tested_steps[i];
                const Scalar tested_value = tested_values[j];
                const Scalar tested_derivative = tested_derivatives[k];
                const mfa::NSConstrainedLineSearchAcceptance accepted =
                    mfa::nsConstrainedClassifyMoreThuenteReturn(
                        0.0, -1.0, tested_step, 1.0, 10.0,
                        tested_value, tested_derivative, 1.0e-4, 0.9);
                if (accepted ==
                    mfa::NS_CONSTRAINED_LINE_SEARCH_STRONG_WOLFE)
                {
                    require(
                        tested_step < 1.0 &&
                        tested_value <= -tested_step * 1.0e-4 &&
                        std::abs(tested_derivative) <= 0.9,
                        "classifier admitted an invalid strong-Wolfe step");
                }
                else if (
                    accepted ==
                    mfa::NS_CONSTRAINED_LINE_SEARCH_BOX_ENDPOINT_ARMIJO_KKT)
                {
                    require(
                        tested_step == 1.0 &&
                        tested_value < 0.0 &&
                        tested_value <= -1.0e-4 &&
                        tested_derivative <= 0.0,
                        "classifier admitted an invalid box endpoint");
                }
                else
                {
                    require(
                        accepted ==
                            mfa::NS_CONSTRAINED_LINE_SEARCH_REJECT,
                        "classifier returned an unknown acceptance type");
                }
            }
}

void testV17MinimumStepBoundaries()
{
    const LBFGSpp::LBFGSBParam<Scalar> parameters =
        lineSearchParametersForCandidateCap(80);
    const Scalar minimum = parameters.min_step;
    const Scalar below = std::nextafter(minimum, Scalar(0));
    const Scalar above = std::nextafter(
        minimum, std::numeric_limits<Scalar>::infinity());
    require(
        below > Scalar(0) && below < minimum && above > minimum,
        "minimum-step adjacent-value precondition drifted");

    {
        CenteredOneCallQuadratic oracle(below);
        Vector previous(1), direction(1), gradient(1), current(1);
        previous(0) = Scalar(0);
        direction(0) = Scalar(1);
        gradient(0) = -below;
        current = previous;
        Scalar step = below;
        Scalar value = Scalar(0.5) * below * below;
        Scalar derivative = -below;
        TestBoundAwareLineSearch::resetTelemetry();
        bool caught = false;
        try
        {
            TestBoundAwareLineSearch::LineSearch(
                oracle, parameters, previous, direction, Scalar(1),
                step, value, gradient, derivative, current);
        }
        catch (const TestGenerator::GuardException& error)
        {
            caught = true;
            require(
                error.reason() ==
                    LBFGSpp::LBFGSPP_MT_GUARD_BELOW_MIN_STEP_PROPOSAL,
                "below-minimum initial step used the wrong guard");
        }
        require(caught, "below-minimum initial step did not fail closed");
        require(
            oracle.calls == 0 && oracle.observed.empty(),
            "below-minimum initial step reached the provider");

        const mfa::NSConstrainedLineSearchTelemetry telemetry =
            TestBoundAwareLineSearch::currentTelemetry();
        const TestBoundAwareLineSearch::CallSnapshot snapshot =
            TestBoundAwareLineSearch::currentCallSnapshot();
        requireResetCallAccounting(
            telemetry, 0, 0, "below-minimum initial step");
        require(
            telemetry.calls == 1 &&
            !telemetry.last_attempted_step_present &&
            !telemetry.last_completed_step_present &&
            !telemetry.last_evaluated_step_present &&
            snapshot.primary_present &&
            !snapshot.retry_present &&
            snapshot.exception_class ==
                "GENERATOR_GUARD_EXCEPTION" &&
            snapshot.guard_reason ==
                "BELOW_MIN_STEP_PROPOSAL" &&
            snapshot.termination_reason == "GUARD_EXCEPTION",
            "below-minimum wrapper diagnostics drifted");
        const TestGenerator::Diagnostics& diagnostics =
            snapshot.primary_diagnostics;
        require(
            diagnostics.proposal_origin ==
                LBFGSpp::LBFGSPP_MT_PROPOSAL_INITIAL &&
            diagnostics.raw_proposed_step_present &&
            diagnostics.raw_proposed_step_class ==
                LBFGSpp::LBFGSPP_MT_RAW_FINITE &&
            diagnostics.safeguarded_proposed_step_present &&
            diagnostics.midpoint_trigger ==
                LBFGSpp::LBFGSPP_MT_MIDPOINT_NONE &&
            diagnostics.guard_failure ==
                LBFGSpp::LBFGSPP_MT_GUARD_BELOW_MIN_STEP_PROPOSAL &&
            diagnostics.candidate_attempts == 0 &&
            diagnostics.candidate_completions == 0 &&
            !diagnostics.last_attempted_step_present &&
            !diagnostics.last_completed_step_present,
            "below-minimum generator diagnostics drifted");
        requireScalarBitsEqual(
            diagnostics.raw_proposed_step, below,
            "below-minimum raw proposal drifted");
        requireScalarBitsEqual(
            diagnostics.safeguarded_proposed_step, below,
            "below-minimum safeguarded proposal drifted");
    }

    const Scalar accepted_steps[] = {minimum, above};
    for (std::size_t i = 0;
         i < sizeof(accepted_steps) / sizeof(accepted_steps[0]); ++i)
    {
        const Scalar tested_step = accepted_steps[i];
        CenteredOneCallQuadratic oracle(tested_step);
        Vector previous(1), direction(1), gradient(1), current(1);
        previous(0) = Scalar(0);
        direction(0) = Scalar(1);
        gradient(0) = -tested_step;
        current = previous;
        Scalar step = tested_step;
        Scalar value =
            Scalar(0.5) * tested_step * tested_step;
        Scalar derivative = -tested_step;
        TestBoundAwareLineSearch::resetTelemetry();
        TestBoundAwareLineSearch::LineSearch(
            oracle, parameters, previous, direction, Scalar(1),
            step, value, gradient, derivative, current);

        require(
            oracle.calls == 1 && oracle.observed.size() == 1 &&
            value == Scalar(0) && derivative == Scalar(0) &&
            current.size() == 1,
            "minimum-boundary strong-Wolfe oracle drifted");
        requireScalarBitsEqual(
            step, tested_step,
            "minimum-boundary returned step drifted");
        requireScalarBitsEqual(
            current(0), tested_step,
            "minimum-boundary exact trial vector drifted");

        const mfa::NSConstrainedLineSearchTelemetry telemetry =
            TestBoundAwareLineSearch::currentTelemetry();
        const TestBoundAwareLineSearch::CallSnapshot snapshot =
            TestBoundAwareLineSearch::currentCallSnapshot();
        requireResetCallAccounting(
            telemetry, 1, 1, "minimum-boundary success");
        require(
            telemetry.calls == 1 &&
            telemetry.primary_candidate_attempts == 1 &&
            telemetry.primary_candidate_completions == 1 &&
            telemetry.retry_candidate_attempts == 0 &&
            telemetry.strong_wolfe_accepts == 1 &&
            telemetry.box_endpoint_accepts == 0 &&
            telemetry.last_attempted_step_present &&
            telemetry.last_completed_step_present &&
            telemetry.last_evaluated_step_present &&
            snapshot.primary_present &&
            !snapshot.retry_present &&
            snapshot.exception_class == "NONE" &&
            snapshot.termination_reason == "STRONG_WOLFE",
            "minimum-boundary wrapper accounting drifted");
        requireScalarBitsEqual(
            telemetry.last_attempted_step, tested_step,
            "minimum-boundary last attempted step drifted");
        requireScalarBitsEqual(
            telemetry.last_completed_step, tested_step,
            "minimum-boundary last completed step drifted");
        requireScalarBitsEqual(
            telemetry.last_evaluated_step, tested_step,
            "minimum-boundary last evaluated step drifted");
        const TestGenerator::Diagnostics& diagnostics =
            snapshot.primary_diagnostics;
        require(
            diagnostics.proposal_origin ==
                LBFGSpp::LBFGSPP_MT_PROPOSAL_INITIAL &&
            diagnostics.raw_proposed_step_present &&
            diagnostics.raw_proposed_step_class ==
                LBFGSpp::LBFGSPP_MT_RAW_FINITE &&
            diagnostics.safeguarded_proposed_step_present &&
            diagnostics.midpoint_trigger ==
                LBFGSpp::LBFGSPP_MT_MIDPOINT_NONE &&
            diagnostics.guard_failure ==
                LBFGSpp::LBFGSPP_MT_GUARD_NONE &&
            diagnostics.termination_reason ==
                LBFGSpp::LBFGSPP_MT_TERMINATION_STRONG_WOLFE &&
            diagnostics.candidate_attempts == 1 &&
            diagnostics.candidate_completions == 1 &&
            diagnostics.last_attempted_step_present &&
            diagnostics.last_completed_step_present &&
            diagnostics.last_evaluated_step_present,
            "minimum-boundary generator diagnostics drifted");
        requireScalarBitsEqual(
            diagnostics.raw_proposed_step, tested_step,
            "minimum-boundary raw proposal drifted");
        requireScalarBitsEqual(
            diagnostics.safeguarded_proposed_step, tested_step,
            "minimum-boundary safeguarded proposal drifted");
    }
}

void testV17NonfiniteTrialGuard()
{
    const LBFGSpp::LBFGSBParam<Scalar> parameters =
        lineSearchParametersForCandidateCap(80);
    NeverCalledLineFunction oracle;
    Vector previous(1), direction(1), gradient(1), current(1);
    previous(0) = (std::numeric_limits<Scalar>::max)();
    direction(0) = (std::numeric_limits<Scalar>::max)();
    gradient(0) = Scalar(-1);
    current = previous;
    Scalar step = Scalar(1);
    Scalar value = Scalar(0);
    Scalar derivative = gradient.dot(direction);
    require(
        previous.allFinite() && direction.allFinite() &&
        gradient.allFinite() && std::isfinite(derivative) &&
        !std::isfinite(previous(0) + step * direction(0)),
        "nonfinite-trial precondition drifted");

    TestBoundAwareLineSearch::resetTelemetry();
    bool caught = false;
    try
    {
        TestBoundAwareLineSearch::LineSearch(
            oracle, parameters, previous, direction, Scalar(1),
            step, value, gradient, derivative, current);
    }
    catch (const TestGenerator::GuardException& error)
    {
        caught = true;
        require(
            error.reason() ==
                LBFGSpp::LBFGSPP_MT_GUARD_NONFINITE_TRIAL_VECTOR,
            "nonfinite trial used the wrong guard");
    }
    require(caught, "nonfinite trial did not fail closed");
    require(oracle.calls == 0, "nonfinite trial reached the provider");

    const mfa::NSConstrainedLineSearchTelemetry telemetry =
        TestBoundAwareLineSearch::currentTelemetry();
    const TestBoundAwareLineSearch::CallSnapshot snapshot =
        TestBoundAwareLineSearch::currentCallSnapshot();
    requireResetCallAccounting(
        telemetry, 0, 0, "nonfinite trial");
    require(
        telemetry.calls == 1 &&
        !telemetry.last_attempted_step_present &&
        !telemetry.last_completed_step_present &&
        snapshot.primary_present &&
        !snapshot.retry_present &&
        snapshot.exception_class ==
            "GENERATOR_GUARD_EXCEPTION" &&
        snapshot.guard_reason == "NONFINITE_TRIAL_VECTOR" &&
        snapshot.termination_reason == "GUARD_EXCEPTION",
        "nonfinite-trial wrapper diagnostics drifted");
    const TestGenerator::Diagnostics& diagnostics =
        snapshot.primary_diagnostics;
    require(
        diagnostics.proposal_origin ==
            LBFGSpp::LBFGSPP_MT_PROPOSAL_INITIAL &&
        diagnostics.raw_proposed_step_present &&
        diagnostics.raw_proposed_step_class ==
            LBFGSpp::LBFGSPP_MT_RAW_FINITE &&
        diagnostics.safeguarded_proposed_step_present &&
        diagnostics.midpoint_trigger ==
            LBFGSpp::LBFGSPP_MT_MIDPOINT_NONE &&
        diagnostics.guard_failure ==
            LBFGSpp::LBFGSPP_MT_GUARD_NONFINITE_TRIAL_VECTOR &&
        diagnostics.candidate_attempts == 0 &&
        diagnostics.candidate_completions == 0,
        "nonfinite-trial generator diagnostics drifted");
    requireScalarBitsEqual(
        diagnostics.raw_proposed_step, Scalar(1),
        "nonfinite-trial raw proposal drifted");
    requireScalarBitsEqual(
        diagnostics.safeguarded_proposed_step, Scalar(1),
        "nonfinite-trial safeguarded proposal drifted");
}

void testV17RemainingPreProviderGuards()
{
    const LBFGSpp::LBFGSBParam<Scalar> parameters =
        lineSearchParametersForCandidateCap(80);

    {
        NeverCalledLineFunction oracle;
        TestGenerator::Diagnostics diagnostics;
        TestGenerator::CandidateAccounting accounting(80);
        bool caught = false;
        try
        {
            TestGenerator::ValidateScalarCandidatePreProviderForTesting(
                Scalar(4), parameters.min_step, Scalar(1),
                true, Scalar(1), false, Scalar(0), Scalar(0),
                diagnostics);
        }
        catch (const TestGenerator::GuardException& error)
        {
            caught = true;
            require(
                error.reason() ==
                    LBFGSpp::LBFGSPP_MT_GUARD_DUPLICATE_SCALAR_STEP,
                "duplicate scalar used the wrong guard");
        }
        require(
            caught && oracle.calls == 0,
            "duplicate scalar diagnostics seam reached a provider");
        requireDirectGuardAccounting(
            diagnostics, accounting,
            LBFGSpp::LBFGSPP_MT_GUARD_DUPLICATE_SCALAR_STEP,
            0, 0, "duplicate scalar");
        require(
            !diagnostics.raw_proposed_step_present &&
            !diagnostics.safeguarded_proposed_step_present &&
            !diagnostics.last_attempted_step_present &&
            !diagnostics.last_completed_step_present &&
            !diagnostics.last_evaluated_step_present,
            "duplicate scalar diagnostics seam mutated candidate state");
    }

    {
        NeverCalledLineFunction oracle;
        Vector previous(1), direction(1), gradient(1), current(1);
        previous(0) = Scalar(0);
        direction(0) = Scalar(1);
        gradient(0) = Scalar(-1);
        current = previous;
        Scalar step = std::nextafter(
            Scalar(4), std::numeric_limits<Scalar>::infinity());
        Scalar value = Scalar(0);
        Scalar derivative = Scalar(-1);
        TestGenerator::Diagnostics diagnostics;
        TestGenerator::CandidateAccounting accounting(80);
        TestGenerator::InvocationContext context(&accounting, 0);
        bool caught = false;
        try
        {
            TestGenerator::LineSearchWithDiagnostics(
                oracle, parameters, previous, direction, Scalar(4),
                step, value, gradient, derivative, current,
                &diagnostics, &context);
        }
        catch (const TestGenerator::GuardException& error)
        {
            caught = true;
            require(
                error.reason() ==
                    LBFGSpp::LBFGSPP_MT_GUARD_OUT_OF_RANGE_PROPOSAL,
                "out-of-range initial proposal used the wrong guard");
        }
        require(
            caught && oracle.calls == 0,
            "out-of-range initial proposal reached the provider");
        requireDirectGuardAccounting(
            diagnostics, accounting,
            LBFGSpp::LBFGSPP_MT_GUARD_OUT_OF_RANGE_PROPOSAL,
            0, 0, "out-of-range initial proposal");
        require(
            diagnostics.proposal_origin ==
                LBFGSpp::LBFGSPP_MT_PROPOSAL_INITIAL &&
            diagnostics.raw_proposed_step_present &&
            diagnostics.raw_proposed_step_class ==
                LBFGSpp::LBFGSPP_MT_RAW_FINITE &&
            diagnostics.safeguarded_proposed_step_present &&
            diagnostics.midpoint_trigger ==
                LBFGSpp::LBFGSPP_MT_MIDPOINT_NONE,
            "out-of-range proposal diagnostics drifted");
        requireScalarBitsEqual(
            diagnostics.raw_proposed_step, step,
            "out-of-range raw proposal drifted");
        requireScalarBitsEqual(
            diagnostics.safeguarded_proposed_step, step,
            "out-of-range safeguarded proposal drifted");
    }

    {
        NeverCalledLineFunction oracle;
        Vector previous(1), direction(1), gradient(1), current(1);
        previous(0) = Scalar(1.0e20);
        direction(0) = Scalar(1);
        gradient(0) = Scalar(-1);
        current = previous;
        Scalar step = Scalar(1);
        Scalar value = Scalar(0);
        Scalar derivative = Scalar(-1);
        require(
            previous(0) + step * direction(0) == previous(0),
            "base-state-equal trial precondition drifted");
        TestGenerator::Diagnostics diagnostics;
        TestGenerator::CandidateAccounting accounting(80);
        TestGenerator::InvocationContext context(&accounting, 0);
        bool caught = false;
        try
        {
            TestGenerator::LineSearchWithDiagnostics(
                oracle, parameters, previous, direction, Scalar(4),
                step, value, gradient, derivative, current,
                &diagnostics, &context);
        }
        catch (const TestGenerator::GuardException& error)
        {
            caught = true;
            require(
                error.reason() ==
                    LBFGSpp::LBFGSPP_MT_GUARD_TRIAL_EQUALS_PREVIOUS_STATE,
                "base-state-equal trial used the wrong guard");
        }
        require(
            caught && oracle.calls == 0,
            "base-state-equal trial reached the provider");
        requireDirectGuardAccounting(
            diagnostics, accounting,
            LBFGSpp::LBFGSPP_MT_GUARD_TRIAL_EQUALS_PREVIOUS_STATE,
            0, 0, "base-state-equal trial");
        require(
            diagnostics.proposal_origin ==
                LBFGSpp::LBFGSPP_MT_PROPOSAL_INITIAL &&
            diagnostics.raw_proposed_step_present &&
            diagnostics.safeguarded_proposed_step_present &&
            diagnostics.midpoint_trigger ==
                LBFGSpp::LBFGSPP_MT_MIDPOINT_NONE,
            "base-state-equal proposal diagnostics drifted");
    }

    const Scalar one_ulp =
        std::nextafter(
            Scalar(1), std::numeric_limits<Scalar>::infinity()) -
        Scalar(1);
    const Scalar narrow_direction = Scalar(0.6) * one_ulp;
    const Scalar narrow_gradient = -Scalar(1) / narrow_direction;
    const Scalar second_scalar =
        Scalar(1) +
        Scalar(1.1) * (Scalar(1) - Scalar(0));
    require(
        one_ulp > Scalar(0) &&
        second_scalar == Scalar(2.1) &&
        Scalar(1) + narrow_direction != Scalar(1) &&
        Scalar(1) + narrow_direction ==
            Scalar(1) + second_scalar * narrow_direction,
        "duplicate-trial rounding precondition drifted");

    {
        OneCallScriptedOracle oracle(
            Scalar(-1), narrow_gradient);
        Vector previous(1), direction(1), gradient(1), current(1);
        previous(0) = Scalar(1);
        direction(0) = narrow_direction;
        gradient(0) = narrow_gradient;
        current = previous;
        Scalar step = Scalar(1);
        Scalar value = Scalar(0);
        Scalar derivative = gradient.dot(direction);
        require(
            derivative < Scalar(0) &&
            std::abs(derivative) >
                parameters.wolfe * std::abs(derivative),
            "duplicate-trial curvature precondition drifted");
        TestGenerator::Diagnostics diagnostics;
        TestGenerator::CandidateAccounting accounting(80);
        TestGenerator::InvocationContext context(&accounting, 0);
        bool caught = false;
        try
        {
            TestGenerator::LineSearchWithDiagnostics(
                oracle, parameters, previous, direction, Scalar(4),
                step, value, gradient, derivative, current,
                &diagnostics, &context);
        }
        catch (const TestGenerator::GuardException& error)
        {
            caught = true;
            require(
                error.reason() ==
                    LBFGSpp::LBFGSPP_MT_GUARD_DUPLICATE_TRIAL_VECTOR,
                "consecutive duplicate trial used the wrong guard");
        }
        require(
            caught && oracle.calls == 1 &&
            oracle.observed.size() == 1 &&
            oracle.observed[0].size() == 1,
            "consecutive duplicate trial changed provider count");
        requireScalarBitsEqual(
            oracle.observed[0](0),
            Scalar(1) + narrow_direction,
            "consecutive duplicate first trial vector drifted");
        requireDirectGuardAccounting(
            diagnostics, accounting,
            LBFGSpp::LBFGSPP_MT_GUARD_DUPLICATE_TRIAL_VECTOR,
            1, 1, "consecutive duplicate trial");
        require(
            diagnostics.proposal_origin ==
                LBFGSpp::LBFGSPP_MT_PROPOSAL_CASE_2 &&
            diagnostics.raw_proposed_step_present &&
            diagnostics.raw_proposed_step_class ==
                LBFGSpp::LBFGSPP_MT_RAW_FINITE &&
            diagnostics.safeguarded_proposed_step_present &&
            diagnostics.midpoint_trigger ==
                LBFGSpp::LBFGSPP_MT_MIDPOINT_NONE &&
            !diagnostics.final_bracketed,
            "consecutive duplicate-trial diagnostics drifted");
        requireScalarBitsEqual(
            diagnostics.raw_proposed_step, second_scalar,
            "duplicate-trial raw proposal drifted");
        requireScalarBitsEqual(
            diagnostics.safeguarded_proposed_step, second_scalar,
            "duplicate-trial safeguarded proposal drifted");
        requireScalarBitsEqual(
            diagnostics.last_completed_step, Scalar(1),
            "duplicate-trial last completion drifted");
    }

    require(
        Scalar(1) + Scalar(2) * narrow_direction ==
            Scalar(1) + narrow_direction &&
        Scalar(1) + narrow_direction != Scalar(1),
        "bracket-endpoint rounding precondition drifted");
    {
        OneCallScriptedOracle oracle(
            std::numeric_limits<Scalar>::infinity(),
            narrow_gradient);
        Vector previous(1), direction(1), gradient(1), current(1);
        previous(0) = Scalar(1);
        direction(0) = narrow_direction;
        gradient(0) = narrow_gradient;
        current = previous;
        Scalar step = Scalar(2);
        Scalar value = Scalar(0);
        Scalar derivative = gradient.dot(direction);
        TestGenerator::Diagnostics diagnostics;
        TestGenerator::CandidateAccounting accounting(80);
        TestGenerator::InvocationContext context(&accounting, 0);
        bool caught = false;
        try
        {
            TestGenerator::LineSearchWithDiagnostics(
                oracle, parameters, previous, direction, Scalar(4),
                step, value, gradient, derivative, current,
                &diagnostics, &context);
        }
        catch (const TestGenerator::GuardException& error)
        {
            caught = true;
            require(
                error.reason() ==
                    LBFGSpp::LBFGSPP_MT_GUARD_TRIAL_EQUALS_BRACKET_ENDPOINT,
                "stored-bracket-endpoint trial used the wrong guard");
        }
        require(
            caught && oracle.calls == 1 &&
            oracle.observed.size() == 1 &&
            oracle.observed[0].size() == 1,
            "stored-bracket-endpoint trial changed provider count");
        requireScalarBitsEqual(
            oracle.observed[0](0),
            Scalar(1) + Scalar(2) * narrow_direction,
            "stored bracket high-endpoint vector drifted");
        requireDirectGuardAccounting(
            diagnostics, accounting,
            LBFGSpp::LBFGSPP_MT_GUARD_TRIAL_EQUALS_BRACKET_ENDPOINT,
            1, 1, "stored-bracket-endpoint trial");
        require(
            diagnostics.proposal_origin ==
                LBFGSpp::LBFGSPP_MT_PROPOSAL_CASE_1 &&
            diagnostics.raw_proposed_step_present &&
            diagnostics.raw_proposed_step_class ==
                LBFGSpp::LBFGSPP_MT_RAW_FINITE &&
            diagnostics.safeguarded_proposed_step_present &&
            diagnostics.midpoint_trigger ==
                LBFGSpp::LBFGSPP_MT_MIDPOINT_NONE &&
            diagnostics.final_bracketed &&
            diagnostics.final_bracket_lo == Scalar(0) &&
            diagnostics.final_bracket_hi == Scalar(2) &&
            diagnostics.final_bracket_width == Scalar(2),
            "stored-bracket-endpoint diagnostics drifted");
        requireScalarBitsEqual(
            diagnostics.raw_proposed_step, Scalar(1),
            "stored-bracket-endpoint raw proposal drifted");
        requireScalarBitsEqual(
            diagnostics.safeguarded_proposed_step, Scalar(1),
            "stored-bracket-endpoint safeguarded proposal drifted");
        requireScalarBitsEqual(
            diagnostics.last_completed_step, Scalar(2),
            "stored-bracket-endpoint last completion drifted");
    }
}

void testV17SquaredHingeBudgetAndBracket()
{
    const Scalar alpha_max =
        C1SquaredHingeOracle::scale() *
        Scalar(299.00529848664178);

    {
        const LBFGSpp::LBFGSBParam<Scalar> parameters =
            lineSearchParametersForCandidateCap(50);
        C1SquaredHingeOracle oracle;
        Vector previous(1), direction(1), gradient(1), current(1);
        previous(0) = Scalar(0);
        direction(0) = Scalar(1);
        gradient(0) = -C1SquaredHingeOracle::g();
        current = previous;
        Scalar step = C1SquaredHingeOracle::scale();
        Scalar value = Scalar(0);
        Scalar derivative = -C1SquaredHingeOracle::g();
        TestBoundAwareLineSearch::resetTelemetry();
        bool caught = false;
        std::string message;
        try
        {
            TestBoundAwareLineSearch::LineSearch(
                oracle, parameters, previous, direction, alpha_max,
                step, value, gradient, derivative, current);
        }
        catch (const std::exception& error)
        {
            caught = true;
            message = error.what();
        }
        require(
            caught &&
            message.find("termination=MAX_EVAL_NO_SW") !=
                std::string::npos,
            "C1 50-candidate fixture did not fail closed");
        require(
            oracle.steps.size() == 50,
            "C1 50-candidate fixture used the wrong provider count");

        const mfa::NSConstrainedLineSearchTelemetry telemetry =
            TestBoundAwareLineSearch::currentTelemetry();
        const TestBoundAwareLineSearch::CallSnapshot snapshot =
            TestBoundAwareLineSearch::currentCallSnapshot();
        requireResetCallAccounting(
            telemetry, 50, 50, "C1 50-candidate fixture");
        require(
            telemetry.calls == 1 &&
            telemetry.primary_candidate_attempts == 50 &&
            telemetry.primary_candidate_completions == 50 &&
            telemetry.retry_candidate_attempts == 0 &&
            telemetry.call_forced_midpoints >= 1 &&
            telemetry.strong_wolfe_accepts == 0 &&
            telemetry.box_endpoint_accepts == 0 &&
            snapshot.primary_present &&
            !snapshot.retry_present &&
            snapshot.exception_class == "LINE_SEARCH_REJECTION" &&
            snapshot.guard_reason == "NONE" &&
            snapshot.termination_reason == "MAX_EVAL_NO_SW",
            "C1 50-candidate wrapper diagnostics drifted");
        const TestGenerator::Diagnostics& diagnostics =
            snapshot.primary_diagnostics;
        require(
            diagnostics.candidate_attempts == 50 &&
            diagnostics.candidate_completions == 50 &&
            diagnostics.forced_midpoints >= 1 &&
            diagnostics.last_completed_step_present &&
            diagnostics.termination_reason ==
                LBFGSpp::LBFGSPP_MT_TERMINATION_MAX_ITERATIONS,
            "C1 50-candidate generator diagnostics drifted");
        requireFiniteContainingBracket(
            diagnostics, diagnostics.last_completed_step,
            "C1 50-candidate fixture");
    }

    {
        const LBFGSpp::LBFGSBParam<Scalar> parameters =
            lineSearchParametersForCandidateCap(80);
        C1SquaredHingeOracle oracle;
        Vector previous(1), direction(1), gradient(1), current(1);
        previous(0) = Scalar(0);
        direction(0) = Scalar(1);
        gradient(0) = -C1SquaredHingeOracle::g();
        current = previous;
        Scalar step = C1SquaredHingeOracle::scale();
        Scalar value = Scalar(0);
        Scalar derivative = -C1SquaredHingeOracle::g();
        TestBoundAwareLineSearch::resetTelemetry();
        TestBoundAwareLineSearch::LineSearch(
            oracle, parameters, previous, direction, alpha_max,
            step, value, gradient, derivative, current);

        const Scalar strong_wolfe_lo =
            C1SquaredHingeOracle::tau() +
            (Scalar(1) - parameters.wolfe) *
                C1SquaredHingeOracle::g() /
                C1SquaredHingeOracle::kappa();
        const Scalar strong_wolfe_hi =
            C1SquaredHingeOracle::tau() +
            (Scalar(1) + parameters.wolfe) *
                C1SquaredHingeOracle::g() /
                C1SquaredHingeOracle::kappa();
        require(
            oracle.steps.size() > 50 &&
            oracle.steps.size() <= 80 &&
            step > Scalar(0) && step < alpha_max &&
            step >= strong_wolfe_lo &&
            step <= strong_wolfe_hi &&
            current.size() == 1 && current(0) == step &&
            value <=
                step * parameters.ftol *
                    (-C1SquaredHingeOracle::g()) &&
            std::abs(derivative) <=
                parameters.wolfe *
                    C1SquaredHingeOracle::g(),
            "C1 80-candidate exact strong-Wolfe result drifted");

        const mfa::NSConstrainedLineSearchTelemetry telemetry =
            TestBoundAwareLineSearch::currentTelemetry();
        const TestBoundAwareLineSearch::CallSnapshot snapshot =
            TestBoundAwareLineSearch::currentCallSnapshot();
        requireResetCallAccounting(
            telemetry,
            static_cast<std::uint64_t>(oracle.steps.size()),
            static_cast<std::uint64_t>(oracle.steps.size()),
            "C1 80-candidate fixture");
        require(
            telemetry.calls == 1 &&
            telemetry.call_candidate_attempts > 50 &&
            telemetry.call_candidate_attempts <= 80 &&
            telemetry.retry_candidate_attempts == 0 &&
            telemetry.call_forced_midpoints >= 1 &&
            telemetry.strong_wolfe_accepts == 1 &&
            telemetry.box_endpoint_accepts == 0 &&
            snapshot.primary_present &&
            !snapshot.retry_present &&
            snapshot.exception_class == "NONE" &&
            snapshot.guard_reason == "NONE" &&
            snapshot.termination_reason == "STRONG_WOLFE",
            "C1 80-candidate wrapper diagnostics drifted");
        const TestGenerator::Diagnostics& diagnostics =
            snapshot.primary_diagnostics;
        require(
            diagnostics.candidate_attempts ==
                oracle.steps.size() &&
            diagnostics.candidate_completions ==
                oracle.steps.size() &&
            diagnostics.forced_midpoints >= 1 &&
            diagnostics.last_completed_step_present &&
            diagnostics.termination_reason ==
                LBFGSpp::LBFGSPP_MT_TERMINATION_STRONG_WOLFE,
            "C1 80-candidate generator diagnostics drifted");
        requireFiniteContainingBracket(
            diagnostics, step, "C1 80-candidate fixture");
        requireScalarBitsEqual(
            diagnostics.last_completed_step, step,
            "C1 accepted candidate was not the last completion");
    }
}

void testV17ExactLinearEightyAttemptCap()
{
    const LBFGSpp::LBFGSBParam<Scalar> parameters =
        lineSearchParametersForCandidateCap(80);
    RecordingLinearLineFunction oracle;
    Vector previous(1), direction(1), gradient(1), current(1);
    previous(0) = Scalar(0);
    direction(0) = Scalar(1);
    gradient(0) = Scalar(-1);
    current = previous;
    Scalar step = Scalar(1);
    Scalar value = Scalar(0);
    Scalar derivative = Scalar(-1);
    TestBoundAwareLineSearch::resetTelemetry();
    bool caught = false;
    std::string message;
    try
    {
        TestBoundAwareLineSearch::LineSearch(
            oracle, parameters, previous, direction,
            parameters.max_step, step, value, gradient,
            derivative, current);
    }
    catch (const std::exception& error)
    {
        caught = true;
        message = error.what();
    }
    require(
        caught &&
        message.find("termination=MAX_EVAL_NO_SW") !=
            std::string::npos,
        "linear exact-80 fixture did not terminate MAX_EVAL_NO_SW");
    require(
        oracle.steps.size() == 80,
        "linear fixture made fewer than 80 or an 81st provider call");
    for (std::size_t i = 0; i < oracle.steps.size(); ++i)
    {
        require(
            std::isfinite(oracle.steps[i]) &&
            oracle.steps[i] > Scalar(0) &&
            oracle.steps[i] <= parameters.max_step,
            "linear fixture observed an invalid candidate");
        if (i > 0)
            require(
                oracle.steps[i] != oracle.steps[i - 1],
                "linear fixture evaluated a duplicate consecutive scalar");
    }

    const mfa::NSConstrainedLineSearchTelemetry telemetry =
        TestBoundAwareLineSearch::currentTelemetry();
    const TestBoundAwareLineSearch::CallSnapshot snapshot =
        TestBoundAwareLineSearch::currentCallSnapshot();
    requireResetCallAccounting(
        telemetry, 80, 80, "linear exact-80 fixture");
    require(
        telemetry.calls == 1 &&
        telemetry.primary_candidate_attempts == 80 &&
        telemetry.primary_candidate_completions == 80 &&
        telemetry.retry_candidate_attempts == 0 &&
        telemetry.retry_candidate_completions == 0 &&
        telemetry.strong_wolfe_accepts == 0 &&
        telemetry.box_endpoint_accepts == 0 &&
        telemetry.last_attempted_step_present &&
        telemetry.last_completed_step_present &&
        telemetry.last_evaluated_step_present &&
        snapshot.primary_present &&
        !snapshot.retry_present &&
        snapshot.exception_class == "LINE_SEARCH_REJECTION" &&
        snapshot.guard_reason == "NONE" &&
        snapshot.original_exception_type == "NONE" &&
        snapshot.termination_reason == "MAX_EVAL_NO_SW" &&
        snapshot.call_accounting.configured_attempt_cap == 80 &&
        snapshot.call_accounting.call_attempts == 80 &&
        snapshot.call_accounting.call_completions == 80,
        "linear exact-80 accounting drifted");
    requireScalarBitsEqual(
        telemetry.last_attempted_step, oracle.steps.back(),
        "linear last attempted step drifted");
    requireScalarBitsEqual(
        telemetry.last_completed_step, oracle.steps.back(),
        "linear last completed step drifted");
    requireScalarBitsEqual(
        telemetry.last_evaluated_step, oracle.steps.back(),
        "linear last evaluated step drifted");
    require(
        snapshot.primary_diagnostics.candidate_attempts == 80 &&
        snapshot.primary_diagnostics.candidate_completions == 80 &&
        snapshot.primary_diagnostics.termination_reason ==
            LBFGSpp::LBFGSPP_MT_TERMINATION_MAX_ITERATIONS,
        "linear generator termination diagnostics drifted");
}

void requireV17SafeguardCase(
    LBFGSpp::LineSearchMoreThuenteProposalOrigin origin,
    Scalar raw_proposal,
    Scalar initial_width1,
    LBFGSpp::LineSearchMoreThuenteMidpointTrigger
        expected_trigger,
    Scalar expected_safeguarded,
    std::uint64_t expected_forced_midpoints,
    const std::string& label)
{
    Scalar width = Scalar(3);
    Scalar width1 = initial_width1;
    std::uint64_t cumulative_attempt_count = 0;
    std::uint64_t cumulative_completion_count = 0;
    std::uint64_t cumulative_midpoint_count = 0;
    TestGenerator::Diagnostics diagnostics;
    TestGenerator::CandidateAccounting accounting(80);
    accounting.cumulative_attempts = &cumulative_attempt_count;
    accounting.cumulative_completions =
        &cumulative_completion_count;
    accounting.cumulative_forced_midpoints =
        &cumulative_midpoint_count;
    const Scalar safeguarded =
        TestGenerator::SafeguardBracketedProposal(
            Scalar(0), Scalar(2), raw_proposal,
            width, width1, origin, diagnostics, &accounting);
    requireScalarBitsEqual(
        safeguarded, expected_safeguarded,
        label + ": returned safeguard drifted");
    requireScalarBitsEqual(
        diagnostics.raw_proposed_step, raw_proposal,
        label + ": raw proposal drifted");
    requireScalarBitsEqual(
        diagnostics.safeguarded_proposed_step,
        expected_safeguarded,
        label + ": safeguarded proposal drifted");
    require(
        diagnostics.proposal_origin == origin &&
        diagnostics.raw_proposed_step_present &&
        diagnostics.raw_proposed_step_class ==
            LBFGSpp::LBFGSPP_MT_RAW_FINITE &&
        diagnostics.safeguarded_proposed_step_present &&
        diagnostics.midpoint_trigger == expected_trigger &&
        diagnostics.forced_midpoints ==
            expected_forced_midpoints &&
        diagnostics.final_bracketed &&
        diagnostics.final_bracket_lo == Scalar(0) &&
        diagnostics.final_bracket_hi == Scalar(2) &&
        diagnostics.final_bracket_width == Scalar(2) &&
        diagnostics.guard_failure ==
            LBFGSpp::LBFGSPP_MT_GUARD_NONE &&
        diagnostics.candidate_attempts == 0 &&
        diagnostics.candidate_completions == 0 &&
        accounting.call_attempts == 0 &&
        accounting.call_completions == 0 &&
        accounting.call_forced_midpoints ==
            expected_forced_midpoints &&
        cumulative_attempt_count == 0 &&
        cumulative_completion_count == 0 &&
        cumulative_midpoint_count ==
            expected_forced_midpoints &&
        width == Scalar(2) &&
        width1 == Scalar(3),
        label + ": safeguard diagnostics or width order drifted");
}

void testV17ProposalStagesAndMidpointTriggers()
{
    requireV17SafeguardCase(
        LBFGSpp::LBFGSPP_MT_PROPOSAL_CASE_1,
        Scalar(0.5), Scalar(4),
        LBFGSpp::LBFGSPP_MT_MIDPOINT_NONE,
        Scalar(0.5), 0, "midpoint NONE");
    requireV17SafeguardCase(
        LBFGSpp::LBFGSPP_MT_PROPOSAL_CASE_1,
        Scalar(3), Scalar(4),
        LBFGSpp::LBFGSPP_MT_MIDPOINT_NONINTERIOR,
        Scalar(1), 1, "midpoint NONINTERIOR");
    requireV17SafeguardCase(
        LBFGSpp::LBFGSPP_MT_PROPOSAL_CASE_2,
        Scalar(0.5), Scalar(3),
        LBFGSpp::LBFGSPP_MT_MIDPOINT_WIDTH_0P66,
        Scalar(1), 1, "midpoint WIDTH_0P66");
    requireV17SafeguardCase(
        LBFGSpp::LBFGSPP_MT_PROPOSAL_CASE_3,
        Scalar(3), Scalar(3),
        LBFGSpp::LBFGSPP_MT_MIDPOINT_BOTH,
        Scalar(1), 1, "midpoint BOTH");

    {
        Scalar width = Scalar(3);
        Scalar width1 = Scalar(4);
        TestGenerator::Diagnostics diagnostics;
        TestGenerator::CandidateAccounting accounting(80);
        bool caught = false;
        try
        {
            TestGenerator::SafeguardBracketedProposal(
                Scalar(0), Scalar(2),
                std::numeric_limits<Scalar>::quiet_NaN(),
                width, width1,
                LBFGSpp::LBFGSPP_MT_PROPOSAL_CASE_3,
                diagnostics, &accounting);
        }
        catch (const TestGenerator::GuardException& error)
        {
            caught = true;
            require(
                error.reason() ==
                    LBFGSpp::LBFGSPP_MT_GUARD_NONFINITE_PROPOSAL,
                "nonfinite raw proposal used the wrong guard");
        }
        require(
            caught &&
            diagnostics.proposal_origin ==
                LBFGSpp::LBFGSPP_MT_PROPOSAL_CASE_3 &&
            diagnostics.raw_proposed_step_present &&
            diagnostics.raw_proposed_step_class ==
                LBFGSpp::LBFGSPP_MT_RAW_NAN &&
            !diagnostics.safeguarded_proposed_step_present &&
            diagnostics.midpoint_trigger ==
                LBFGSpp::LBFGSPP_MT_MIDPOINT_NONE &&
            diagnostics.guard_failure ==
                LBFGSpp::LBFGSPP_MT_GUARD_NONFINITE_PROPOSAL &&
            diagnostics.candidate_attempts == 0 &&
            diagnostics.candidate_completions == 0 &&
            accounting.call_attempts == 0 &&
            accounting.call_completions == 0 &&
            width == Scalar(3) && width1 == Scalar(4),
            "nonfinite raw proposal diagnostics drifted");
    }

    {
        const Scalar lo = Scalar(1);
        const Scalar hi = std::nextafter(
            lo, std::numeric_limits<Scalar>::infinity());
        const Scalar ulp = hi - lo;
        require(
            ulp > Scalar(0) &&
            lo + Scalar(0.5) * ulp == lo,
            "adjacent-float midpoint precondition drifted");
        Scalar width = Scalar(3) * ulp;
        Scalar width1 = Scalar(4) * ulp;
        std::uint64_t cumulative_attempts = 0;
        std::uint64_t cumulative_completions = 0;
        std::uint64_t cumulative_midpoints = 0;
        TestGenerator::Diagnostics diagnostics;
        TestGenerator::CandidateAccounting accounting(80);
        accounting.cumulative_attempts = &cumulative_attempts;
        accounting.cumulative_completions =
            &cumulative_completions;
        accounting.cumulative_forced_midpoints =
            &cumulative_midpoints;
        bool caught = false;
        try
        {
            TestGenerator::SafeguardBracketedProposal(
                lo, hi, hi, width, width1,
                LBFGSpp::LBFGSPP_MT_PROPOSAL_CASE_1,
                diagnostics, &accounting);
        }
        catch (const TestGenerator::GuardException& error)
        {
            caught = true;
            require(
                error.reason() ==
                    LBFGSpp::LBFGSPP_MT_GUARD_NO_REPRESENTABLE_BRACKET_INTERIOR,
                "adjacent-float midpoint used the wrong guard");
        }
        require(
            caught &&
            diagnostics.raw_proposed_step_present &&
            diagnostics.raw_proposed_step_class ==
                LBFGSpp::LBFGSPP_MT_RAW_FINITE &&
            diagnostics.safeguarded_proposed_step_present &&
            diagnostics.midpoint_trigger ==
                LBFGSpp::LBFGSPP_MT_MIDPOINT_NONINTERIOR &&
            diagnostics.forced_midpoints == 1 &&
            diagnostics.guard_failure ==
                LBFGSpp::LBFGSPP_MT_GUARD_NO_REPRESENTABLE_BRACKET_INTERIOR &&
            diagnostics.candidate_attempts == 0 &&
            diagnostics.candidate_completions == 0 &&
            accounting.call_attempts == 0 &&
            accounting.call_completions == 0 &&
            accounting.call_forced_midpoints == 1 &&
            cumulative_attempts == 0 &&
            cumulative_completions == 0 &&
            cumulative_midpoints == 1,
            "adjacent-float midpoint diagnostics drifted");
        requireScalarBitsEqual(
            diagnostics.raw_proposed_step, hi,
            "adjacent-float raw proposal drifted");
        requireScalarBitsEqual(
            diagnostics.safeguarded_proposed_step, lo,
            "adjacent-float rounded midpoint was not retained");
    }

    {
        const LBFGSpp::LBFGSBParam<Scalar> parameters =
            lineSearchParametersForCandidateCap(80);
        RecordingLinearLineFunction oracle;
        Vector previous(1), direction(1), gradient(1), current(1);
        previous(0) = Scalar(0);
        direction(0) = Scalar(1);
        gradient(0) = Scalar(-1);
        current = previous;
        Scalar step = Scalar(2);
        Scalar value = Scalar(0);
        Scalar derivative = Scalar(-1);
        std::uint64_t cumulative_attempts = 0;
        std::uint64_t cumulative_completions = 0;
        std::uint64_t cumulative_midpoints = 0;
        TestGenerator::Diagnostics diagnostics;
        TestGenerator::CandidateAccounting accounting(80);
        accounting.cumulative_attempts = &cumulative_attempts;
        accounting.cumulative_completions =
            &cumulative_completions;
        accounting.cumulative_forced_midpoints =
            &cumulative_midpoints;
        TestGenerator::InvocationContext context(&accounting, 0);
        TestGenerator::LineSearchWithDiagnostics(
            oracle, parameters, previous, direction, Scalar(2),
            step, value, gradient, derivative, current,
            &diagnostics, &context);
        require(
            oracle.steps.size() == 1 &&
            diagnostics.proposal_origin ==
                LBFGSpp::LBFGSPP_MT_PROPOSAL_CASE_2 &&
            diagnostics.raw_proposed_step_present &&
            diagnostics.raw_proposed_step_class ==
                LBFGSpp::LBFGSPP_MT_RAW_FINITE &&
            !diagnostics.safeguarded_proposed_step_present &&
            diagnostics.midpoint_trigger ==
                LBFGSpp::LBFGSPP_MT_MIDPOINT_NONE &&
            diagnostics.guard_failure ==
                LBFGSpp::LBFGSPP_MT_GUARD_NONE &&
            diagnostics.termination_reason ==
                LBFGSpp::LBFGSPP_MT_TERMINATION_GENERATOR_MAXIMUM &&
            diagnostics.candidate_attempts == 1 &&
            diagnostics.candidate_completions == 1 &&
            accounting.call_attempts == 1 &&
            accounting.call_completions == 1 &&
            cumulative_attempts == 1 &&
            cumulative_completions == 1 &&
            cumulative_midpoints == 0,
            "generator-maximum diagnostics drifted");
        requireScalarBitsEqual(
            diagnostics.raw_proposed_step, Scalar(2),
            "generator-maximum raw proposal drifted");
    }
}

void testV17ThrowingOracleAndCanonicalAbortBytes()
{
    std::string message;
    message.push_back('q');
    message.push_back('"');
    message.push_back('\\');
    message.push_back('\n');
    message.push_back(static_cast<char>(0xff));
    require(
        mfa::nsConstrainedLowerHexWhatBytes(message.c_str()) ==
            "71225c0aff",
        "throwing-oracle message precondition drifted");

    const LBFGSpp::LBFGSBParam<Scalar> parameters =
        lineSearchParametersForCandidateCap(80);
    PIMFATestThrowingOracle oracle(message);
    Vector previous(1), direction(1), gradient(1), current(1);
    previous(0) = Scalar(0);
    direction(0) = Scalar(1);
    gradient(0) = Scalar(-1);
    current = previous;
    Scalar step = Scalar(1);
    Scalar value = Scalar(0.5);
    Scalar derivative = Scalar(-1);
    TestBoundAwareLineSearch::resetTelemetry(7);

    RecordingStreambuf recording;
    bool caught_exact_type = false;
    bool caught_wrong_type = false;
    {
        ScopedCerrRedirect redirect(recording);
        try
        {
            TestBoundAwareLineSearch::LineSearch(
                oracle, parameters, previous, direction, Scalar(10),
                step, value, gradient, derivative, current);
        }
        catch (
            const mfa::NSConstrainedTestThrowingOracleException& error)
        {
            caught_exact_type = true;
            require(
                std::string(error.what()) == message,
                "throwing-oracle what() bytes changed");
        }
        catch (...)
        {
            caught_wrong_type = true;
        }
        require(
            caught_exact_type && !caught_wrong_type,
            "throwing oracle did not preserve its dynamic exception type");
        require(
            recording.syncCount() >= 1,
            "abort record was not flushed before exception propagation");
    }

    const std::string expected = expectedThrowingOracleAbortRecord();
    require(
        recording.bytes() == expected &&
        !recording.bytes().empty() &&
        recording.bytes()[recording.bytes().size() - 1] == '\n' &&
        std::count(
            recording.bytes().begin(),
            recording.bytes().end(), '\n') == 1,
        "throwing-oracle canonical abort bytes drifted");
    require(
        oracle.calls == 1,
        "throwing oracle was not attempted exactly once");

    const mfa::NSConstrainedLineSearchTelemetry telemetry =
        TestBoundAwareLineSearch::currentTelemetry();
    const TestBoundAwareLineSearch::CallSnapshot snapshot =
        TestBoundAwareLineSearch::currentCallSnapshot();
    requireResetCallAccounting(
        telemetry, 1, 0, "throwing oracle");
    require(
        telemetry.outer_iteration == 7 &&
        telemetry.calls == 1 &&
        telemetry.primary_candidate_attempts == 1 &&
        telemetry.primary_candidate_completions == 0 &&
        telemetry.retry_candidate_attempts == 0 &&
        telemetry.retry_candidate_completions == 0 &&
        telemetry.last_attempted_step_present &&
        !telemetry.last_completed_step_present &&
        !telemetry.last_evaluated_step_present &&
        snapshot.primary_present &&
        !snapshot.retry_present &&
        snapshot.invocation == "PRIMARY" &&
        snapshot.exception_class ==
            "OBJECTIVE_PROVIDER_EXCEPTION" &&
        snapshot.guard_reason == "NONE" &&
        snapshot.original_exception_type ==
            "PIMFA_TEST_THROWING_ORACLE" &&
        snapshot.original_exception_message_hex ==
            "71225c0aff" &&
        snapshot.termination_reason ==
            "OBJECTIVE_PROVIDER_EXCEPTION" &&
        snapshot.call_accounting.call_attempts == 1 &&
        snapshot.call_accounting.call_completions == 0,
        "throwing-oracle wrapper diagnostics drifted");
    requireScalarBitsEqual(
        telemetry.last_attempted_step, Scalar(1),
        "throwing-oracle last attempted step drifted");
    const TestGenerator::Diagnostics& diagnostics =
        snapshot.primary_diagnostics;
    require(
        diagnostics.proposal_origin ==
            LBFGSpp::LBFGSPP_MT_PROPOSAL_INITIAL &&
        diagnostics.raw_proposed_step_present &&
        diagnostics.raw_proposed_step_class ==
            LBFGSpp::LBFGSPP_MT_RAW_FINITE &&
        diagnostics.safeguarded_proposed_step_present &&
        diagnostics.midpoint_trigger ==
            LBFGSpp::LBFGSPP_MT_MIDPOINT_NONE &&
        diagnostics.guard_failure ==
            LBFGSpp::LBFGSPP_MT_GUARD_NONE &&
        diagnostics.termination_reason ==
            LBFGSpp::LBFGSPP_MT_TERMINATION_OBJECTIVE_PROVIDER_EXCEPTION &&
        diagnostics.candidate_attempts == 1 &&
        diagnostics.candidate_completions == 0 &&
        diagnostics.last_attempted_step_present &&
        !diagnostics.last_completed_step_present &&
        !diagnostics.last_evaluated_step_present &&
        !diagnostics.final_bracketed,
        "throwing-oracle generator diagnostics drifted");
    requireScalarBitsEqual(
        diagnostics.raw_proposed_step, Scalar(1),
        "throwing-oracle raw proposal drifted");
    requireScalarBitsEqual(
        diagnostics.safeguarded_proposed_step, Scalar(1),
        "throwing-oracle safeguarded proposal drifted");
}

void testRawComponentsAndPerComponentGradients()
{
    ManufacturedProvider provider;
    Vector points[2];
    points[0] = provider.center();
    points[1] = provider.center();
    points[1] << 0.23, -0.28, 0.36, -0.45;
    Vector direction(4);
    direction << 0.3, -0.4, 0.2, 0.5;
    direction.normalize();

    for (int point_index = 0; point_index < 2; ++point_index)
    {
        const Vector point = points[point_index];
        const mfa::NSConstrainedRawEvaluation<Scalar> raw =
            provider.evaluate(point);
        require(raw.gauge.size() == 1001, "raw gauge count mismatch");
        for (int metric = 0; metric < mfa::NS_V2_METRIC_COUNT; ++metric)
        {
            require(raw.metrics[metric] > 0.0,
                    "manufactured raw MSE is not positive");
            std::array<Scalar, mfa::NS_V2_METRIC_COUNT> weights;
            weights.fill(0.0);
            weights[metric] = 1.0;
            provider.evaluate(point);
            const Vector gradient =
                provider.weightedGradient(weights, Vector::Zero(1001));
            const Scalar h = 1.0e-6;
            const Scalar plus =
                provider.evaluate(point + h * direction).metrics[metric];
            const Scalar minus =
                provider.evaluate(point - h * direction).metrics[metric];
            const Scalar centered = (plus - minus) / (2.0 * h);
            requireNear(
                centered, gradient.dot(direction), 2.0e-8,
                std::string("centered component gradient failed: ") +
                mfa::nsConstrainedMetricName(metric));

            provider.evaluate(point);
            const Vector exact_gradient =
                provider.weightedGradient(weights, Vector::Zero(1001));
            const Scalar f0 = raw.metrics[metric];
            const Scalar h1 = 1.0e-3;
            const Scalar h2 = 5.0e-4;
            const Scalar r1 = std::abs(
                provider.evaluate(point + h1 * direction).metrics[metric] -
                f0 - h1 * exact_gradient.dot(direction));
            const Scalar r2 = std::abs(
                provider.evaluate(point + h2 * direction).metrics[metric] -
                f0 - h2 * exact_gradient.dot(direction));
            require(r1 > 0.0 && r2 > 0.0,
                    "Taylor remainder unexpectedly vanished");
            const Scalar slope = std::log(r1 / r2) / std::log(2.0);
            requireNear(
                slope, 2.0, 2.0e-6,
                std::string("second-order Taylor slope failed: ") +
                mfa::nsConstrainedMetricName(metric));
        }
    }
}

void testGaugeAdjoint()
{
    ManufacturedProvider provider;
    Vector point = provider.center();
    point << 0.22, -0.31, 0.42, -0.47;
    const mfa::NSConstrainedRawEvaluation<Scalar> raw =
        provider.evaluate(point);
    Vector direction(4);
    direction << -0.4, 0.2, 0.1, 0.3;
    Vector gauge_weights(1001);
    for (int i = 0; i < gauge_weights.size(); ++i)
        gauge_weights(i) = 0.001 * ((i % 9) - 4);
    std::array<Scalar, mfa::NS_V2_METRIC_COUNT> zero_weights;
    zero_weights.fill(0.0);
    const Vector adjoint =
        provider.weightedGradient(zero_weights, gauge_weights);
    const Scalar step = 1.0e-7;
    const Vector gauge_plus =
        provider.evaluate(point + step * direction).gauge;
    const Vector gauge_minus =
        provider.evaluate(point - step * direction).gauge;
    const Scalar lhs =
        gauge_weights.dot((gauge_plus - gauge_minus) / (2.0 * step));
    requireNear(lhs, adjoint.dot(direction), 1.0e-10,
                "gauge Jacobian/adjoint identity failed");
    require(raw.gauge.allFinite(), "gauge contains nonfinite values");
    const mfa::NSConstrainedRawEvaluation<Scalar> zero =
        provider.evaluate(provider.center());
    require(zero.gauge.norm() == 0.0,
            "zero-pressure manufactured gauge is not exact zero");
}

void testBaselineNormalizationAndKnownStart()
{
    ManufacturedProvider provider;
    const mfa::NSConstrainedRawEvaluation<Scalar> raw =
        provider.evaluate(provider.center());
    const mfa::NSConstrainedBaseline<Scalar> baseline =
        mfa::NSConstrainedBaseline<Scalar>::capture(raw);
    const mfa::NSConstrainedConstraints<Scalar> constraints =
        mfa::nsConstrainedEvaluateConstraints(raw, baseline, 100.0);
    const Scalar expected_fidelity = 1.0 / 1.21 - 1.0;
    for (int i = 0; i < mfa::NS_V2_FIDELITY_COUNT; ++i)
        requireNear(
            constraints.inequalities[i], expected_fidelity, 1.0e-15,
            "Stage-A fidelity normalization mismatch");
    for (int i = 0; i < 3; ++i)
        requireNear(
            constraints.inequalities[mfa::NS_V2_FIDELITY_COUNT + i],
            0.0, 5.0e-14,
            "Stage-A strong epigraph is not at the hinge");
    for (int i = 3; i < mfa::NS_V2_MECHANICS_COUNT; ++i)
        requireNear(
            constraints.inequalities[mfa::NS_V2_FIDELITY_COUNT + i],
            1.5625 - 100.0, 1.0e-14,
            "Stage-A integral epigraph normalization mismatch");
    require(mfa::nsConstrainedPrimalInfeasibility(constraints) == 0.0,
            "known Stage-A start is not feasible");

    mfa::NSConstrainedRawEvaluation<Scalar> bad = raw;
    bad.metrics[mfa::NS_V2_DATA_U] = 0.0;
    requireThrows(
        [&bad]() {
            mfa::NSConstrainedBaseline<Scalar>::capture(bad);
        },
        "zero Stage-A denominator was accepted");
    bad = raw;
    bad.metrics[mfa::NS_V2_LOCAL_X] =
        std::numeric_limits<Scalar>::infinity();
    requireThrows(
        [&bad]() {
            mfa::NSConstrainedBaseline<Scalar>::capture(bad);
        },
        "nonfinite Stage-A denominator was accepted");
}

void checkALDirectionalGradient(
    ManufacturedProvider& provider,
    const mfa::NSConstrainedBaseline<Scalar>& baseline,
    const mfa::NSConstrainedOuterState<Scalar>& state,
    const Vector& z,
    const Vector& direction,
    Scalar tolerance,
    const std::string& label)
{
    mfa::NSConstrainedAugmentedLagrangian<ManufacturedProvider> objective(
        provider, baseline, state);
    Vector gradient;
    const Scalar value = objective(z, gradient);
    const Scalar h = 1.0e-6;
    Vector throwaway;
    const Scalar plus = objective(z + h * direction, throwaway);
    const Scalar minus = objective(z - h * direction, throwaway);
    const Scalar centered = (plus - minus) / (2.0 * h);
    requireNear(centered, gradient.dot(direction), tolerance, label);
    require(std::isfinite(value), label + " objective is nonfinite");
}

void testFullAugmentedLagrangianGradient()
{
    ManufacturedProvider provider;
    const mfa::NSConstrainedRawEvaluation<Scalar> raw =
        provider.evaluate(provider.center());
    const mfa::NSConstrainedBaseline<Scalar> baseline =
        mfa::NSConstrainedBaseline<Scalar>::capture(raw);
    mfa::NSConstrainedOuterState<Scalar> inactive(1001);
    Vector z(5);
    z.head(4) = provider.center();
    z(4) = 100.0;
    Vector direction(5);
    direction << 0.2, -0.3, 0.4, 0.1, -0.5;
    direction.normalize();
    checkALDirectionalGradient(
        provider, baseline, inactive, z, direction, 5.0e-7,
        "inactive/hinge AL gradient failed");

    mfa::NSConstrainedOuterState<Scalar> active(1001);
    active.rho = 7.0;
    for (int i = 0; i < mfa::NS_V2_INEQUALITY_COUNT; ++i)
        active.lambda[i] = 0.05 * (i + 1);
    for (int i = 0; i < active.mu.size(); ++i)
        active.mu(i) = 0.001 * ((i % 5) - 2);
    z.head(4) << 0.24, -0.27, 0.35, -0.44;
    z(4) = 0.7;
    checkALDirectionalGradient(
        provider, baseline, active, z, direction, 2.0e-5,
        "active AL gradient failed");
}

void testSyntheticKKTAndStateMachine()
{
    ManufacturedProvider provider;
    const mfa::NSConstrainedRawEvaluation<Scalar> raw =
        provider.evaluate(provider.center());
    const mfa::NSConstrainedBaseline<Scalar> baseline =
        mfa::NSConstrainedBaseline<Scalar>::capture(raw);
    mfa::NSConstrainedOuterState<Scalar> state(1001);
    state.lambda[mfa::NS_V2_FIDELITY_COUNT] = 1.0;
    const Vector kkt_gradient =
        mfa::nsConstrainedKKTGradient(
            provider, baseline, state, raw, 100.0);
    require(kkt_gradient.norm() < 1.0e-13,
            "known synthetic KKT stationarity is not zero");
    Vector z(5);
    z.head(4) = provider.center();
    z(4) = 100.0;
    Vector lower = Vector::Constant(5, -100.0);
    Vector upper = Vector::Constant(5, 100.0);
    lower(4) = 0.0;
    require(
        mfa::nsConstrainedProjectedInfinityNorm(
            z, kkt_gradient, lower, upper) < 1.0e-13,
        "known synthetic projected KKT residual failed");
    const mfa::NSConstrainedConstraints<Scalar> constraints =
        mfa::nsConstrainedEvaluateConstraints(raw, baseline, 100.0);
    require(
        mfa::nsConstrainedComplementarity(
            state.lambda, constraints) < 1.0e-13,
        "known synthetic complementarity failed");

    mfa::NSConstrainedOuterState<Scalar> machine(1001);
    mfa::NSConstrainedConstraints<Scalar> first;
    first.gauge = Vector::Zero(1001);
    first.inequalities.fill(-1.0);
    first.inequalities[0] = 0.1;
    mfa::nsConstrainedUpdateMultipliersAndPenalty(first, machine, 0.1);
    require(machine.rho == 1.0, "first penalty should remain one");
    requireNear(machine.lambda[0], 0.1, 0.0,
                "first multiplier update failed");
    mfa::NSConstrainedConstraints<Scalar> second = first;
    second.inequalities[0] = 0.03;
    mfa::nsConstrainedUpdateMultipliersAndPenalty(second, machine, 0.03);
    require(machine.rho == 10.0,
            "insufficient-progress penalty growth failed");
    requireNear(machine.lambda[0], 0.13, 1.0e-15,
                "second multiplier update failed");
    second.inequalities[0] = 0.007;
    mfa::nsConstrainedUpdateMultipliersAndPenalty(second, machine, 0.007);
    require(machine.rho == 10.0,
            "sufficient-progress penalty should remain fixed");
}

void testStateRestartAndSerializationHandshake()
{
    require(
        mfa::nsConstrainedSHA256("abc") ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 known vector failed");
    ManufacturedProvider provider;
    const mfa::NSConstrainedRawEvaluation<Scalar> raw =
        provider.evaluate(provider.center());
    const mfa::NSConstrainedBaseline<Scalar> baseline =
        mfa::NSConstrainedBaseline<Scalar>::capture(raw);
    mfa::NSConstrainedOuterState<Scalar> state(1001);
    state.rho = 100.0;
    state.previous_theta = 0.0125;
    state.completed_outer_iterations = 3;
    state.total_inner_iterations = 417;
    state.consecutive_kkt_passes = 1;
    state.elapsed_seconds = 9.25;
    for (int i = 0; i < mfa::NS_V2_INEQUALITY_COUNT; ++i)
        state.lambda[i] = 0.01 * i;
    for (int i = 0; i < state.mu.size(); ++i)
        state.mu(i) = 1.0e-4 * (i % 11);
    Vector z(5);
    z.head(4) = provider.center();
    z(4) = 2.5;
    const std::string state_path = temporaryPath("_state.txt");
    const std::string final_path = temporaryPath("_final.txt");
    const mfa::NSConstrainedArtifactAnchors anchors =
        testArtifactAnchors();
    mfa::nsConstrainedWriteState(
        state_path, z, baseline, state, anchors);

    Vector loaded;
    mfa::NSConstrainedBaseline<Scalar> loaded_baseline;
    mfa::NSConstrainedOuterState<Scalar> loaded_state;
    mfa::nsConstrainedLoadState(
        state_path, loaded, loaded_baseline, loaded_state, 4, 1001,
        anchors);
    require(loaded == z, "state reload changed z");
    require(loaded_state.rho == state.rho,
            "state reload changed rho");
    require(loaded_state.mu == state.mu,
            "state reload changed gauge multipliers");
    require(loaded_state.lambda == state.lambda,
            "state reload changed inequality multipliers");
    require(loaded_baseline.metrics == baseline.metrics,
            "state reload changed Stage-A baseline");
    mfa::NSConstrainedArtifactAnchors wrong_anchors = anchors;
    wrong_anchors.implementation_sha256 = std::string(64, 'f');
    requireThrows(
        [&state_path, &wrong_anchors]() {
            Vector rejected_z;
            mfa::NSConstrainedBaseline<Scalar> rejected_baseline;
            mfa::NSConstrainedOuterState<Scalar> rejected_state;
            mfa::nsConstrainedLoadState(
                state_path, rejected_z, rejected_baseline, rejected_state,
                4, 1001, wrong_anchors);
        },
        "state reload accepted implementation-hash drift");

    const std::string decoded_sha =
        mfa::nsConstrainedSHA256("deterministic decoded points");
    mfa::nsConstrainedWriteFinalCheckpoint(
        final_path, z.head(4), z(4), true, decoded_sha, anchors);
    std::ifstream final_stream(final_path.c_str());
    std::string header;
    std::getline(final_stream, header);
    require(
        header.find("PIMFA_NS_CONSTRAINED_MINIMAX_AL_V2") !=
            std::string::npos &&
        header.find(mfa::nsConstrainedCoefficientSHA256(z.head(4))) !=
            std::string::npos &&
        header.find(decoded_sha) != std::string::npos &&
        header.find(anchors.stage_a_checkpoint_sha256) !=
            std::string::npos &&
        header.find(anchors.optimization_protocol_sha256) !=
            std::string::npos &&
        header.find(anchors.optimization_protocol_parent_sha256) !=
            std::string::npos &&
        header.find(anchors.optimization_protocol_ancestor_2_sha256) !=
            std::string::npos &&
        header.find(anchors.optimization_protocol_ancestor_3_sha256) !=
            std::string::npos &&
        header.find(anchors.inherited_optimization_protocol_sha256) !=
            std::string::npos &&
        header.find(anchors.post_selection_protocol_v5_sha256) !=
            std::string::npos &&
        header.find(anchors.post_selection_protocol_v4_parent_sha256) !=
            std::string::npos,
        "final coefficient/model handshake header is incomplete");
    std::remove(state_path.c_str());
    std::remove(final_path.c_str());
}

void testDeterministicOuterDriverWithRepresentableInteriorRetry()
{
    // The zero center and tiny nonzero residual shifts give the analytic
    // search direction a coefficient component.  That component remains
    // representably distinct when the retry cap is one ULP below the true
    // feasible maximum.  The separate rounded-endpoint regression above
    // covers the required one-primary/zero-retry-provider fail-closed path.
    ManufacturedProvider provider(true);
    const Vector initial = provider.center();
    const std::string state_path = temporaryPath("_driver_state.txt");
    const std::string record_path = temporaryPath("_driver_record.tsv");
    std::remove(state_path.c_str());
    std::remove(record_path.c_str());
    mfa::NSConstrainedRunOptions<Scalar> options;
    options.state_path = state_path;
    options.outer_record_path = record_path;
    options.anchors = testArtifactAnchors();
    const mfa::NSConstrainedRunResult<Scalar> result =
        mfa::runNSConstrainedMinimaxAL(
            provider, initial, options);
    require(result.converged &&
            result.state.consecutive_kkt_passes == 2,
            "synthetic outer driver did not enforce two consecutive passes");
    require(result.state.completed_outer_iterations <= 8,
            "outer driver exceeded frozen outer cap");
    require(result.state.total_inner_iterations <= 1600,
            "outer driver exceeded frozen inner cap");
    require(result.eta >= 0.0 && result.eta <= 100.0,
            "outer driver violated eta bounds");
    require(result.coefficients.minCoeff() >= -100.0 &&
            result.coefficients.maxCoeff() <= 100.0,
            "outer driver violated coefficient bounds");
    require(pathExists(state_path),
            "outer driver did not write state");
    require(pathExists(record_path),
            "outer driver did not write outer record");
    const std::string record_before_resume = readText(record_path);
    options.resume = true;
    const mfa::NSConstrainedRunResult<Scalar> resumed =
        mfa::runNSConstrainedMinimaxAL(
            provider, initial, options);
    require(
        resumed.converged &&
        resumed.state.completed_outer_iterations ==
            result.state.completed_outer_iterations &&
        resumed.state.total_inner_iterations ==
            result.state.total_inner_iterations &&
        resumed.coefficients == result.coefficients &&
        readText(record_path) == record_before_resume,
        "converged resume changed deterministic outer state/record");
    std::remove(state_path.c_str());
    std::remove(record_path.c_str());
}

void testCanonicalPressureDataWeightEnvironment()
{
    const std::vector<std::string> canonical(
        1, "MFA_LAMBDA_DATA_P=0");
    mfa::nsConstrainedValidateRawEnvironmentEntries(canonical);

    requireThrowsContaining(
        []() {
            mfa::nsConstrainedValidateRawEnvironmentEntries(
                std::vector<std::string>());
        },
        "requires exactly one MFA_LAMBDA_DATA_P=0",
        "constrained environment accepted a missing pressure-data weight");

    static const char* noncanonical_values[] = {
        "", "0.0", "+0", "-0", "00", " 0", "0 ", "1"
    };
    for (std::size_t i = 0;
         i < sizeof(noncanonical_values) / sizeof(noncanonical_values[0]);
         ++i)
    {
        const std::string entry =
            std::string("MFA_LAMBDA_DATA_P=") + noncanonical_values[i];
        requireThrowsContaining(
            [entry]() {
                mfa::nsConstrainedValidateRawEnvironmentEntries(
                    std::vector<std::string>(1, entry));
            },
            "exactly the literal 0",
            "constrained environment accepted noncanonical entry " + entry);
    }

    std::vector<std::string> duplicate_valid;
    duplicate_valid.push_back("MFA_LAMBDA_DATA_P=0");
    duplicate_valid.push_back("MFA_LAMBDA_DATA_P=0");
    requireThrowsContaining(
        [duplicate_valid]() {
            mfa::nsConstrainedValidateRawEnvironmentEntries(
                duplicate_valid);
        },
        "requires exactly one MFA_LAMBDA_DATA_P=0",
        "constrained environment accepted duplicate pressure-data weights");

    std::vector<std::string> duplicate_bad;
    duplicate_bad.push_back("MFA_LAMBDA_DATA_P=0");
    duplicate_bad.push_back("MFA_LAMBDA_DATA_P=0.0");
    requireThrowsContaining(
        [duplicate_bad]() {
            mfa::nsConstrainedValidateRawEnvironmentEntries(duplicate_bad);
        },
        "exactly the literal 0",
        "constrained environment accepted a bad duplicate pressure weight");

    static const char* forbidden_weight_names[] = {
        "MFA_LAMBDA_BC",
        "MFA_LAMBDA_BC_P",
        "MFA_LAMBDA_BC_U",
        "MFA_LAMBDA_BC_V",
        "MFA_LAMBDA_DATA",
        "MFA_LAMBDA_DATA_U",
        "MFA_LAMBDA_DATA_V",
        "MFA_LAMBDA_DIV",
        "MFA_LAMBDA_GAUGE",
        "MFA_LAMBDA_IC",
        "MFA_LAMBDA_IC_P",
        "MFA_LAMBDA_IC_U",
        "MFA_LAMBDA_IC_V",
        "MFA_LAMBDA_INT_MASS",
        "MFA_LAMBDA_INT_XMOM",
        "MFA_LAMBDA_INT_YMOM",
        "MFA_LAMBDA_MOM",
        "MFA_LAMBDA_PDE",
        "MFA_LAMBDA_XMOM",
        "MFA_LAMBDA_YMOM",
        "MFA_LAMBDA_FUTURE"
    };
    for (std::size_t i = 0;
         i < sizeof(forbidden_weight_names) /
             sizeof(forbidden_weight_names[0]);
         ++i)
    {
        std::vector<std::string> entries(canonical);
        entries.push_back(
            std::string(forbidden_weight_names[i]) + "=0");
        requireThrowsContaining(
            [entries]() {
                mfa::nsConstrainedValidateRawEnvironmentEntries(entries);
            },
            std::string("forbids legacy scalar weights: ") +
                forbidden_weight_names[i],
            std::string("constrained environment accepted ") +
                forbidden_weight_names[i]);
    }
}

void setExactStartupEnvironment()
{
    static const char* names[] = {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "OMPI_MCA_btl",
        "MFA_INIT_PRESSURE_MODE",
        "MFA_LAMBDA_DATA_P",
        "MFA_REQUIRE_ZERO_PRESSURE_BC",
        "MFA_NS_CONSTRAINED_V2_RESUME",
        "MFA_NS_CONSTRAINED_V2_CANDIDATE_ID",
        "MFA_NS_CONSTRAINED_V2_PROTOCOL_SHA256",
        "MFA_NS_CONSTRAINED_V2_PROTOCOL_PARENT_SHA256",
        "MFA_NS_CONSTRAINED_V2_PROTOCOL_ANCESTOR_2_SHA256",
        "MFA_NS_CONSTRAINED_V2_PROTOCOL_ANCESTOR_3_SHA256",
        "MFA_NS_CONSTRAINED_V2_INHERITED_PROTOCOL_SHA256",
        "MFA_NS_CONSTRAINED_V3_INFRASTRUCTURE_PROTOCOL_SHA256",
        "MFA_NS_CONSTRAINED_V3_LINEAGE_ADDENDUM_SHA256"
    };
    static const char* values[] = {
        "1", "1", "1", "1", "1", "1", "self", "zero", "0", "1", "0",
        "ns_fit_guarded_minimax_al_v2",
        "14648ed7c01d8e34b8869232bd20268e1d9f4c4a788e8091f3a02aac41692bf0",
        "ea33c7e22d89a589c8bc3b90051b6e003b53d420cb9ea5c600aa8a7868e54ad0",
        "4eb6fff402c270f659b8230859edc7afde3539306f1a007fe65569f6403bd431",
        "07102dc337849f602b002da3160a61d791a6dd68659a48416926b2c112b2879e",
        "eb26fbf050bf543043aa3be7e64ff707d2cdb784d2dd17b0d7c856ceafe62072",
        "17c3a2cb4f14bac71576e5d95f9f3f6cc0628b03e8ba3abbc0baa63b14af8ab7",
        "67d5db7973de1974d6dc120f2630a9773dda18facdcd3924164925e3e63e1b16"
    };
    require(
        sizeof(names) / sizeof(names[0]) ==
            sizeof(values) / sizeof(values[0]),
        "startup test environment table drifted");
    for (std::size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i)
        setenv(names[i], values[i], 1);
}

void testFullStartupContractEnvironment()
{
    static const char* fixed_names[] = {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "OMPI_MCA_btl",
        "MFA_INIT_PRESSURE_MODE",
        "MFA_LAMBDA_DATA_P",
        "MFA_REQUIRE_ZERO_PRESSURE_BC",
        "MFA_NS_CONSTRAINED_V2_RESUME",
        "MFA_NS_CONSTRAINED_V2_CANDIDATE_ID",
        "MFA_NS_CONSTRAINED_V2_PROTOCOL_SHA256",
        "MFA_NS_CONSTRAINED_V2_PROTOCOL_PARENT_SHA256",
        "MFA_NS_CONSTRAINED_V2_PROTOCOL_ANCESTOR_2_SHA256",
        "MFA_NS_CONSTRAINED_V2_PROTOCOL_ANCESTOR_3_SHA256",
        "MFA_NS_CONSTRAINED_V2_INHERITED_PROTOCOL_SHA256",
        "MFA_NS_CONSTRAINED_V3_INFRASTRUCTURE_PROTOCOL_SHA256",
        "MFA_NS_CONSTRAINED_V3_LINEAGE_ADDENDUM_SHA256"
    };
    static const char* legacy_controls[] = {
        "MFA_INT_DIAGNOSTICS",
        "MFA_PREFILTERED_PDE_BENCHMARK",
        "MFA_PREFILTERED_PDE_EVALUATIONS",
        "MFA_MAX_ITERATIONS",
        "MFA_OPTIMIZER_DELTA",
        "MFA_OPTIMIZER_EPSILON",
        "MFA_OPTIMIZER_EPSILON_REL",
        "MFA_OPTIMIZER_PAST",
        "MFA_CHECK_GRADIENT",
        "MFA_GRADIENT_CHECK_ONLY",
        "MFA_INITIALIZER_ONLY",
        "MFA_INIT_CHECKPOINT_IN",
        "MFA_INIT_CHECKPOINT_OUT",
        "MFA_FINAL_COEFFICIENT_CHECKPOINT_OUT",
        "MFA_FINAL_STATE_HANDSHAKE"
    };
    static const char* prohibited_roles[] = {
        "MFA_DATA_VALID_INDEX_FILE",
        "MFA_SEALED_DATA",
        "MFA_REFERENCE_DATA",
        "MFA_REFINED_DATA",
        "MFA_HIGH_RES_DATA",
        "MFA_QOI_DATA",
        "MFA_FINAL_TEST_DATA"
    };
    std::vector<std::string> restored_names;
    for (std::size_t i = 0;
         i < sizeof(fixed_names) / sizeof(fixed_names[0]); ++i)
        restored_names.push_back(fixed_names[i]);
    for (std::size_t i = 0;
         i < sizeof(legacy_controls) / sizeof(legacy_controls[0]); ++i)
        restored_names.push_back(legacy_controls[i]);
    for (std::size_t i = 0;
         i < sizeof(prohibited_roles) / sizeof(prohibited_roles[0]); ++i)
        restored_names.push_back(prohibited_roles[i]);
    ScopedEnvironmentRestore restore(restored_names);
    for (std::size_t i = 0;
         i < sizeof(legacy_controls) / sizeof(legacy_controls[0]); ++i)
        unsetenv(legacy_controls[i]);
    for (std::size_t i = 0;
         i < sizeof(prohibited_roles) / sizeof(prohibited_roles[0]); ++i)
        unsetenv(prohibited_roles[i]);

    setExactStartupEnvironment();
    mfa::nsConstrainedStartupContractPreflight();
    require(
        std::string(
            mfa::nsConstrainedFrozenInfrastructureProtocolV3SHA256()) ==
            "17c3a2cb4f14bac71576e5d95f9f3f6cc0628b03e8ba3abbc0baa63b14af8ab7",
        "V3 infrastructure protocol accessor drifted");
    require(
        std::string(
            mfa::nsConstrainedFrozenInfrastructureLineageAddendumSHA256()) ==
            "67d5db7973de1974d6dc120f2630a9773dda18facdcd3924164925e3e63e1b16",
        "V3 lineage-addendum accessor drifted");

    for (std::size_t i = 0;
         i < sizeof(fixed_names) / sizeof(fixed_names[0]); ++i)
    {
        setExactStartupEnvironment();
        unsetenv(fixed_names[i]);
        requireThrows(
            []() {
                mfa::nsConstrainedStartupContractPreflight();
            },
            std::string("startup contract accepted missing ") +
                fixed_names[i]);
    }
    for (std::size_t i = 0;
         i < sizeof(fixed_names) / sizeof(fixed_names[0]); ++i)
    {
        setExactStartupEnvironment();
        setenv(fixed_names[i], "WRONG", 1);
        requireThrows(
            []() {
                mfa::nsConstrainedStartupContractPreflight();
            },
            std::string("startup contract accepted wrong ") +
                fixed_names[i]);
    }
    for (std::size_t i = 0;
         i < sizeof(legacy_controls) / sizeof(legacy_controls[0]); ++i)
    {
        setExactStartupEnvironment();
        setenv(legacy_controls[i], "1", 1);
        requireThrowsContaining(
            []() {
                mfa::nsConstrainedStartupContractPreflight();
            },
            i == 0 ? "forbids V1 integral diagnostics" :
                legacy_controls[i],
            std::string("startup contract accepted legacy control ") +
                legacy_controls[i]);
        unsetenv(legacy_controls[i]);
    }
    for (std::size_t i = 0;
         i < sizeof(prohibited_roles) / sizeof(prohibited_roles[0]); ++i)
    {
        setExactStartupEnvironment();
        setenv(prohibited_roles[i], "/prohibited", 1);
        requireThrowsContaining(
            []() {
                mfa::nsConstrainedStartupContractPreflight();
            },
            prohibited_roles[i],
            std::string("startup contract accepted prohibited role ") +
                prohibited_roles[i]);
        unsetenv(prohibited_roles[i]);
    }
}

void testInformationBoundaryAndStageACheckpoint()
{
    require(
        mfa::nsConstrainedContainsForbiddenRoleWord(
            "MFA_DATA_VALID_INDEX_FILE"),
        "validation role name was not recognized as forbidden");
    require(
        mfa::nsConstrainedContainsForbiddenRoleWord(
            "PIMFA_REFINED_TRAJECTORY"),
        "refined role name was not recognized as forbidden");
    require(
        !mfa::nsConstrainedContainsForbiddenRoleWord(
            "MFA_DATA_ACTIVE_INDEX_FILE"),
        "fit role name was incorrectly forbidden");

    setenv("MFA_LAMBDA_DATA_P", "0", 1);
    mfa::nsConstrainedRejectForbiddenEnvironment();
    setenv("MFA_DATA_VALID_INDEX_FILE", "/prohibited", 1);
    requireThrowsContaining(
        []() {
            mfa::nsConstrainedRejectForbiddenEnvironment();
        },
        "contains a prohibited role: MFA_DATA_VALID_INDEX_FILE",
        "optimizer environment accepted a validation role");
    unsetenv("MFA_DATA_VALID_INDEX_FILE");
    setenv("MFA_PREFILTERED_PDE_BENCHMARK", "1", 1);
    requireThrowsContaining(
        []() {
            mfa::nsConstrainedRejectForbiddenEnvironment();
        },
        "forbids legacy control: MFA_PREFILTERED_PDE_BENCHMARK",
        "optimizer environment accepted a legacy prefiltered path");
    unsetenv("MFA_PREFILTERED_PDE_BENCHMARK");
    unsetenv("MFA_LAMBDA_DATA_P");

    require(
        std::string(mfa::nsConstrainedFrozenStageASHA256()) ==
            "ff7456a87e6dc389cb73f22f0778d5a926510d3adb5f940e9eea0534bc4421ba",
        "frozen Stage-A hash drifted");
    require(
        std::string(
            mfa::nsConstrainedFrozenOptimizationProtocolSHA256()) ==
            "14648ed7c01d8e34b8869232bd20268e1d9f4c4a788e8091f3a02aac41692bf0",
        "frozen V17 optimization protocol hash drifted");
    require(
        std::string(
            mfa::nsConstrainedFrozenOptimizationProtocolParentSHA256()) ==
            "ea33c7e22d89a589c8bc3b90051b6e003b53d420cb9ea5c600aa8a7868e54ad0",
        "frozen V17 protocol-parent hash drifted");
    require(
        std::string(
            mfa::nsConstrainedFrozenOptimizationProtocolAncestor2SHA256()) ==
            "4eb6fff402c270f659b8230859edc7afde3539306f1a007fe65569f6403bd431",
        "frozen V17 protocol ancestor-2 hash drifted");
    require(
        std::string(
            mfa::nsConstrainedFrozenOptimizationProtocolAncestor3SHA256()) ==
            "07102dc337849f602b002da3160a61d791a6dd68659a48416926b2c112b2879e",
        "frozen V17 protocol ancestor-3 hash drifted");
    require(
        std::string(
            mfa::nsConstrainedFrozenInheritedOptimizationProtocolSHA256()) ==
            "eb26fbf050bf543043aa3be7e64ff707d2cdb784d2dd17b0d7c856ceafe62072",
        "frozen inherited optimization protocol hash drifted");
    require(
        std::string(
            mfa::nsConstrainedFrozenHistoricalImplementationSHA256()) ==
            "69ecca0dc4fb736525fa1738ce0b8fc8ef6d704c7cb8dc3ccec111cc60da4a40",
        "frozen historical Stage-A implementation hash drifted");
    {
        const mfa::NSConstrainedArtifactAnchors current =
            testArtifactAnchors();
        const mfa::NSConstrainedArtifactAnchors historical =
            mfa::nsConstrainedHistoricalStageABaselineAnchors(current);
        require(
            historical.optimization_protocol_sha256 ==
                mfa::nsConstrainedFrozenInheritedOptimizationProtocolSHA256() &&
            historical.optimization_protocol_parent_sha256 ==
                std::string(64, '0') &&
            historical.optimization_protocol_ancestor_2_sha256 ==
                std::string(64, '0') &&
            historical.optimization_protocol_ancestor_3_sha256 ==
                std::string(64, '0') &&
            historical.inherited_optimization_protocol_sha256 ==
                std::string(64, '0') &&
            historical.implementation_sha256 ==
                mfa::nsConstrainedFrozenHistoricalImplementationSHA256() &&
            current.optimization_protocol_sha256 !=
                historical.optimization_protocol_sha256 &&
            current.implementation_sha256 !=
                historical.implementation_sha256,
            "current and historical Stage-A anchor contexts were conflated");
    }
    require(
        std::string(
            mfa::nsConstrainedFrozenPostSelectionV5SHA256()) ==
            "c4e22c560a029565af39c636e1dbfb33edfd290946f7fb92dd8f6997033f1dc2",
        "frozen post-selection V5 protocol hash drifted");
    require(
        std::string(
            mfa::nsConstrainedFrozenPostSelectionV4ParentSHA256()) ==
            "a82761a7373fb563a757269f41a3121d6a26f86d60494fc15daf6bb7d7e15c03",
        "frozen post-selection V4 parent hash drifted");
    unsetenv("MFA_NS_CONSTRAINED_V2_RESUME");
    requireThrows(
        []() {
            mfa::nsConstrainedRequireProductionNoResume();
        },
        "production executable accepted a missing RESUME control");
    setenv("MFA_NS_CONSTRAINED_V2_RESUME", "1", 1);
    requireThrows(
        []() {
            mfa::nsConstrainedRequireProductionNoResume();
        },
        "production executable accepted in-place resume");
    setenv("MFA_NS_CONSTRAINED_V2_RESUME", "false", 1);
    requireThrows(
        []() {
            mfa::nsConstrainedRequireProductionNoResume();
        },
        "production executable accepted a noncanonical RESUME control");
    setenv("MFA_NS_CONSTRAINED_V2_RESUME", "0", 1);
    mfa::nsConstrainedRequireProductionNoResume();
    unsetenv("MFA_NS_CONSTRAINED_V2_RESUME");
    const mfa::BaselineCheckpointMetadata frozen =
        mfa::nsConstrainedFrozenStageAMetadata();
    require(
        frozen.case_name == "ns_lid_cavity" &&
        frozen.kind == mfa::BaselineKind::data_only &&
        frozen.lattice == "20,20,50" &&
        frozen.degree == "3,3,3" &&
        frozen.pressure_regime == "zero" &&
        frozen.field_order == "u,v,p" &&
        frozen.coefficient_count_per_field == 20000 &&
        frozen.coefficient_count_total == 60000,
        "frozen Stage-A metadata drifted");
    if (const char* frozen_path =
            std::getenv("PIMFA_TEST_FROZEN_STAGE_A_CHECKPOINT"))
    {
        require(
            mfa::nsConstrainedFileSHA256(frozen_path) ==
                mfa::nsConstrainedFrozenStageASHA256(),
            "production Stage-A file hash drifted");
        const Vector production_stage_a =
            mfa::nsConstrainedReadStageACheckpoint<Scalar>(
                frozen_path, 60000);
        require(
            production_stage_a.size() == 60000 &&
            production_stage_a.tail(20000).isZero(0.0),
            "production Stage-A payload failed exact ingestion");
        if (const char* model_dump =
                std::getenv("PIMFA_TEST_STAGE_A_V3_MODEL_DUMP"))
        {
            require(
                readText(model_dump) ==
                    mfa::nsConstrainedCoefficientPayload(
                        production_stage_a),
                "Stage-A V3 model dump is not the exact canonical "
                "checkpoint coefficient payload");
            require(
                mfa::nsConstrainedFileSHA256(model_dump) ==
                    mfa::nsConstrainedCoefficientSHA256(
                        production_stage_a),
                "Stage-A V3 model/checkpoint canonical coefficient "
                "SHA-256 mismatch");
        }
    }

    const std::string checkpoint = temporaryPath("_stage_a.ckpt");
    const mfa::BaselineCheckpointMetadata metadata = smallStageAMetadata();
    writeSmallStageA(checkpoint, metadata);
    const Vector coefficients =
        mfa::nsConstrainedReadStageACheckpointContract<Scalar>(
            checkpoint, metadata,
            mfa::nsConstrainedFileSHA256(checkpoint));
    require(coefficients.size() == 6 &&
            coefficients(4) == 0.0 &&
            !std::signbit(coefficients(4)) &&
            coefficients(5) == 0.0 &&
            !std::signbit(coefficients(5)),
            "valid Stage-A checkpoint did not round-trip");

    requireThrows(
        [&checkpoint, &metadata]() {
            mfa::nsConstrainedReadStageACheckpointContract<Scalar>(
                checkpoint, metadata, std::string(64, 'f'));
        },
        "Stage-A loader accepted a file-hash mismatch");

    std::string malformed = readText(checkpoint);
    malformed.replace(
        0, std::string("# PIMFA_BASELINE_V2").size(),
        "# PIMFA_BASELINE_V1");
    writeText(checkpoint, malformed);
    requireThrows(
        [&checkpoint, &metadata]() {
            mfa::nsConstrainedReadStageACheckpointContract<Scalar>(
                checkpoint, metadata,
                mfa::nsConstrainedFileSHA256(checkpoint));
        },
        "Stage-A loader accepted a wrong header");

    mfa::BaselineCheckpointMetadata changed = metadata;
    changed.lattice = "3,1,1";
    writeSmallStageA(checkpoint, changed);
    requireThrows(
        [&checkpoint, &metadata]() {
            mfa::nsConstrainedReadStageACheckpointContract<Scalar>(
                checkpoint, metadata,
                mfa::nsConstrainedFileSHA256(checkpoint));
        },
        "Stage-A loader accepted a metadata mismatch");

    writeSmallStageA(checkpoint, metadata);
    {
        std::ofstream stream(checkpoint.c_str(), std::ios::app);
        stream << "7\n";
    }
    requireThrows(
        [&checkpoint, &metadata]() {
            mfa::nsConstrainedReadStageACheckpointContract<Scalar>(
                checkpoint, metadata,
                mfa::nsConstrainedFileSHA256(checkpoint));
        },
        "Stage-A loader accepted an extra payload token");

    writeSmallStageA(checkpoint, metadata);
    replaceSmallStageAPayload(
        checkpoint, {"0.1", "-0.2", "0.3", "0.4", "0", "nan"});
    requireThrows(
        [&checkpoint, &metadata]() {
            mfa::nsConstrainedReadStageACheckpointContract<Scalar>(
                checkpoint, metadata,
                mfa::nsConstrainedFileSHA256(checkpoint));
        },
        "Stage-A loader accepted a nonfinite coefficient");

    writeSmallStageA(checkpoint, metadata);
    replaceSmallStageAPayload(
        checkpoint, {"101", "-0.2", "0.3", "0.4", "0", "0"});
    requireThrows(
        [&checkpoint, &metadata]() {
            mfa::nsConstrainedReadStageACheckpointContract<Scalar>(
                checkpoint, metadata,
                mfa::nsConstrainedFileSHA256(checkpoint));
        },
        "Stage-A loader accepted an out-of-bounds coefficient");

    writeSmallStageA(checkpoint, metadata);
    replaceSmallStageAPayload(
        checkpoint, {"0.1", "-0.2", "0.3", "0.4", "-0", "0"});
    requireThrows(
        [&checkpoint, &metadata]() {
            mfa::nsConstrainedReadStageACheckpointContract<Scalar>(
                checkpoint, metadata,
                mfa::nsConstrainedFileSHA256(checkpoint));
        },
        "Stage-A loader accepted negative-zero pressure");

    writeSmallStageA(checkpoint, metadata);
    replaceSmallStageAPayload(
        checkpoint, {"0.1", "-0.2", "0.3", "0.4", "0", "1e-30"});
    requireThrows(
        [&checkpoint, &metadata]() {
            mfa::nsConstrainedReadStageACheckpointContract<Scalar>(
                checkpoint, metadata,
                mfa::nsConstrainedFileSHA256(checkpoint));
        },
        "Stage-A loader accepted a nonzero pressure coefficient");
    std::remove(checkpoint.c_str());

    ManufacturedProvider provider;
    const mfa::NSConstrainedRunResult<Scalar> baseline_result =
        mfa::nsConstrainedEvaluateStageABaseline(
            provider, provider.center());
    const std::string baseline_record =
        mfa::nsConstrainedStageABaselineRecordPayload(
            baseline_result,
            testArtifactAnchors(),
            std::string(64, 'e'));
    require(
        baseline_record.find("known_feasible_start=1\n") !=
            std::string::npos &&
        baseline_record.find("gauge_count=1001\n") !=
            std::string::npos &&
        baseline_record.find("payload_sha256=") != std::string::npos,
        "Stage-A baseline record omitted feasibility/hash metadata");
    for (int metric = 0; metric < mfa::NS_V2_METRIC_COUNT; ++metric)
        require(
            baseline_record.find(
                std::string("raw_") +
                mfa::nsConstrainedMetricName(metric) + "=") !=
                std::string::npos,
            "Stage-A baseline record omitted a raw metric");
}

} // namespace

int main()
{
    try
    {
        testFrozenInnerSolverAndStrongWolfe();
        testV17MinimumStepBoundaries();
        testV17NonfiniteTrialGuard();
        testV17RemainingPreProviderGuards();
        testV17SquaredHingeBudgetAndBracket();
        testV17ExactLinearEightyAttemptCap();
        testV17ProposalStagesAndMidpointTriggers();
        testV17ThrowingOracleAndCanonicalAbortBytes();
        testRawComponentsAndPerComponentGradients();
        testGaugeAdjoint();
        testBaselineNormalizationAndKnownStart();
        testFullAugmentedLagrangianGradient();
        testSyntheticKKTAndStateMachine();
        testStateRestartAndSerializationHandshake();
        testDeterministicOuterDriverWithRepresentableInteriorRetry();
        testCanonicalPressureDataWeightEnvironment();
        testFullStartupContractEnvironment();
        testInformationBoundaryAndStageACheckpoint();
        std::cout << "NS constrained minimax V17 tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "NS constrained minimax V17 test failure: "
            << error.what() << "\n";
        return 1;
    }
}

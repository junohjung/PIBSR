// Copyright (C) 2020-2025 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSPP_LINE_SEARCH_MORE_THUENTE_H
#define LBFGSPP_LINE_SEARCH_MORE_THUENTE_H

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>  // std::invalid_argument, std::runtime_error
#include <string>
#include "Param.h"

namespace LBFGSpp {

enum LineSearchMoreThuenteProposalOrigin
{
    LBFGSPP_MT_PROPOSAL_INITIAL = 0,
    LBFGSPP_MT_PROPOSAL_CASE_1 = 1,
    LBFGSPP_MT_PROPOSAL_CASE_2 = 2,
    LBFGSPP_MT_PROPOSAL_CASE_3 = 3
};

enum LineSearchMoreThuenteMidpointTrigger
{
    LBFGSPP_MT_MIDPOINT_NONE = 0,
    LBFGSPP_MT_MIDPOINT_NONINTERIOR = 1,
    LBFGSPP_MT_MIDPOINT_WIDTH_0P66 = 2,
    LBFGSPP_MT_MIDPOINT_BOTH = 3
};

enum LineSearchMoreThuenteRawStepClass
{
    LBFGSPP_MT_RAW_ABSENT = 0,
    LBFGSPP_MT_RAW_FINITE = 1,
    LBFGSPP_MT_RAW_NAN = 2,
    LBFGSPP_MT_RAW_POS_INF = 3,
    LBFGSPP_MT_RAW_NEG_INF = 4
};

enum LineSearchMoreThuenteGuardReason
{
    LBFGSPP_MT_GUARD_NONE = 0,
    LBFGSPP_MT_GUARD_NONFINITE_PROPOSAL,
    LBFGSPP_MT_GUARD_OUT_OF_RANGE_PROPOSAL,
    LBFGSPP_MT_GUARD_BELOW_MIN_STEP_PROPOSAL,
    LBFGSPP_MT_GUARD_DUPLICATE_SCALAR_STEP,
    LBFGSPP_MT_GUARD_NO_REPRESENTABLE_BRACKET_INTERIOR,
    LBFGSPP_MT_GUARD_NONFINITE_TRIAL_VECTOR,
    LBFGSPP_MT_GUARD_TRIAL_EQUALS_PREVIOUS_STATE,
    LBFGSPP_MT_GUARD_TRIAL_EQUALS_BRACKET_ENDPOINT,
    LBFGSPP_MT_GUARD_TRIAL_EQUALS_TRUE_BOX_ENDPOINT,
    LBFGSPP_MT_GUARD_DUPLICATE_TRIAL_VECTOR,
    LBFGSPP_MT_GUARD_INVALID_BRACKET_WIDTH,
    LBFGSPP_MT_GUARD_INVALID_TRUE_BOX_ENDPOINT,
    LBFGSPP_MT_GUARD_CANDIDATE_ATTEMPT_CAP
};

enum LineSearchMoreThuenteTerminationReason
{
    LBFGSPP_MT_TERMINATION_NONE = 0,
    LBFGSPP_MT_TERMINATION_STRONG_WOLFE,
    LBFGSPP_MT_TERMINATION_GENERATOR_MAXIMUM,
    LBFGSPP_MT_TERMINATION_STEP_MAXIMUM,
    LBFGSPP_MT_TERMINATION_MAX_ITERATIONS,
    LBFGSPP_MT_TERMINATION_GUARD_EXCEPTION,
    LBFGSPP_MT_TERMINATION_OBJECTIVE_PROVIDER_EXCEPTION
};

inline const char* lineSearchMoreThuenteProposalOriginName(
    LineSearchMoreThuenteProposalOrigin value)
{
    switch (value)
    {
    case LBFGSPP_MT_PROPOSAL_INITIAL:
        return "INITIAL";
    case LBFGSPP_MT_PROPOSAL_CASE_1:
        return "CASE_1";
    case LBFGSPP_MT_PROPOSAL_CASE_2:
        return "CASE_2";
    case LBFGSPP_MT_PROPOSAL_CASE_3:
        return "CASE_3";
    }
    return "INITIAL";
}

inline const char* lineSearchMoreThuenteMidpointTriggerName(
    LineSearchMoreThuenteMidpointTrigger value)
{
    switch (value)
    {
    case LBFGSPP_MT_MIDPOINT_NONE:
        return "NONE";
    case LBFGSPP_MT_MIDPOINT_NONINTERIOR:
        return "NONINTERIOR";
    case LBFGSPP_MT_MIDPOINT_WIDTH_0P66:
        return "WIDTH_0P66";
    case LBFGSPP_MT_MIDPOINT_BOTH:
        return "BOTH";
    }
    return "NONE";
}

inline const char* lineSearchMoreThuenteRawStepClassName(
    LineSearchMoreThuenteRawStepClass value)
{
    switch (value)
    {
    case LBFGSPP_MT_RAW_ABSENT:
        return "ABSENT";
    case LBFGSPP_MT_RAW_FINITE:
        return "FINITE";
    case LBFGSPP_MT_RAW_NAN:
        return "NAN";
    case LBFGSPP_MT_RAW_POS_INF:
        return "POS_INF";
    case LBFGSPP_MT_RAW_NEG_INF:
        return "NEG_INF";
    }
    return "ABSENT";
}

inline const char* lineSearchMoreThuenteGuardReasonName(
    LineSearchMoreThuenteGuardReason value)
{
    switch (value)
    {
    case LBFGSPP_MT_GUARD_NONE:
        return "NONE";
    case LBFGSPP_MT_GUARD_NONFINITE_PROPOSAL:
        return "NONFINITE_PROPOSAL";
    case LBFGSPP_MT_GUARD_OUT_OF_RANGE_PROPOSAL:
        return "OUT_OF_RANGE_PROPOSAL";
    case LBFGSPP_MT_GUARD_BELOW_MIN_STEP_PROPOSAL:
        return "BELOW_MIN_STEP_PROPOSAL";
    case LBFGSPP_MT_GUARD_DUPLICATE_SCALAR_STEP:
        return "DUPLICATE_SCALAR_STEP";
    case LBFGSPP_MT_GUARD_NO_REPRESENTABLE_BRACKET_INTERIOR:
        return "NO_REPRESENTABLE_BRACKET_INTERIOR";
    case LBFGSPP_MT_GUARD_NONFINITE_TRIAL_VECTOR:
        return "NONFINITE_TRIAL_VECTOR";
    case LBFGSPP_MT_GUARD_TRIAL_EQUALS_PREVIOUS_STATE:
        return "TRIAL_EQUALS_PREVIOUS_STATE";
    case LBFGSPP_MT_GUARD_TRIAL_EQUALS_BRACKET_ENDPOINT:
        return "TRIAL_EQUALS_BRACKET_ENDPOINT";
    case LBFGSPP_MT_GUARD_TRIAL_EQUALS_TRUE_BOX_ENDPOINT:
        return "TRIAL_EQUALS_TRUE_BOX_ENDPOINT";
    case LBFGSPP_MT_GUARD_DUPLICATE_TRIAL_VECTOR:
        return "DUPLICATE_TRIAL_VECTOR";
    case LBFGSPP_MT_GUARD_INVALID_BRACKET_WIDTH:
        return "INVALID_BRACKET_WIDTH";
    case LBFGSPP_MT_GUARD_INVALID_TRUE_BOX_ENDPOINT:
        return "INVALID_TRUE_BOX_ENDPOINT";
    case LBFGSPP_MT_GUARD_CANDIDATE_ATTEMPT_CAP:
        return "CANDIDATE_ATTEMPT_CAP";
    }
    return "NONE";
}

inline const char* lineSearchMoreThuenteTerminationReasonName(
    LineSearchMoreThuenteTerminationReason value)
{
    switch (value)
    {
    case LBFGSPP_MT_TERMINATION_NONE:
        return "NONE";
    case LBFGSPP_MT_TERMINATION_STRONG_WOLFE:
        return "STRONG_WOLFE";
    case LBFGSPP_MT_TERMINATION_GENERATOR_MAXIMUM:
        return "GENERATOR_MAXIMUM";
    case LBFGSPP_MT_TERMINATION_STEP_MAXIMUM:
        return "STEP_MAXIMUM";
    case LBFGSPP_MT_TERMINATION_MAX_ITERATIONS:
        return "MAX_ITERATIONS";
    case LBFGSPP_MT_TERMINATION_GUARD_EXCEPTION:
        return "GUARD_EXCEPTION";
    case LBFGSPP_MT_TERMINATION_OBJECTIVE_PROVIDER_EXCEPTION:
        return "OBJECTIVE_PROVIDER_EXCEPTION";
    }
    return "NONE";
}

class LineSearchMoreThuenteGuardException : public std::runtime_error
{
private:
    LineSearchMoreThuenteGuardReason m_reason;

public:
    explicit LineSearchMoreThuenteGuardException(
        LineSearchMoreThuenteGuardReason reason)
        : std::runtime_error(
              std::string("Moré--Thuente generator guard: ") +
              lineSearchMoreThuenteGuardReasonName(reason)),
          m_reason(reason)
    {
    }

    LineSearchMoreThuenteGuardReason reason() const
    {
        return m_reason;
    }
};

template <typename Scalar>
struct LineSearchMoreThuenteDiagnostics
{
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    LineSearchMoreThuenteProposalOrigin proposal_origin;
    Scalar raw_proposed_step;
    bool raw_proposed_step_present;
    LineSearchMoreThuenteRawStepClass raw_proposed_step_class;
    Scalar safeguarded_proposed_step;
    bool safeguarded_proposed_step_present;
    LineSearchMoreThuenteMidpointTrigger midpoint_trigger;
    std::uint64_t forced_midpoints;
    bool final_bracketed;
    Scalar final_bracket_lo;
    Scalar final_bracket_hi;
    Scalar final_bracket_width;
    LineSearchMoreThuenteGuardReason guard_failure;
    LineSearchMoreThuenteTerminationReason termination_reason;
    std::uint64_t candidate_attempts;
    std::uint64_t candidate_completions;
    Scalar last_attempted_step;
    bool last_attempted_step_present;
    Scalar last_completed_step;
    bool last_completed_step_present;
    Scalar last_evaluated_step;
    bool last_evaluated_step_present;
    Vector true_box_endpoint;
    bool true_box_endpoint_present;

    LineSearchMoreThuenteDiagnostics()
        : proposal_origin(LBFGSPP_MT_PROPOSAL_INITIAL),
          raw_proposed_step(Scalar(0)),
          raw_proposed_step_present(false),
          raw_proposed_step_class(LBFGSPP_MT_RAW_ABSENT),
          safeguarded_proposed_step(Scalar(0)),
          safeguarded_proposed_step_present(false),
          midpoint_trigger(LBFGSPP_MT_MIDPOINT_NONE),
          forced_midpoints(0),
          final_bracketed(false),
          final_bracket_lo(Scalar(0)),
          final_bracket_hi(Scalar(0)),
          final_bracket_width(Scalar(0)),
          guard_failure(LBFGSPP_MT_GUARD_NONE),
          termination_reason(LBFGSPP_MT_TERMINATION_NONE),
          candidate_attempts(0),
          candidate_completions(0),
          last_attempted_step(Scalar(0)),
          last_attempted_step_present(false),
          last_completed_step(Scalar(0)),
          last_completed_step_present(false),
          last_evaluated_step(Scalar(0)),
          last_evaluated_step_present(false),
          true_box_endpoint(),
          true_box_endpoint_present(false)
    {
    }
};

template <typename Scalar>
struct LineSearchMoreThuenteCandidateAccounting
{
    std::uint64_t configured_attempt_cap;
    std::uint64_t call_attempts;
    std::uint64_t call_completions;
    std::uint64_t call_forced_midpoints;
    Scalar last_attempted_step;
    bool last_attempted_step_present;
    Scalar last_completed_step;
    bool last_completed_step_present;
    std::uint64_t* cumulative_attempts;
    std::uint64_t* cumulative_completions;
    std::uint64_t* cumulative_forced_midpoints;

    explicit LineSearchMoreThuenteCandidateAccounting(
        std::uint64_t attempt_cap =
            (std::numeric_limits<std::uint64_t>::max)())
        : configured_attempt_cap(attempt_cap),
          call_attempts(0),
          call_completions(0),
          call_forced_midpoints(0),
          last_attempted_step(Scalar(0)),
          last_attempted_step_present(false),
          last_completed_step(Scalar(0)),
          last_completed_step_present(false),
          cumulative_attempts(0),
          cumulative_completions(0),
          cumulative_forced_midpoints(0)
    {
    }
};

template <typename Scalar>
struct LineSearchMoreThuenteInvocationContext
{
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    LineSearchMoreThuenteCandidateAccounting<Scalar>* accounting;
    const Vector* true_box_endpoint;

    LineSearchMoreThuenteInvocationContext(
        LineSearchMoreThuenteCandidateAccounting<Scalar>* shared_accounting = 0,
        const Vector* exact_true_box_endpoint = 0)
        : accounting(shared_accounting),
          true_box_endpoint(exact_true_box_endpoint)
    {
    }
};

///
/// The line search algorithm by Moré and Thuente (1994), currently used for the L-BFGS-B algorithm.
///
/// The target of this line search algorithm is to find a step size \f$\alpha\f$ that satisfies the strong Wolfe condition
/// \f$f(x+\alpha d) \le f(x) + \alpha\mu g(x)^T d\f$ and \f$|g(x+\alpha d)^T d| \le \eta|g(x)^T d|\f$.
/// Our implementation is a simplified version of the algorithm in [1]. We assume that \f$0<\mu<\eta<1\f$, while in [1]
/// they do not assume \f$\eta>\mu\f$. As a result, the algorithm in [1] has two stages, but in our implementation we
/// only need the first stage to guarantee the convergence.
///
/// Reference:
/// [1] Moré, J. J., & Thuente, D. J. (1994). Line search algorithms with guaranteed sufficient decrease.
///
template <typename Scalar>
class LineSearchMoreThuente
{
public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Diagnostics = LineSearchMoreThuenteDiagnostics<Scalar>;
    using CandidateAccounting =
        LineSearchMoreThuenteCandidateAccounting<Scalar>;
    using InvocationContext =
        LineSearchMoreThuenteInvocationContext<Scalar>;
    using GuardException = LineSearchMoreThuenteGuardException;

private:

    // Minimizer of a quadratic function q(x) = c0 + c1 * x + c2 * x^2
    // that interpolates fa, ga, and fb, assuming the minimizer exists
    // For case I: fb >= fa and ga * (b - a) < 0
    static Scalar quadratic_minimizer(const Scalar& a, const Scalar& b, const Scalar& fa, const Scalar& ga, const Scalar& fb)
    {
        const Scalar ba = b - a;
        const Scalar w = Scalar(0.5) * ba * ga / (fa - fb + ba * ga);
        return a + w * ba;
    }

    // Minimizer of a quadratic function q(x) = c0 + c1 * x + c2 * x^2
    // that interpolates fa, ga and gb, assuming the minimizer exists
    // The result actually does not depend on fa
    // For case II: ga * (b - a) < 0, ga * gb < 0
    // For case III: ga * (b - a) < 0, ga * ga >= 0, |gb| <= |ga|
    static Scalar quadratic_minimizer(const Scalar& a, const Scalar& b, const Scalar& ga, const Scalar& gb)
    {
        const Scalar w = ga / (ga - gb);
        return a + w * (b - a);
    }

    // Local minimizer of a cubic function q(x) = c0 + c1 * x + c2 * x^2 + c3 * x^3
    // that interpolates fa, ga, fb and gb, assuming a != b
    // Also sets a flag indicating whether the minimizer exists
    static Scalar cubic_minimizer(const Scalar& a, const Scalar& b, const Scalar& fa, const Scalar& fb,
                                  const Scalar& ga, const Scalar& gb, bool& exists)
    {
        using std::abs;
        using std::sqrt;

        const Scalar apb = a + b;
        const Scalar ba = b - a;
        const Scalar ba2 = ba * ba;
        const Scalar fba = fb - fa;
        const Scalar gba = gb - ga;
        // z3 = c3 * (b-a)^3, z2 = c2 * (b-a)^3, z1 = c1 * (b-a)^3
        const Scalar z3 = (ga + gb) * ba - Scalar(2) * fba;
        const Scalar z2 = Scalar(0.5) * (gba * ba2 - Scalar(3) * apb * z3);
        const Scalar z1 = fba * ba2 - apb * z2 - (a * apb + b * b) * z3;
        // If c3 = z/(b-a)^3 == 0, reduce to quadratic problem
        const Scalar eps = std::numeric_limits<Scalar>::epsilon();
        if (abs(z3) < eps * abs(z2) || abs(z3) < eps * abs(z1))
        {
            // Minimizer exists if c2 > 0
            exists = (z2 * ba > Scalar(0));
            // Return the end point if the minimizer does not exist
            return exists ? (-Scalar(0.5) * z1 / z2) : b;
        }

        // Now we can assume z3 > 0
        // The minimizer is a solution to the equation c1 + 2*c2 * x + 3*c3 * x^2 = 0
        // roots = -(z2/z3) / 3 (+-) sqrt((z2/z3)^2 - 3 * (z1/z3)) / 3
        //
        // Let u = z2/(3z3) and v = z1/z2
        // The minimizer exists if v/u <= 1
        const Scalar u = z2 / (Scalar(3) * z3), v = z1 / z2;
        const Scalar vu = v / u;
        exists = (vu <= Scalar(1));
        if (!exists)
            return b;

        // We need to find a numerically stable way to compute the roots, as z3 may still be small
        //
        // If |u| >= |v|, let w = 1 + sqrt(1-v/u), and then
        // r1 = -u * w, r2 = -v / w, r1 does not need to be the smaller one
        //
        // If |u| < |v|, we must have uv <= 0, and then
        // r = -u (+-) sqrt(delta), where
        // sqrt(delta) = sqrt(|u|) * sqrt(|v|) * sqrt(1-u/v)
        Scalar r1 = Scalar(0), r2 = Scalar(0);
        if (abs(u) >= abs(v))
        {
            const Scalar w = Scalar(1) + sqrt(Scalar(1) - vu);
            r1 = -u * w;
            r2 = -v / w;
        }
        else
        {
            const Scalar sqrtd = sqrt(abs(u)) * sqrt(abs(v)) * sqrt(1 - u / v);
            r1 = -u - sqrtd;
            r2 = -u + sqrtd;
        }
        return (z3 * ba > Scalar(0)) ? ((std::max)(r1, r2)) : ((std::min)(r1, r2));
    }

    // Select the next step size according to the current step sizes,
    // function values, and derivatives
    static Scalar step_selection(
        const Scalar& al, const Scalar& au, const Scalar& at,
        const Scalar& fl, const Scalar& fu, const Scalar& ft,
        const Scalar& gl, const Scalar& gu, const Scalar& gt)
    {
        using std::abs;

        if (al == au)
            return al;

        // If ft = Inf or gt = Inf, we return the middle point of al and at
        if (!std::isfinite(ft) || !std::isfinite(gt))
            return (al + at) / Scalar(2);

        // ac: cubic interpolation of fl, ft, gl, gt
        // aq: quadratic interpolation of fl, gl, ft
        bool ac_exists;
        const Scalar ac = cubic_minimizer(al, at, fl, ft, gl, gt, ac_exists);
        const Scalar aq = quadratic_minimizer(al, at, fl, gl, ft);
        // Case 1: ft > fl
        if (ft > fl)
        {
            // This should not happen if ft > fl, but just to be safe
            if (!ac_exists)
                return aq;
            // Then use the scheme described in the paper
            return (abs(ac - al) < abs(aq - al)) ? ac : ((aq + ac) / Scalar(2));
        }

        // as: quadratic interpolation of gl and gt
        const Scalar as = quadratic_minimizer(al, at, gl, gt);
        // Case 2: ft <= fl, gt * gl < 0
        if (gt * gl < Scalar(0))
            return (abs(ac - at) >= abs(as - at)) ? ac : as;

        // Case 3: ft <= fl, gt * gl >= 0, |gt| < |gl|
        const Scalar deltal = Scalar(1.1), deltau = Scalar(0.66);
        if (abs(gt) < abs(gl))
        {
            // We choose either ac or as
            // The case for ac: 1. It exists, and
            //                  2. ac is farther than at from al, and
            //                  3. ac is closer to at than as
            // Cases for as: otherwise
            const Scalar res = (ac_exists &&
                                (ac - at) * (at - al) > Scalar(0) &&
                                abs(ac - at) < abs(as - at)) ?
                ac :
                as;
            // Postprocessing the chosen step
            return (at > al) ?
                std::min(at + deltau * (au - at), res) :
                std::max(at + deltau * (au - at), res);
        }

        // Simple extrapolation if au, fu, or gu is infinity
        if ((!std::isfinite(au)) || (!std::isfinite(fu)) || (!std::isfinite(gu)))
            return at + deltal * (at - al);

        // ae: cubic interpolation of ft, fu, gt, gu
        bool ae_exists;
        const Scalar ae = cubic_minimizer(at, au, ft, fu, gt, gu, ae_exists);
        // Case 4: ft <= fl, gt * gl >= 0, |gt| >= |gl|
        // The following is not used in the paper, but it seems to be a reasonable safeguard
        return (at > al) ?
            std::min(at + deltau * (au - at), ae) :
            std::max(at + deltau * (au - at), ae);
    }

    static LineSearchMoreThuenteRawStepClass raw_step_class(
        const Scalar& value)
    {
        if (std::isnan(value))
            return LBFGSPP_MT_RAW_NAN;
        if (std::isfinite(value))
            return LBFGSPP_MT_RAW_FINITE;
        return value > Scalar(0) ?
            LBFGSPP_MT_RAW_POS_INF :
            LBFGSPP_MT_RAW_NEG_INF;
    }

    static bool vectors_equal_exactly(
        const Vector& first, const Vector& second)
    {
        return
            first.size() == second.size() &&
            (first.array() == second.array()).all();
    }

    static void record_raw_proposal(
        Diagnostics& diagnostics,
        LineSearchMoreThuenteProposalOrigin origin,
        const Scalar& raw_proposal)
    {
        diagnostics.proposal_origin = origin;
        diagnostics.raw_proposed_step = raw_proposal;
        diagnostics.raw_proposed_step_present = true;
        diagnostics.raw_proposed_step_class =
            raw_step_class(raw_proposal);
        diagnostics.safeguarded_proposed_step = Scalar(0);
        diagnostics.safeguarded_proposed_step_present = false;
        diagnostics.midpoint_trigger = LBFGSPP_MT_MIDPOINT_NONE;
    }

    static void record_safeguarded_proposal(
        Diagnostics& diagnostics,
        const Scalar& safeguarded_proposal)
    {
        diagnostics.safeguarded_proposed_step =
            safeguarded_proposal;
        diagnostics.safeguarded_proposed_step_present = true;
    }

    static void record_bracket_if_valid(
        Diagnostics& diagnostics,
        const Scalar& first_endpoint,
        const Scalar& second_endpoint)
    {
        diagnostics.final_bracketed = false;
        diagnostics.final_bracket_lo = Scalar(0);
        diagnostics.final_bracket_hi = Scalar(0);
        diagnostics.final_bracket_width = Scalar(0);
        if (!std::isfinite(first_endpoint) ||
            !std::isfinite(second_endpoint))
            return;

        const Scalar lo = (std::min)(first_endpoint, second_endpoint);
        const Scalar hi = (std::max)(first_endpoint, second_endpoint);
        const Scalar bracket_width = hi - lo;
        if (!std::isfinite(bracket_width) ||
            !(bracket_width > Scalar(0)))
            return;

        diagnostics.final_bracketed = true;
        diagnostics.final_bracket_lo = lo;
        diagnostics.final_bracket_hi = hi;
        diagnostics.final_bracket_width = bracket_width;
    }

    static void fail_guard(
        Diagnostics& diagnostics,
        LineSearchMoreThuenteGuardReason reason)
    {
        diagnostics.guard_failure = reason;
        diagnostics.termination_reason =
            LBFGSPP_MT_TERMINATION_GUARD_EXCEPTION;
        throw GuardException(reason);
    }

    static void validate_scalar_candidate_before_provider(
        const Scalar& invocation_alpha_max,
        const Scalar& minimum_step,
        const Scalar& alpha,
        bool preceding_scalar_present,
        const Scalar& preceding_scalar,
        bool finite_bracket_present,
        const Scalar& bracket_lo,
        const Scalar& bracket_hi,
        Diagnostics& diagnostics)
    {
        if (!(alpha > Scalar(0)) ||
            !(alpha <= invocation_alpha_max))
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_OUT_OF_RANGE_PROPOSAL);
        if (alpha < minimum_step)
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_BELOW_MIN_STEP_PROPOSAL);
        if (preceding_scalar_present && alpha == preceding_scalar)
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_DUPLICATE_SCALAR_STEP);
        if (finite_bracket_present &&
            !(alpha > bracket_lo && alpha < bracket_hi))
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_NO_REPRESENTABLE_BRACKET_INTERIOR);
    }

    static void record_forced_midpoint(
        Diagnostics& diagnostics,
        CandidateAccounting* accounting)
    {
        ++diagnostics.forced_midpoints;
        if (!accounting)
            return;
        ++accounting->call_forced_midpoints;
        if (accounting->cumulative_forced_midpoints)
            ++(*accounting->cumulative_forced_midpoints);
    }

    static Scalar safeguard_bracketed_proposal(
        const Scalar& interval_lo,
        const Scalar& interval_hi,
        const Scalar& raw_proposal,
        Scalar& width,
        Scalar& width1,
        LineSearchMoreThuenteProposalOrigin origin,
        Diagnostics& diagnostics,
        CandidateAccounting* accounting)
    {
        record_raw_proposal(diagnostics, origin, raw_proposal);
        record_bracket_if_valid(
            diagnostics, interval_lo, interval_hi);

        if (!std::isfinite(raw_proposal))
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_NONFINITE_PROPOSAL);

        if (!std::isfinite(interval_lo) ||
            !std::isfinite(interval_hi))
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_INVALID_BRACKET_WIDTH);

        const Scalar lo = (std::min)(interval_lo, interval_hi);
        const Scalar hi = (std::max)(interval_lo, interval_hi);
        const Scalar new_width = hi - lo;
        if (!std::isfinite(new_width) ||
            !(new_width > Scalar(0)) ||
            !std::isfinite(width) ||
            !(width > Scalar(0)) ||
            !std::isfinite(width1) ||
            !(width1 > Scalar(0)))
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_INVALID_BRACKET_WIDTH);

        const bool noninterior =
            !(raw_proposal > lo && raw_proposal < hi);
        const bool width_trigger =
            new_width >= Scalar(0.66) * width1;
        Scalar safeguarded_proposal = raw_proposal;
        if (noninterior || width_trigger)
        {
            diagnostics.midpoint_trigger =
                noninterior && width_trigger ?
                    LBFGSPP_MT_MIDPOINT_BOTH :
                noninterior ?
                    LBFGSPP_MT_MIDPOINT_NONINTERIOR :
                    LBFGSPP_MT_MIDPOINT_WIDTH_0P66;
            safeguarded_proposal =
                interval_lo +
                Scalar(0.5) * (interval_hi - interval_lo);
            record_forced_midpoint(diagnostics, accounting);
            if (!std::isfinite(safeguarded_proposal))
                fail_guard(
                    diagnostics,
                    LBFGSPP_MT_GUARD_NO_REPRESENTABLE_BRACKET_INTERIOR);
            record_safeguarded_proposal(
                diagnostics, safeguarded_proposal);
            if (!(safeguarded_proposal > lo &&
                  safeguarded_proposal < hi))
                fail_guard(
                    diagnostics,
                    LBFGSPP_MT_GUARD_NO_REPRESENTABLE_BRACKET_INTERIOR);
        }
        else
        {
            record_safeguarded_proposal(
                diagnostics, safeguarded_proposal);
        }
        width1 = width;
        width = new_width;
        return safeguarded_proposal;
    }

    template <typename Foo>
    static void evaluate_candidate(
        Foo& function,
        const Scalar& invocation_alpha_max,
        const Scalar& minimum_step,
        const Vector& previous,
        const Vector& direction,
        const Scalar& alpha,
        bool preceding_scalar_present,
        const Scalar& preceding_scalar,
        bool finite_bracket_present,
        const Scalar& bracket_lo,
        const Scalar& bracket_hi,
        const Vector* first_bracket_endpoint,
        const Vector* second_bracket_endpoint,
        bool preceding_trial_present,
        const Vector* preceding_trial,
        const InvocationContext* context,
        CandidateAccounting& accounting,
        Diagnostics& diagnostics,
        Scalar& value,
        Vector& gradient,
        Scalar& directional_derivative,
        Vector& current)
    {
        validate_scalar_candidate_before_provider(
            invocation_alpha_max, minimum_step, alpha,
            preceding_scalar_present, preceding_scalar,
            finite_bracket_present, bracket_lo, bracket_hi,
            diagnostics);

        Vector guarded_trial(previous.size());
        guarded_trial.noalias() = previous + alpha * direction;
        if (!guarded_trial.allFinite())
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_NONFINITE_TRIAL_VECTOR);
        if (vectors_equal_exactly(guarded_trial, previous))
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_TRIAL_EQUALS_PREVIOUS_STATE);
        if (finite_bracket_present &&
            ((first_bracket_endpoint &&
              vectors_equal_exactly(
                  guarded_trial, *first_bracket_endpoint)) ||
             (second_bracket_endpoint &&
              vectors_equal_exactly(
                  guarded_trial, *second_bracket_endpoint))))
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_TRIAL_EQUALS_BRACKET_ENDPOINT);

        const Vector* true_box_endpoint =
            context ? context->true_box_endpoint : 0;
        if (true_box_endpoint)
        {
            if (true_box_endpoint->size() != previous.size() ||
                !true_box_endpoint->allFinite())
                fail_guard(
                    diagnostics,
                    LBFGSPP_MT_GUARD_INVALID_TRUE_BOX_ENDPOINT);
            if (vectors_equal_exactly(
                    guarded_trial, *true_box_endpoint))
                fail_guard(
                    diagnostics,
                    LBFGSPP_MT_GUARD_TRIAL_EQUALS_TRUE_BOX_ENDPOINT);
        }
        if (preceding_trial_present &&
            preceding_trial &&
            vectors_equal_exactly(guarded_trial, *preceding_trial))
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_DUPLICATE_TRIAL_VECTOR);

        if (accounting.call_attempts >=
            accounting.configured_attempt_cap)
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_CANDIDATE_ATTEMPT_CAP);

        current = guarded_trial;

        ++diagnostics.candidate_attempts;
        diagnostics.last_attempted_step = alpha;
        diagnostics.last_attempted_step_present = true;
        ++accounting.call_attempts;
        accounting.last_attempted_step = alpha;
        accounting.last_attempted_step_present = true;
        if (accounting.cumulative_attempts)
            ++(*accounting.cumulative_attempts);

        try
        {
            value = function(current, gradient);
        }
        catch (...)
        {
            diagnostics.termination_reason =
                LBFGSPP_MT_TERMINATION_OBJECTIVE_PROVIDER_EXCEPTION;
            throw;
        }

        ++diagnostics.candidate_completions;
        diagnostics.last_completed_step = alpha;
        diagnostics.last_completed_step_present = true;
        diagnostics.last_evaluated_step = alpha;
        diagnostics.last_evaluated_step_present = true;
        ++accounting.call_completions;
        accounting.last_completed_step = alpha;
        accounting.last_completed_step_present = true;
        if (accounting.cumulative_completions)
            ++(*accounting.cumulative_completions);

        directional_derivative = gradient.dot(direction);
    }

public:
    /// Diagnostics-only test seam for the exact scalar pre-provider validator
    /// used by evaluate_candidate(). It cannot access candidate/provider
    /// accounting and does not construct or evaluate a trial vector.
    static void ValidateScalarCandidatePreProviderForTesting(
        const Scalar& invocation_alpha_max,
        const Scalar& minimum_step,
        const Scalar& alpha,
        bool preceding_scalar_present,
        const Scalar& preceding_scalar,
        bool finite_bracket_present,
        const Scalar& bracket_lo,
        const Scalar& bracket_hi,
        Diagnostics& diagnostics)
    {
        validate_scalar_candidate_before_provider(
            invocation_alpha_max, minimum_step, alpha,
            preceding_scalar_present, preceding_scalar,
            finite_bracket_present, bracket_lo, bracket_hi,
            diagnostics);
    }

    ///
    /// Line search by Moré and Thuente (1994).
    ///
    /// \param f        A function object such that `f(x, grad)` returns the
    ///                 objective function value at `x`, and overwrites `grad` with
    ///                 the gradient.
    /// \param param    An `LBFGSParam` or `LBFGSBParam` object that stores the
    ///                 parameters of the solver.
    /// \param xp       The current point.
    /// \param drt      The current moving direction.
    /// \param step_max The upper bound for the step size that makes x feasible.
    /// \param step     In: The initial step length.
    ///                 Out: The calculated step length.
    /// \param fx       In: The objective function value at the current point.
    ///                 Out: The function value at the new point.
    /// \param grad     In: The current gradient vector.
    ///                 Out: The gradient at the new point.
    /// \param dg       In: The inner product between drt and grad.
    ///                 Out: The inner product between drt and the new gradient.
    /// \param x        Out: The new point moved to.
    ///
    static Scalar SafeguardBracketedProposal(
        const Scalar& interval_lo,
        const Scalar& interval_hi,
        const Scalar& raw_proposal,
        Scalar& width,
        Scalar& width1,
        LineSearchMoreThuenteProposalOrigin origin,
        Diagnostics& diagnostics,
        CandidateAccounting* accounting = 0)
    {
        return safeguard_bracketed_proposal(
            interval_lo, interval_hi, raw_proposal,
            width, width1, origin, diagnostics, accounting);
    }

    template <typename Foo, typename SolverParam>
    static void LineSearch(Foo& f, const SolverParam& param,
                           const Vector& xp, const Vector& drt, const Scalar& step_max,
                           Scalar& step, Scalar& fx, Vector& grad, Scalar& dg, Vector& x)
    {
        const std::uint64_t inferred_cap =
            param.max_linesearch > 0 ?
                static_cast<std::uint64_t>(param.max_linesearch) + 1 :
                std::uint64_t(1);
        Diagnostics diagnostics;
        CandidateAccounting accounting(inferred_cap);
        InvocationContext context(&accounting, 0);
        LineSearchWithDiagnostics(
            f, param, xp, drt, step_max, step, fx, grad, dg, x,
            &diagnostics, &context);
    }

    template <typename Foo, typename SolverParam>
    static void LineSearchWithDiagnostics(
        Foo& function,
        const SolverParam& parameters,
        const Vector& previous,
        const Vector& direction,
        const Scalar& invocation_alpha_max,
        Scalar& step,
        Scalar& value,
        Vector& gradient,
        Scalar& directional_derivative,
        Vector& current,
        Diagnostics* output_diagnostics,
        const InvocationContext* invocation_context = 0)
    {
        using std::abs;

        Diagnostics local_diagnostics;
        Diagnostics& diagnostics =
            output_diagnostics ?
                *output_diagnostics :
                local_diagnostics;
        diagnostics = Diagnostics();

        if (previous.size() != direction.size() ||
            previous.size() != gradient.size())
            throw std::invalid_argument(
                "Moré--Thuente vector dimensions do not match");
        if (!std::isfinite(parameters.min_step) ||
            !(parameters.min_step > Scalar(0)))
            throw std::invalid_argument(
                "'param.min_step' must be finite and positive");

        const std::uint64_t inferred_cap =
            parameters.max_linesearch > 0 ?
                static_cast<std::uint64_t>(
                    parameters.max_linesearch) + 1 :
                std::uint64_t(1);
        CandidateAccounting local_accounting(inferred_cap);
        CandidateAccounting& accounting =
            invocation_context && invocation_context->accounting ?
                *invocation_context->accounting :
                local_accounting;

        if (invocation_context &&
            invocation_context->true_box_endpoint)
        {
            diagnostics.true_box_endpoint =
                *invocation_context->true_box_endpoint;
            diagnostics.true_box_endpoint_present = true;
        }

        const Scalar initial_value = value;
        const Scalar initial_derivative = directional_derivative;
        if (!(initial_derivative < Scalar(0)))
            throw std::logic_error(
                "the moving direction does not decrease the objective function value");

        record_raw_proposal(
            diagnostics, LBFGSPP_MT_PROPOSAL_INITIAL, step);
        if (!std::isfinite(step))
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_NONFINITE_PROPOSAL);
        record_safeguarded_proposal(diagnostics, step);

        Scalar width = invocation_alpha_max;
        Scalar width1 = Scalar(2) * width;
        if (!std::isfinite(width) ||
            !(width > Scalar(0)) ||
            !std::isfinite(width1) ||
            !(width1 > Scalar(0)))
            fail_guard(
                diagnostics,
                LBFGSPP_MT_GUARD_INVALID_BRACKET_WIDTH);

        const Scalar test_decr =
            parameters.ftol * initial_derivative;
        const Scalar test_curv =
            -parameters.wolfe * initial_derivative;

        Scalar interval_lo = Scalar(0);
        Scalar interval_hi =
            std::numeric_limits<Scalar>::infinity();
        Scalar function_lo = Scalar(0);
        Scalar function_hi =
            std::numeric_limits<Scalar>::infinity();
        Scalar derivative_lo =
            (Scalar(1) - parameters.ftol) *
            initial_derivative;
        Scalar derivative_hi =
            std::numeric_limits<Scalar>::infinity();

        Vector vector_lo = previous;
        Vector gradient_lo = gradient;
        Vector vector_hi;
        bool vector_hi_present = false;
        Scalar value_lo = initial_value;
        Scalar directional_derivative_lo =
            initial_derivative;

        evaluate_candidate(
            function, invocation_alpha_max,
            parameters.min_step, previous, direction, step,
            false, Scalar(0), false, Scalar(0), Scalar(0),
            0, 0, false, 0, invocation_context,
            accounting, diagnostics, value, gradient,
            directional_derivative, current);

        Scalar preceding_scalar = step;
        bool preceding_scalar_present = true;
        Vector preceding_trial = current;
        bool preceding_trial_present = true;

        if (value <= initial_value + step * test_decr &&
            abs(directional_derivative) <= test_curv)
        {
            diagnostics.termination_reason =
                LBFGSPP_MT_TERMINATION_STRONG_WOLFE;
            return;
        }

        const Scalar extrapolation_factor = Scalar(1.1);
        int iteration = 0;
        for (iteration = 0;
             iteration < parameters.max_linesearch;
             ++iteration)
        {
            const Scalar shifted_value =
                value - initial_value - step * test_decr;
            const Scalar shifted_derivative =
                directional_derivative -
                parameters.ftol * initial_derivative;

            const Vector evaluated_vector = current;
            const Vector evaluated_gradient = gradient;
            const Scalar evaluated_step = step;

            Scalar raw_proposal;
            LineSearchMoreThuenteProposalOrigin origin;
            bool low_endpoint_was_updated = false;
            if (shifted_value > function_lo)
            {
                origin = LBFGSPP_MT_PROPOSAL_CASE_1;
                raw_proposal = step_selection(
                    interval_lo, interval_hi, step,
                    function_lo, function_hi, shifted_value,
                    derivative_lo, derivative_hi,
                    shifted_derivative);

                interval_hi = step;
                function_hi = shifted_value;
                derivative_hi = shifted_derivative;
                vector_hi = evaluated_vector;
                vector_hi_present = true;
            }
            else if (
                shifted_derivative *
                    (interval_lo - step) >
                Scalar(0))
            {
                origin = LBFGSPP_MT_PROPOSAL_CASE_2;
                raw_proposal = (std::min)(
                    invocation_alpha_max,
                    step + extrapolation_factor *
                        (step - interval_lo));

                interval_lo = step;
                function_lo = shifted_value;
                derivative_lo = shifted_derivative;
                vector_lo.swap(current);
                gradient_lo.swap(gradient);
                value_lo = value;
                directional_derivative_lo =
                    directional_derivative;
                low_endpoint_was_updated = true;
            }
            else
            {
                origin = LBFGSPP_MT_PROPOSAL_CASE_3;
                raw_proposal = step_selection(
                    interval_lo, interval_hi, step,
                    function_lo, function_hi, shifted_value,
                    derivative_lo, derivative_hi,
                    shifted_derivative);

                interval_hi = interval_lo;
                function_hi = function_lo;
                derivative_hi = derivative_lo;
                vector_hi = vector_lo;
                vector_hi_present = true;

                interval_lo = step;
                function_lo = shifted_value;
                derivative_lo = shifted_derivative;
                vector_lo.swap(current);
                gradient_lo.swap(gradient);
                value_lo = value;
                directional_derivative_lo =
                    directional_derivative;
                low_endpoint_was_updated = true;
            }

            record_raw_proposal(
                diagnostics, origin, raw_proposal);
            record_bracket_if_valid(
                diagnostics, interval_lo, interval_hi);

            if (evaluated_step == invocation_alpha_max &&
                raw_proposal >= invocation_alpha_max)
            {
                if (low_endpoint_was_updated)
                {
                    current.swap(vector_lo);
                    gradient.swap(gradient_lo);
                }
                else
                {
                    current = evaluated_vector;
                    gradient = evaluated_gradient;
                }
                step = evaluated_step;
                diagnostics.termination_reason =
                    LBFGSPP_MT_TERMINATION_GENERATOR_MAXIMUM;
                return;
            }

            Scalar safeguarded_proposal;
            const bool finite_bracket_present =
                std::isfinite(interval_hi);
            if (finite_bracket_present)
            {
                if (!vector_hi_present)
                    fail_guard(
                        diagnostics,
                        LBFGSPP_MT_GUARD_INVALID_BRACKET_WIDTH);
                safeguarded_proposal =
                    safeguard_bracketed_proposal(
                        interval_lo, interval_hi,
                        raw_proposal, width, width1, origin,
                        diagnostics, &accounting);
            }
            else
            {
                if (!std::isfinite(raw_proposal))
                    fail_guard(
                        diagnostics,
                        LBFGSPP_MT_GUARD_NONFINITE_PROPOSAL);
                record_safeguarded_proposal(
                    diagnostics, raw_proposal);
                safeguarded_proposal = raw_proposal;
            }

            step = safeguarded_proposal;
            const Scalar bracket_lo =
                finite_bracket_present ?
                    (std::min)(interval_lo, interval_hi) :
                    Scalar(0);
            const Scalar bracket_hi =
                finite_bracket_present ?
                    (std::max)(interval_lo, interval_hi) :
                    Scalar(0);
            evaluate_candidate(
                function, invocation_alpha_max,
                parameters.min_step, previous, direction, step,
                preceding_scalar_present, preceding_scalar,
                finite_bracket_present, bracket_lo, bracket_hi,
                finite_bracket_present ? &vector_lo : 0,
                finite_bracket_present ? &vector_hi : 0,
                preceding_trial_present, &preceding_trial,
                invocation_context, accounting, diagnostics,
                value, gradient, directional_derivative,
                current);

            preceding_scalar = step;
            preceding_scalar_present = true;
            preceding_trial = current;
            preceding_trial_present = true;

            if (value <= initial_value + step * test_decr &&
                abs(directional_derivative) <= test_curv)
            {
                diagnostics.termination_reason =
                    LBFGSPP_MT_TERMINATION_STRONG_WOLFE;
                return;
            }

            if (step >= invocation_alpha_max)
            {
                const Scalar bound_shifted_value =
                    value - initial_value - step * test_decr;
                if (bound_shifted_value <= function_lo)
                {
                    diagnostics.termination_reason =
                        LBFGSPP_MT_TERMINATION_STEP_MAXIMUM;
                    return;
                }
            }
        }

        if (iteration >= parameters.max_linesearch)
        {
            diagnostics.termination_reason =
                LBFGSPP_MT_TERMINATION_MAX_ITERATIONS;
            record_bracket_if_valid(
                diagnostics, interval_lo, interval_hi);

            const Scalar shifted_value =
                value - initial_value - step * test_decr;
            if (shifted_value <= function_lo)
                return;

            if (interval_lo <= Scalar(0))
                throw std::runtime_error(
                    "the line search routine is unable to sufficiently decrease the function value");

            step = interval_lo;
            value = value_lo;
            directional_derivative =
                directional_derivative_lo;
            current.swap(vector_lo);
            gradient.swap(gradient_lo);
            return;
        }
    }
};

}  // namespace LBFGSpp

#endif  // LBFGSPP_LINE_SEARCH_MORE_THUENTE_H

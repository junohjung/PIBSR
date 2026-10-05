//--------------------------------------------------------------
// Read-only final fit evaluator for the prospective cavity V2 method.
//
// The evaluator strict-loads a final checkpoint, evaluates the frozen
// fit-only provider exactly once, and emits no model serialization.
//--------------------------------------------------------------

#ifndef MFA_NS_CONSTRAINED_FIT_EVALUATOR_HPP
#define MFA_NS_CONSTRAINED_FIT_EVALUATOR_HPP

#include <mfa/ns_constrained_minimax_executable.hpp>
#include <mfa/ns_constrained_model_tools.hpp>

#include <Eigen/Core>

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mfa
{

inline std::string nsConstrainedReadHeaderField(
    std::istream& stream,
    const char* expected)
{
    std::string token;
    if (!(stream >> token))
        throw std::runtime_error(
            std::string("Truncated constrained V2 checkpoint before ") +
            expected);
    const std::string prefix = std::string(expected) + "=";
    if (token.compare(0, prefix.size(), prefix) != 0)
        throw std::runtime_error(
            std::string("Expected constrained V2 checkpoint field ") +
            expected);
    return token.substr(prefix.size());
}

template <typename T>
struct NSConstrainedLoadedFinalCheckpoint
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    Vector coefficients;
    T eta;
    bool converged;
    std::string coefficient_sha256;
    std::string decoded_handshake_sha256;
    NSConstrainedArtifactAnchors anchors;
};

template <typename Derived>
inline std::string nsConstrainedFinalCheckpointCanonicalPayload(
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
    return stream.str();
}

template <typename T>
inline NSConstrainedLoadedFinalCheckpoint<T>
nsConstrainedReadFinalCheckpointStrict(
    const std::string& path,
    const NSConstrainedArtifactAnchors& expected_anchors,
    int expected_coefficient_count)
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    if (expected_coefficient_count != 60000)
        throw std::invalid_argument(
            "Constrained V2 final checkpoint requires 60000 coefficients");
    nsConstrainedRequireRegularNonSymlink(path, "final checkpoint");
    const std::string payload = nsConstrainedReadFileText(path);
    std::istringstream stream(payload);
    std::string header;
    if (!std::getline(stream, header))
        throw std::runtime_error(
            "Missing constrained V2 final-checkpoint header");
    std::istringstream fields(header);
    std::string marker;
    std::string magic;
    if (!(fields >> marker >> magic) ||
        marker != "#" ||
        magic != "PIMFA_NS_CONSTRAINED_MINIMAX_AL_V2")
        throw std::runtime_error(
            "Invalid constrained V2 final-checkpoint magic");

    NSConstrainedLoadedFinalCheckpoint<T> loaded;
    if (nsConstrainedReadHeaderField(fields, "candidate_id") !=
        nsConstrainedCandidateId())
        throw std::runtime_error(
            "Constrained V2 final-checkpoint candidate mismatch");
    loaded.anchors.stage_a_checkpoint_sha256 =
        nsConstrainedReadHeaderField(
            fields, "stage_a_checkpoint_sha256");
    loaded.anchors.stage_a_baseline_record_sha256 =
        nsConstrainedReadHeaderField(
            fields, "stage_a_baseline_record_sha256");
    loaded.anchors.optimization_protocol_sha256 =
        nsConstrainedReadHeaderField(
            fields, "optimization_protocol_sha256");
    loaded.anchors.optimization_protocol_parent_sha256 =
        nsConstrainedReadHeaderField(
            fields, "optimization_protocol_parent_sha256");
    loaded.anchors.optimization_protocol_ancestor_2_sha256 =
        nsConstrainedReadHeaderField(
            fields, "optimization_protocol_ancestor_2_sha256");
    loaded.anchors.optimization_protocol_ancestor_3_sha256 =
        nsConstrainedReadHeaderField(
            fields, "optimization_protocol_ancestor_3_sha256");
    loaded.anchors.inherited_optimization_protocol_sha256 =
        nsConstrainedReadHeaderField(
            fields, "inherited_optimization_protocol_sha256");
    loaded.anchors.implementation_sha256 =
        nsConstrainedReadHeaderField(
            fields, "implementation_sha256");
    loaded.anchors.post_selection_protocol_v5_sha256 =
        nsConstrainedReadHeaderField(
            fields, "post_selection_protocol_v5_sha256");
    loaded.anchors.post_selection_protocol_v4_parent_sha256 =
        nsConstrainedReadHeaderField(
            fields, "post_selection_protocol_v4_parent_sha256");
    nsConstrainedValidateArtifactAnchors(loaded.anchors);
    if (loaded.anchors.stage_a_checkpoint_sha256 !=
            expected_anchors.stage_a_checkpoint_sha256 ||
        loaded.anchors.stage_a_baseline_record_sha256 !=
            expected_anchors.stage_a_baseline_record_sha256 ||
        loaded.anchors.optimization_protocol_sha256 !=
            expected_anchors.optimization_protocol_sha256 ||
        loaded.anchors.optimization_protocol_parent_sha256 !=
            expected_anchors.optimization_protocol_parent_sha256 ||
        loaded.anchors.optimization_protocol_ancestor_2_sha256 !=
            expected_anchors.optimization_protocol_ancestor_2_sha256 ||
        loaded.anchors.optimization_protocol_ancestor_3_sha256 !=
            expected_anchors.optimization_protocol_ancestor_3_sha256 ||
        loaded.anchors.inherited_optimization_protocol_sha256 !=
            expected_anchors.inherited_optimization_protocol_sha256 ||
        loaded.anchors.implementation_sha256 !=
            expected_anchors.implementation_sha256 ||
        loaded.anchors.post_selection_protocol_v5_sha256 !=
            expected_anchors.post_selection_protocol_v5_sha256 ||
        loaded.anchors.post_selection_protocol_v4_parent_sha256 !=
            expected_anchors.post_selection_protocol_v4_parent_sha256)
        throw std::runtime_error(
            "Constrained V2 final-checkpoint anchor mismatch");

    const int coefficient_count = nsConstrainedParseInteger(
        nsConstrainedReadHeaderField(fields, "n"), "n", 1);
    if (coefficient_count != expected_coefficient_count)
        throw std::runtime_error(
            "Constrained V2 final-checkpoint coefficient-count mismatch");
    loaded.eta = nsConstrainedParseFinite<T>(
        nsConstrainedReadHeaderField(fields, "eta"), "eta");
    const int converged = nsConstrainedParseInteger(
        nsConstrainedReadHeaderField(fields, "converged"),
        "converged", 0);
    if (converged != 0 && converged != 1)
        throw std::runtime_error(
            "Constrained V2 final-checkpoint converged flag is not 0 or 1");
    loaded.converged = converged == 1;
    loaded.coefficient_sha256 =
        nsConstrainedReadHeaderField(fields, "coefficient_sha256");
    loaded.decoded_handshake_sha256 =
        nsConstrainedReadHeaderField(
            fields, "decoded_handshake_sha256");
    std::string extra;
    if (fields >> extra)
        throw std::runtime_error(
            "Constrained V2 final-checkpoint header has trailing fields");
    if (!nsConstrainedIsLowerHexDigest(loaded.coefficient_sha256) ||
        !nsConstrainedIsLowerHexDigest(
            loaded.decoded_handshake_sha256) ||
        loaded.eta < T(0) ||
        loaded.eta > T(100))
        throw std::runtime_error(
            "Constrained V2 final-checkpoint scalar/hash contract failed");

    loaded.coefficients.resize(coefficient_count);
    for (int i = 0; i < coefficient_count; ++i)
    {
        std::string line;
        if (!std::getline(stream, line))
            throw std::runtime_error(
                "Truncated constrained V2 final-checkpoint payload");
        const T value =
            nsConstrainedParseFinite<T>(line, "coefficient");
        if (value < T(-100) || value > T(100))
            throw std::runtime_error(
                "Constrained V2 final-checkpoint coefficient is out "
                "of frozen bounds");
        loaded.coefficients(i) = value;
    }
    std::string trailing;
    if (std::getline(stream, trailing))
        throw std::runtime_error(
            "Constrained V2 final-checkpoint has trailing content");
    if (nsConstrainedCoefficientSHA256(loaded.coefficients) !=
        loaded.coefficient_sha256)
        throw std::runtime_error(
            "Constrained V2 final-checkpoint coefficient hash mismatch");
    if (payload != nsConstrainedFinalCheckpointCanonicalPayload(
            loaded.coefficients,
            loaded.eta,
            loaded.converged,
            loaded.decoded_handshake_sha256,
            loaded.anchors))
        throw std::runtime_error(
            "Constrained V2 final-checkpoint is not canonical");
    return loaded;
}

template <typename T>
inline void nsConstrainedValidateStageAInitialTheta(
    T observed,
    T recomputed)
{
    if (!std::isfinite(observed) ||
        !std::isfinite(recomputed) ||
        observed != recomputed ||
        recomputed < T(0) ||
        recomputed >
            T(64) * std::numeric_limits<T>::epsilon())
        throw std::runtime_error(
            "Constrained V2 Stage-A baseline-record initial theta "
            "mismatch/nonfeasibility");
}

template <typename T>
inline NSConstrainedBaseline<T>
nsConstrainedReadStageABaselineRecordStrict(
    const std::string& path,
    const NSConstrainedArtifactAnchors& expected_anchors,
    const std::string& expected_coefficient_sha256)
{
    nsConstrainedRequireRegularNonSymlink(path, "Stage-A baseline record");
    const std::string payload = nsConstrainedReadFileText(path);
    const std::string marker = "payload_sha256=";
    const std::size_t marker_position = payload.rfind(marker);
    if (marker_position == std::string::npos ||
        marker_position > 0 && payload[marker_position - 1] != '\n' ||
        payload.empty() ||
        payload[payload.size() - 1] != '\n')
        throw std::runtime_error(
            "Malformed constrained V2 Stage-A baseline record digest");
    const std::string body = payload.substr(0, marker_position);
    const std::string declared_digest = payload.substr(
        marker_position + marker.size(),
        payload.size() - marker_position - marker.size() - 1);
    if (!nsConstrainedIsLowerHexDigest(declared_digest) ||
        nsConstrainedSHA256(body) != declared_digest)
        throw std::runtime_error(
            "Constrained V2 Stage-A baseline record payload hash mismatch");

    std::istringstream stream(body);
    std::string magic;
    if (!std::getline(stream, magic) ||
        magic !=
            "PIMFA_NS_CONSTRAINED_MINIMAX_AL_V2_STAGE_A_BASELINE")
        throw std::runtime_error(
            "Invalid constrained V2 Stage-A baseline-record magic");
    if (nsConstrainedReadKey(stream, "candidate_id") !=
            nsConstrainedCandidateId() ||
        nsConstrainedReadKey(stream, "stage_a_checkpoint_sha256") !=
            expected_anchors.stage_a_checkpoint_sha256 ||
        nsConstrainedReadKey(
            stream, "stage_a_initializer_role_sha256") !=
            nsConstrainedFrozenStageARoleSHA256() ||
        nsConstrainedReadKey(stream, "stage_a_code_sha256") !=
            nsConstrainedFrozenStageACodeSHA256() ||
        nsConstrainedReadKey(stream, "optimization_protocol_sha256") !=
            expected_anchors.optimization_protocol_sha256 ||
        nsConstrainedReadKey(stream, "implementation_sha256") !=
            expected_anchors.implementation_sha256 ||
        nsConstrainedReadKey(
            stream, "post_selection_protocol_v5_sha256") !=
            expected_anchors.post_selection_protocol_v5_sha256 ||
        nsConstrainedReadKey(
            stream, "post_selection_protocol_v4_parent_sha256") !=
            expected_anchors.post_selection_protocol_v4_parent_sha256 ||
        nsConstrainedReadKey(stream, "coefficient_sha256") !=
            expected_coefficient_sha256)
        throw std::runtime_error(
            "Constrained V2 Stage-A baseline-record anchor mismatch");
    const std::string decoded =
        nsConstrainedReadKey(stream, "decoded_handshake_sha256");
    if (!nsConstrainedIsLowerHexDigest(decoded) ||
        nsConstrainedReadKey(stream, "coefficient_count") != "60000" ||
        nsConstrainedReadKey(stream, "gauge_count") != "1001" ||
        nsConstrainedReadKey(stream, "data_fit_count") != "100813" ||
        nsConstrainedReadKey(stream, "pde_fit_count") != "101508" ||
        nsConstrainedReadKey(
            stream, "integral_fit_time_count") != "560" ||
        nsConstrainedReadKey(
            stream, "initial_condition_count") != "441" ||
        nsConstrainedReadKey(
            stream, "boundary_x_positive_time_count") != "42000" ||
        nsConstrainedReadKey(
            stream, "boundary_y_positive_time_count") != "42000" ||
        nsConstrainedReadKey(stream, "bc_mse_definition") !=
            "sum_x_face_mean_plus_y_face_mean" ||
        nsConstrainedReadKey(stream, "physics_t_min") != "0.01" ||
        nsConstrainedReadKey(stream, "viscosity") != "0.001" ||
        nsConstrainedReadKey(
            stream, "local_integral_partition") != "4,4" ||
        nsConstrainedReadKey(
            stream, "global_integral_partition") != "1,1" ||
        nsConstrainedReadKey(stream, "quadrature") != "trapezoid")
        throw std::runtime_error(
            "Constrained V2 Stage-A baseline-record role/count mismatch");
    const T eta = nsConstrainedParseFinite<T>(
        nsConstrainedReadKey(stream, "eta"), "eta");
    const T gauge_max_abs = nsConstrainedParseFinite<T>(
        nsConstrainedReadKey(stream, "gauge_max_abs"), "gauge_max_abs");
    const T gauge_rms = nsConstrainedParseFinite<T>(
        nsConstrainedReadKey(stream, "gauge_rms"), "gauge_rms");
    const T initial_theta = nsConstrainedParseFinite<T>(
        nsConstrainedReadKey(stream, "initial_theta"), "initial_theta");
    if (eta != T(100) ||
        gauge_max_abs != T(0) ||
        gauge_rms != T(0) ||
        nsConstrainedReadKey(stream, "known_feasible_start") != "1")
        throw std::runtime_error(
            "Constrained V2 Stage-A baseline record is not the exact "
            "known feasible start");

    NSConstrainedRawEvaluation<T> raw;
    raw.gauge =
        Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(1001);
    for (int i = 0; i < NS_V2_METRIC_COUNT; ++i)
    {
        const std::string key =
            std::string("raw_") + nsConstrainedMetricName(i);
        raw.metrics[i] = nsConstrainedParseFinite<T>(
            nsConstrainedReadKey(stream, key.c_str()),
            key.c_str());
    }
    const NSConstrainedBaseline<T> baseline =
        NSConstrainedBaseline<T>::capture(raw);
    const NSConstrainedConstraints<T> recomputed =
        nsConstrainedEvaluateConstraints(raw, baseline, T(100));
    const T recomputed_theta =
        nsConstrainedPrimalInfeasibility(recomputed);
    nsConstrainedValidateStageAInitialTheta(
        initial_theta, recomputed_theta);
    for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
    {
        const std::string key =
            std::string("constraint_") + std::to_string(i);
        const T observed = nsConstrainedParseFinite<T>(
            nsConstrainedReadKey(stream, key.c_str()),
            key.c_str());
        if (observed != recomputed.inequalities[i])
            throw std::runtime_error(
                "Constrained V2 Stage-A baseline-record constraint "
                "mismatch");
    }
    std::string trailing;
    if (std::getline(stream, trailing))
        throw std::runtime_error(
            "Constrained V2 Stage-A baseline record has trailing content");
    return baseline;
}

inline NSConstrainedArtifactAnchors
nsConstrainedFitEvaluatorPreflight()
{
    nsConstrainedRejectForbiddenEnvironment();
    nsConstrainedRequireProductionNoResume();
    if (nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_CANDIDATE_ID") !=
        nsConstrainedCandidateId())
        throw std::invalid_argument(
            "Constrained V2 evaluator candidate ID mismatch");
    nsConstrainedRequiredEnvironment("MFA_DATA_ACTIVE_INDEX_FILE");
    nsConstrainedRequiredEnvironment("MFA_PDE_ACTIVE_INDEX_FILE");
    nsConstrainedRequiredEnvironment("MFA_INT_FIT_TIME_INDEX_FILE");
    const std::string stage_a_path =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_STAGE_A_CHECKPOINT_IN");
    const std::string stage_a_sha =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_STAGE_A_CHECKPOINT_SHA256");
    if (stage_a_sha != nsConstrainedFrozenStageASHA256())
        throw std::invalid_argument(
            "Constrained V2 evaluator Stage-A declared hash drifted");
    nsConstrainedRequireRegularNonSymlink(
        stage_a_path, "Stage-A checkpoint");
    if (nsConstrainedFileSHA256(stage_a_path) != stage_a_sha)
        throw std::runtime_error(
            "Constrained V2 evaluator Stage-A file hash mismatch");

    const std::string baseline_path =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_BASELINE_RECORD_IN");
    const std::string baseline_sha =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_BASELINE_RECORD_SHA256");
    nsConstrainedRequireRegularNonSymlink(
        baseline_path, "Stage-A baseline record");
    if (!nsConstrainedIsLowerHexSHA256(baseline_sha) ||
        nsConstrainedFileSHA256(baseline_path) != baseline_sha)
        throw std::runtime_error(
            "Constrained V2 evaluator baseline-record hash mismatch");

    const std::string final_path =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_FINAL_CHECKPOINT_IN");
    const std::string final_sha =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_FINAL_CHECKPOINT_SHA256");
    nsConstrainedRequireRegularNonSymlink(
        final_path, "final checkpoint");
    if (!nsConstrainedIsLowerHexSHA256(final_sha) ||
        nsConstrainedFileSHA256(final_path) != final_sha)
        throw std::runtime_error(
            "Constrained V2 evaluator final-checkpoint hash mismatch");
    nsConstrainedRequireAbsentPath(
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_FINAL_FIT_RECORD_OUT"),
        "final fit record");

    static const char* prohibited_outputs[] = {
        "MFA_NS_CONSTRAINED_V2_BASELINE_EVALUATION_ONLY",
        "MFA_NS_CONSTRAINED_V2_BASELINE_RECORD_OUT",
        "MFA_NS_CONSTRAINED_V2_OUTER_RECORD_OUT",
        "MFA_NS_CONSTRAINED_V2_STATE_OUT",
        "MFA_NS_CONSTRAINED_V2_FINAL_CHECKPOINT_OUT"
    };
    for (std::size_t i = 0;
         i < sizeof(prohibited_outputs) / sizeof(prohibited_outputs[0]);
         ++i)
        if (nsConstrainedEnvironmentIsDefined(prohibited_outputs[i]))
            throw std::invalid_argument(
                std::string("Constrained V2 fit evaluator forbids "
                            "optimizer output control: ") +
                prohibited_outputs[i]);

    const std::string protocol_sha =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_PROTOCOL_SHA256");
    const std::string implementation_sha =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_CODE_SHA256");
    const std::string post_selection_v5_sha =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_POST_SELECTION_PROTOCOL_V5_SHA256");
    const std::string post_selection_v4_parent_sha =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_POST_SELECTION_PROTOCOL_V4_PARENT_SHA256");
    if (protocol_sha !=
            nsConstrainedFrozenOptimizationProtocolSHA256() ||
        !nsConstrainedIsLowerHexSHA256(implementation_sha) ||
        post_selection_v5_sha !=
            nsConstrainedFrozenPostSelectionV5SHA256() ||
        post_selection_v4_parent_sha !=
            nsConstrainedFrozenPostSelectionV4ParentSHA256())
        throw std::invalid_argument(
            "Constrained V2 evaluator protocol/code anchors drifted");
    if (nsConstrainedRequiredFiniteScalar("MFA_NS_PHYSICS_T_MIN") !=
            0.01 ||
        nsConstrainedRequiredFiniteScalar("MFA_NS_NU") != 0.001 ||
        nsConstrainedRequiredEnvironment("OMP_NUM_THREADS") != "1")
        throw std::invalid_argument(
            "Constrained V2 evaluator physics/thread controls drifted");
    if (nsConstrainedEnvironmentIsDefined("MFA_INT_PARTS_X") &&
        nsConstrainedRequiredEnvironment("MFA_INT_PARTS_X") != "4")
        throw std::invalid_argument(
            "Constrained V2 evaluator local x partition drifted");
    if (nsConstrainedEnvironmentIsDefined("MFA_INT_PARTS_Y") &&
        nsConstrainedRequiredEnvironment("MFA_INT_PARTS_Y") != "4")
        throw std::invalid_argument(
            "Constrained V2 evaluator local y partition drifted");
    if (nsConstrainedEnvironmentIsDefined("MFA_INT_QUADRATURE") &&
        nsConstrainedRequiredEnvironment("MFA_INT_QUADRATURE") !=
            "trapezoid")
        throw std::invalid_argument(
            "Constrained V2 evaluator quadrature drifted");
    return nsConstrainedExecutableAnchors();
}

template <typename T>
inline std::string nsConstrainedFinalFitRecordPayload(
    const NSConstrainedLoadedFinalCheckpoint<T>& checkpoint,
    const NSConstrainedRawEvaluation<T>& raw,
    const NSConstrainedConstraints<T>& constraints,
    const std::string& final_checkpoint_file_sha256)
{
    if (raw.gauge.size() != 1001 ||
        constraints.gauge.size() != 1001 ||
        (raw.gauge - constraints.gauge).squaredNorm() != T(0) ||
        !raw.gauge.allFinite() ||
        !nsConstrainedIsLowerHexDigest(final_checkpoint_file_sha256))
        throw std::invalid_argument(
            "Constrained V2 final fit record dimension/hash mismatch");
    std::ostringstream stream;
    stream
        << "PIMFA_NS_CONSTRAINED_MINIMAX_AL_V2_FINAL_FIT_EVALUATION_V1\n"
        << "candidate_id=" << nsConstrainedCandidateId() << "\n"
        << "stage_a_checkpoint_sha256="
        << checkpoint.anchors.stage_a_checkpoint_sha256 << "\n"
        << "stage_a_baseline_record_sha256="
        << checkpoint.anchors.stage_a_baseline_record_sha256 << "\n"
        << "optimization_protocol_sha256="
        << checkpoint.anchors.optimization_protocol_sha256 << "\n"
        << "optimization_protocol_parent_sha256="
        << checkpoint.anchors.optimization_protocol_parent_sha256 << "\n"
        << "optimization_protocol_ancestor_2_sha256="
        << checkpoint.anchors.optimization_protocol_ancestor_2_sha256
        << "\n"
        << "optimization_protocol_ancestor_3_sha256="
        << checkpoint.anchors.optimization_protocol_ancestor_3_sha256
        << "\n"
        << "inherited_optimization_protocol_sha256="
        << checkpoint.anchors.inherited_optimization_protocol_sha256 << "\n"
        << "implementation_sha256="
        << checkpoint.anchors.implementation_sha256 << "\n"
        << "post_selection_protocol_v5_sha256="
        << checkpoint.anchors.post_selection_protocol_v5_sha256 << "\n"
        << "post_selection_protocol_v4_parent_sha256="
        << checkpoint.anchors.post_selection_protocol_v4_parent_sha256
        << "\n"
        << "final_checkpoint_file_sha256="
        << final_checkpoint_file_sha256 << "\n"
        << "coefficient_sha256="
        << checkpoint.coefficient_sha256 << "\n"
        << "decoded_handshake_sha256="
        << checkpoint.decoded_handshake_sha256 << "\n"
        << "coefficient_count=60000\n"
        << "gauge_count=1001\n"
        << "provider_evaluations=1\n"
        << "data_fit_count=100813\n"
        << "pde_fit_count=101508\n"
        << "integral_fit_time_count=560\n"
        << "initial_condition_count=441\n"
        << "boundary_x_positive_time_count=42000\n"
        << "boundary_y_positive_time_count=42000\n"
        << std::setprecision(std::numeric_limits<T>::max_digits10)
        << "eta=" << checkpoint.eta << "\n"
        << "converged=" << (checkpoint.converged ? 1 : 0) << "\n"
        << "primal_infeasibility="
        << nsConstrainedPrimalInfeasibility(constraints) << "\n";
    for (int i = 0; i < NS_V2_METRIC_COUNT; ++i)
        stream << "raw_" << nsConstrainedMetricName(i)
               << "=" << raw.metrics[i] << "\n";
    for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
        stream << "constraint_" << i
               << "=" << constraints.inequalities[i] << "\n";
    for (Eigen::Index i = 0; i < raw.gauge.size(); ++i)
        stream << "gauge_" << i << "=" << raw.gauge(i) << "\n";
    const std::string body = stream.str();
    return body + "payload_sha256=" +
        nsConstrainedSHA256(body) + "\n";
}

template <typename Provider>
inline std::string nsConstrainedEvaluateFinalFitOnce(
    Provider& provider,
    const NSConstrainedLoadedFinalCheckpoint<
        typename Provider::Scalar>& checkpoint,
    const NSConstrainedBaseline<typename Provider::Scalar>& baseline,
    const std::string& final_checkpoint_file_sha256)
{
    typedef typename Provider::Scalar T;
    if (provider.coefficientCount() != 60000 ||
        provider.gaugeCount() != 1001 ||
        provider.dataFitCount() != 100813 ||
        provider.pdeFitCount() != 101508 ||
        provider.initialConditionCount() != 441 ||
        provider.boundaryXCount() != 42000 ||
        provider.boundaryYCount() != 42000 ||
        provider.integralFitTimeCount() != 560 ||
        checkpoint.coefficients.size() != provider.coefficientCount())
        throw std::runtime_error(
            "Constrained V2 final evaluator fit-role dimensions drifted");
    const NSConstrainedRawEvaluation<T> raw =
        provider.evaluate(checkpoint.coefficients);
    const std::string decoded =
        provider.decodedHandshakeSHA256FromCache();
    if (decoded != checkpoint.decoded_handshake_sha256)
        throw std::runtime_error(
            "Constrained V2 final checkpoint/provider decoded handshake "
            "mismatch");
    const NSConstrainedConstraints<T> constraints =
        nsConstrainedEvaluateConstraints(
            raw, baseline, checkpoint.eta);
    return nsConstrainedFinalFitRecordPayload(
        checkpoint,
        raw,
        constraints,
        final_checkpoint_file_sha256);
}

} // namespace mfa

#endif

//--------------------------------------------------------------
// Fail-closed executable contract for the prospective cavity V2 core.
//--------------------------------------------------------------

#ifndef MFA_NS_CONSTRAINED_MINIMAX_EXECUTABLE_HPP
#define MFA_NS_CONSTRAINED_MINIMAX_EXECUTABLE_HPP

#include <mfa/ns_constrained_minimax.hpp>
#include <mfa/ns_constrained_minimax_ns_adapter.hpp>
#include <mfa/reg_mfa_tikhonov.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <crt_externs.h>
#else
extern char** environ;
#endif

namespace mfa
{

inline const char* nsConstrainedFrozenStageASHA256()
{
    return
        "ff7456a87e6dc389cb73f22f0778d5a926510d3adb5f940e9eea0534bc4421ba";
}

inline const char* nsConstrainedFrozenStageARoleSHA256()
{
    return
        "f5a57978d5991a104742112b0be3c1596aa669e7e2dd4e50a7c825e2cff7b391";
}

inline const char* nsConstrainedFrozenStageACodeSHA256()
{
    return
        "903a801715b56283abc45ad04a4831d8d04c1bdacda5fe19be956d9a02d1e4d6";
}

inline const char* nsConstrainedFrozenPostSelectionV5SHA256()
{
    return
        "c4e22c560a029565af39c636e1dbfb33edfd290946f7fb92dd8f6997033f1dc2";
}

inline const char* nsConstrainedFrozenPostSelectionV4ParentSHA256()
{
    return
        "a82761a7373fb563a757269f41a3121d6a26f86d60494fc15daf6bb7d7e15c03";
}

inline const char* nsConstrainedFrozenOptimizationProtocolSHA256()
{
    return
        "14648ed7c01d8e34b8869232bd20268e1d9f4c4a788e8091f3a02aac41692bf0";
}

inline const char* nsConstrainedFrozenOptimizationProtocolParentSHA256()
{
    return
        "ea33c7e22d89a589c8bc3b90051b6e003b53d420cb9ea5c600aa8a7868e54ad0";
}

inline const char* nsConstrainedFrozenOptimizationProtocolAncestor2SHA256()
{
    return
        "4eb6fff402c270f659b8230859edc7afde3539306f1a007fe65569f6403bd431";
}

inline const char* nsConstrainedFrozenOptimizationProtocolAncestor3SHA256()
{
    return
        "07102dc337849f602b002da3160a61d791a6dd68659a48416926b2c112b2879e";
}

inline const char* nsConstrainedFrozenInheritedOptimizationProtocolSHA256()
{
    return
        "eb26fbf050bf543043aa3be7e64ff707d2cdb784d2dd17b0d7c856ceafe62072";
}

inline const char* nsConstrainedFrozenHistoricalImplementationSHA256()
{
    return
        "69ecca0dc4fb736525fa1738ce0b8fc8ef6d704c7cb8dc3ccec111cc60da4a40";
}

inline const char* nsConstrainedFrozenInfrastructureProtocolV3SHA256()
{
    return
        "17c3a2cb4f14bac71576e5d95f9f3f6cc0628b03e8ba3abbc0baa63b14af8ab7";
}

inline const char* nsConstrainedFrozenInfrastructureLineageAddendumSHA256()
{
    return
        "67d5db7973de1974d6dc120f2630a9773dda18facdcd3924164925e3e63e1b16";
}

inline BaselineCheckpointMetadata nsConstrainedFrozenStageAMetadata()
{
    BaselineCheckpointMetadata metadata;
    metadata.case_name = "ns_lid_cavity";
    metadata.kind = BaselineKind::data_only;
    metadata.alpha1 = 0.0;
    metadata.alpha2 = 0.0;
    metadata.normalization = "mean_rows_v1";
    metadata.derivative_orders = "none";
    metadata.derivative_dims = "x,y,t";
    metadata.lattice = "20,20,50";
    metadata.degree = "3,3,3";
    metadata.initializer_role_sha256 =
        nsConstrainedFrozenStageARoleSHA256();
    metadata.pressure_regime = "zero";
    metadata.field_order = "u,v,p";
    metadata.coefficient_count_per_field = 20000;
    metadata.coefficient_count_total = 60000;
    metadata.code_sha256 = nsConstrainedFrozenStageACodeSHA256();
    metadata.payload_encoding = "float_text_max_digits10";
    return metadata;
}

inline bool nsConstrainedPathExists(const std::string& path)
{
    std::ifstream stream(path.c_str(), std::ios::binary);
    return static_cast<bool>(stream);
}

inline std::string nsConstrainedReadFileText(const std::string& path)
{
    std::ifstream stream(path.c_str(), std::ios::in | std::ios::binary);
    if (!stream)
        throw std::runtime_error(
            "Cannot open constrained V2 text input: " + path);
    std::ostringstream payload;
    payload << stream.rdbuf();
    if (stream.bad())
        throw std::runtime_error(
            "Failed reading constrained V2 text input: " + path);
    return payload.str();
}

inline std::string nsConstrainedRequiredEnvironment(const char* name)
{
    const char* value = std::getenv(name);
    if (!value || !*value)
        throw std::invalid_argument(
            std::string("Constrained V2 requires ") + name);
    return std::string(value);
}

inline void nsConstrainedRequireExactEnvironment(
    const char* name,
    const char* expected)
{
    const std::string value = nsConstrainedRequiredEnvironment(name);
    if (value != expected)
        throw std::invalid_argument(
            std::string("Constrained V2 requires ") + name +
            "=" + expected);
}

inline bool nsConstrainedEnvironmentIsDefined(const char* name)
{
    const char* value = std::getenv(name);
    return value && *value;
}

inline bool nsConstrainedContainsForbiddenRoleWord(
    const std::string& environment_name)
{
    static const char* forbidden[] = {
        "VALID", "SEALED", "REFERENCE", "REFINED", "HIGH_RES", "HIGHRES",
        "QOI", "FINAL_TEST"
    };
    if (environment_name.compare(0, 4, "MFA_") != 0 &&
        environment_name.compare(0, 6, "PIMFA_") != 0)
        return false;
    for (size_t i = 0; i < sizeof(forbidden) / sizeof(forbidden[0]); ++i)
        if (environment_name.find(forbidden[i]) != std::string::npos)
            return true;
    return false;
}

inline void nsConstrainedValidateRawEnvironmentEntries(
    const std::vector<std::string>& entries)
{
    std::size_t pressure_data_weight_count = 0;
    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        const std::string& entry = entries[i];
        const size_t equals = entry.find('=');
        const std::string name = entry.substr(0, equals);
        if (nsConstrainedContainsForbiddenRoleWord(name))
            throw std::invalid_argument(
                "Constrained V2 optimizer environment contains a prohibited "
                "role: " + name);
        if (name.compare(0, 11, "MFA_LAMBDA_") == 0)
        {
            if (name != "MFA_LAMBDA_DATA_P")
                throw std::invalid_argument(
                    "Constrained V2 forbids legacy scalar weights: " + name);
            ++pressure_data_weight_count;
            if (equals == std::string::npos ||
                entry.substr(equals + 1) != "0")
                throw std::invalid_argument(
                    "Constrained V2 requires MFA_LAMBDA_DATA_P to be "
                    "exactly the literal 0");
        }
    }
    if (pressure_data_weight_count != 1)
        throw std::invalid_argument(
            "Constrained V2 requires exactly one "
            "MFA_LAMBDA_DATA_P=0 environment entry");
}

inline void nsConstrainedRejectForbiddenEnvironment()
{
#if defined(__APPLE__)
    char** environment = *_NSGetEnviron();
#else
    char** environment = ::environ;
#endif
    std::vector<std::string> entries;
    for (char** item = environment; item && *item; ++item)
        entries.push_back(std::string(*item));
    nsConstrainedValidateRawEnvironmentEntries(entries);
    if (nsConstrainedEnvironmentIsDefined("MFA_INT_DIAGNOSTICS"))
        throw std::invalid_argument(
            "Constrained V2 optimizer forbids V1 integral diagnostics");
    static const char* legacy_controls[] = {
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
    for (std::size_t i = 0;
         i < sizeof(legacy_controls) / sizeof(legacy_controls[0]); ++i)
        if (nsConstrainedEnvironmentIsDefined(legacy_controls[i]))
            throw std::invalid_argument(
                std::string("Constrained V2 forbids legacy control: ") +
                legacy_controls[i]);
}

inline double nsConstrainedRequiredFiniteScalar(
    const char* name)
{
    const std::string text = nsConstrainedRequiredEnvironment(name);
    std::istringstream stream(text);
    double value;
    char extra = '\0';
    if (!(stream >> value) || (stream >> extra) || !std::isfinite(value))
        throw std::invalid_argument(
            std::string("Constrained V2 requires a finite scalar in ") + name);
    return value;
}

inline bool nsConstrainedIsLowerHexSHA256(const std::string& value)
{
    if (value.size() != 64)
        return false;
    for (size_t i = 0; i < value.size(); ++i)
        if (!((value[i] >= '0' && value[i] <= '9') ||
              (value[i] >= 'a' && value[i] <= 'f')))
            return false;
    return true;
}

inline bool nsConstrainedStrictBooleanEnvironment(
    const char* name,
    bool default_value)
{
    const char* value = std::getenv(name);
    if (!value)
        return default_value;
    if (std::string(value) == "0")
        return false;
    if (std::string(value) == "1")
        return true;
    throw std::invalid_argument(
        std::string(name) + " must be exactly 0 or 1");
}

inline void nsConstrainedRequireProductionNoResume()
{
    const std::string value =
        nsConstrainedRequiredEnvironment("MFA_NS_CONSTRAINED_V2_RESUME");
    if (value != "0")
        throw std::invalid_argument(
            "Constrained V2 production requires "
            "MFA_NS_CONSTRAINED_V2_RESUME=0; in-place resume is forbidden");
}

inline void nsConstrainedStartupContractPreflight()
{
    nsConstrainedRejectForbiddenEnvironment();
    nsConstrainedRequireExactEnvironment("OMP_NUM_THREADS", "1");
    nsConstrainedRequireExactEnvironment("OPENBLAS_NUM_THREADS", "1");
    nsConstrainedRequireExactEnvironment("MKL_NUM_THREADS", "1");
    nsConstrainedRequireExactEnvironment(
        "VECLIB_MAXIMUM_THREADS", "1");
    nsConstrainedRequireExactEnvironment("NUMEXPR_NUM_THREADS", "1");
    nsConstrainedRequireExactEnvironment("BLIS_NUM_THREADS", "1");
    nsConstrainedRequireExactEnvironment("OMPI_MCA_btl", "self");
    nsConstrainedRequireExactEnvironment(
        "MFA_INIT_PRESSURE_MODE", "zero");
    nsConstrainedRequireExactEnvironment("MFA_LAMBDA_DATA_P", "0");
    nsConstrainedRequireExactEnvironment(
        "MFA_REQUIRE_ZERO_PRESSURE_BC", "1");
    nsConstrainedRequireProductionNoResume();
    nsConstrainedRequireExactEnvironment(
        "MFA_NS_CONSTRAINED_V2_CANDIDATE_ID",
        nsConstrainedCandidateId());
    nsConstrainedRequireExactEnvironment(
        "MFA_NS_CONSTRAINED_V2_PROTOCOL_SHA256",
        nsConstrainedFrozenOptimizationProtocolSHA256());
    nsConstrainedRequireExactEnvironment(
        "MFA_NS_CONSTRAINED_V2_PROTOCOL_PARENT_SHA256",
        nsConstrainedFrozenOptimizationProtocolParentSHA256());
    nsConstrainedRequireExactEnvironment(
        "MFA_NS_CONSTRAINED_V2_PROTOCOL_ANCESTOR_2_SHA256",
        nsConstrainedFrozenOptimizationProtocolAncestor2SHA256());
    nsConstrainedRequireExactEnvironment(
        "MFA_NS_CONSTRAINED_V2_PROTOCOL_ANCESTOR_3_SHA256",
        nsConstrainedFrozenOptimizationProtocolAncestor3SHA256());
    nsConstrainedRequireExactEnvironment(
        "MFA_NS_CONSTRAINED_V2_INHERITED_PROTOCOL_SHA256",
        nsConstrainedFrozenInheritedOptimizationProtocolSHA256());
    nsConstrainedRequireExactEnvironment(
        "MFA_NS_CONSTRAINED_V3_INFRASTRUCTURE_PROTOCOL_SHA256",
        nsConstrainedFrozenInfrastructureProtocolV3SHA256());
    nsConstrainedRequireExactEnvironment(
        "MFA_NS_CONSTRAINED_V3_LINEAGE_ADDENDUM_SHA256",
        nsConstrainedFrozenInfrastructureLineageAddendumSHA256());
}

inline void nsConstrainedExecutablePreflight()
{
    nsConstrainedStartupContractPreflight();
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
            "Constrained V2 Stage-A declared hash is not the frozen hash");
    if (nsConstrainedFileSHA256(stage_a_path) != stage_a_sha)
        throw std::runtime_error(
            "Constrained V2 Stage-A file hash mismatch");
    const bool baseline_only =
        nsConstrainedStrictBooleanEnvironment(
            "MFA_NS_CONSTRAINED_V2_BASELINE_EVALUATION_ONLY", false);
    if (baseline_only)
    {
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_BASELINE_RECORD_OUT");
    }
    else
    {
        const std::string baseline_record_path =
            nsConstrainedRequiredEnvironment(
                "MFA_NS_CONSTRAINED_V2_BASELINE_RECORD_IN");
        const std::string baseline_record_sha =
            nsConstrainedRequiredEnvironment(
                "MFA_NS_CONSTRAINED_V2_BASELINE_RECORD_SHA256");
        if (!nsConstrainedIsLowerHexSHA256(baseline_record_sha) ||
            nsConstrainedFileSHA256(baseline_record_path) !=
                baseline_record_sha)
            throw std::runtime_error(
                "Constrained V2 frozen baseline-record hash mismatch");
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_OUTER_RECORD_OUT");
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_STATE_OUT");
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_FINAL_CHECKPOINT_OUT");
    }
    const std::string protocol_sha =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_PROTOCOL_SHA256");
    const std::string code_sha =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_CODE_SHA256");
    const std::string post_selection_v5_sha =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_POST_SELECTION_PROTOCOL_V5_SHA256");
    const std::string post_selection_v4_parent_sha =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_POST_SELECTION_PROTOCOL_V4_PARENT_SHA256");
    if (protocol_sha != nsConstrainedFrozenOptimizationProtocolSHA256() ||
        !nsConstrainedIsLowerHexSHA256(code_sha) ||
        post_selection_v5_sha !=
            nsConstrainedFrozenPostSelectionV5SHA256() ||
        post_selection_v4_parent_sha !=
            nsConstrainedFrozenPostSelectionV4ParentSHA256())
        throw std::invalid_argument(
            "Constrained V2 protocol/code/post-selection hashes do not "
            "match the frozen contract");
    const double physics_t_min =
        nsConstrainedRequiredFiniteScalar("MFA_NS_PHYSICS_T_MIN");
    if (physics_t_min != 0.01)
        throw std::invalid_argument(
            "Constrained V2 prescribed BC threshold must be exactly 0.01");
    const double viscosity =
        nsConstrainedRequiredFiniteScalar("MFA_NS_NU");
    if (viscosity != 0.001)
        throw std::invalid_argument(
            "Constrained V2 cavity viscosity must be exactly 0.001");
    if (nsConstrainedEnvironmentIsDefined("MFA_INT_PARTS_X") &&
        nsConstrainedRequiredEnvironment("MFA_INT_PARTS_X") != "4")
        throw std::invalid_argument(
            "Constrained V2 local partition is frozen at 4 x 4");
    if (nsConstrainedEnvironmentIsDefined("MFA_INT_PARTS_Y") &&
        nsConstrainedRequiredEnvironment("MFA_INT_PARTS_Y") != "4")
        throw std::invalid_argument(
            "Constrained V2 local partition is frozen at 4 x 4");
    if (nsConstrainedEnvironmentIsDefined("MFA_INT_QUADRATURE") &&
        nsConstrainedRequiredEnvironment("MFA_INT_QUADRATURE") !=
            "trapezoid")
        throw std::invalid_argument(
            "Constrained V2 quadrature is frozen at trapezoid");
}

inline NSConstrainedArtifactAnchors nsConstrainedExecutableAnchors()
{
    NSConstrainedArtifactAnchors anchors;
    anchors.stage_a_checkpoint_sha256 =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_STAGE_A_CHECKPOINT_SHA256");
    if (nsConstrainedEnvironmentIsDefined(
            "MFA_NS_CONSTRAINED_V2_BASELINE_RECORD_SHA256"))
        anchors.stage_a_baseline_record_sha256 =
            nsConstrainedRequiredEnvironment(
                "MFA_NS_CONSTRAINED_V2_BASELINE_RECORD_SHA256");
    else
        anchors.stage_a_baseline_record_sha256 = std::string(64, '0');
    anchors.optimization_protocol_sha256 =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_PROTOCOL_SHA256");
    anchors.optimization_protocol_parent_sha256 =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_PROTOCOL_PARENT_SHA256");
    anchors.optimization_protocol_ancestor_2_sha256 =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_PROTOCOL_ANCESTOR_2_SHA256");
    anchors.optimization_protocol_ancestor_3_sha256 =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_PROTOCOL_ANCESTOR_3_SHA256");
    anchors.inherited_optimization_protocol_sha256 =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_INHERITED_PROTOCOL_SHA256");
    anchors.implementation_sha256 =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_CODE_SHA256");
    anchors.post_selection_protocol_v5_sha256 =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_POST_SELECTION_PROTOCOL_V5_SHA256");
    anchors.post_selection_protocol_v4_parent_sha256 =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_POST_SELECTION_PROTOCOL_V4_PARENT_SHA256");
    nsConstrainedValidateArtifactAnchors(anchors);
    if (anchors.stage_a_checkpoint_sha256 !=
            nsConstrainedFrozenStageASHA256() ||
        anchors.optimization_protocol_sha256 !=
            nsConstrainedFrozenOptimizationProtocolSHA256() ||
        anchors.optimization_protocol_parent_sha256 !=
            nsConstrainedFrozenOptimizationProtocolParentSHA256() ||
        anchors.optimization_protocol_ancestor_2_sha256 !=
            nsConstrainedFrozenOptimizationProtocolAncestor2SHA256() ||
        anchors.optimization_protocol_ancestor_3_sha256 !=
            nsConstrainedFrozenOptimizationProtocolAncestor3SHA256() ||
        anchors.inherited_optimization_protocol_sha256 !=
            nsConstrainedFrozenInheritedOptimizationProtocolSHA256() ||
        anchors.post_selection_protocol_v5_sha256 !=
            nsConstrainedFrozenPostSelectionV5SHA256() ||
        anchors.post_selection_protocol_v4_parent_sha256 !=
            nsConstrainedFrozenPostSelectionV4ParentSHA256())
        throw std::invalid_argument(
            "Constrained V2 executable artifact anchors drifted");
    return anchors;
}

inline NSConstrainedArtifactAnchors
nsConstrainedHistoricalStageABaselineAnchors(
    const NSConstrainedArtifactAnchors& current)
{
    NSConstrainedArtifactAnchors historical = current;
    historical.optimization_protocol_sha256 =
        nsConstrainedFrozenInheritedOptimizationProtocolSHA256();
    historical.optimization_protocol_parent_sha256 =
        std::string(64, '0');
    historical.optimization_protocol_ancestor_2_sha256 =
        std::string(64, '0');
    historical.optimization_protocol_ancestor_3_sha256 =
        std::string(64, '0');
    historical.inherited_optimization_protocol_sha256 =
        std::string(64, '0');
    historical.implementation_sha256 =
        nsConstrainedFrozenHistoricalImplementationSHA256();
    nsConstrainedValidateArtifactAnchors(historical);
    return historical;
}

template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1>
nsConstrainedReadStageACheckpointContract(
    const std::string& path,
    const BaselineCheckpointMetadata& expected,
    const std::string& expected_file_sha256)
{
    if (!nsConstrainedIsLowerHexSHA256(expected_file_sha256))
        throw std::invalid_argument(
            "Constrained V2 Stage-A expected hash is not lowercase SHA-256");
    if (expected.pressure_regime != "zero" ||
        expected.field_order != "u,v,p" ||
        expected.coefficient_count_total !=
            3 * expected.coefficient_count_per_field)
        throw std::invalid_argument(
            "Constrained V2 Stage-A contract requires zero-pressure u,v,p");
    const std::string actual_file_sha256 =
        nsConstrainedFileSHA256(path);
    if (actual_file_sha256 != expected_file_sha256)
        throw std::runtime_error(
            "Constrained V2 Stage-A file hash mismatch");

    const std::vector<double> loaded =
        ReadBaselineCheckpointV2(path, expected);
    Eigen::Matrix<T, Eigen::Dynamic, 1> coefficients(
        static_cast<Eigen::Index>(loaded.size()));
    for (std::size_t i = 0; i < loaded.size(); ++i)
    {
        const T value = static_cast<T>(loaded[i]);
        if (!std::isfinite(value) ||
            value < T(-100) ||
            value > T(100))
            throw std::runtime_error(
                "Nonfinite or out-of-bounds constrained V2 Stage-A "
                "coefficient");
        coefficients(static_cast<Eigen::Index>(i)) = value;
    }
    const Eigen::Index field_count =
        static_cast<Eigen::Index>(expected.coefficient_count_per_field);
    for (Eigen::Index i = 2 * field_count;
         i < coefficients.size(); ++i)
        if (coefficients(i) != T(0) ||
            std::signbit(static_cast<double>(coefficients(i))))
            throw std::runtime_error(
                "Constrained V2 Stage-A pressure must be exact +0");
    return coefficients;
}

template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1>
nsConstrainedReadStageACheckpoint(
    const std::string& path,
    int expected_count)
{
    const BaselineCheckpointMetadata expected =
        nsConstrainedFrozenStageAMetadata();
    if (expected_count !=
        static_cast<int>(expected.coefficient_count_total))
        throw std::runtime_error(
            "Constrained V2 runtime lattice does not match frozen Stage-A");
    return nsConstrainedReadStageACheckpointContract<T>(
        path, expected, nsConstrainedFrozenStageASHA256());
}

template <typename Provider>
inline NSConstrainedRunResult<typename Provider::Scalar>
nsConstrainedEvaluateStageABaseline(
    Provider& provider,
    const Eigen::Matrix<
        typename Provider::Scalar, Eigen::Dynamic, 1>& coefficients)
{
    typedef typename Provider::Scalar T;
    if (coefficients.size() != provider.coefficientCount())
        throw std::invalid_argument(
            "Constrained V2 Stage-A/provider coefficient mismatch");
    NSConstrainedRunResult<T> result;
    result.coefficients = coefficients;
    result.eta = T(100);
    result.converged = false;
    result.termination = "baseline_evaluation_only";
    result.state = NSConstrainedOuterState<T>(provider.gaugeCount());
    result.raw = provider.evaluate(coefficients);
    result.baseline = NSConstrainedBaseline<T>::capture(result.raw);
    result.constraints = nsConstrainedEvaluateConstraints(
        result.raw, result.baseline, result.eta);
    const T tolerance =
        T(64) * std::numeric_limits<T>::epsilon();
    for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
        if (result.constraints.inequalities[i] > tolerance)
            throw std::runtime_error(
                "Constrained V2 Stage-A/eta=100 is not the preregistered "
                "known feasible start");
    return result;
}

template <typename T>
inline std::string nsConstrainedStageABaselineRecordPayload(
    const NSConstrainedRunResult<T>& result,
    const NSConstrainedArtifactAnchors& anchors,
    const std::string& decoded_handshake_sha256)
{
    nsConstrainedValidateArtifactAnchors(anchors);
    if (!nsConstrainedIsLowerHexSHA256(decoded_handshake_sha256) ||
        anchors.stage_a_checkpoint_sha256 !=
            nsConstrainedFrozenStageASHA256() ||
        anchors.post_selection_protocol_v5_sha256 !=
            nsConstrainedFrozenPostSelectionV5SHA256() ||
        anchors.post_selection_protocol_v4_parent_sha256 !=
            nsConstrainedFrozenPostSelectionV4ParentSHA256())
        throw std::invalid_argument(
            "Constrained V2 baseline record anchors must be SHA-256 values");
    if (result.raw.gauge.size() != 1001)
        throw std::invalid_argument(
            "Constrained V2 baseline record requires 1001 gauges");
    const T gauge_max_abs = result.raw.gauge.cwiseAbs().maxCoeff();
    const T gauge_rms = std::sqrt(
        result.raw.gauge.squaredNorm() /
        static_cast<T>(result.raw.gauge.size()));
    const T initial_theta =
        nsConstrainedPrimalInfeasibility(result.constraints);
    std::ostringstream stream;
    stream
        << "PIMFA_NS_CONSTRAINED_MINIMAX_AL_V2_STAGE_A_BASELINE\n"
        << "candidate_id=" << nsConstrainedCandidateId() << "\n"
        << "stage_a_checkpoint_sha256="
        << anchors.stage_a_checkpoint_sha256 << "\n"
        << "stage_a_initializer_role_sha256="
        << nsConstrainedFrozenStageARoleSHA256() << "\n"
        << "stage_a_code_sha256="
        << nsConstrainedFrozenStageACodeSHA256() << "\n"
        << "optimization_protocol_sha256="
        << anchors.optimization_protocol_sha256 << "\n"
        << "implementation_sha256="
        << anchors.implementation_sha256 << "\n"
        << "post_selection_protocol_v5_sha256="
        << anchors.post_selection_protocol_v5_sha256 << "\n"
        << "post_selection_protocol_v4_parent_sha256="
        << anchors.post_selection_protocol_v4_parent_sha256 << "\n"
        << "coefficient_sha256="
        << nsConstrainedCoefficientSHA256(result.coefficients) << "\n"
        << "decoded_handshake_sha256="
        << decoded_handshake_sha256 << "\n"
        << "coefficient_count=" << result.coefficients.size() << "\n"
        << "gauge_count=" << result.raw.gauge.size() << "\n"
        << "data_fit_count=100813\n"
        << "pde_fit_count=101508\n"
        << "integral_fit_time_count=560\n"
        << "initial_condition_count=441\n"
        << "boundary_x_positive_time_count=42000\n"
        << "boundary_y_positive_time_count=42000\n"
        << "bc_mse_definition="
           "sum_x_face_mean_plus_y_face_mean\n"
        << "physics_t_min=0.01\n"
        << "viscosity=0.001\n"
        << "local_integral_partition=4,4\n"
        << "global_integral_partition=1,1\n"
        << "quadrature=trapezoid\n"
        << std::setprecision(std::numeric_limits<T>::max_digits10)
        << "eta=" << result.eta << "\n"
        << "gauge_max_abs=" << gauge_max_abs << "\n"
        << "gauge_rms=" << gauge_rms << "\n"
        << "initial_theta=" << initial_theta << "\n"
        << "known_feasible_start=1\n";
    for (int i = 0; i < NS_V2_METRIC_COUNT; ++i)
        stream << "raw_" << nsConstrainedMetricName(i)
               << "=" << result.raw.metrics[i] << "\n";
    for (int i = 0; i < NS_V2_INEQUALITY_COUNT; ++i)
        stream << "constraint_" << i
               << "=" << result.constraints.inequalities[i] << "\n";
    const std::string body = stream.str();
    return body + "payload_sha256=" +
        nsConstrainedSHA256(body) + "\n";
}

template <typename T, typename Objective>
inline NSConstrainedRunResult<T> nsConstrainedExecute(
    Objective& objective,
    Eigen::Matrix<T, Eigen::Dynamic, 1>& stage_a)
{
    // Defense in depth: the executable preflight checks this before input
    // construction, and the execution entry point checks it again before
    // any baseline or optimizer operation.  The optional core resume parser
    // is deliberately unreachable from the production executable.
    nsConstrainedRequireProductionNoResume();
    NSConstrainedNSFitProvider<T, Objective> provider(objective);
    if (stage_a.size() != provider.coefficientCount())
        throw std::invalid_argument(
            "Constrained V2 Stage-A/provider coefficient mismatch");
    if (provider.coefficientCount() != 60000 ||
        provider.gaugeCount() != 1001 ||
        provider.dataFitCount() != 100813 ||
        provider.pdeFitCount() != 101508 ||
        provider.initialConditionCount() != 441 ||
        provider.boundaryXCount() != 42000 ||
        provider.boundaryYCount() != 42000 ||
        provider.integralFitTimeCount() != 560)
        throw std::runtime_error(
            "Constrained V2 production fit-role dimensions drifted");
    const NSConstrainedArtifactAnchors anchors =
        nsConstrainedExecutableAnchors();
    const NSConstrainedArtifactAnchors baseline_anchors =
        nsConstrainedHistoricalStageABaselineAnchors(anchors);
    const bool baseline_only =
        nsConstrainedStrictBooleanEnvironment(
            "MFA_NS_CONSTRAINED_V2_BASELINE_EVALUATION_ONLY", false);
    if (baseline_only)
    {
        const std::string record_path =
            nsConstrainedRequiredEnvironment(
                "MFA_NS_CONSTRAINED_V2_BASELINE_RECORD_OUT");
        if (nsConstrainedPathExists(record_path))
            throw std::runtime_error(
                "Constrained V2 baseline record must not already exist");
        NSConstrainedRunResult<T> baseline_result =
            nsConstrainedEvaluateStageABaseline(provider, stage_a);
        const std::string decoded_sha =
            provider.decodedHandshakeSHA256(stage_a);
        nsConstrainedAtomicWrite(
            record_path,
            nsConstrainedStageABaselineRecordPayload(
                baseline_result,
                baseline_anchors,
                decoded_sha));
        std::cout
            << "PIMFA_NS_CONSTRAINED_V2_STAGE_A_BASELINE"
            << " candidate_id=" << nsConstrainedCandidateId()
            << " coefficient_sha256="
            << nsConstrainedCoefficientSHA256(stage_a)
            << " decoded_handshake_sha256=" << decoded_sha
            << " optimization_protocol_sha256="
            << baseline_anchors.optimization_protocol_sha256
            << " optimization_protocol_parent_sha256="
            << baseline_anchors.optimization_protocol_parent_sha256
            << " optimization_protocol_ancestor_2_sha256="
            << baseline_anchors.optimization_protocol_ancestor_2_sha256
            << " optimization_protocol_ancestor_3_sha256="
            << baseline_anchors.optimization_protocol_ancestor_3_sha256
            << " inherited_optimization_protocol_sha256="
            << baseline_anchors.inherited_optimization_protocol_sha256
            << " record=" << record_path
            << " record_sha256="
            << nsConstrainedFileSHA256(record_path)
            << "\n";
        return baseline_result;
    }
    const std::string baseline_record_path =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_BASELINE_RECORD_IN");
    const std::string baseline_record_sha =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_BASELINE_RECORD_SHA256");
    if (nsConstrainedFileSHA256(baseline_record_path) !=
        baseline_record_sha)
        throw std::runtime_error(
            "Constrained V2 baseline record changed after preflight");
    const NSConstrainedRunResult<T> recomputed_baseline =
        nsConstrainedEvaluateStageABaseline(provider, stage_a);
    const std::string stage_a_decoded_sha =
        provider.decodedHandshakeSHA256(stage_a);
    const std::string expected_baseline_payload =
        nsConstrainedStageABaselineRecordPayload(
            recomputed_baseline, baseline_anchors, stage_a_decoded_sha);
    if (nsConstrainedReadFileText(baseline_record_path) !=
        expected_baseline_payload)
        throw std::runtime_error(
            "Constrained V2 recomputed Stage-A metrics do not match the "
            "frozen baseline record");

    NSConstrainedRunOptions<T> options;
    options.outer_record_path =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_OUTER_RECORD_OUT");
    options.state_path =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_STATE_OUT");
    options.anchors = anchors;
    options.resume = false;
    const std::string final_path =
        nsConstrainedRequiredEnvironment(
            "MFA_NS_CONSTRAINED_V2_FINAL_CHECKPOINT_OUT");
    if (nsConstrainedPathExists(options.outer_record_path) ||
        nsConstrainedPathExists(options.state_path) ||
        nsConstrainedPathExists(final_path))
        throw std::runtime_error(
            "Constrained V2 production requires fresh primary outputs");

    NSConstrainedRunResult<T> result =
        runNSConstrainedMinimaxAL(provider, stage_a, options);
    const std::string decoded_sha =
        provider.decodedHandshakeSHA256(result.coefficients);
    nsConstrainedWriteFinalCheckpoint(
        final_path,
        result.coefficients,
        result.eta,
        result.converged,
        decoded_sha,
        anchors);
    std::cout
        << "PIMFA_NS_CONSTRAINED_V2_FINAL"
        << " candidate_id=" << nsConstrainedCandidateId()
        << " converged=" << (result.converged ? 1 : 0)
        << " termination=" << result.termination
        << " eta=" << std::setprecision(17) << result.eta
        << " theta=" << result.kkt.theta
        << " projected_stationarity="
        << result.kkt.projected_stationarity
        << " complementarity=" << result.kkt.complementarity
        << " outer_iterations="
        << result.state.completed_outer_iterations
        << " total_inner_iterations="
        << result.state.total_inner_iterations
        << " coefficient_sha256="
        << nsConstrainedCoefficientSHA256(result.coefficients)
        << " decoded_handshake_sha256=" << decoded_sha
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
        << " final_checkpoint=" << final_path
        << "\n";
    return result;
}

} // namespace mfa

#endif

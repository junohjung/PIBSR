#include <mfa/ns_constrained_fit_evaluator.hpp>

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace
{

typedef double Scalar;
typedef Eigen::VectorXd Vector;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
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

std::string temporaryPath(const std::string& suffix)
{
    return std::string("/tmp/pimfa_ns_constrained_v2_evaluator_") +
        std::to_string(static_cast<long long>(getpid())) + suffix;
}

void writeText(const std::string& path, const std::string& text)
{
    std::ofstream stream(
        path.c_str(),
        std::ios::out | std::ios::binary | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("Cannot write evaluator test file");
    stream << text;
    if (!stream)
        throw std::runtime_error("Cannot finish evaluator test file");
}

std::size_t countMarker(
    const std::string& text,
    const std::string& marker)
{
    std::size_t count = 0;
    std::size_t position = 0;
    while ((position = text.find(marker, position)) != std::string::npos)
    {
        ++count;
        position += marker.size();
    }
    return count;
}

mfa::NSConstrainedArtifactAnchors testAnchors()
{
    mfa::NSConstrainedArtifactAnchors anchors;
    anchors.stage_a_checkpoint_sha256 =
        mfa::nsConstrainedFrozenStageASHA256();
    anchors.stage_a_baseline_record_sha256 = std::string(64, 'b');
    anchors.optimization_protocol_sha256 = std::string(64, 'c');
    anchors.optimization_protocol_parent_sha256 = std::string(64, 'e');
    anchors.optimization_protocol_ancestor_2_sha256 = std::string(64, '1');
    anchors.optimization_protocol_ancestor_3_sha256 = std::string(64, '2');
    anchors.inherited_optimization_protocol_sha256 = std::string(64, 'f');
    anchors.implementation_sha256 = std::string(64, 'd');
    anchors.post_selection_protocol_v5_sha256 =
        mfa::nsConstrainedFrozenPostSelectionV5SHA256();
    anchors.post_selection_protocol_v4_parent_sha256 =
        mfa::nsConstrainedFrozenPostSelectionV4ParentSHA256();
    return anchors;
}

class CountingProvider
{
public:
    typedef double Scalar;

    CountingProvider()
        : evaluations_(0),
          evaluated_(false)
    {
    }

    int coefficientCount() const { return 60000; }
    int gaugeCount() const { return 1001; }
    int dataFitCount() const { return 100813; }
    int pdeFitCount() const { return 101508; }
    int initialConditionCount() const { return 441; }
    int boundaryXCount() const { return 42000; }
    int boundaryYCount() const { return 42000; }
    int integralFitTimeCount() const { return 560; }

    mfa::NSConstrainedRawEvaluation<Scalar>
    evaluate(const Vector& coefficients)
    {
        require(
            coefficients.size() == 60000,
            "Counting provider received wrong coefficient count");
        ++evaluations_;
        evaluated_ = true;
        mfa::NSConstrainedRawEvaluation<Scalar> raw;
        for (int i = 0; i < mfa::NS_V2_METRIC_COUNT; ++i)
            raw.metrics[i] = Scalar(i + 1) * 0.75;
        raw.gauge.resize(1001);
        for (int i = 0; i < 1001; ++i)
            raw.gauge(i) = Scalar(i - 500) * 1.0e-9;
        return raw;
    }

    std::string decodedHandshakeSHA256FromCache() const
    {
        if (!evaluated_)
            throw std::logic_error("No counting-provider cache");
        return std::string(64, '7');
    }

    int evaluations() const { return evaluations_; }

private:
    int evaluations_;
    bool evaluated_;
};

void testStrictCheckpointAndSingleEvaluation()
{
    const mfa::NSConstrainedArtifactAnchors anchors = testAnchors();
    Vector coefficients(60000);
    for (Eigen::Index i = 0; i < coefficients.size(); ++i)
        coefficients(i) =
            Scalar((i % 101) - 50) / Scalar(1000);
    const std::string checkpoint_path =
        temporaryPath("_final.ckpt");
    const std::string baseline_path =
        temporaryPath("_baseline.txt");
    std::remove(checkpoint_path.c_str());
    std::remove(baseline_path.c_str());

    mfa::nsConstrainedWriteFinalCheckpoint(
        checkpoint_path,
        coefficients,
        Scalar(3.25),
        false,
        std::string(64, '7'),
        anchors);
    const mfa::NSConstrainedLoadedFinalCheckpoint<Scalar> loaded =
        mfa::nsConstrainedReadFinalCheckpointStrict<Scalar>(
            checkpoint_path, anchors, 60000);
    require(
        (loaded.coefficients - coefficients).isZero(0.0) &&
        loaded.eta == Scalar(3.25) &&
        !loaded.converged &&
        loaded.decoded_handshake_sha256 == std::string(64, '7'),
        "Strict final-checkpoint round trip failed");

    mfa::NSConstrainedRunResult<Scalar> stage_a;
    stage_a.coefficients = coefficients;
    stage_a.eta = Scalar(100);
    stage_a.raw.gauge = Vector::Zero(1001);
    for (int i = 0; i < mfa::NS_V2_METRIC_COUNT; ++i)
        stage_a.raw.metrics[i] = Scalar(i + 1);
    const Scalar positive_roundoff_theta =
        std::numeric_limits<Scalar>::epsilon();
    mfa::nsConstrainedValidateStageAInitialTheta(
        positive_roundoff_theta, positive_roundoff_theta);
    requireThrows(
        [positive_roundoff_theta]() {
            mfa::nsConstrainedValidateStageAInitialTheta(
                Scalar(0), positive_roundoff_theta);
        },
        "Stage-A theta validator accepted an observed/recomputed mismatch");
    requireThrows(
        []() {
            const Scalar too_large =
                Scalar(65) * std::numeric_limits<Scalar>::epsilon();
            mfa::nsConstrainedValidateStageAInitialTheta(
                too_large, too_large);
        },
        "Stage-A theta validator accepted an infeasible positive theta");
    stage_a.baseline =
        mfa::NSConstrainedBaseline<Scalar>::capture(stage_a.raw);
    stage_a.constraints = mfa::nsConstrainedEvaluateConstraints(
        stage_a.raw, stage_a.baseline, stage_a.eta);
    writeText(
        baseline_path,
        mfa::nsConstrainedStageABaselineRecordPayload(
            stage_a,
            anchors,
            std::string(64, '6')));
    const mfa::NSConstrainedBaseline<Scalar> baseline =
        mfa::nsConstrainedReadStageABaselineRecordStrict<Scalar>(
            baseline_path,
            anchors,
            mfa::nsConstrainedCoefficientSHA256(coefficients));

    CountingProvider provider;
    const std::string record =
        mfa::nsConstrainedEvaluateFinalFitOnce(
            provider,
            loaded,
            baseline,
            mfa::nsConstrainedFileSHA256(checkpoint_path));
    require(
        provider.evaluations() == 1,
        "Final fit evaluator did not call provider exactly once");
    require(
        countMarker(record, "\nraw_") == 15 &&
        countMarker(record, "\nconstraint_") == 15 &&
        countMarker(record, "\ngauge_") == 1002 &&
        record.find("\ngauge_count=1001\n") != std::string::npos &&
        record.find("\nprovider_evaluations=1\n") != std::string::npos,
        "Final fit record does not contain the exact metric/constraint/"
        "gauge cardinalities");
    const std::size_t digest_position =
        record.rfind("payload_sha256=");
    require(
        digest_position != std::string::npos &&
        mfa::nsConstrainedSHA256(
            record.substr(0, digest_position)) ==
            record.substr(
                digest_position +
                    std::string("payload_sha256=").size(),
                64),
        "Final fit record payload digest mismatch");

    writeText(
        checkpoint_path,
        mfa::nsConstrainedFinalCheckpointCanonicalPayload(
            coefficients,
            Scalar(3.25),
            false,
            std::string(64, '7'),
            anchors) +
            "trailing\n");
    requireThrows(
        [&checkpoint_path, &anchors]() {
            mfa::nsConstrainedReadFinalCheckpointStrict<Scalar>(
                checkpoint_path, anchors, 60000);
        },
        "Strict final-checkpoint reader accepted trailing content");

    std::remove(checkpoint_path.c_str());
    std::remove(baseline_path.c_str());
}

} // namespace

int main()
{
    try
    {
        testStrictCheckpointAndSingleEvaluation();
        std::cout
            << "PIMFA_NS_CONSTRAINED_V2_FIT_EVALUATOR_TEST PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "PIMFA_NS_CONSTRAINED_V2_FIT_EVALUATOR_TEST FAIL: "
            << error.what() << "\n";
        return 1;
    }
}

#include <mfa/reg_mfa_tikhonov.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace
{
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;

    void Require(bool condition, const std::string& message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    void RequireThrows(
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
        Require(threw, message);
    }

    Vector SolveAugmented(
        const Matrix& data,
        const Vector& target,
        const Matrix& derivative1,
        const Matrix& derivative2,
        double alpha1,
        double alpha2)
    {
        const std::size_t n_train =
            static_cast<std::size_t>(data.rows());
        const int columns1 = alpha1 > 0.0 ? derivative1.rows() : 0;
        const int columns2 = alpha2 > 0.0 ? derivative2.rows() : 0;
        Matrix transpose(
            data.cols(),
            data.rows() + columns1 + columns2);
        transpose.leftCols(data.rows()) = data.transpose();
        int column = data.rows();
        if (alpha1 > 0.0)
        {
            const double scale = mfa::TikhonovColumnScale(
                alpha1,
                n_train,
                static_cast<std::size_t>(derivative1.rows()));
            transpose.middleCols(column, derivative1.rows()) =
                scale * derivative1.transpose();
            column += derivative1.rows();
        }
        if (alpha2 > 0.0)
        {
            const double scale = mfa::TikhonovColumnScale(
                alpha2,
                n_train,
                static_cast<std::size_t>(derivative2.rows()));
            transpose.middleCols(column, derivative2.rows()) =
                scale * derivative2.transpose();
        }
        return (transpose * transpose.transpose()).ldlt().solve(
            data.transpose() * target);
    }

    mfa::BaselineCheckpointMetadata Metadata()
    {
        mfa::BaselineCheckpointMetadata metadata;
        metadata.case_name = "2d_burgers";
        metadata.kind = mfa::BaselineKind::reg_mfa_tikhonov;
        metadata.alpha1 = 1.0e-4;
        metadata.alpha2 = 1.0e-4;
        metadata.derivative_orders = "1,2";
        metadata.derivative_dims = "x,y,t";
        metadata.lattice = "3,2,2";
        metadata.degree = "2,2,2";
        metadata.initializer_role_sha256 =
            std::string(64, 'a');
        metadata.pressure_regime = "not_applicable";
        metadata.field_order = "u,v";
        metadata.coefficient_count_per_field = 3;
        metadata.coefficient_count_total = 6;
        metadata.code_sha256 = std::string(64, 'b');
        return metadata;
    }

    std::string TemporaryPath(const std::string& suffix)
    {
        return std::string("/tmp/pimfa_reg_mfa_tikhonov_") +
            std::to_string(static_cast<long long>(getpid())) +
            suffix;
    }

    void TestAlgebraAndScaling()
    {
        const double scale = mfa::TikhonovColumnScale(0.25, 8, 2);
        Require(scale == 1.0, "sqrt-alpha scaling is not exact");

        Matrix data(5, 3);
        data << 1.0, 0.0, 0.2,
                0.0, 1.0, 0.3,
                1.0, 1.0, 0.4,
                0.4, 0.1, 1.0,
                0.2, 0.7, 0.5;
        Vector target(5);
        target << 1.2, -0.7, 0.4, 1.1, -0.2;
        Matrix derivative1(4, 3);
        derivative1 << 1.0, -1.0, 0.0,
                       0.0, 1.0, -1.0,
                       0.5, 0.0, -0.5,
                       1.0, 0.5, -1.5;
        Matrix derivative2(2, 3);
        derivative2 << 1.0, -2.0, 1.0,
                       0.5, -1.0, 0.5;

        const double alpha1 = 0.3;
        const double alpha2 = 0.08;
        const Vector augmented = SolveAugmented(
            data,
            target,
            derivative1,
            derivative2,
            alpha1,
            alpha2);
        const Matrix normal =
            data.transpose() * data +
            alpha1 * static_cast<double>(data.rows()) /
                static_cast<double>(derivative1.rows()) *
                derivative1.transpose() * derivative1 +
            alpha2 * static_cast<double>(data.rows()) /
                static_cast<double>(derivative2.rows()) *
                derivative2.transpose() * derivative2;
        const Vector independent = normal.ldlt().solve(
            data.transpose() * target);
        Require(
            (augmented - independent).norm() /
                independent.norm() <= 1.0e-12,
            "augmented and independent normal-equation solves disagree");

        const Vector alpha_zero = SolveAugmented(
            data,
            target,
            derivative1,
            derivative2,
            0.0,
            0.0);
        const Vector data_only =
            (data.transpose() * data).ldlt().solve(
                data.transpose() * target);
        Require(
            std::memcmp(
                alpha_zero.data(),
                data_only.data(),
                sizeof(double) *
                    static_cast<std::size_t>(data_only.size())) == 0,
            "alpha-zero path is not byte-identical on the dense fixture");

        const Vector alpha_one = SolveAugmented(
            data,
            target,
            derivative1,
            derivative2,
            1.0,
            1.0);
        Require(
            std::memcmp(
                alpha_zero.data(),
                alpha_one.data(),
                sizeof(double) *
                    static_cast<std::size_t>(alpha_zero.size())) != 0,
            "nonzero regularization did not change coefficients");
        const double roughness_zero =
            (derivative1 * alpha_zero).squaredNorm() /
                derivative1.rows() +
            (derivative2 * alpha_zero).squaredNorm() /
                derivative2.rows();
        const double roughness_one =
            (derivative1 * alpha_one).squaredNorm() /
                derivative1.rows() +
            (derivative2 * alpha_one).squaredNorm() /
                derivative2.rows();
        Require(
            roughness_one <
                roughness_zero * (1.0 - 1.0e-10),
            "nonzero regularization did not lower roughness");
        Require(
            (data * alpha_one - target).squaredNorm() + 1.0e-10 >=
                (data * alpha_zero - target).squaredNorm(),
            "regularization unexpectedly lowered training SSE");
    }

    void TestRoleIsolation()
    {
        Matrix all_rows(4, 2);
        all_rows << 1.0, 0.0,
                    0.0, 1.0,
                    1.0, 1.0,
                    0.5, 0.5;
        Vector targets_a(4);
        targets_a << 1.0, -1.0, 0.2, 4.0;
        Vector targets_b = targets_a;
        targets_b(3) = -9000.0;
        const Matrix training = all_rows.topRows(3);
        const Vector fit_a =
            (training.transpose() * training).ldlt().solve(
                training.transpose() * targets_a.head(3));
        const Vector fit_b =
            (training.transpose() * training).ldlt().solve(
                training.transpose() * targets_b.head(3));
        Require(
            std::memcmp(
                fit_a.data(),
                fit_b.data(),
                sizeof(double) * 2) == 0,
            "validation-only target perturbation changed coefficients");
    }

    void TestCheckpointContract()
    {
        const std::string path = TemporaryPath(".ckpt");
        const std::string mismatch_path =
            TemporaryPath(".mismatch.ckpt");
        const std::string malformed_path =
            TemporaryPath(".malformed.ckpt");
        const mfa::BaselineCheckpointMetadata metadata = Metadata();
        std::vector<double> coefficients;
        coefficients.push_back(0.1);
        coefficients.push_back(-0.2);
        coefficients.push_back(0.3);
        coefficients.push_back(-0.4);
        coefficients.push_back(0.5);
        coefficients.push_back(-0.6);

        mfa::WriteBaselineCheckpointV2(
            path,
            metadata,
            coefficients);
        const std::vector<double> loaded =
            mfa::ReadBaselineCheckpointV2(path, metadata);
        Require(
            loaded.size() == coefficients.size() &&
            std::memcmp(
                loaded.data(),
                coefficients.data(),
                sizeof(double) * coefficients.size()) == 0,
            "V2 checkpoint round trip changed its payload");

        std::vector<mfa::BaselineCheckpointMetadata> mismatches;
        mfa::BaselineCheckpointMetadata changed = metadata;
        changed.case_name = "wrong";
        mismatches.push_back(changed);
        changed = metadata;
        changed.kind = mfa::BaselineKind::data_only;
        mismatches.push_back(changed);
        changed = metadata;
        changed.alpha1 = 2.0e-4;
        mismatches.push_back(changed);
        changed = metadata;
        changed.alpha2 = 2.0e-4;
        mismatches.push_back(changed);
        changed = metadata;
        changed.normalization = "wrong";
        mismatches.push_back(changed);
        changed = metadata;
        changed.derivative_orders = "none";
        mismatches.push_back(changed);
        changed = metadata;
        changed.derivative_dims = "x,y";
        mismatches.push_back(changed);
        changed = metadata;
        changed.lattice = "3,3,2";
        mismatches.push_back(changed);
        changed = metadata;
        changed.degree = "1,2,2";
        mismatches.push_back(changed);
        changed = metadata;
        changed.initializer_role_sha256 = std::string(64, 'c');
        mismatches.push_back(changed);
        changed = metadata;
        changed.pressure_regime = "zero";
        mismatches.push_back(changed);
        changed = metadata;
        changed.field_order = "v,u";
        mismatches.push_back(changed);
        changed = metadata;
        changed.coefficient_count_per_field = 2;
        mismatches.push_back(changed);
        changed = metadata;
        changed.coefficient_count_total = 5;
        mismatches.push_back(changed);
        changed = metadata;
        changed.code_sha256 = std::string(64, 'd');
        mismatches.push_back(changed);
        changed = metadata;
        changed.payload_encoding = "wrong";
        mismatches.push_back(changed);

        for (std::size_t i = 0; i < mismatches.size(); ++i)
            RequireThrows(
                [path, mismatches, i]() {
                    mfa::ReadBaselineCheckpointV2(
                        path,
                        mismatches[i]);
                },
                "V2 checkpoint accepted a metadata mismatch");

        {
            std::ofstream malformed(malformed_path.c_str());
            malformed << "MFA_BURGERS_INITIALIZER_V1 3\n"
                      << "0\n0\n0\n0\n0\n0\n";
        }
        RequireThrows(
            [malformed_path, metadata]() {
                mfa::ReadBaselineCheckpointV2(
                    malformed_path,
                    metadata);
            },
            "V2 loader accepted a V1 checkpoint");

        mfa::WriteBaselineCheckpointV2(
            mismatch_path,
            metadata,
            coefficients);
        {
            std::ofstream extra(
                mismatch_path.c_str(),
                std::ios::app);
            extra << "7\n";
        }
        RequireThrows(
            [mismatch_path, metadata]() {
                mfa::ReadBaselineCheckpointV2(
                    mismatch_path,
                    metadata);
            },
            "V2 loader accepted an extra coefficient");

        {
            std::ifstream original(path.c_str());
            std::ofstream truncated(mismatch_path.c_str());
            std::string line;
            int payload_values = 0;
            bool in_payload = false;
            while (std::getline(original, line))
            {
                if (line == "--coefficients--")
                    in_payload = true;
                if (!in_payload || payload_values < 5)
                    truncated << line << "\n";
                if (in_payload && line != "--coefficients--")
                    ++payload_values;
            }
        }
        RequireThrows(
            [mismatch_path, metadata]() {
                mfa::ReadBaselineCheckpointV2(
                    mismatch_path,
                    metadata);
            },
            "V2 loader accepted a truncated payload");

        std::remove(path.c_str());
        std::remove(mismatch_path.c_str());
        std::remove(malformed_path.c_str());
    }

    void TestConfigurationGuards()
    {
        mfa::RegMFAConfig config;
        config.baseline_requested = true;
        config.initializer_only = true;
        config.kind = mfa::BaselineKind::reg_mfa_tikhonov;
        config.role_sha256 = std::string(64, 'a');
        config.code_sha256 = std::string(64, 'b');
        mfa::ValidateRegMFAConfig(config, 0.0, false);

        mfa::RegMFAConfig bad = config;
        bad.initializer_only = false;
        RequireThrows(
            [bad]() {
                mfa::ValidateRegMFAConfig(bad, 0.0, false);
            },
            "configuration accepted non-initializer Reg-MFA");
        bad = config;
        bad.role_sha256 = std::string(64, 'A');
        RequireThrows(
            [bad]() {
                mfa::ValidateRegMFAConfig(bad, 0.0, false);
            },
            "configuration accepted a malformed role hash");
        bad = config;
        bad.checkpoint_in = "in";
        bad.checkpoint_out = "out";
        RequireThrows(
            [bad]() {
                mfa::ValidateRegMFAConfig(bad, 0.0, false);
            },
            "configuration accepted checkpoint input and output together");
        RequireThrows(
            [config]() {
                mfa::ValidateRegMFAConfig(config, 1.0, false);
            },
            "configuration accepted legacy -b");
        RequireThrows(
            [config]() {
                mfa::ValidateRegMFAConfig(config, 0.0, true);
            },
            "configuration accepted legacy -k");

        bad = config;
        bad.kind = mfa::BaselineKind::data_only;
        bad.alpha1 = 1.0;
        RequireThrows(
            [bad]() {
                mfa::ValidateRegMFAConfig(bad, 0.0, false);
            },
            "data_only accepted a nonzero alpha");

        RequireThrows(
            []() {
                mfa::TikhonovColumnScale(-1.0, 2, 2);
            },
            "negative alpha was accepted");
        RequireThrows(
            []() {
                mfa::TikhonovColumnScale(
                    std::numeric_limits<double>::infinity(),
                    2,
                    2);
            },
            "infinite alpha was accepted");
    }
}

int main()
{
    try
    {
        TestAlgebraAndScaling();
        TestRoleIsolation();
        TestCheckpointContract();
        TestConfigurationGuards();
        std::cout << "Reg-MFA Tikhonov tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Reg-MFA Tikhonov test failure: "
                  << error.what() << "\n";
        return 1;
    }
}

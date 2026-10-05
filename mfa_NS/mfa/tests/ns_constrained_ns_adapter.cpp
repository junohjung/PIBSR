#include <mfa/ns_constrained_minimax_ns_adapter.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace
{

typedef double Scalar;
typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::SparseMatrix<Scalar> Sparse;
typedef Eigen::Triplet<Scalar> Triplet;

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
        throw std::runtime_error(message);
}

template <typename Provider>
void checkAdapterGradients(
    Provider& provider,
    const Vector& coefficients,
    const Vector& direction,
    const std::string& state_label)
{
    const Scalar centered_step = 2.0e-6;
    const Scalar taylor_step_1 = 2.0e-4;
    const Scalar taylor_step_2 = 1.0e-4;
    for (int metric = 0; metric < mfa::NS_V2_METRIC_COUNT; ++metric)
    {
        const Scalar base =
            provider.evaluate(coefficients).metrics[metric];
        const Vector gradient = provider.componentGradient(metric);
        const Scalar directional = gradient.dot(direction);
        const Scalar plus =
            provider.evaluate(
                coefficients + centered_step * direction).metrics[metric];
        const Scalar minus =
            provider.evaluate(
                coefficients - centered_step * direction).metrics[metric];
        const Scalar centered =
            (plus - minus) / (2.0 * centered_step);
        const Scalar tolerance =
            metric >= mfa::NS_V2_LOCAL_MASS ? 2.0e-6 : 2.0e-7;
        requireNear(
            centered, directional, tolerance,
            state_label + " adapter centered gradient mismatch: " +
            mfa::nsConstrainedMetricName(metric));

        const Scalar first_value =
            provider.evaluate(
                coefficients + taylor_step_1 * direction).metrics[metric];
        const Scalar second_value =
            provider.evaluate(
                coefficients + taylor_step_2 * direction).metrics[metric];
        const Scalar first_remainder =
            std::abs(
                first_value - base - taylor_step_1 * directional);
        const Scalar second_remainder =
            std::abs(
                second_value - base - taylor_step_2 * directional);
        require(
            first_remainder > 1.0e-15 &&
            second_remainder > 1.0e-15,
            state_label + " adapter Taylor remainder reached roundoff: " +
            mfa::nsConstrainedMetricName(metric));
        const Scalar slope =
            std::log(first_remainder / second_remainder) /
            std::log(taylor_step_1 / taylor_step_2);
        require(
            std::isfinite(slope) && slope >= 1.7 && slope <= 2.3,
            state_label + " adapter Taylor slope is not second order: " +
            mfa::nsConstrainedMetricName(metric));
    }
}

struct FakePointSet
{
    int npts;
    Matrix domain;
};

Sparse affineRows(const Matrix& coordinates)
{
    std::vector<Triplet> entries;
    entries.reserve(static_cast<size_t>(coordinates.rows()) * 4);
    for (int row = 0; row < coordinates.rows(); ++row)
    {
        entries.push_back(Triplet(row, 0, 1.0));
        if (coordinates(row, 0) != 0.0)
            entries.push_back(Triplet(row, 1, coordinates(row, 0)));
        if (coordinates(row, 1) != 0.0)
            entries.push_back(Triplet(row, 2, coordinates(row, 1)));
        if (coordinates(row, 2) != 0.0)
            entries.push_back(Triplet(row, 3, coordinates(row, 2)));
    }
    Sparse result(coordinates.rows(), 4);
    result.setFromTriplets(entries.begin(), entries.end());
    result.makeCompressed();
    return result;
}

Sparse constantDerivative(int rows, int column)
{
    std::vector<Triplet> entries;
    entries.reserve(rows);
    for (int row = 0; row < rows; ++row)
        entries.push_back(Triplet(row, column, 1.0));
    Sparse result(rows, 4);
    result.setFromTriplets(entries.begin(), entries.end());
    result.makeCompressed();
    return result;
}

struct FakeObjective
{
    Sparse N1, N1t, N1_x, N1t_x, N1_y, N1t_y;
    Sparse N1_t, N1t_t, N1_xx, N1t_xx, N1_yy, N1t_yy;
    Sparse N2, N3;
    Sparse G, Gt;
    Sparse IC, ICt, Bx, Bxt, By, Byt;
    Vector Q, Q2, Q_IC, Q_IC2, Q_BCx, Q_BCx2, Q_BCy, Q_BCy2;
    Vector data_active, pde_active;
    Vector bcx_physics_mask, bcy_physics_mask;
    Scalar data_active_count;
    Scalar pde_active_count;
    Scalar data_validation_count;
    Scalar pde_validation_count;
    Scalar active_bcx_count;
    Scalar active_bcy_count;
    bool physics_time_mask_active;
    double nu;
    mfa::NSIntegralBalanceConfig<Scalar> integral_config;
    std::shared_ptr<mfa::NSIntegralBalanceOperators<Scalar> >
        integral_operators;
    std::shared_ptr<mfa::NSIntegralBalanceOperators<Scalar> >
        global_integral_operators;
    FakePointSet input, input_IC, input_BCx, input_BCy;

    FakeObjective()
        : data_active_count(0.0),
          pde_active_count(0.0),
          data_validation_count(0.0),
          pde_validation_count(0.0),
          active_bcx_count(0.0),
          active_bcy_count(0.0),
          physics_time_mask_active(false),
          nu(0.001)
    {
        const int nx = 21;
        const int ny = 21;
        const int nt = 1001;
        input.npts = nx * ny * nt;
        input.domain.resize(input.npts, 3);
        int row = 0;
        for (int it = 0; it < nt; ++it)
            for (int iy = 0; iy < ny; ++iy)
                for (int ix = 0; ix < nx; ++ix, ++row)
                {
                    input.domain(row, 0) =
                        static_cast<Scalar>(ix) / 20.0;
                    input.domain(row, 1) =
                        static_cast<Scalar>(iy) / 20.0;
                    input.domain(row, 2) =
                        static_cast<Scalar>(it) / 100.0;
                }
        N1 = affineRows(input.domain);
        N1t = N1.transpose();
        N1_x = constantDerivative(input.npts, 1);
        N1t_x = N1_x.transpose();
        N1_y = constantDerivative(input.npts, 2);
        N1t_y = N1_y.transpose();
        N1_t = constantDerivative(input.npts, 3);
        N1t_t = N1_t.transpose();
        N1_xx.resize(input.npts, 4);
        N1t_xx.resize(4, input.npts);
        N1_yy.resize(input.npts, 4);
        N1t_yy.resize(4, input.npts);
        N2.resize(input.npts, 4);
        N3.resize(input.npts, 4);

        std::vector<Triplet> gauge_entries;
        gauge_entries.reserve(input.npts);
        for (int it = 0; it < nt; ++it)
            for (int spatial = 0; spatial < nx * ny; ++spatial)
                gauge_entries.push_back(
                    Triplet(
                        it, it * nx * ny + spatial,
                        1.0 / static_cast<Scalar>(nx * ny)));
        G.resize(nt, input.npts);
        G.setFromTriplets(gauge_entries.begin(), gauge_entries.end());
        G.makeCompressed();
        Gt = G.transpose();

        data_active = Vector::Ones(input.npts);
        pde_active = Vector::Ones(input.npts);
        data_active_count = input.npts;
        pde_active_count = input.npts;

        Vector target_u(4);
        target_u << 0.4, 0.1, -0.2, 0.03;
        Vector target_v(4);
        target_v << -0.1, 0.25, 0.12, -0.02;
        Q = N1 * target_u;
        Q2 = N1 * target_v;

        input_IC.npts = 5;
        input_IC.domain.resize(5, 3);
        input_IC.domain <<
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            0.5, 0.5, 0.0;
        IC = affineRows(input_IC.domain);
        ICt = IC.transpose();
        Q_IC = IC * target_u + Vector::Constant(5, 0.015);
        Q_IC2 = IC * target_v - Vector::Constant(5, 0.02);

        input_BCx.npts = 6;
        input_BCy.npts = 6;
        input_BCx.domain.resize(6, 3);
        input_BCy.domain.resize(6, 3);
        input_BCx.domain <<
            0.0, 0.0, 0.01,
            0.0, 0.5, 0.01,
            0.0, 1.0, 0.01,
            1.0, 0.0, 1.00,
            1.0, 0.5, 1.00,
            1.0, 1.0, 1.00;
        input_BCy.domain <<
            0.0, 0.0, 0.01,
            0.5, 0.0, 0.01,
            1.0, 0.0, 0.01,
            0.0, 1.0, 1.00,
            0.5, 1.0, 1.00,
            1.0, 1.0, 1.00;
        Bx = affineRows(input_BCx.domain);
        Bxt = Bx.transpose();
        By = affineRows(input_BCy.domain);
        Byt = By.transpose();
        Q_BCx = Bx * target_u + Vector::Constant(6, 0.01);
        Q_BCx2 = Bx * target_v - Vector::Constant(6, 0.012);
        Q_BCy = By * target_u - Vector::Constant(6, 0.008);
        Q_BCy2 = By * target_v + Vector::Constant(6, 0.014);
        bcx_physics_mask = Vector::Ones(6);
        bcy_physics_mask = Vector::Ones(6);
        active_bcx_count = 6;
        active_bcy_count = 6;
        integral_config.physics_t_min = 0.01;
    }
};

void runAdapterTest()
{
    const std::string time_path =
        std::string("/tmp/pimfa_ns_constrained_adapter_") +
        std::to_string(static_cast<long long>(getpid())) + ".txt";
    {
        std::ofstream stream(time_path.c_str());
        stream << "1\n500\n";
    }
    setenv("MFA_INT_FIT_TIME_INDEX_FILE", time_path.c_str(), 1);
    FakeObjective objective;
    mfa::NSConstrainedNSFitProvider<Scalar, FakeObjective> provider(
        objective);
    require(provider.coefficientCount() == 12,
            "adapter coefficient count mismatch");
    require(provider.gaugeCount() == 1001,
            "adapter gauge count mismatch");

    Vector coefficients(12);
    coefficients <<
        0.45, 0.18, -0.12, 0.04,
        -0.04, 0.21, 0.17, -0.01,
        0.3, 0.08, -0.06, 0.02;
    const mfa::NSConstrainedRawEvaluation<Scalar> raw =
        provider.evaluate(coefficients);
    for (int metric = 0; metric < mfa::NS_V2_METRIC_COUNT; ++metric)
        require(
            std::isfinite(raw.metrics[metric]) &&
            raw.metrics[metric] > 0.0,
            std::string("adapter raw metric is not positive: ") +
            mfa::nsConstrainedMetricName(metric));
    const Vector coefficient_u = coefficients.segment(0, 4);
    const Vector coefficient_v = coefficients.segment(4, 4);
    const Scalar expected_bc_u =
        (objective.Bx * coefficient_u - objective.Q_BCx).squaredNorm() /
            objective.active_bcx_count +
        (objective.By * coefficient_u - objective.Q_BCy).squaredNorm() /
            objective.active_bcy_count;
    const Scalar expected_bc_v =
        (objective.Bx * coefficient_v - objective.Q_BCx2).squaredNorm() /
            objective.active_bcx_count +
        (objective.By * coefficient_v - objective.Q_BCy2).squaredNorm() /
            objective.active_bcy_count;
    requireNear(
        raw.metrics[mfa::NS_V2_BC_U], expected_bc_u, 1.0e-15,
        "u BC metric is not the frozen sum of facewise means");
    requireNear(
        raw.metrics[mfa::NS_V2_BC_V], expected_bc_v, 1.0e-15,
        "v BC metric is not the frozen sum of facewise means");
    requireNear(
        raw.gauge(1),
        0.3 + 0.5 * 0.08 - 0.5 * 0.06 + 0.02 * 0.01,
        5.0e-14,
        "adapter timewise pressure mean mismatch");

    Vector direction(12);
    direction <<
        0.3, -0.2, 0.1, 0.05,
        -0.1, 0.25, -0.15, 0.08,
        0.2, -0.12, 0.17, -0.04;
    direction.normalize();
    checkAdapterGradients(
        provider, coefficients, direction, "Stage-A");
    Vector perturbation(12);
    perturbation <<
        0.02, -0.01, 0.015, -0.008,
        -0.012, 0.018, 0.006, -0.011,
        0.01, -0.007, 0.009, 0.005;
    checkAdapterGradients(
        provider, coefficients + perturbation, direction, "perturbed");

    provider.evaluate(coefficients);
    Vector gauge_weights(1001);
    for (int i = 0; i < gauge_weights.size(); ++i)
        gauge_weights(i) = 1.0e-4 * ((i % 7) - 3);
    const Vector gauge_adjoint = provider.gaugeAdjoint(gauge_weights);
    const Vector gauge_plus =
        provider.evaluate(coefficients + 2.0e-6 * direction).gauge;
    const Vector gauge_minus =
        provider.evaluate(coefficients - 2.0e-6 * direction).gauge;
    requireNear(
        gauge_weights.dot((gauge_plus - gauge_minus) / (4.0e-6)),
        gauge_adjoint.dot(direction),
        2.0e-10,
        "adapter gauge adjoint identity mismatch");
    unsetenv("MFA_INT_FIT_TIME_INDEX_FILE");
    std::remove(time_path.c_str());
}

} // namespace

int main()
{
    try
    {
        runAdapterTest();
        std::cout << "NS constrained V2 fit-adapter tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "NS constrained V2 fit-adapter test failure: "
                  << error.what() << "\n";
        return 1;
    }
}

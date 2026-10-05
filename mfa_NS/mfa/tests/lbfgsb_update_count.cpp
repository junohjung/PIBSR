#include <LBFGSB.h>

#include <Eigen/Core>

#include <cmath>
#include <iostream>

namespace
{
using Vector = Eigen::VectorXd;

struct Quadratic
{
    double scale;
    double target;

    double operator()(const Vector& x, Vector& gradient) const
    {
        const double displacement = x(0) - target;
        gradient.resize(1);
        gradient(0) = scale * displacement;
        return 0.5 * scale * displacement * displacement;
    }
};

bool close(double lhs, double rhs, double tolerance = 1.0e-14)
{
    return std::abs(lhs - rhs) <= tolerance;
}
}

int main()
{
    const Vector lower = Vector::Constant(1, -100.0);
    const Vector upper = Vector::Constant(1, 100.0);

    // Reproduce the important semantic distinction: LBFGS++ historically
    // returns one from an initial-point relative-gradient exit, even though it
    // has accepted no line-search update.
    {
        LBFGSpp::LBFGSBParam<double> parameters;
        parameters.epsilon = 1.0e-6;
        parameters.epsilon_rel = 1.0e-6;
        parameters.past = 0;
        parameters.max_iterations = 50;
        LBFGSpp::LBFGSBSolver<double> solver(parameters);
        Vector x = Vector::Constant(1, 76.41079149758633);
        const double initial_x = x(0);
        Quadratic objective{3.13e-5 / initial_x, 0.0};
        double value = 0.0;
        const int historical_iterations =
            solver.minimize(objective, x, value, lower, upper);
        if (historical_iterations != 1 || solver.update_count() != 0 ||
            !close(x(0), initial_x) ||
            !(solver.initial_grad_norm() > parameters.epsilon) ||
            !(solver.initial_grad_norm() <=
              parameters.epsilon_rel * std::abs(initial_x)))
        {
            std::cerr << "initial relative-gradient exit was not identified as zero-update\n";
            return 1;
        }
    }

    // With the norm-scaled relative test disabled, a nonstationary initial
    // point must take at least one accepted update and satisfy the unchanged
    // absolute projected-gradient tolerance.
    {
        LBFGSpp::LBFGSBParam<double> parameters;
        parameters.epsilon = 1.0e-6;
        parameters.epsilon_rel = 0.0;
        parameters.past = 0;
        parameters.max_iterations = 50;
        LBFGSpp::LBFGSBSolver<double> solver(parameters);
        Vector x = Vector::Constant(1, 10.0);
        Quadratic objective{1.0, 0.0};
        double value = 0.0;
        solver.minimize(objective, x, value, lower, upper);
        if (solver.update_count() < 1 ||
            !(solver.initial_grad_norm() > parameters.epsilon) ||
            solver.final_grad_norm() > parameters.epsilon)
        {
            std::cerr << "absolute-only solve did not perform and verify an accepted update\n";
            return 1;
        }
    }

    return 0;
}

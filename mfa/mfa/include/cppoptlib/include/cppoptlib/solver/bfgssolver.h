// CppNumericalSolver
#include <iostream>
#include <cmath>
#include <limits>
#include <Eigen/LU>
#include "isolver.h"
#include "../linesearch/morethuente.h"

#ifndef BFGSSOLVER_H_
#define BFGSSOLVER_H_

namespace cppoptlib {

template<typename ProblemType>
class BfgsSolver : public ISolver<ProblemType, 1> {
  public:
    using Superclass = ISolver<ProblemType, 1>;
    using typename Superclass::Scalar;
    using typename Superclass::TVector;
    using typename Superclass::THessian;

    void minimize(ProblemType &objFunc, TVector & x0) {
        const size_t DIM = x0.rows();
        THessian H = THessian::Identity(DIM, DIM);
        TVector grad(DIM);
        TVector x_old = x0;

        this->m_current.reset();
        objFunc.gradient(x0, grad);
        size_t stagnantSteps = 0;

        // Tighter step tolerance => smaller steps required before "too small" exit.
        const Scalar stepTol = Scalar(1e-12);  // you can also try 1e-13 or 1e-14

        do {
            TVector searchDir = -H * grad;

            // check "positive definite"
            Scalar phi = grad.dot(searchDir);
            if ((phi > 0) || (phi != phi)) {
                // reset Hessian approximation
                H = THessian::Identity(DIM, DIM);
                searchDir = -grad;
            }

            // line search
            const Scalar rate =
                MoreThuente<ProblemType, 1>::linesearch(x0, searchDir, objFunc);

            TVector s = rate * searchDir;  // step
            x0 = x0 + s;

            TVector grad_old = grad;
            objFunc.gradient(x0, grad);
            TVector y = grad - grad_old;

            // update diagnostics
            this->m_current.iterations++;
            this->m_current.gradNorm = grad.template lpNorm<Eigen::Infinity>();

            // step-based stopping: only stop if step is tiny *and* gradient already small
            const Scalar stepNormInf =
                (x_old - x0).template lpNorm<Eigen::Infinity>();
            const Scalar parameterScale = std::max(
                Scalar(1), x_old.template lpNorm<Eigen::Infinity>());

            if (stepNormInf <= Scalar(1e-14) * parameterScale &&
                this->m_current.gradNorm > this->m_stop.gradNorm)
                ++stagnantSteps;
            else
                stagnantSteps = 0;

            if (stagnantSteps >= 20) {
                this->m_status = Status::UserDefined;
                break;
            }

            const Scalar ys = y.dot(s);
            if (std::isfinite(ys) &&
                std::abs(ys) > std::numeric_limits<Scalar>::epsilon()) {
                const Scalar rho = Scalar(1) / ys;
                H = H
                  - rho * (s * (y.transpose() * H) + (H * y) * s.transpose())
                  + rho * (rho * y.dot(H * y) + Scalar(1))
                    * (s * s.transpose());
            } else {
                H = THessian::Identity(DIM, DIM);
            }

            if (stepNormInf < stepTol &&
                this->m_current.gradNorm <= this->m_stop.gradNorm) {
                // consider converged; no need to set a special Status enum
                break;
            }

            x_old = x0;

            // main convergence check (gradient, fDelta, xDelta, maxIter, ...)
            this->m_status = checkConvergence(this->m_stop, this->m_current);

        } while (objFunc.callback(this->m_current, x0)
                 && (this->m_status == Status::Continue));
    }

};

} // namespace cppoptlib

#endif /* BFGSSOLVER_H_ */

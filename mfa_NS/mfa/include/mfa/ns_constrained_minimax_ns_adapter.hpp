//--------------------------------------------------------------
// Fit-only Navier--Stokes adapter for ns_constrained_minimax.hpp.
//
// The adapter deliberately accepts the historical objective as an already
// constructed matrix/operator container but never calls its scalar objective.
// It exposes the 15 raw V2 MSEs and one exact dynamically weighted adjoint.
// No validation role is read or evaluated.
//--------------------------------------------------------------

#ifndef MFA_NS_CONSTRAINED_MINIMAX_NS_ADAPTER_HPP
#define MFA_NS_CONSTRAINED_MINIMAX_NS_ADAPTER_HPP

#include <mfa/ns_constrained_minimax.hpp>
#include <mfa/ns_integral_balance.hpp>

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace mfa
{

template <typename T, typename Objective>
class NSConstrainedNSFitProvider
{
public:
    typedef T Scalar;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    explicit NSConstrainedNSFitProvider(Objective& objective)
        : objective_(objective),
          cache_valid_(false)
    {
        if (objective_.N1.cols() <= 0 ||
            objective_.N1.cols() != objective_.N2.cols() ||
            objective_.N1.cols() != objective_.N3.cols())
            throw std::invalid_argument(
                "Constrained V2 requires identical nonempty u/v/p lattices");
        if (objective_.G.rows() != 1001)
            throw std::invalid_argument(
                "Constrained V2 requires exactly 1001 represented times");
        if (!(objective_.data_active_count > T(0)) ||
            !(objective_.pde_active_count > T(0)) ||
            !(objective_.active_bcx_count > T(0)) ||
            !(objective_.active_bcy_count > T(0)) ||
            objective_.input_IC.npts <= 0)
            throw std::invalid_argument(
                "Constrained V2 fit roles contain no active rows");
        if (objective_.data_validation_count != T(0) ||
            objective_.pde_validation_count != T(0) ||
            objective_.integral_operators ||
            objective_.global_integral_operators)
            throw std::invalid_argument(
                "Constrained V2 matrix container must not hold validation "
                "rows or legacy integral operators");

        NSIntegralBalanceConfig<T> local_config;
        local_config.lambda_mass = T(0);
        local_config.lambda_xmom = T(0);
        local_config.lambda_ymom = T(0);
        local_config.parts_x = 4;
        local_config.parts_y = 4;
        local_config.quadrature = "trapezoid";
        local_config.physics_t_min =
            objective_.integral_config.physics_t_min;
        local_config.viscosity =
            static_cast<T>(objective_.nu);
        local_config.diagnostics = true;
        local_config.check_closure = true;
        local_config.closure_tolerance = T(1.0e-12);
        local_config.fit_time_indices = nsIntegralReadIndices(
            nsIntegralRequiredPath("MFA_INT_FIT_TIME_INDEX_FILE"),
            "MFA_INT_FIT_TIME_INDEX_FILE");

        // NSIntegralBalance predates the fit-only V2 executable and owns a
        // two-role container.  Supplying the same *fit* times to its unused
        // second slot constructs the exact fit operator without reading or
        // naming any validation file.  Only .fit is retained below.
        local_config.validation_time_indices =
            local_config.fit_time_indices;
        Matrix physical_coordinates(objective_.input.npts, 3);
        physical_coordinates = objective_.input.domain.leftCols(3);
        const NSIntegralBalanceOperators<T> local_container =
            NSIntegralBalance<T>::build(
                physical_coordinates, local_config);
        local_ = local_container.fit;

        NSIntegralBalanceConfig<T> global_config = local_config;
        global_config.parts_x = 1;
        global_config.parts_y = 1;
        const NSIntegralBalanceOperators<T> global_container =
            NSIntegralBalance<T>::build(
                physical_coordinates, global_config);
        global_ = global_container.fit;

        const int expected_local_rows =
            16 * static_cast<int>(local_config.fit_time_indices.size());
        const int expected_global_rows =
            static_cast<int>(local_config.fit_time_indices.size());
        if (local_.rows() != expected_local_rows ||
            global_.rows() != expected_global_rows)
            throw std::runtime_error(
                "Constrained V2 fit-only integral row count mismatch");
    }

    int coefficientCount() const
    {
        return 3 * objective_.N1.cols();
    }

    int gaugeCount() const
    {
        return objective_.G.rows();
    }

    int dataFitCount() const
    {
        return static_cast<int>(objective_.data_active_count);
    }

    int pdeFitCount() const
    {
        return static_cast<int>(objective_.pde_active_count);
    }

    int initialConditionCount() const
    {
        return objective_.input_IC.npts;
    }

    int boundaryXCount() const
    {
        return static_cast<int>(objective_.active_bcx_count);
    }

    int boundaryYCount() const
    {
        return static_cast<int>(objective_.active_bcy_count);
    }

    int integralFitTimeCount() const
    {
        return global_.rows();
    }

    NSConstrainedRawEvaluation<T> evaluate(const Vector& coefficients)
    {
        if (coefficients.size() != coefficientCount())
            throw std::invalid_argument(
                "Constrained V2 coefficient vector has the wrong size");
        if (!coefficients.allFinite())
            throw std::invalid_argument(
                "Constrained V2 coefficient vector contains a nonfinite value");
        const int count = objective_.N1.cols();
        cache_.cu = coefficients.segment(0, count);
        cache_.cv = coefficients.segment(count, count);
        cache_.cp = coefficients.segment(2 * count, count);

        cache_.u = objective_.N1 * cache_.cu;
        cache_.ut = objective_.N1_t * cache_.cu;
        cache_.ux = objective_.N1_x * cache_.cu;
        cache_.uy = objective_.N1_y * cache_.cu;
        cache_.ulap =
            objective_.N1_xx * cache_.cu +
            objective_.N1_yy * cache_.cu;
        cache_.v = objective_.N1 * cache_.cv;
        cache_.vt = objective_.N1_t * cache_.cv;
        cache_.vx = objective_.N1_x * cache_.cv;
        cache_.vy = objective_.N1_y * cache_.cv;
        cache_.vlap =
            objective_.N1_xx * cache_.cv +
            objective_.N1_yy * cache_.cv;
        cache_.p = objective_.N1 * cache_.cp;
        cache_.px = objective_.N1_x * cache_.cp;
        cache_.py = objective_.N1_y * cache_.cp;

        cache_.ru =
            cache_.ut +
            cache_.u.cwiseProduct(cache_.ux) +
            cache_.v.cwiseProduct(cache_.uy) +
            cache_.px -
            static_cast<T>(objective_.nu) * cache_.ulap;
        cache_.rv =
            cache_.vt +
            cache_.u.cwiseProduct(cache_.vx) +
            cache_.v.cwiseProduct(cache_.vy) +
            cache_.py -
            static_cast<T>(objective_.nu) * cache_.vlap;
        cache_.rc = cache_.ux + cache_.vy;
        cache_.data_u =
            (cache_.u - objective_.Q).cwiseProduct(objective_.data_active);
        cache_.data_v =
            (cache_.v - objective_.Q2).cwiseProduct(objective_.data_active);
        cache_.strong_x =
            cache_.ru.cwiseProduct(objective_.pde_active);
        cache_.strong_y =
            cache_.rv.cwiseProduct(objective_.pde_active);
        cache_.strong_c =
            cache_.rc.cwiseProduct(objective_.pde_active);
        cache_.ic_u = objective_.IC * cache_.cu - objective_.Q_IC;
        cache_.ic_v = objective_.IC * cache_.cv - objective_.Q_IC2;
        cache_.bcx_u = objective_.Bx * cache_.cu - objective_.Q_BCx;
        cache_.bcy_u = objective_.By * cache_.cu - objective_.Q_BCy;
        cache_.bcx_v = objective_.Bx * cache_.cv - objective_.Q_BCx2;
        cache_.bcy_v = objective_.By * cache_.cv - objective_.Q_BCy2;
        if (objective_.physics_time_mask_active)
        {
            cache_.bcx_u =
                cache_.bcx_u.cwiseProduct(objective_.bcx_physics_mask);
            cache_.bcy_u =
                cache_.bcy_u.cwiseProduct(objective_.bcy_physics_mask);
            cache_.bcx_v =
                cache_.bcx_v.cwiseProduct(objective_.bcx_physics_mask);
            cache_.bcy_v =
                cache_.bcy_v.cwiseProduct(objective_.bcy_physics_mask);
        }
        cache_.gauge = objective_.G * cache_.p;

        const NSIntegralBalanceEvaluation<T> local_values =
            NSIntegralBalance<T>::evaluate(
                local_,
                cache_.u, cache_.ut, cache_.ux, cache_.uy,
                cache_.v, cache_.vt, cache_.vx, cache_.vy, cache_.p,
                static_cast<T>(objective_.nu),
                T(0), T(0), T(0), false);
        const NSIntegralBalanceEvaluation<T> global_values =
            NSIntegralBalance<T>::evaluate(
                global_,
                cache_.u, cache_.ut, cache_.ux, cache_.uy,
                cache_.v, cache_.vt, cache_.vx, cache_.vy, cache_.p,
                static_cast<T>(objective_.nu),
                T(0), T(0), T(0), false);

        NSConstrainedRawEvaluation<T> raw;
        raw.metrics[NS_V2_DATA_U] =
            cache_.data_u.squaredNorm() / objective_.data_active_count;
        raw.metrics[NS_V2_DATA_V] =
            cache_.data_v.squaredNorm() / objective_.data_active_count;
        raw.metrics[NS_V2_IC_U] =
            cache_.ic_u.squaredNorm() /
            static_cast<T>(objective_.input_IC.npts);
        raw.metrics[NS_V2_IC_V] =
            cache_.ic_v.squaredNorm() /
            static_cast<T>(objective_.input_IC.npts);

        const T count_bcx = objective_.physics_time_mask_active
            ? objective_.active_bcx_count
            : static_cast<T>(objective_.input_BCx.npts);
        const T count_bcy = objective_.physics_time_mask_active
            ? objective_.active_bcy_count
            : static_cast<T>(objective_.input_BCy.npts);
        // This is the frozen historical boundary MSE definition used by
        // ObjectiveFunction_LBFGS: one mean over the prescribed x-normal
        // faces plus one mean over the prescribed y-normal faces.  It is not
        // a pooled mean over the concatenated (and corner-duplicated) rows.
        raw.metrics[NS_V2_BC_U] =
            cache_.bcx_u.squaredNorm() / count_bcx +
            cache_.bcy_u.squaredNorm() / count_bcy;
        raw.metrics[NS_V2_BC_V] =
            cache_.bcx_v.squaredNorm() / count_bcx +
            cache_.bcy_v.squaredNorm() / count_bcy;
        raw.metrics[NS_V2_STRONG_X] =
            cache_.strong_x.squaredNorm() / objective_.pde_active_count;
        raw.metrics[NS_V2_STRONG_Y] =
            cache_.strong_y.squaredNorm() / objective_.pde_active_count;
        raw.metrics[NS_V2_STRONG_C] =
            cache_.strong_c.squaredNorm() / objective_.pde_active_count;
        raw.metrics[NS_V2_LOCAL_MASS] = local_values.mse_mass;
        raw.metrics[NS_V2_LOCAL_X] = local_values.mse_xmom;
        raw.metrics[NS_V2_LOCAL_Y] = local_values.mse_ymom;
        raw.metrics[NS_V2_GLOBAL_MASS] = global_values.mse_mass;
        raw.metrics[NS_V2_GLOBAL_X] = global_values.mse_xmom;
        raw.metrics[NS_V2_GLOBAL_Y] = global_values.mse_ymom;
        raw.gauge = cache_.gauge;
        for (int i = 0; i < NS_V2_METRIC_COUNT; ++i)
            if (!std::isfinite(raw.metrics[i]) || raw.metrics[i] < T(0))
                throw std::runtime_error(
                    std::string("Nonfinite/negative constrained V2 raw MSE: ") +
                    nsConstrainedMetricName(i));
        if (!raw.gauge.allFinite())
            throw std::runtime_error(
                "Nonfinite constrained V2 pressure gauge");
        cache_valid_ = true;
        return raw;
    }

    Vector weightedGradient(
        const std::array<T, NS_V2_METRIC_COUNT>& weights,
        const Vector& gauge_weights)
    {
        if (!cache_valid_)
            throw std::logic_error(
                "Constrained V2 weightedGradient requires evaluate first");
        if (gauge_weights.size() != gaugeCount() ||
            !gauge_weights.allFinite())
            throw std::invalid_argument(
                "Constrained V2 gauge-adjoint weight mismatch");
        for (int i = 0; i < NS_V2_METRIC_COUNT; ++i)
            if (!std::isfinite(weights[i]))
                throw std::invalid_argument(
                    "Constrained V2 metric weight is nonfinite");

        const int count = objective_.N1.cols();
        Vector grad_u = Vector::Zero(count);
        Vector grad_v = Vector::Zero(count);
        Vector grad_p = Vector::Zero(count);
        const T inverse_data = T(1) / objective_.data_active_count;
        const T inverse_pde = T(1) / objective_.pde_active_count;
        const T inverse_ic = T(1) /
            static_cast<T>(objective_.input_IC.npts);
        const T count_bcx = objective_.physics_time_mask_active
            ? objective_.active_bcx_count
            : static_cast<T>(objective_.input_BCx.npts);
        const T count_bcy = objective_.physics_time_mask_active
            ? objective_.active_bcy_count
            : static_cast<T>(objective_.input_BCy.npts);

        grad_u.noalias() +=
            (T(2) * weights[NS_V2_DATA_U] * inverse_data) *
            (objective_.N1t * cache_.data_u);
        grad_v.noalias() +=
            (T(2) * weights[NS_V2_DATA_V] * inverse_data) *
            (objective_.N1t * cache_.data_v);
        grad_u.noalias() +=
            (T(2) * weights[NS_V2_IC_U] * inverse_ic) *
            (objective_.ICt * cache_.ic_u);
        grad_v.noalias() +=
            (T(2) * weights[NS_V2_IC_V] * inverse_ic) *
            (objective_.ICt * cache_.ic_v);
        grad_u.noalias() +=
            T(2) * weights[NS_V2_BC_U] *
            ((objective_.Bxt * cache_.bcx_u) / count_bcx +
             (objective_.Byt * cache_.bcy_u) / count_bcy);
        grad_v.noalias() +=
            T(2) * weights[NS_V2_BC_V] *
            ((objective_.Bxt * cache_.bcx_v) / count_bcx +
             (objective_.Byt * cache_.bcy_v) / count_bcy);

        const Vector zx =
            (T(2) * weights[NS_V2_STRONG_X] * inverse_pde) *
            cache_.strong_x;
        const Vector zy =
            (T(2) * weights[NS_V2_STRONG_Y] * inverse_pde) *
            cache_.strong_y;
        const Vector zc =
            (T(2) * weights[NS_V2_STRONG_C] * inverse_pde) *
            cache_.strong_c;
        grad_u.noalias() +=
            objective_.N1t_t * zx +
            objective_.N1t * cache_.ux.cwiseProduct(zx) +
            objective_.N1t_x * cache_.u.cwiseProduct(zx) +
            objective_.N1t_y * cache_.v.cwiseProduct(zx) -
            static_cast<T>(objective_.nu) *
                (objective_.N1t_xx + objective_.N1t_yy) * zx +
            objective_.N1t * cache_.vx.cwiseProduct(zy) +
            objective_.N1t_x * zc;
        grad_v.noalias() +=
            objective_.N1t * cache_.uy.cwiseProduct(zx) +
            objective_.N1t_t * zy +
            objective_.N1t_x * cache_.u.cwiseProduct(zy) +
            objective_.N1t * cache_.vy.cwiseProduct(zy) +
            objective_.N1t_y * cache_.v.cwiseProduct(zy) -
            static_cast<T>(objective_.nu) *
                (objective_.N1t_xx + objective_.N1t_yy) * zy +
            objective_.N1t_y * zc;
        grad_p.noalias() +=
            objective_.N1t_x * zx +
            objective_.N1t_y * zy;

        const bool local_active =
            weights[NS_V2_LOCAL_MASS] != T(0) ||
            weights[NS_V2_LOCAL_X] != T(0) ||
            weights[NS_V2_LOCAL_Y] != T(0);
        if (local_active)
        {
            const NSIntegralBalanceEvaluation<T> integral =
                NSIntegralBalance<T>::evaluate(
                    local_,
                    cache_.u, cache_.ut, cache_.ux, cache_.uy,
                    cache_.v, cache_.vt, cache_.vx, cache_.vy, cache_.p,
                    static_cast<T>(objective_.nu),
                    weights[NS_V2_LOCAL_MASS],
                    weights[NS_V2_LOCAL_X],
                    weights[NS_V2_LOCAL_Y],
                    true);
            addIntegralAdjoint(integral, grad_u, grad_v, grad_p);
        }
        const bool global_active =
            weights[NS_V2_GLOBAL_MASS] != T(0) ||
            weights[NS_V2_GLOBAL_X] != T(0) ||
            weights[NS_V2_GLOBAL_Y] != T(0);
        if (global_active)
        {
            const NSIntegralBalanceEvaluation<T> integral =
                NSIntegralBalance<T>::evaluate(
                    global_,
                    cache_.u, cache_.ut, cache_.ux, cache_.uy,
                    cache_.v, cache_.vt, cache_.vx, cache_.vy, cache_.p,
                    static_cast<T>(objective_.nu),
                    weights[NS_V2_GLOBAL_MASS],
                    weights[NS_V2_GLOBAL_X],
                    weights[NS_V2_GLOBAL_Y],
                    true);
            addIntegralAdjoint(integral, grad_u, grad_v, grad_p);
        }
        grad_p.noalias() +=
            objective_.N1t * (objective_.Gt * gauge_weights);

        Vector gradient(coefficientCount());
        gradient.segment(0, count) = grad_u;
        gradient.segment(count, count) = grad_v;
        gradient.segment(2 * count, count) = grad_p;
        if (!gradient.allFinite())
            throw std::runtime_error(
                "Nonfinite constrained V2 coefficient adjoint");
        return gradient;
    }

    Vector componentGradient(int metric)
    {
        if (metric < 0 || metric >= NS_V2_METRIC_COUNT)
            throw std::out_of_range(
                "Constrained V2 component-gradient metric index");
        std::array<T, NS_V2_METRIC_COUNT> weights;
        weights.fill(T(0));
        weights[metric] = T(1);
        return weightedGradient(weights, Vector::Zero(gaugeCount()));
    }

    Vector gaugeAdjoint(const Vector& gauge_weights)
    {
        std::array<T, NS_V2_METRIC_COUNT> weights;
        weights.fill(T(0));
        return weightedGradient(weights, gauge_weights);
    }

    std::string decodedHandshakeSHA256(const Vector& coefficients)
    {
        evaluate(coefficients);
        return decodedHandshakeSHA256FromCache();
    }

    std::string decodedHandshakeSHA256FromCache() const
    {
        if (!cache_valid_)
            throw std::logic_error(
                "Constrained V2 decoded handshake requires evaluate first");
        const Eigen::Index middle = cache_.u.size() / 2;
        const Eigen::Index last = cache_.u.size() - 1;
        std::ostringstream stream;
        stream
            << "# PIMFA_NS_CONSTRAINED_V2_DECODED_HANDSHAKE rows=0,"
            << middle << "," << last << "\n"
            << std::setprecision(std::numeric_limits<T>::max_digits10);
        const Eigen::Index rows[3] = {0, middle, last};
        for (int i = 0; i < 3; ++i)
            stream << cache_.u(rows[i]) << " "
                   << cache_.v(rows[i]) << " "
                   << cache_.p(rows[i]) << "\n";
        return nsConstrainedSHA256(stream.str());
    }

private:
    struct Cache
    {
        Vector cu, cv, cp;
        Vector u, ut, ux, uy, ulap;
        Vector v, vt, vx, vy, vlap;
        Vector p, px, py;
        Vector ru, rv, rc;
        Vector data_u, data_v;
        Vector strong_x, strong_y, strong_c;
        Vector ic_u, ic_v;
        Vector bcx_u, bcy_u, bcx_v, bcy_v;
        Vector gauge;
    };

    Objective& objective_;
    NSIntegralBlockOperators<T> local_;
    NSIntegralBlockOperators<T> global_;
    Cache cache_;
    bool cache_valid_;

    void addIntegralAdjoint(
        const NSIntegralBalanceEvaluation<T>& integral,
        Vector& grad_u,
        Vector& grad_v,
        Vector& grad_p)
    {
        grad_u.noalias() +=
            objective_.N1t * integral.adjoints.u +
            objective_.N1t_t * integral.adjoints.ut +
            objective_.N1t_x * integral.adjoints.ux +
            objective_.N1t_y * integral.adjoints.uy;
        grad_v.noalias() +=
            objective_.N1t * integral.adjoints.v +
            objective_.N1t_t * integral.adjoints.vt +
            objective_.N1t_x * integral.adjoints.vx +
            objective_.N1t_y * integral.adjoints.vy;
        grad_p.noalias() += objective_.N1t * integral.adjoints.p;
    }
};

} // namespace mfa

#endif

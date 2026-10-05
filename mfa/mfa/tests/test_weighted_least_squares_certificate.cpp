#include <mfa/weighted_least_squares_certificate.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

typedef double Real;
typedef mfa::WeightedLeastSquaresCertificate<Real> Certificate;
typedef Certificate::SparseMatrix SparseMatrix;
typedef Certificate::Vector Vector;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void requireNear(
    Real actual,
    Real expected,
    Real tolerance,
    const std::string& message)
{
    const Real scale = std::max<Real>(
        Real(1), std::max(std::abs(actual), std::abs(expected)));
    if (std::abs(actual - expected) > tolerance * scale)
    {
        std::ostringstream detail;
        detail.precision(17);
        detail << message << ": actual=" << actual
               << " expected=" << expected;
        throw std::runtime_error(detail.str());
    }
}

SparseMatrix sparse(const Eigen::MatrixXd& dense)
{
    SparseMatrix result = dense.sparseView();
    result.makeCompressed();
    return result;
}

void requireBlockEquals(
    const Certificate::System& system,
    std::size_t block_index,
    const SparseMatrix& source,
    const Vector& rhs,
    Real scale)
{
    const Certificate::BlockRecord& record = system.blocks[block_index];
    const Eigen::MatrixXd actual = Eigen::MatrixXd(
        system.A.middleRows(record.row_offset, record.rows));
    const Eigen::MatrixXd expected = scale * Eigen::MatrixXd(source);
    require((actual - expected).norm() < 2.0e-15,
            "augmented block matrix has the wrong scaling");
    require((system.b.segment(record.row_offset, record.rows) - scale * rhs)
                .norm() < 2.0e-15,
            "augmented block rhs has the wrong scaling");
}

void testFullRankAgainstDenseReference()
{
    Eigen::MatrixXd data_dense(3, 3);
    data_dense << 1.0, 0.0, 0.5,
                  0.0, 2.0, 0.0,
                  1.0, 1.0, 1.0;
    Eigen::MatrixXd pde_dense(2, 3);
    pde_dense << 1.0, -1.0, 0.0,
                 0.0, 1.0, -1.0;
    Eigen::MatrixXd ic_dense(1, 3);
    ic_dense << 1.0, 0.0, 0.0;
    Eigen::MatrixXd bc_dense(2, 3);
    bc_dense << 0.0, 1.0, 0.0,
                0.0, 0.0, 1.0;
    Eigen::MatrixXd integral_dense(2, 3);
    integral_dense << 1.0, 1.0, 0.0,
                      0.0, 1.0, 1.0;

    const SparseMatrix data = sparse(data_dense);
    const SparseMatrix pde = sparse(pde_dense);
    const SparseMatrix ic = sparse(ic_dense);
    const SparseMatrix bc = sparse(bc_dense);
    const SparseMatrix integral = sparse(integral_dense);
    Vector q_data(3); q_data << 1.0, -0.5, 0.75;
    Vector q_pde(2); q_pde << 0.1, -0.2;
    Vector q_ic(1); q_ic << 0.9;
    Vector q_bc(2); q_bc << -0.3, 0.4;
    Vector q_integral(2); q_integral << 0.0, 0.0;

    const Real weights[] = {2.0, 0.5, 3.0, 0.25, 1.2};
    std::vector<Certificate::Block> blocks;
    blocks.push_back(Certificate::Block("data", data, q_data, weights[0]));
    blocks.push_back(Certificate::Block("pde", pde, q_pde, weights[1]));
    blocks.push_back(Certificate::Block("ic", ic, q_ic, weights[2]));
    blocks.push_back(Certificate::Block("bc", bc, q_bc, weights[3]));
    blocks.push_back(Certificate::Block(
        "integral", integral, q_integral, weights[4]));

    const Certificate::System system = Certificate::assemble(blocks);
    require(system.A.rows() == 10 && system.A.cols() == 3,
            "full-rank augmented system has incorrect dimensions");
    require(system.blocks.size() == 5,
            "full-rank augmented system lost a block");
    Eigen::Index expected_offset = 0;
    for (std::size_t index = 0; index < system.blocks.size(); ++index)
    {
        require(system.blocks[index].row_offset == expected_offset,
                "block row offsets are not contiguous");
        expected_offset += system.blocks[index].rows;
    }

    requireBlockEquals(
        system, 0, data, q_data, std::sqrt(weights[0] / 3.0));
    requireBlockEquals(
        system, 1, pde, q_pde, std::sqrt(weights[1] / 2.0));
    requireBlockEquals(
        system, 2, ic, q_ic, std::sqrt(weights[2] / 1.0));
    requireBlockEquals(
        system, 3, bc, q_bc, std::sqrt(weights[3] / 2.0));
    requireBlockEquals(
        system, 4, integral, q_integral,
        std::sqrt(weights[4] / 2.0));

    const Eigen::MatrixXd dense_A(system.A);
    const Vector dense_reference =
        dense_A.colPivHouseholderQr().solve(system.b);
    Vector primary = dense_reference;
    primary(0) += 2.0e-3;
    primary(1) -= 1.0e-3;
    primary(2) += 3.0e-3;
    const Certificate::Result result =
        Certificate::solveAndCompare(system, primary);

    require(result.solver_status == "success", "SparseQR did not succeed");
    require(result.numerical_rank == 3 && !result.rank_deficient,
            "full-rank system was classified incorrectly");
    require((result.qr_coefficients - dense_reference).norm() < 2.0e-12,
            "SparseQR disagrees with the dense full-rank reference");
    require(result.primary_minus_qr_objective > 0.0,
            "perturbed primary objective did not exceed the QR optimum");
    require(result.primary_raw_gradient_inf > result.qr_raw_gradient_inf,
            "raw-gradient diagnostic did not distinguish the perturbed point");
    require(result.qr_kkt_backward_error < 2.0e-14,
            "full-rank SparseQR KKT backward error is too large");
    require(result.coefficient_relative_l2 > 0.0,
            "coefficient comparison was not reported");
}

void testRankDeficientAgainstDenseReference()
{
    Eigen::MatrixXd dense(5, 3);
    dense << 1.0, 0.0, 1.0,
             0.0, 1.0, 1.0,
             1.0, 1.0, 2.0,
             2.0, -1.0, 1.0,
             -1.0, 2.0, 1.0;
    const SparseMatrix matrix = sparse(dense);
    Vector rhs(5); rhs << 1.0, -2.0, 0.4, 1.5, -0.7;
    std::vector<Certificate::Block> blocks;
    blocks.push_back(Certificate::Block("rank_deficient", matrix, rhs, 1.0));
    const Certificate::System system = Certificate::assemble(blocks);

    const Vector dense_reference = Eigen::JacobiSVD<Eigen::MatrixXd>(
        Eigen::MatrixXd(system.A), Eigen::ComputeThinU | Eigen::ComputeThinV)
        .solve(system.b);
    const Certificate::Result result =
        Certificate::solveAndCompare(system, dense_reference);
    const Real dense_objective =
        (system.A * dense_reference - system.b).squaredNorm();

    require(result.numerical_rank == 2 && result.rank_deficient,
            "rank-deficient system was not detected");
    requireNear(result.qr_objective, dense_objective, 2.0e-13,
                "SparseQR rank-deficient objective disagrees with dense SVD");
    require((system.A * result.qr_coefficients -
             system.A * dense_reference).norm() < 2.0e-12,
            "rank-deficient SparseQR and dense SVD fits disagree");
    require(result.qr_kkt_backward_error < 2.0e-14,
            "rank-deficient SparseQR KKT backward error is too large");
}

void testZeroWeightRowsAndDeterministicExports()
{
    const SparseMatrix active = sparse(Eigen::MatrixXd::Identity(2, 2));
    Eigen::MatrixXd zero_dense(3, 2);
    zero_dense << 7.0, -3.0,
                  2.0, 5.0,
                  -4.0, 9.0;
    const SparseMatrix zero_weight = sparse(zero_dense);
    Vector active_rhs(2); active_rhs << 1.0, -2.0;
    Vector ignored_rhs(3); ignored_rhs << 11.0, -13.0, 17.0;
    std::vector<Certificate::Block> blocks;
    blocks.push_back(Certificate::Block("data", active, active_rhs, 2.0));
    blocks.push_back(Certificate::Block(
        "integral", zero_weight, ignored_rhs, 0.0));
    const Certificate::System system = Certificate::assemble(blocks);

    require(system.A.rows() == 5 && system.A.cols() == 2,
            "zero-weight block rows were omitted");
    require(system.blocks[1].row_offset == 2 &&
            system.blocks[1].rows == 3,
            "zero-weight block row range is incorrect");
    require(system.blocks[1].row_scale == 0.0 &&
            system.blocks[1].augmented_nnz == 0,
            "zero-weight block is not structurally zero");
    require(Eigen::MatrixXd(system.A.bottomRows(3)).norm() == 0.0,
            "zero-weight block contributed matrix entries");
    require(system.b.tail(3).norm() == 0.0,
            "zero-weight block contributed rhs entries");

    const Certificate::Result result =
        Certificate::solveAndCompare(system, active_rhs);
    require((result.qr_coefficients - active_rhs).norm() < 2.0e-14,
            "zero-weight rows changed the active solution");

    std::ostringstream matrix_first;
    std::ostringstream matrix_second;
    Certificate::writeMatrixMarket(matrix_first, system.A);
    Certificate::writeMatrixMarket(matrix_second, system.A);
    require(matrix_first.str() == matrix_second.str(),
            "MatrixMarket export is not byte deterministic");
    require(matrix_first.str().find(
        "ordering: compressed-column-major, then row") != std::string::npos,
        "MatrixMarket export does not declare its ordering");

    std::ostringstream rhs_first;
    std::ostringstream rhs_second;
    Certificate::writeVectorText(rhs_first, system.b, "augmented_rhs");
    Certificate::writeVectorText(rhs_second, system.b, "augmented_rhs");
    require(rhs_first.str() == rhs_second.str(),
            "rhs export is not byte deterministic");

    std::ostringstream certificate_first;
    std::ostringstream certificate_second;
    Certificate::writeCertificateText(certificate_first, system, result);
    Certificate::writeCertificateText(certificate_second, system, result);
    require(certificate_first.str() == certificate_second.str(),
            "certificate export is not byte deterministic");
    require(certificate_first.str().find(
        "format=mfa_augmented_sparse_qr_certificate_v1") !=
            std::string::npos,
        "certificate format tag is missing");
    require(certificate_first.str().find(
        "block_1_augmented_nnz=0") != std::string::npos,
        "certificate does not expose the zero-weight row convention");

    bool negative_rejected = false;
    try
    {
        std::vector<Certificate::Block> invalid;
        invalid.push_back(Certificate::Block(
            "negative", active, active_rhs, -1.0));
        Certificate::assemble(invalid);
    }
    catch (const std::invalid_argument&)
    {
        negative_rejected = true;
    }
    require(negative_rejected, "negative block weight was accepted");
}

} // namespace

int main()
{
    try
    {
        testFullRankAgainstDenseReference();
        testRankDeficientAgainstDenseReference();
        testZeroWeightRowsAndDeterministicExports();
        std::cout << "weighted_least_squares_certificate_tests=PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "weighted_least_squares_certificate_tests=FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}

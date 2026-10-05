//--------------------------------------------------------------
// Direct sparse-QR certificate for an augmented weighted
// least-squares objective.
//
// If block k represents ||M_k p - q_k||_2^2 with objective weight
// lambda_k / n_k, the augmented rows are
//
//   A_k = sqrt(lambda_k / n_k) M_k,
//   b_k = sqrt(lambda_k / n_k) q_k.
//
// The primary solution is obtained from the rectangular A directly
// with Eigen::SparseQR.  A^T A is formed only after that solve, and
// only to report the requested KKT backward-error diagnostics.
//--------------------------------------------------------------

#ifndef MFA_WEIGHTED_LEAST_SQUARES_CERTIFICATE_HPP
#define MFA_WEIGHTED_LEAST_SQUARES_CERTIFICATE_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfa
{

template <typename T>
class WeightedLeastSquaresCertificate
{
public:
    typedef Eigen::SparseMatrix<T, Eigen::ColMajor> SparseMatrix;
    typedef typename SparseMatrix::StorageIndex StorageIndex;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Triplet<T> Triplet;

    // Non-owning view.  The matrix and rhs must outlive assemble().
    struct Block
    {
        std::string name;
        const SparseMatrix* matrix;
        const Vector* rhs;
        T lambda;

        Block(
            const std::string& name_,
            const SparseMatrix& matrix_,
            const Vector& rhs_,
            T lambda_)
            : name(name_), matrix(&matrix_), rhs(&rhs_), lambda(lambda_)
        {
        }
    };

    struct BlockRecord
    {
        std::string name;
        Eigen::Index row_offset;
        Eigen::Index rows;
        Eigen::Index input_nnz;
        Eigen::Index augmented_nnz;
        T lambda;
        T row_scale;

        BlockRecord()
            : row_offset(0), rows(0), input_nnz(0), augmented_nnz(0),
              lambda(T(0)), row_scale(T(0))
        {
        }
    };

    struct System
    {
        SparseMatrix A;
        Vector b;
        std::vector<BlockRecord> blocks;
    };

    struct Options
    {
        // A negative value requests the formula used by Eigen::SparseQR:
        // 20 (m+n) max_j ||A(:,j)||_2 epsilon.  The resulting value is
        // explicitly set on the solver so it can be reported and frozen.
        T pivot_threshold;

        Options() : pivot_threshold(T(-1)) {}
    };

    struct Result
    {
        std::string solver;
        std::string solver_status;
        Eigen::Index rows;
        Eigen::Index cols;
        Eigen::Index nnz;
        Eigen::Index numerical_rank;
        bool rank_deficient;
        T pivot_threshold;
        Vector qr_coefficients;

        T qr_objective;
        T primary_objective;
        T primary_minus_qr_objective;
        T objective_gap_absolute;
        T objective_gap_relative;
        T coefficient_relative_l2;
        T coefficient_relative_l2_denominator;

        T primary_raw_gradient_inf;
        T qr_raw_gradient_inf;
        T primary_kkt_numerator_inf;
        T qr_kkt_numerator_inf;
        T primary_kkt_backward_error;
        T qr_kkt_backward_error;
        T normal_matrix_inf;
        T normal_rhs_inf;
        T primary_coefficient_inf;
        T qr_coefficient_inf;

        Result()
            : rows(0), cols(0), nnz(0), numerical_rank(0),
              rank_deficient(false), pivot_threshold(T(0)),
              qr_objective(T(0)), primary_objective(T(0)),
              primary_minus_qr_objective(T(0)),
              objective_gap_absolute(T(0)),
              objective_gap_relative(T(0)),
              coefficient_relative_l2(T(0)),
              coefficient_relative_l2_denominator(T(0)),
              primary_raw_gradient_inf(T(0)), qr_raw_gradient_inf(T(0)),
              primary_kkt_numerator_inf(T(0)), qr_kkt_numerator_inf(T(0)),
              primary_kkt_backward_error(T(0)),
              qr_kkt_backward_error(T(0)), normal_matrix_inf(T(0)),
              normal_rhs_inf(T(0)), primary_coefficient_inf(T(0)),
              qr_coefficient_inf(T(0))
        {
        }
    };

    // Zero-weight blocks deliberately retain their row ranges.  Their
    // augmented matrix and rhs entries are exactly zero, which preserves a
    // frozen row layout between gamma=0 and positive-weight candidates.
    static System assemble(const std::vector<Block>& blocks)
    {
        if (blocks.empty())
            throw std::invalid_argument(
                "Weighted least-squares assembly requires at least one block");

        Eigen::Index total_rows = 0;
        Eigen::Index columns = -1;
        std::size_t triplet_reserve = 0;
        std::set<std::string> names;

        for (std::size_t index = 0; index < blocks.size(); ++index)
        {
            validateBlock(blocks[index]);
            const SparseMatrix& matrix = *blocks[index].matrix;
            if (columns < 0)
                columns = matrix.cols();
            else if (matrix.cols() != columns)
                throw std::invalid_argument(
                    "Weighted least-squares blocks have inconsistent column counts");
            if (!names.insert(blocks[index].name).second)
                throw std::invalid_argument(
                    "Weighted least-squares block names must be unique");
            const Eigen::Index storage_max = static_cast<Eigen::Index>(
                std::numeric_limits<StorageIndex>::max());
            if (matrix.cols() > storage_max || matrix.rows() > storage_max ||
                total_rows > storage_max - matrix.rows())
                throw std::overflow_error(
                    "Weighted least-squares dimensions exceed sparse-index range");
            total_rows += matrix.rows();
            if (blocks[index].lambda > T(0))
                triplet_reserve += static_cast<std::size_t>(matrix.nonZeros());
        }
        if (columns <= 0 || total_rows <= 0)
            throw std::invalid_argument(
                "Weighted least-squares system must have positive dimensions");

        System result;
        result.A.resize(total_rows, columns);
        result.b = Vector::Zero(total_rows);
        result.blocks.reserve(blocks.size());
        std::vector<Triplet> triplets;
        triplets.reserve(triplet_reserve);

        Eigen::Index row_offset = 0;
        for (std::size_t index = 0; index < blocks.size(); ++index)
        {
            const Block& block = blocks[index];
            const SparseMatrix& matrix = *block.matrix;
            const Vector& rhs = *block.rhs;
            const T row_scale = std::sqrt(
                block.lambda / static_cast<T>(matrix.rows()));

            BlockRecord record;
            record.name = block.name;
            record.row_offset = row_offset;
            record.rows = matrix.rows();
            record.input_nnz = matrix.nonZeros();
            record.lambda = block.lambda;
            record.row_scale = row_scale;

            if (block.lambda > T(0))
            {
                for (Eigen::Index column = 0; column < matrix.outerSize();
                     ++column)
                    for (typename SparseMatrix::InnerIterator entry(
                             matrix, static_cast<int>(column));
                         entry; ++entry)
                    {
                        const T value = row_scale * entry.value();
                        if (!std::isfinite(value))
                            throw std::overflow_error(
                                "Weighted least-squares matrix scaling overflow");
                        if (value != T(0))
                        {
                            triplets.push_back(Triplet(
                                static_cast<StorageIndex>(
                                    row_offset + entry.row()),
                                static_cast<StorageIndex>(entry.col()), value));
                            ++record.augmented_nnz;
                        }
                    }
                for (Eigen::Index row = 0; row < rhs.size(); ++row)
                {
                    const T value = row_scale * rhs(row);
                    if (!std::isfinite(value))
                        throw std::overflow_error(
                            "Weighted least-squares rhs scaling overflow");
                    result.b(row_offset + row) = canonicalZero(value);
                }
            }
            // For lambda=0, the pre-zeroed rhs and the absence of triplets
            // encode the exact zero-scaled rows without evaluating 0*x.

            result.blocks.push_back(record);
            row_offset += matrix.rows();
        }

        result.A.setFromTriplets(triplets.begin(), triplets.end());
        result.A.makeCompressed();
        return result;
    }

    static Result solveAndCompare(
        const System& system,
        const Vector& primary_coefficients,
        const Options& options = Options())
    {
        validateSystem(system);
        requireFiniteVector(primary_coefficients, "primary coefficients");
        if (primary_coefficients.size() != system.A.cols())
            throw std::invalid_argument(
                "Primary coefficient count does not match augmented system");

        Result result;
        result.solver = "Eigen::SparseQR_rectangular_A";
        result.rows = system.A.rows();
        result.cols = system.A.cols();
        result.nnz = system.A.nonZeros();
        result.pivot_threshold = resolvedPivotThreshold(system.A, options);

        Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int> > solver;
        solver.setPivotThreshold(result.pivot_threshold);
        solver.compute(system.A);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error(
                std::string("SparseQR factorization failed: ") +
                solver.lastErrorMessage());

        result.numerical_rank = solver.rank();
        result.rank_deficient = result.numerical_rank < system.A.cols();
        result.qr_coefficients = solver.solve(system.b);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error(
                std::string("SparseQR least-squares solve failed: ") +
                solver.lastErrorMessage());
        requireFiniteVector(result.qr_coefficients, "SparseQR coefficients");
        result.solver_status = "success";

        const Vector primary_residual =
            system.A * primary_coefficients - system.b;
        const Vector qr_residual =
            system.A * result.qr_coefficients - system.b;
        result.primary_objective = primary_residual.squaredNorm();
        result.qr_objective = qr_residual.squaredNorm();
        result.primary_minus_qr_objective =
            result.primary_objective - result.qr_objective;
        result.objective_gap_absolute =
            std::abs(result.primary_minus_qr_objective);
        result.objective_gap_relative = safeRatio(
            result.objective_gap_absolute,
            std::max(result.qr_objective,
                     std::numeric_limits<T>::epsilon()));

        const T qr_l2 = result.qr_coefficients.norm();
        result.coefficient_relative_l2_denominator =
            std::max(qr_l2, std::numeric_limits<T>::epsilon());
        result.coefficient_relative_l2 =
            (primary_coefficients - result.qr_coefficients).norm() /
            result.coefficient_relative_l2_denominator;

        const Vector primary_gradient =
            T(2) * (system.A.transpose() * primary_residual);
        const Vector qr_gradient =
            T(2) * (system.A.transpose() * qr_residual);
        result.primary_raw_gradient_inf = infinityNorm(primary_gradient);
        result.qr_raw_gradient_inf = infinityNorm(qr_gradient);

        // Diagnostic only.  This matrix is never passed to the primary solve.
        SparseMatrix normal_matrix = system.A.transpose() * system.A;
        normal_matrix.makeCompressed();
        const Vector normal_rhs = system.A.transpose() * system.b;
        result.normal_matrix_inf = sparseInfinityNorm(normal_matrix);
        result.normal_rhs_inf = infinityNorm(normal_rhs);
        result.primary_coefficient_inf = infinityNorm(primary_coefficients);
        result.qr_coefficient_inf = infinityNorm(result.qr_coefficients);

        const Vector primary_kkt =
            normal_matrix * primary_coefficients - normal_rhs;
        const Vector qr_kkt =
            normal_matrix * result.qr_coefficients - normal_rhs;
        result.primary_kkt_numerator_inf = infinityNorm(primary_kkt);
        result.qr_kkt_numerator_inf = infinityNorm(qr_kkt);
        result.primary_kkt_backward_error = backwardError(
            result.primary_kkt_numerator_inf,
            result.normal_matrix_inf,
            result.primary_coefficient_inf,
            result.normal_rhs_inf);
        result.qr_kkt_backward_error = backwardError(
            result.qr_kkt_numerator_inf,
            result.normal_matrix_inf,
            result.qr_coefficient_inf,
            result.normal_rhs_inf);

        return result;
    }

    // MatrixMarket coordinate entries are emitted in the deterministic order
    // of a compressed column-major Eigen matrix: column, then row.
    static void writeMatrixMarket(
        std::ostream& output,
        const SparseMatrix& matrix)
    {
        if (!output)
            throw std::runtime_error("MatrixMarket output stream is not writable");
        requireFiniteSparse(matrix, "MatrixMarket matrix");
        output.imbue(std::locale::classic());
        output << "%%MatrixMarket matrix coordinate real general\n"
               << "% mfa_weighted_least_squares_v1\n"
               << "% ordering: compressed-column-major, then row\n"
               << matrix.rows() << ' ' << matrix.cols() << ' '
               << matrix.nonZeros() << '\n'
               << std::scientific << std::setprecision(17);
        for (Eigen::Index column = 0; column < matrix.outerSize(); ++column)
            for (typename SparseMatrix::InnerIterator entry(
                     matrix, static_cast<int>(column));
                 entry; ++entry)
                output << entry.row() + 1 << ' ' << entry.col() + 1 << ' '
                       << canonicalZero(entry.value()) << '\n';
        if (!output)
            throw std::runtime_error("Failed while writing MatrixMarket matrix");
    }

    static void writeVectorText(
        std::ostream& output,
        const Vector& vector,
        const std::string& name)
    {
        if (!output)
            throw std::runtime_error("Vector output stream is not writable");
        validateName(name);
        requireFiniteVector(vector, "export vector");
        output.imbue(std::locale::classic());
        output << "format=mfa_vector_text_v1\n"
               << "name=" << name << '\n'
               << "size=" << vector.size() << '\n'
               << std::scientific << std::setprecision(17);
        for (Eigen::Index index = 0; index < vector.size(); ++index)
            output << index << ' ' << canonicalZero(vector(index)) << '\n';
        if (!output)
            throw std::runtime_error("Failed while writing vector text");
    }

    static void writeCertificateText(
        std::ostream& output,
        const System& system,
        const Result& result)
    {
        if (!output)
            throw std::runtime_error("Certificate output stream is not writable");
        output.imbue(std::locale::classic());
        output << std::scientific << std::setprecision(17)
               << "format=mfa_augmented_sparse_qr_certificate_v1\n"
               << "solver=" << result.solver << '\n'
               << "solver_status=" << result.solver_status << '\n'
               << "rows=" << result.rows << '\n'
               << "cols=" << result.cols << '\n'
               << "nnz=" << result.nnz << '\n'
               << "numerical_rank=" << result.numerical_rank << '\n'
               << "rank_deficient=" << (result.rank_deficient ? 1 : 0) << '\n'
               << "pivot_threshold=" << result.pivot_threshold << '\n'
               << "qr_objective=" << result.qr_objective << '\n'
               << "primary_objective=" << result.primary_objective << '\n'
               << "primary_minus_qr_objective="
               << result.primary_minus_qr_objective << '\n'
               << "objective_gap_absolute="
               << result.objective_gap_absolute << '\n'
               << "objective_gap_relative="
               << result.objective_gap_relative << '\n'
               << "coefficient_relative_l2="
               << result.coefficient_relative_l2 << '\n'
               << "coefficient_relative_l2_denominator="
               << result.coefficient_relative_l2_denominator << '\n'
               << "primary_raw_gradient_inf="
               << result.primary_raw_gradient_inf << '\n'
               << "qr_raw_gradient_inf=" << result.qr_raw_gradient_inf << '\n'
               << "primary_kkt_numerator_inf="
               << result.primary_kkt_numerator_inf << '\n'
               << "qr_kkt_numerator_inf="
               << result.qr_kkt_numerator_inf << '\n'
               << "primary_kkt_backward_error="
               << result.primary_kkt_backward_error << '\n'
               << "qr_kkt_backward_error="
               << result.qr_kkt_backward_error << '\n'
               << "normal_matrix_inf=" << result.normal_matrix_inf << '\n'
               << "normal_rhs_inf=" << result.normal_rhs_inf << '\n'
               << "primary_coefficient_inf="
               << result.primary_coefficient_inf << '\n'
               << "qr_coefficient_inf=" << result.qr_coefficient_inf << '\n'
               << "block_count=" << system.blocks.size() << '\n';
        for (std::size_t index = 0; index < system.blocks.size(); ++index)
        {
            const BlockRecord& block = system.blocks[index];
            output << "block_" << index << "_name=" << block.name << '\n'
                   << "block_" << index << "_row_offset="
                   << block.row_offset << '\n'
                   << "block_" << index << "_rows=" << block.rows << '\n'
                   << "block_" << index << "_input_nnz="
                   << block.input_nnz << '\n'
                   << "block_" << index << "_augmented_nnz="
                   << block.augmented_nnz << '\n'
                   << "block_" << index << "_lambda="
                   << block.lambda << '\n'
                   << "block_" << index << "_row_scale="
                   << block.row_scale << '\n';
        }
        if (!output)
            throw std::runtime_error("Failed while writing certificate text");
    }

    // Writes exactly four deterministic artifacts:
    //   <prefix>_A.mtx
    //   <prefix>_b.txt
    //   <prefix>_qr_coefficients.txt
    //   <prefix>_certificate.txt
    static void exportArtifacts(
        const std::string& prefix,
        const System& system,
        const Result& result)
    {
        if (prefix.empty())
            throw std::invalid_argument("Certificate artifact prefix is empty");
        std::ofstream matrix_file(
            (prefix + "_A.mtx").c_str(), std::ios::out | std::ios::binary);
        if (!matrix_file)
            throw std::runtime_error("Cannot open certificate MatrixMarket file");
        writeMatrixMarket(matrix_file, system.A);
        matrix_file.close();
        if (!matrix_file)
            throw std::runtime_error("Failed to close certificate MatrixMarket file");

        std::ofstream rhs_file(
            (prefix + "_b.txt").c_str(), std::ios::out | std::ios::binary);
        if (!rhs_file)
            throw std::runtime_error("Cannot open certificate rhs file");
        writeVectorText(rhs_file, system.b, "augmented_rhs");
        rhs_file.close();
        if (!rhs_file)
            throw std::runtime_error("Failed to close certificate rhs file");

        std::ofstream coefficient_file(
            (prefix + "_qr_coefficients.txt").c_str(),
            std::ios::out | std::ios::binary);
        if (!coefficient_file)
            throw std::runtime_error("Cannot open certificate coefficient file");
        writeVectorText(
            coefficient_file, result.qr_coefficients,
            "sparse_qr_coefficients");
        coefficient_file.close();
        if (!coefficient_file)
            throw std::runtime_error("Failed to close certificate coefficient file");

        std::ofstream certificate_file(
            (prefix + "_certificate.txt").c_str(),
            std::ios::out | std::ios::binary);
        if (!certificate_file)
            throw std::runtime_error("Cannot open certificate summary file");
        writeCertificateText(certificate_file, system, result);
        certificate_file.close();
        if (!certificate_file)
            throw std::runtime_error("Failed to close certificate summary file");
    }

private:
    static T canonicalZero(T value)
    {
        return value == T(0) ? T(0) : value;
    }

    static void validateName(const std::string& name)
    {
        if (name.empty())
            throw std::invalid_argument("Certificate name must be nonempty");
        for (std::size_t index = 0; index < name.size(); ++index)
        {
            const char value = name[index];
            const bool accepted =
                (value >= 'a' && value <= 'z') ||
                (value >= 'A' && value <= 'Z') ||
                (value >= '0' && value <= '9') ||
                value == '_' || value == '-' || value == '.';
            if (!accepted)
                throw std::invalid_argument(
                    "Certificate names may contain only [A-Za-z0-9_.-]");
        }
    }

    static void validateBlock(const Block& block)
    {
        validateName(block.name);
        if (!block.matrix || !block.rhs)
            throw std::invalid_argument(
                "Weighted least-squares block has a null reference");
        if (block.matrix->rows() <= 0 || block.matrix->cols() <= 0)
            throw std::invalid_argument(
                "Weighted least-squares block must have positive dimensions");
        if (block.rhs->size() != block.matrix->rows())
            throw std::invalid_argument(
                "Weighted least-squares block rhs has the wrong length");
        if (!std::isfinite(block.lambda) || block.lambda < T(0))
            throw std::invalid_argument(
                "Weighted least-squares block weight must be finite and nonnegative");
        requireFiniteSparse(*block.matrix, "weighted least-squares block");
        requireFiniteVector(*block.rhs, "weighted least-squares block rhs");
    }

    static void validateSystem(const System& system)
    {
        if (system.A.rows() <= 0 || system.A.cols() <= 0 ||
            system.b.size() != system.A.rows())
            throw std::invalid_argument(
                "Augmented weighted least-squares system is inconsistent");
        if (!system.A.isCompressed())
            throw std::invalid_argument(
                "Augmented weighted least-squares matrix must be compressed");
        requireFiniteSparse(system.A, "augmented matrix");
        requireFiniteVector(system.b, "augmented rhs");
    }

    static void requireFiniteSparse(
        const SparseMatrix& matrix,
        const char* label)
    {
        for (Eigen::Index column = 0; column < matrix.outerSize(); ++column)
            for (typename SparseMatrix::InnerIterator entry(
                     matrix, static_cast<int>(column));
                 entry; ++entry)
                if (!std::isfinite(entry.value()))
                    throw std::invalid_argument(
                        std::string(label) + " contains a nonfinite value");
    }

    static void requireFiniteVector(const Vector& vector, const char* label)
    {
        for (Eigen::Index index = 0; index < vector.size(); ++index)
            if (!std::isfinite(vector(index)))
                throw std::invalid_argument(
                    std::string(label) + " contains a nonfinite value");
    }

    static T resolvedPivotThreshold(
        const SparseMatrix& matrix,
        const Options& options)
    {
        if (std::isfinite(options.pivot_threshold) &&
            options.pivot_threshold > T(0))
            return options.pivot_threshold;
        if (options.pivot_threshold != T(-1))
            throw std::invalid_argument(
                "SparseQR pivot threshold must be positive or exactly -1");
        T maximum_column_norm = T(0);
        for (Eigen::Index column = 0; column < matrix.cols(); ++column)
            maximum_column_norm = std::max<T>(
                maximum_column_norm, matrix.col(column).norm());
        if (maximum_column_norm == T(0))
            maximum_column_norm = T(1);
        const T threshold =
            T(20) * static_cast<T>(matrix.rows() + matrix.cols()) *
            maximum_column_norm * std::numeric_limits<T>::epsilon();
        if (!(threshold > T(0)) || !std::isfinite(threshold))
            throw std::overflow_error(
                "Computed SparseQR pivot threshold is invalid");
        return threshold;
    }

    static T infinityNorm(const Vector& vector)
    {
        if (vector.size() == 0)
            return T(0);
        return vector.template lpNorm<Eigen::Infinity>();
    }

    static T sparseInfinityNorm(const SparseMatrix& matrix)
    {
        Vector absolute_row_sums = Vector::Zero(matrix.rows());
        for (Eigen::Index column = 0; column < matrix.outerSize(); ++column)
            for (typename SparseMatrix::InnerIterator entry(
                     matrix, static_cast<int>(column));
                 entry; ++entry)
                absolute_row_sums(entry.row()) += std::abs(entry.value());
        return infinityNorm(absolute_row_sums);
    }

    static T safeRatio(T numerator, T denominator)
    {
        if (denominator > T(0))
            return numerator / denominator;
        return numerator == T(0)
            ? T(0) : std::numeric_limits<T>::infinity();
    }

    static T backwardError(
        T numerator,
        T matrix_inf,
        T coefficient_inf,
        T rhs_inf)
    {
        return safeRatio(
            numerator, matrix_inf * coefficient_inf + rhs_inf);
    }
};

} // namespace mfa

#endif // MFA_WEIGHTED_LEAST_SQUARES_CERTIFICATE_HPP

#ifndef MFA_REG_MFA_TIKHONOV_HPP
#define MFA_REG_MFA_TIKHONOV_HPP

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfa
{
    enum class BaselineKind
    {
        data_only,
        reg_mfa_tikhonov
    };

    struct RegMFAConfig
    {
        BaselineKind kind;
        double alpha1;
        double alpha2;
        std::string role_sha256;
        std::string code_sha256;
        std::string checkpoint_in;
        std::string checkpoint_out;
        bool initializer_only;
        bool baseline_requested;

        RegMFAConfig() :
            kind(BaselineKind::data_only),
            alpha1(0.0),
            alpha2(0.0),
            initializer_only(false),
            baseline_requested(false)
        {}
    };

    struct BaselineCheckpointMetadata
    {
        std::string case_name;
        BaselineKind kind;
        double alpha1;
        double alpha2;
        std::string normalization;
        std::string derivative_orders;
        std::string derivative_dims;
        std::string lattice;
        std::string degree;
        std::string initializer_role_sha256;
        std::string pressure_regime;
        std::string field_order;
        std::size_t coefficient_count_per_field;
        std::size_t coefficient_count_total;
        std::string code_sha256;
        std::string payload_encoding;

        BaselineCheckpointMetadata() :
            kind(BaselineKind::data_only),
            alpha1(0.0),
            alpha2(0.0),
            normalization("mean_rows_v1"),
            derivative_orders("none"),
            derivative_dims("x,y,t"),
            pressure_regime("not_applicable"),
            field_order("u,v"),
            coefficient_count_per_field(0),
            coefficient_count_total(0),
            payload_encoding("float_text_max_digits10")
        {}
    };

    inline const char* BaselineKindName(BaselineKind kind)
    {
        return kind == BaselineKind::data_only
            ? "data_only"
            : "reg_mfa_tikhonov";
    }

    inline bool RegMFAEnvPresent(const char* name)
    {
        const char* value = std::getenv(name);
        return value != NULL && *value != '\0';
    }

    inline bool RegMFAEnvDefined(const char* name)
    {
        return std::getenv(name) != NULL;
    }

    inline std::string RegMFAEnvString(const char* name)
    {
        const char* value = std::getenv(name);
        return value == NULL ? std::string() : std::string(value);
    }

    inline double ParseStrictFiniteDouble(
        const std::string& text,
        const char* label)
    {
        if (text.empty())
            throw std::invalid_argument(
                std::string(label) + " must not be empty");

        char* end = NULL;
        errno = 0;
        const double value = std::strtod(text.c_str(), &end);
        if (errno == ERANGE ||
            end == text.c_str() ||
            end == NULL ||
            *end != '\0' ||
            !std::isfinite(value))
            throw std::invalid_argument(
                std::string(label) + " must be a finite scalar");
        return value;
    }

    inline std::size_t ParseStrictSize(
        const std::string& text,
        const char* label)
    {
        if (text.empty())
            throw std::invalid_argument(
                std::string(label) + " must not be empty");
        for (std::size_t i = 0; i < text.size(); ++i)
            if (text[i] < '0' || text[i] > '9')
                throw std::invalid_argument(
                    std::string(label) + " must be an unsigned integer");

        char* end = NULL;
        errno = 0;
        const unsigned long long value =
            std::strtoull(text.c_str(), &end, 10);
        if (errno == ERANGE ||
            end == text.c_str() ||
            end == NULL ||
            *end != '\0' ||
            value > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument(
                std::string(label) + " is out of range");
        return static_cast<std::size_t>(value);
    }

    inline bool IsLowerHexSHA256(const std::string& value)
    {
        if (value.size() != 64)
            return false;
        for (std::size_t i = 0; i < value.size(); ++i)
            if (!((value[i] >= '0' && value[i] <= '9') ||
                  (value[i] >= 'a' && value[i] <= 'f')))
                return false;
        return true;
    }

    inline bool SameDoubleBits(double lhs, double rhs)
    {
        unsigned char lhs_bytes[sizeof(double)];
        unsigned char rhs_bytes[sizeof(double)];
        std::memcpy(lhs_bytes, &lhs, sizeof(double));
        std::memcpy(rhs_bytes, &rhs, sizeof(double));
        return std::memcmp(lhs_bytes, rhs_bytes, sizeof(double)) == 0;
    }

    inline RegMFAConfig ReadRegMFAConfig()
    {
        RegMFAConfig config;

        const bool kind_present = RegMFAEnvDefined("MFA_BASELINE_KIND");
        const bool alpha1_present = RegMFAEnvDefined("MFA_REG_ALPHA1");
        const bool alpha2_present = RegMFAEnvDefined("MFA_REG_ALPHA2");
        const bool role_present =
            RegMFAEnvDefined("MFA_BASELINE_ROLE_SHA256");
        const bool code_present = RegMFAEnvDefined("MFA_CODE_SHA256");
        const bool checkpoint_in_present =
            RegMFAEnvDefined("MFA_BASELINE_CHECKPOINT_IN");
        const bool checkpoint_out_present =
            RegMFAEnvDefined("MFA_BASELINE_CHECKPOINT_OUT");

        config.baseline_requested =
            kind_present ||
            alpha1_present ||
            alpha2_present ||
            role_present ||
            code_present ||
            checkpoint_in_present ||
            checkpoint_out_present;

        if (config.baseline_requested && !kind_present)
            throw std::invalid_argument(
                "MFA_BASELINE_KIND is required when corrected baseline "
                "variables are used");

        if (kind_present)
        {
            const std::string kind = RegMFAEnvString("MFA_BASELINE_KIND");
            if (kind == "data_only")
                config.kind = BaselineKind::data_only;
            else if (kind == "reg_mfa_tikhonov")
                config.kind = BaselineKind::reg_mfa_tikhonov;
            else
                throw std::invalid_argument(
                    "MFA_BASELINE_KIND must be data_only or "
                    "reg_mfa_tikhonov");
        }

        if (alpha1_present)
            config.alpha1 = ParseStrictFiniteDouble(
                RegMFAEnvString("MFA_REG_ALPHA1"),
                "MFA_REG_ALPHA1");
        if (alpha2_present)
            config.alpha2 = ParseStrictFiniteDouble(
                RegMFAEnvString("MFA_REG_ALPHA2"),
                "MFA_REG_ALPHA2");

        if (config.alpha1 < 0.0 || std::signbit(config.alpha1))
            throw std::invalid_argument(
                "MFA_REG_ALPHA1 must be nonnegative");
        if (config.alpha2 < 0.0 || std::signbit(config.alpha2))
            throw std::invalid_argument(
                "MFA_REG_ALPHA2 must be nonnegative");

        config.role_sha256 =
            RegMFAEnvString("MFA_BASELINE_ROLE_SHA256");
        config.code_sha256 = RegMFAEnvString("MFA_CODE_SHA256");
        config.checkpoint_in =
            RegMFAEnvString("MFA_BASELINE_CHECKPOINT_IN");
        config.checkpoint_out =
            RegMFAEnvString("MFA_BASELINE_CHECKPOINT_OUT");

        if (config.baseline_requested)
        {
            const std::string initializer_only =
                RegMFAEnvString("MFA_INITIALIZER_ONLY");
            if (initializer_only.empty())
                config.initializer_only = false;
            else if (initializer_only == "1")
                config.initializer_only = true;
            else if (initializer_only == "0")
                config.initializer_only = false;
            else
                throw std::invalid_argument(
                    "MFA_INITIALIZER_ONLY must be exactly 0 or 1");
        }

        return config;
    }

    inline void ValidateRegMFAConfig(
        const RegMFAConfig& config,
        double legacy_regularization,
        bool legacy_reg1and2)
    {
        if (!config.baseline_requested)
            return;

        if (!std::isfinite(config.alpha1) ||
            config.alpha1 < 0.0 ||
            std::signbit(config.alpha1) ||
            !std::isfinite(config.alpha2) ||
            config.alpha2 < 0.0 ||
            std::signbit(config.alpha2))
            throw std::invalid_argument(
                "Corrected baseline alphas must be finite and nonnegative");
        if (!config.initializer_only)
            throw std::invalid_argument(
                "Corrected baseline mode requires MFA_INITIALIZER_ONLY=1");
        if (!IsLowerHexSHA256(config.role_sha256))
            throw std::invalid_argument(
                "MFA_BASELINE_ROLE_SHA256 must contain exactly 64 "
                "lowercase hexadecimal characters");
        if (!IsLowerHexSHA256(config.code_sha256))
            throw std::invalid_argument(
                "MFA_CODE_SHA256 must contain exactly 64 lowercase "
                "hexadecimal characters");
        if (!config.checkpoint_in.empty() &&
            !config.checkpoint_out.empty())
            throw std::invalid_argument(
                "Set only one of MFA_BASELINE_CHECKPOINT_IN and "
                "MFA_BASELINE_CHECKPOINT_OUT");

        if (config.kind == BaselineKind::data_only &&
            (config.alpha1 != 0.0 || config.alpha2 != 0.0))
            throw std::invalid_argument(
                "data_only requires MFA_REG_ALPHA1=0 and "
                "MFA_REG_ALPHA2=0");

        if (legacy_regularization != 0.0 || legacy_reg1and2)
            throw std::invalid_argument(
                "Corrected baseline mode rejects nonzero legacy -b/-k "
                "regularization values");
        if (RegMFAEnvPresent("MFA_INITIALIZER_LOAD") ||
            RegMFAEnvPresent("MFA_INITIALIZER_SAVE") ||
            RegMFAEnvPresent("MFA_INIT_CHECKPOINT_IN") ||
            RegMFAEnvPresent("MFA_INIT_CHECKPOINT_OUT"))
            throw std::invalid_argument(
                "Corrected baseline mode rejects legacy initializer "
                "checkpoint variables");
    }

    inline double TikhonovColumnScale(
        double alpha,
        std::size_t n_train,
        std::size_t n_derivative_rows)
    {
        if (!std::isfinite(alpha) ||
            alpha < 0.0 ||
            std::signbit(alpha))
            throw std::invalid_argument(
                "Tikhonov alpha must be finite and nonnegative");
        if (n_train == 0)
            throw std::invalid_argument(
                "Tikhonov scaling requires at least one training row");
        if (n_derivative_rows == 0)
            throw std::invalid_argument(
                "Tikhonov scaling requires derivative rows");
        const double ratio =
            static_cast<double>(n_train) /
            static_cast<double>(n_derivative_rows);
        const double scale = std::sqrt(alpha) * std::sqrt(ratio);
        if (!std::isfinite(scale))
            throw std::overflow_error(
                "Tikhonov column scale is not finite");
        return scale;
    }

    inline std::string ReadCheckpointKey(
        std::istream& input,
        const char* expected_key)
    {
        std::string line;
        if (!std::getline(input, line))
            throw std::runtime_error(
                std::string("Truncated V2 checkpoint before key ") +
                expected_key);
        const std::string prefix = std::string(expected_key) + "=";
        if (line.compare(0, prefix.size(), prefix) != 0)
            throw std::runtime_error(
                std::string("Expected V2 checkpoint key ") +
                expected_key);
        return line.substr(prefix.size());
    }

    inline void RequireCheckpointString(
        const std::string& actual,
        const std::string& expected,
        const char* label)
    {
        if (actual != expected)
            throw std::runtime_error(
                std::string("V2 checkpoint metadata mismatch: ") + label);
    }

    inline void ValidateCheckpointMetadata(
        const BaselineCheckpointMetadata& metadata)
    {
        if (metadata.case_name.empty() ||
            metadata.lattice.empty() ||
            metadata.degree.empty())
            throw std::invalid_argument(
                "V2 checkpoint metadata fields must not be empty");
        if (!std::isfinite(metadata.alpha1) ||
            metadata.alpha1 < 0.0 ||
            std::signbit(metadata.alpha1) ||
            !std::isfinite(metadata.alpha2) ||
            metadata.alpha2 < 0.0 ||
            std::signbit(metadata.alpha2))
            throw std::invalid_argument(
                "V2 checkpoint alphas must be finite and nonnegative");
        if (!IsLowerHexSHA256(metadata.initializer_role_sha256) ||
            !IsLowerHexSHA256(metadata.code_sha256))
            throw std::invalid_argument(
                "V2 checkpoint hashes must be lowercase SHA-256 values");
        if (metadata.coefficient_count_per_field == 0 ||
            metadata.coefficient_count_total == 0)
            throw std::invalid_argument(
                "V2 checkpoint coefficient counts must be positive");
        if (metadata.payload_encoding != "float_text_max_digits10")
            throw std::invalid_argument(
                "Unsupported V2 checkpoint payload encoding");
    }

    inline void WriteBaselineCheckpointV2(
        const std::string& path,
        const BaselineCheckpointMetadata& metadata,
        const std::vector<double>& coefficients)
    {
        ValidateCheckpointMetadata(metadata);
        if (coefficients.size() != metadata.coefficient_count_total)
            throw std::invalid_argument(
                "V2 checkpoint payload size does not match metadata");

        std::ofstream output(path.c_str());
        if (!output)
            throw std::runtime_error(
                std::string("Cannot create V2 baseline checkpoint: ") +
                path);

        output << "# PIMFA_BASELINE_V2\n"
               << "case=" << metadata.case_name << "\n"
               << "kind=" << BaselineKindName(metadata.kind) << "\n"
               << std::setprecision(
                    std::numeric_limits<double>::max_digits10)
               << "alpha1=" << metadata.alpha1 << "\n"
               << "alpha2=" << metadata.alpha2 << "\n"
               << "normalization=" << metadata.normalization << "\n"
               << "derivative_orders=" << metadata.derivative_orders << "\n"
               << "derivative_dims=" << metadata.derivative_dims << "\n"
               << "lattice=" << metadata.lattice << "\n"
               << "degree=" << metadata.degree << "\n"
               << "initializer_role_sha256="
               << metadata.initializer_role_sha256 << "\n"
               << "pressure_regime=" << metadata.pressure_regime << "\n"
               << "field_order=" << metadata.field_order << "\n"
               << "coefficient_count_per_field="
               << metadata.coefficient_count_per_field << "\n"
               << "coefficient_count_total="
               << metadata.coefficient_count_total << "\n"
               << "code_sha256=" << metadata.code_sha256 << "\n"
               << "payload_encoding=" << metadata.payload_encoding << "\n"
               << "--coefficients--\n";

        for (std::size_t i = 0; i < coefficients.size(); ++i)
        {
            if (!std::isfinite(coefficients[i]))
                throw std::invalid_argument(
                    "V2 checkpoint payload contains a non-finite value");
            output << coefficients[i] << "\n";
        }
        if (!output)
            throw std::runtime_error(
                std::string("Failed writing V2 baseline checkpoint: ") +
                path);
    }

    inline std::vector<double> ReadBaselineCheckpointV2(
        const std::string& path,
        const BaselineCheckpointMetadata& expected)
    {
        ValidateCheckpointMetadata(expected);
        std::ifstream input(path.c_str());
        if (!input)
            throw std::runtime_error(
                std::string("Cannot open V2 baseline checkpoint: ") + path);

        std::string magic;
        if (!std::getline(input, magic) || magic != "# PIMFA_BASELINE_V2")
            throw std::runtime_error(
                "Invalid V2 baseline checkpoint magic");

        RequireCheckpointString(
            ReadCheckpointKey(input, "case"),
            expected.case_name,
            "case");
        RequireCheckpointString(
            ReadCheckpointKey(input, "kind"),
            BaselineKindName(expected.kind),
            "kind");

        const double alpha1 = ParseStrictFiniteDouble(
            ReadCheckpointKey(input, "alpha1"),
            "checkpoint alpha1");
        const double alpha2 = ParseStrictFiniteDouble(
            ReadCheckpointKey(input, "alpha2"),
            "checkpoint alpha2");
        if (!SameDoubleBits(alpha1, expected.alpha1))
            throw std::runtime_error(
                "V2 checkpoint metadata mismatch: alpha1");
        if (!SameDoubleBits(alpha2, expected.alpha2))
            throw std::runtime_error(
                "V2 checkpoint metadata mismatch: alpha2");

        RequireCheckpointString(
            ReadCheckpointKey(input, "normalization"),
            expected.normalization,
            "normalization");
        RequireCheckpointString(
            ReadCheckpointKey(input, "derivative_orders"),
            expected.derivative_orders,
            "derivative_orders");
        RequireCheckpointString(
            ReadCheckpointKey(input, "derivative_dims"),
            expected.derivative_dims,
            "derivative_dims");
        RequireCheckpointString(
            ReadCheckpointKey(input, "lattice"),
            expected.lattice,
            "lattice");
        RequireCheckpointString(
            ReadCheckpointKey(input, "degree"),
            expected.degree,
            "degree");
        RequireCheckpointString(
            ReadCheckpointKey(input, "initializer_role_sha256"),
            expected.initializer_role_sha256,
            "initializer_role_sha256");
        RequireCheckpointString(
            ReadCheckpointKey(input, "pressure_regime"),
            expected.pressure_regime,
            "pressure_regime");
        RequireCheckpointString(
            ReadCheckpointKey(input, "field_order"),
            expected.field_order,
            "field_order");

        const std::size_t count_per_field = ParseStrictSize(
            ReadCheckpointKey(input, "coefficient_count_per_field"),
            "coefficient_count_per_field");
        const std::size_t count_total = ParseStrictSize(
            ReadCheckpointKey(input, "coefficient_count_total"),
            "coefficient_count_total");
        if (count_per_field != expected.coefficient_count_per_field)
            throw std::runtime_error(
                "V2 checkpoint metadata mismatch: "
                "coefficient_count_per_field");
        if (count_total != expected.coefficient_count_total)
            throw std::runtime_error(
                "V2 checkpoint metadata mismatch: "
                "coefficient_count_total");

        RequireCheckpointString(
            ReadCheckpointKey(input, "code_sha256"),
            expected.code_sha256,
            "code_sha256");
        RequireCheckpointString(
            ReadCheckpointKey(input, "payload_encoding"),
            expected.payload_encoding,
            "payload_encoding");

        std::string marker;
        if (!std::getline(input, marker) || marker != "--coefficients--")
            throw std::runtime_error(
                "Missing V2 checkpoint coefficient marker");

        std::vector<double> coefficients(expected.coefficient_count_total);
        for (std::size_t i = 0; i < coefficients.size(); ++i)
        {
            if (!(input >> coefficients[i]) ||
                !std::isfinite(coefficients[i]))
                throw std::runtime_error(
                    "Truncated or non-finite V2 checkpoint payload");
        }
        std::string extra;
        if (input >> extra)
            throw std::runtime_error(
                "V2 checkpoint contains an extra payload token");
        return coefficients;
    }
}

#endif

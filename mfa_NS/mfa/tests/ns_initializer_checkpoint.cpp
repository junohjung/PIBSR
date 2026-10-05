#include <mfa/ns_initializer_checkpoint.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void require(bool condition, const std::string& message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    template <typename Function>
    void require_throw(Function function, const std::string& message)
    {
        bool threw = false;
        try
        {
            function();
        }
        catch (const std::exception&)
        {
            threw = true;
        }
        require(threw, message);
    }

    std::string temporary_path(const char* suffix)
    {
        return std::string("ns_initializer_checkpoint_test_") + suffix + ".txt";
    }

    void write_checkpoint(
        const std::string& path,
        const std::string& header,
        const std::vector<std::string>& values)
    {
        std::ofstream output(path.c_str());
        require(static_cast<bool>(output), "cannot create checkpoint fixture");
        output << header << "\n";
        for (std::size_t i = 0; i < values.size(); ++i)
            output << values[i] << "\n";
    }
}

int main()
{
    const std::string valid = temporary_path("valid");
    const std::string wrong_pressure = temporary_path("wrong_pressure");
    const std::string nonfinite = temporary_path("nonfinite");
    const std::string too_large = temporary_path("too_large");
    const std::string negative_zero = temporary_path("negative_zero");
    const std::string extra = temporary_path("extra");

    try
    {
        unsetenv("MFA_INIT_CHECKPOINT_IN");
        mfa::NSInitializerCheckpointPlan plan =
            mfa::ReadNSInitializerCheckpointPlan();
        require(!plan.bypass_data_solve,
                "unset checkpoint unexpectedly bypasses Stage-A solve");

        setenv("MFA_INIT_CHECKPOINT_IN", "", 1);
        require_throw(
            []() { mfa::ReadNSInitializerCheckpointPlan(); },
            "empty checkpoint environment did not fail closed");

        write_checkpoint(
            valid,
            "# PIMFA_NS_INIT_V1 pressure=zero n=6",
            {"1.25", "-2.5", "3.75", "-4", "0", "0"});
        setenv("MFA_INIT_CHECKPOINT_IN", valid.c_str(), 1);
        plan = mfa::ReadNSInitializerCheckpointPlan();
        require(plan.bypass_data_solve && plan.path == valid,
                "checkpoint did not select the Stage-B bypass path");
        const std::vector<double> payload =
            mfa::ReadNSInitializerCheckpointV1Strict(
                plan.path, "zero", 6, 100.0);
        require(payload.size() == 6 && payload[0] == 1.25 && payload[5] == 0.0,
                "valid checkpoint payload changed during parsing");

        require_throw(
            []() {
                mfa::ReadNSInitializerCheckpointV1Strict(
                    "does_not_exist.txt", "zero", 6, 100.0);
            },
            "missing checkpoint did not fail closed");

        write_checkpoint(
            wrong_pressure,
            "# PIMFA_NS_INIT_V1 pressure=data n=6",
            {"1", "2", "3", "4", "0", "0"});
        require_throw(
            [&]() {
                mfa::ReadNSInitializerCheckpointV1Strict(
                    wrong_pressure, "zero", 6, 100.0);
            },
            "pressure-regime mismatch was accepted");

        write_checkpoint(
            nonfinite,
            "# PIMFA_NS_INIT_V1 pressure=zero n=6",
            {"1", "nan", "3", "4", "0", "0"});
        require_throw(
            [&]() {
                mfa::ReadNSInitializerCheckpointV1Strict(
                    nonfinite, "zero", 6, 100.0);
            },
            "non-finite checkpoint coefficient was accepted");

        write_checkpoint(
            too_large,
            "# PIMFA_NS_INIT_V1 pressure=zero n=6",
            {"101", "2", "3", "4", "0", "0"});
        require_throw(
            [&]() {
                mfa::ReadNSInitializerCheckpointV1Strict(
                    too_large, "zero", 6, 100.0);
            },
            "coefficient outside the optimizer box was accepted");

        write_checkpoint(
            negative_zero,
            "# PIMFA_NS_INIT_V1 pressure=zero n=6",
            {"1", "2", "3", "4", "-0", "0"});
        require_throw(
            [&]() {
                mfa::ReadNSInitializerCheckpointV1Strict(
                    negative_zero, "zero", 6, 100.0);
            },
            "negative-zero pressure coefficient was accepted");

        write_checkpoint(
            extra,
            "# PIMFA_NS_INIT_V1 pressure=zero n=6",
            {"1", "2", "3", "4", "0", "0", "7"});
        require_throw(
            [&]() {
                mfa::ReadNSInitializerCheckpointV1Strict(
                    extra, "zero", 6, 100.0);
            },
            "extra checkpoint coefficient was accepted");

        require_throw(
            [&]() {
                mfa::ReadNSInitializerCheckpointV1Strict(
                    valid, "zero", 9, 100.0);
            },
            "coefficient-count mismatch was accepted");

        unsetenv("MFA_INIT_CHECKPOINT_IN");
        std::remove(valid.c_str());
        std::remove(wrong_pressure.c_str());
        std::remove(nonfinite.c_str());
        std::remove(too_large.c_str());
        std::remove(negative_zero.c_str());
        std::remove(extra.c_str());
        std::cout << "NS initializer checkpoint fail-closed tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        unsetenv("MFA_INIT_CHECKPOINT_IN");
        std::cerr << "NS initializer checkpoint test failed: "
                  << error.what() << "\n";
        return 1;
    }
}

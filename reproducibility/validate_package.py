#!/usr/bin/env python3
"""Validate the PI-BSR reproducibility metadata."""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath


SCHEMA_VERSION = 1
CONFIG_FILES = {
    "one_dimensional_convection_diffusion": "one_dimensional_convection_diffusion.json",
    "two_dimensional_burgers": "two_dimensional_burgers.json",
    "lid_driven_cavity": "lid_driven_cavity.json",
}
ARTIFACT_DIRECTORIES = {
    "one_dimensional_convection_diffusion": "one_d_convection_diffusion",
    "two_dimensional_burgers": "two_d_burgers",
    "lid_driven_cavity": "lid_driven_cavity",
}
RESULT_INVENTORY_DIRECTORIES = tuple(ARTIFACT_DIRECTORIES.values()) + (
    "verification",
)
CALCULATION_DIRECTORIES = {
    case_id: artifact_directory
    for case_id, artifact_directory in ARTIFACT_DIRECTORIES.items()
}
PRECISION_LABELS = {"archived_full_precision", "manuscript_rounded"}
MANIFEST_PATTERN = re.compile(r"^([0-9a-f]{64})  (.+)$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SUBSET_MANIFEST_NAME = "PACKAGE_SUBSET_MANIFEST.tsv"
SUBSET_MANIFEST_HEADER = "relative_path\tsha256\tbytes"
REPORTED_METRIC_FILES = {
    "one_d": "calculations/one_d_convection_diffusion/reported_metrics.json",
    "burgers": "calculations/two_d_burgers/reported_metrics.json",
    "cavity": "calculations/lid_driven_cavity/reported_metrics.json",
    "verification": "calculations/verification/reported_metrics.json",
}

ONE_D_MODEL_ALIASES = {
    "mfa_data_only": "data_only",
    "pibsr_strong_beta_3e_5": "matched_strong_gamma_0",
    "pibsr_hybrid_beta_3e_5_gamma_0_1": "selected_hybrid_gamma_0p1",
    "pibsr_hybrid_beta_3e_5_gamma_10": "reporting_only_gamma_10",
}
ONE_D_METRIC_ALIASES = {
    "midpoint_data_mse": "midpoint_data_mse",
    "midpoint_pde_mse": "midpoint_pde_mse",
    "analytical_midpoint_relative_l2": "midpoint_analytic_relative_l2",
    "q4_gl4_balance_mse": "balance_validation_q4_gl4_mse",
    "whole_domain_gl8_balance_rms": "whole_domain_q1_gl8_time_rms",
}
BURGERS_MODEL_ALIASES = {
    "pibsr_strong_beta_0_01_kappa_10": "matched_strong_gamma_0",
    "pibsr_hybrid_beta_0_01_kappa_10_gamma_0_03": (
        "selected_hybrid_gamma_0p03"
    ),
}
BURGERS_METRIC_PATHS = {
    "q4_gl8_balance_rms_u": "balance.{model}.gl8_q4_u_time_rms",
    "q4_gl8_balance_rms_v": "balance.{model}.gl8_q4_v_time_rms",
    "off_node_strong_rms_u": "postselection.{model}.offnode_strong_residual_u_rms",
    "off_node_strong_rms_v": "postselection.{model}.offnode_strong_residual_v_rms",
    "whole_domain_q1_balance_rms_u": "postselection.{model}.legacy_q1_u_time_rms",
    "whole_domain_q1_balance_rms_v": "postselection.{model}.legacy_q1_v_time_rms",
    "off_node_velocity_relative_l2": "postselection.{model}.offnode_velocity_relative_l2",
    "off_node_vorticity_relative_l2": "postselection.{model}.offnode_vorticity_relative_l2",
    "kinetic_energy_trajectory_relative_l2": (
        "postselection.{model}.kinetic_energy_trajectory_relative_l2"
    ),
    "enstrophy_trajectory_relative_l2": (
        "postselection.{model}.enstrophy_trajectory_relative_l2"
    ),
    "test_velocity_relative_l2": "postselection.{model}.final_velocity_relative_l2",
    "test_vorticity_relative_l2": "postselection.{model}.final_vorticity_relative_l2",
}
CAVITY_MODEL_ALIASES = {
    "selected_gl4_hybrid": "selected_gl4_hybrid",
    "matched_strong_control": "matched_strong_gamma_i0",
    "stage_a_canonical_p0": "stage_a",
}

CHECKPOINT_MODELS = {
    "one_d_convection_diffusion": {
        "stage_a_strong_initializer_beta_3e-5": "pibsr_strong_beta_3e_5",
        "matched_strong_gamma_0": "pibsr_strong_beta_3e_5",
        "selected_hybrid_gamma_0.1": "pibsr_hybrid_beta_3e_5_gamma_0_1",
        "high_weight_sensitivity_gamma_10": "pibsr_hybrid_beta_3e_5_gamma_10",
    },
    "two_d_burgers": {
        "stage_a_data_only_initializer": None,
        "matched_strong_beta_0.01_kappa_10_gamma_0": "pibsr_strong_beta_0_01_kappa_10",
        "selected_hybrid_beta_0.01_kappa_10_gamma_0.03": (
            "pibsr_hybrid_beta_0_01_kappa_10_gamma_0_03"
        ),
    },
    "lid_driven_cavity": {
        "stage_a_canonical_zero_pressure_initializer": "stage_a_canonical_p0",
        "matched_strong_gamma_s_0.02_gamma_i_0": "matched_strong_control",
        "selected_gl4_hybrid_gamma_s_0.02_gamma_i_0.02": "selected_gl4_hybrid",
    },
}

INDEX_FILE_COUNTS = {
    "two_d_burgers/data_fit_indices.txt": 106213,
    "two_d_burgers/data_val_indices.txt": 30332,
    "two_d_burgers/collocation_fit_indices.txt": 106669,
    "two_d_burgers/collocation_val_indices.txt": 30312,
    "two_d_burgers/initializer_fit_indices.txt": 139894,
    "two_d_burgers/ic_indices.txt": 1681,
    "two_d_burgers/bc_indices.txt": 32000,
    "two_d_burgers/final_test_indices.txt": 30674,
    "lid_driven_cavity/data_fit_indices.txt": 126017,
    "lid_driven_cavity/data_val_indices.txt": 35932,
    "lid_driven_cavity/collocation_fit_indices.txt": 126886,
    "lid_driven_cavity/collocation_val_indices.txt": 35935,
    "lid_driven_cavity/initializer_fit_indices.txt": 206458,
    "lid_driven_cavity/ic_indices.txt": 441,
    "lid_driven_cavity/bc_indices.txt": 80000,
    "lid_driven_cavity/final_test_indices.txt": 36230,
}

INTEGRAL_TIME_ROLES = {
    "one_d_convection_diffusion": (
        "fit_time_indices.txt",
        279,
        "validation_time_indices.txt",
        120,
    ),
    "two_d_burgers": (
        "integral_balance/fit_time_indices.txt",
        140,
        "integral_balance/validation_time_indices.txt",
        60,
    ),
    "lid_driven_cavity": (
        "integral_time_roles/integral_time_fit_indices.txt",
        700,
        "integral_time_roles/integral_time_validation_indices.txt",
        300,
    ),
}


class ValidationError(Exception):
    pass


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError as exc:
        raise ValidationError("missing file: {}".format(path)) from exc
    except json.JSONDecodeError as exc:
        raise ValidationError("invalid JSON in {}: {}".format(path, exc)) from exc


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def add_check(errors, condition, message):
    if not condition:
        errors.append(message)


def case_checker(errors, case_id):
    def check(condition, message):
        add_check(errors, condition, "{}: {}".format(case_id, message))

    return check


def require_mapping(value, label):
    if not isinstance(value, dict):
        raise ValidationError("{} must be an object".format(label))
    return value


def require_keys(mapping, keys, label):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValidationError("{} is missing keys: {}".format(label, ", ".join(missing)))


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_artifact_paths(config, errors):
    case_id = config["case_id"]
    directory = ARTIFACT_DIRECTORIES[case_id]
    check = case_checker(errors, case_id)
    check(
        config.get("artifact_directory") == directory,
        "unexpected artifact directory",
    )
    paths = require_mapping(config.get("artifact_paths"), "{}.artifact_paths".format(case_id))
    expected = {
        kind: "{}/{}".format(kind, directory)
        for kind in ("data", "checkpoints", "results", "point_sets")
    }
    check(paths == expected, "artifact paths do not match the release layout")
    for value in paths.values():
        path = PurePosixPath(value)
        check(
            not path.is_absolute() and ".." not in path.parts,
            "unsafe artifact path: {}".format(value),
        )


def validate_one_dimensional(config, errors):
    case_id = config["case_id"]
    check = case_checker(errors, case_id)
    output = config["observation_generator"]["output"]
    fitting = config["point_roles"]["fitting"]
    selection = config["point_roles"]["selection"]
    post = config["point_roles"]["post_selection"]
    spline = config["spline"]
    strong = config["selection"]["strong_form"]
    balance_selection = config["selection"]["integral_balance"]

    check(
        config["domain"]["bounds"] == {"x": [0.0, 1.0], "t": [0.0, 0.4]},
        "domain mismatch",
    )
    check(
        output["space_nodes"] == 60 and output["time_nodes"] == 400,
        "output grid mismatch",
    )
    check(
        fitting["observation_rows"] == product(fitting["observation_shape"]),
        "observation-row count mismatch",
    )
    check(fitting["observation_rows"] == 24000, "expected 24,000 observation rows")
    check(fitting["bc_rows"] == 2 * output["time_nodes"], "BC-row count mismatch")
    check(fitting["ic_rows"] == output["space_nodes"], "IC-row count mismatch")
    check(
        fitting["pde_rows"] == 59 * 399 == 23541,
        "deterministic midpoint PDE-row count mismatch",
    )
    check(
        selection["strong_midpoint_rows"]
        == product(selection["strong_midpoint_shape"]),
        "midpoint-row count mismatch",
    )
    check(selection["strong_midpoint_shape"] == [59, 399], "midpoint shape mismatch")
    check(
        selection["balance_fit_times"] + selection["balance_selection_times"] == 399,
        "noninitial balance times must total 399",
    )
    check(post["cellwise_gl2_points"] == 59 * 399 * 4, "cellwise GL2 point count mismatch")
    check(post["whole_domain_gl8_times"] == 400, "GL8 reporting-time count mismatch")
    check(
        spline["coefficients_per_field"] == product(spline["control_lattice"]),
        "coefficient count mismatch",
    )
    check(spline["degree"] == [3, 3], "spline degree mismatch")
    check(len(strong["beta_candidates"]) == 26, "beta family must contain 26 values")
    check(
        sum(value > 0 for value in strong["beta_candidates"]) == 25,
        "beta family must contain 25 positive values",
    )
    check(
        strong["selected_beta"] in strong["beta_candidates"],
        "selected beta is outside its family",
    )
    check(strong["selected_beta"] == 3e-5, "selected beta mismatch")
    check(
        balance_selection["selected_gamma"]
        in balance_selection["gamma_candidates"],
        "selected gamma is outside its family",
    )
    check(balance_selection["selected_gamma"] == 0.1, "selected gamma mismatch")
    check(
        config["analytical_reference"]["role"] == "post_selection_only",
        "analytical reference may not enter selection",
    )


def validate_burgers(config, errors):
    case_id = config["case_id"]
    check = case_checker(errors, case_id)
    generator = config["observation_generator"]
    fitting = config["point_roles"]["fitting"]
    selection = config["point_roles"]["selection"]
    post = config["point_roles"]["post_selection"]
    spline = config["spline"]
    weights = config["selection"]

    raw_samples = 2 * product(generator["space_nodes"]) * generator["time_nodes"]
    check(raw_samples == 675762, "raw scalar-sample count mismatch")
    check(
        spline["coefficients_per_field"]
        == product(spline["control_lattice_per_field"]),
        "per-field coefficient count mismatch",
    )
    check(
        spline["total_coefficients"]
        == len(spline["fields"]) * spline["coefficients_per_field"],
        "total coefficient count mismatch",
    )
    check(fitting["ic_rows"] == product(generator["space_nodes"]), "IC-row count mismatch")
    check(
        fitting["ordered_four_face_bc_rows"] == 4 * 41 * 201,
        "ordered BC-row count mismatch",
    )
    check(fitting["unique_bc_coordinates"] == 160 * 201, "unique BC-coordinate count mismatch")
    balance_fit_times = fitting["balance_fit_times"]
    optimizer_balance_validation_times = selection.get(
        "optimizer_balance_validation_times"
    )
    check(balance_fit_times == 140, "balance-fit time count mismatch")
    check(
        optimizer_balance_validation_times == 60,
        "optimizer balance-validation time count mismatch",
    )
    check(
        (balance_fit_times, optimizer_balance_validation_times) == (140, 60),
        "optimizer balance-fit and balance-validation times must total 200",
    )
    check(
        selection.get("balance_weight_selection_times") == 200,
        "weight-selection balance-time count mismatch",
    )
    check(
        selection.get("balance_weight_selection_design")
        == "sealed_design_a_off_node",
        "weight-selection balance design mismatch",
    )
    check(
        post["fully_off_node_points"] == product(post["fully_off_node_shape"]),
        "fully off-node point count mismatch",
    )
    check(post["fully_off_node_points"] == 320000, "expected 320,000 fully off-node points")
    check(
        post["mechanics_points"] == 40 * 40 * 4 * post["mechanics_times"],
        "mechanics point count mismatch",
    )
    check(post["balance_confirmation_times"] == 200, "reporting-time count mismatch")
    check(
        post.get("balance_confirmation_design") == "sealed_design_b_off_node",
        "balance confirmation design mismatch",
    )
    check(
        post.get("balance_confirmation_disjoint_from_weight_selection") is True,
        "balance confirmation must be disjoint from weight selection",
    )
    check(
        config["balance"]["regions"] == config["balance"]["partition_q"] ** 2,
        "balance-region count mismatch",
    )
    check(
        weights["strong_form"]["selected_beta"]
        in weights["strong_form"]["beta_candidates"],
        "selected beta is outside its family",
    )
    check(
        weights["trace"]["selected_kappa"]
        in weights["trace"]["kappa_candidates"],
        "selected kappa is outside its family",
    )
    check(
        weights["integral_balance"]["selected_gamma"]
        in weights["integral_balance"]["gamma_candidates"],
        "selected gamma is outside its family",
    )
    check(
        weights["integral_balance"].get("archived_sensitivity_candidates")
        == [0.003, 0.3],
        "archived gamma sensitivity candidates mismatch",
    )
    selected_weights = (
        weights["strong_form"]["selected_beta"],
        weights["trace"]["selected_kappa"],
        weights["integral_balance"]["selected_gamma"],
    )
    check(selected_weights == (0.01, 10.0, 0.03), "selected weight tuple mismatch")
    check(
        config["observation_forcing"]["used_by_reconstruction_operator"] is False,
        "forcing must not enter the prescribed operator",
    )
    check(config["observation_forcing"]["seed"] == 1234, "forcing seed mismatch")
    check(
        config["unforced_reference"]["role"] == "post_selection_only",
        "unforced reference may not enter selection",
    )
    check(len(config["contextual_pinn"]["seeds"]) == 5, "contextual PINN seed count mismatch")


def validate_cavity(config, errors):
    case_id = config["case_id"]
    check = case_checker(errors, case_id)
    generator = config["observation_generator"]
    fitting = config["point_roles"]["fitting"]
    selection_roles = config["point_roles"]["selection"]
    post = config["point_roles"]["post_selection"]
    spline = config["spline"]
    pressure = config["pressure_supervision"]
    selection = config["selection"]

    raw_samples = 3 * 21 * 21 * generator["stored_times"]
    check(raw_samples == 1324323, "raw scalar-sample count mismatch")
    check(
        spline["coefficients_per_field"]
        == product(spline["control_lattice_per_field"]),
        "per-field coefficient count mismatch",
    )
    check(
        spline["total_coefficients"]
        == len(spline["fields"]) * spline["coefficients_per_field"],
        "total coefficient count mismatch",
    )
    check(
        spline["spatial_pressure_coefficients"] == 20 * 20,
        "spatial pressure coefficient count mismatch",
    )
    check(fitting["balance_regions"] == 16, "fitting region count mismatch")
    check(
        fitting["balance_fit_region_time_rows_per_component"]
        == fitting["balance_fit_times"] * fitting["balance_regions"],
        "balance-fit row count mismatch",
    )
    check(
        selection_roles["balance_validation_region_time_rows_per_component"]
        == selection_roles["balance_validation_times"]
        * selection_roles["balance_regions"],
        "balance-validation row count mismatch",
    )
    check(
        post["balance_region_time_rows_per_component"]
        == post["balance_times"] * post["balance_regions"],
        "post-selection balance row count mismatch",
    )
    check(
        fitting["balance_fit_times"]
        + selection_roles["balance_validation_times"]
        == 1000,
        "fit and validation balance times must total 1,000",
    )
    check(
        fitting["bc_orientation_groups"]
        * fitting["bc_rows_per_orientation_group"]
        == 84000,
        "BC-row group count mismatch",
    )
    check(
        fitting["gauge_rows"] == pressure["gauge_times"] == 1001,
        "gauge time count mismatch",
    )
    check(pressure["gauge_spatial_points"] == 21 * 21, "gauge spatial-point count mismatch")
    pressure_targets = (
        pressure["pressure_observations"],
        pressure["pressure_initial_targets"],
        pressure["pressure_boundary_targets"],
    )
    check(not any(pressure_targets), "pressure-value supervision must be absent")
    check(
        pressure["q4_integral_plus_gauge_rank"] == 2 * 4 * 4 + 1,
        "q=4 pressure rank mismatch",
    )
    check(
        pressure["q4_integral_plus_gauge_rank"]
        + pressure["q4_integral_plus_gauge_nullity"]
        == spline["spatial_pressure_coefficients"],
        "pressure rank/nullity mismatch",
    )
    check(
        pressure["strong_gradient_plus_gauge_rank"]
        == spline["spatial_pressure_coefficients"],
        "strong pressure block must be full rank",
    )
    check(
        config["balance"]["fit_regions"]
        == config["balance"]["fit_partition_q"] ** 2,
        "q=4 region count mismatch",
    )
    check(
        config["balance"]["canonical_separators"]
        == [0.0, 0.25, 0.5, 0.75, 1.0],
        "canonical partition mismatch",
    )
    candidates = selection["gamma_total_candidates"]
    endpoints = selection["selected_family_endpoints"]
    check(endpoints["strong_only"] in candidates, "strong endpoint outside candidate family")
    check(endpoints["gl4_integral_only"] in candidates, "integral endpoint outside candidate family")
    check(endpoints["gl4_hybrid"] in candidates, "hybrid endpoint outside candidate family")
    check(
        endpoints["trapezoidal_hybrid"] is None,
        "trapezoidal family must have no selected endpoint",
    )
    hybrid = selection["selected_hybrid_allocation"]
    matched = selection["matched_strong_control_allocation"]
    check(
        math.isclose(
            hybrid["gamma_s"] + hybrid["gamma_i"],
            endpoints["gl4_hybrid"],
        ),
        "hybrid budget allocation mismatch",
    )
    check(
        hybrid["gamma_s"] == matched["gamma_s"] == 0.02,
        "matched strong weights are not fixed",
    )
    check(matched["gamma_i"] == 0.0, "matched control must disable the integral branch")
    check(
        config["higher_resolution_comparison"]["role"] == "post_selection_only",
        "higher-resolution comparison may not enter selection",
    )
    role_files = config.get("point_role_files", {})
    check(
        role_files.get("optimization_balance_fit")
        == "point_sets/lid_driven_cavity/integral_time_roles/integral_time_fit_indices.txt",
        "optimization balance-fit path mismatch",
    )
    check(
        role_files.get("optimization_balance_validation")
        == "point_sets/lid_driven_cavity/integral_time_roles/integral_time_validation_indices.txt",
        "optimization balance-validation path mismatch",
    )
    check(
        role_files.get("post_selection_reporting_plan")
        == "data/lid_driven_cavity/reporting_plan.npz",
        "post-selection reporting-plan path mismatch",
    )


def validate_config(config, expected_case_id):
    label = "config {}".format(expected_case_id)
    config = require_mapping(config, label)
    require_keys(
        config,
        [
            "schema_version",
            "case_id",
            "title",
            "artifact_directory",
            "artifact_paths",
            "manuscript_sources",
            "domain",
            "physics",
            "spline",
            "point_roles",
            "selection",
            "optimizer",
            "expected_model_ids",
        ],
        label,
    )
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValidationError("{} has unsupported schema version".format(label))
    if config["case_id"] != expected_case_id:
        raise ValidationError("{} has case_id {!r}".format(label, config["case_id"]))
    if not isinstance(config["title"], str) or not config["title"].strip():
        raise ValidationError("{} has an empty title".format(label))
    if not isinstance(config["manuscript_sources"], list) or not config["manuscript_sources"]:
        raise ValidationError("{} must cite manuscript sources".format(label))
    model_ids = config["expected_model_ids"]
    if (
        not isinstance(model_ids, list)
        or not model_ids
        or any(not isinstance(model_id, str) or not model_id.strip() for model_id in model_ids)
    ):
        raise ValidationError("{}.expected_model_ids must be a nonempty string list".format(label))
    if len(model_ids) != len(set(model_ids)):
        raise ValidationError("{}.expected_model_ids contains duplicates".format(label))

    errors = []
    validate_artifact_paths(config, errors)
    if expected_case_id == "one_dimensional_convection_diffusion":
        validate_one_dimensional(config, errors)
    elif expected_case_id == "two_dimensional_burgers":
        validate_burgers(config, errors)
    elif expected_case_id == "lid_driven_cavity":
        validate_cavity(config, errors)
    if errors:
        raise ValidationError("\n".join(errors))


def lookup(document, dotted_path):
    value = document
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValidationError("unknown result path: {}".format(dotted_path))
        value = value[part]
    if not is_number(value):
        raise ValidationError("result path is not numeric: {}".format(dotted_path))
    return float(value)


def validate_numeric_tree(value, label):
    if isinstance(value, dict):
        if not value:
            raise ValidationError("{} may not be empty".format(label))
        for key, child in value.items():
            validate_numeric_tree(child, "{}.{}".format(label, key))
        return
    if not is_number(value):
        raise ValidationError("{} must be a finite number".format(label))


def metric_groups(document):
    groups = []

    def visit(value, path):
        if not isinstance(value, dict):
            return
        if {"precision", "tolerance", "source", "values"}.issubset(value):
            groups.append((path, value))
            return
        for key, child in value.items():
            if key != "derived_checks":
                visit(child, "{}.{}".format(path, key) if path else key)

    visit(document, "")
    return groups


def validate_expected_results(document):
    document = require_mapping(document, "expected_results")
    require_keys(
        document,
        ["schema_version", "precision_labels", "cases", "verification", "derived_checks"],
        "expected_results",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValidationError("expected_results has unsupported schema version")
    if set(document["precision_labels"]) != PRECISION_LABELS:
        raise ValidationError("expected_results precision labels do not match the validator")
    if set(document["cases"]) != set(CONFIG_FILES):
        raise ValidationError("expected_results case IDs do not match the configurations")

    groups = metric_groups(document)
    if not groups:
        raise ValidationError("expected_results contains no metric groups")
    for path, group in groups:
        if group["precision"] not in PRECISION_LABELS:
            raise ValidationError("{} has an unknown precision label".format(path))
        tolerance = require_mapping(group["tolerance"], "{}.tolerance".format(path))
        require_keys(tolerance, ["relative", "absolute"], "{}.tolerance".format(path))
        if not all(is_number(tolerance[key]) and tolerance[key] >= 0 for key in ("relative", "absolute")):
            raise ValidationError("{} has an invalid tolerance".format(path))
        if not isinstance(group["source"], str) or not group["source"].strip():
            raise ValidationError("{} must cite a source".format(path))
        validate_numeric_tree(group["values"], "{}.values".format(path))

    checks = document["derived_checks"]
    if not isinstance(checks, list) or not checks:
        raise ValidationError("expected_results must contain derived checks")
    names = set()
    computed = {}
    for check in checks:
        require_keys(check, ["name", "operation", "reported", "tolerance_percentage_points"], "derived check")
        name = check["name"]
        if name in names:
            raise ValidationError("duplicate derived check: {}".format(name))
        names.add(name)
        operation = check["operation"]
        if operation in ("percent_change", "percent_reduction"):
            require_keys(check, ["baseline", "candidate"], name)
            baseline = lookup(document, check["baseline"])
            candidate = lookup(document, check["candidate"])
            if baseline == 0:
                raise ValidationError("{} has a zero baseline".format(name))
            if operation == "percent_change":
                value = 100.0 * (candidate / baseline - 1.0)
            else:
                value = 100.0 * (1.0 - candidate / baseline)
        elif operation == "geometric_mean_percent_change":
            require_keys(check, ["baselines", "candidates"], name)
            if len(check["baselines"]) != len(check["candidates"]) or not check["baselines"]:
                raise ValidationError("{} has incompatible path lists".format(name))
            ratios = []
            for baseline_path, candidate_path in zip(check["baselines"], check["candidates"]):
                baseline = lookup(document, baseline_path)
                candidate = lookup(document, candidate_path)
                if baseline <= 0 or candidate < 0:
                    raise ValidationError("{} requires positive baselines and nonnegative candidates".format(name))
                ratios.append(candidate / baseline)
            value = 100.0 * (product(ratios) ** (1.0 / len(ratios)) - 1.0)
        else:
            raise ValidationError("{} has unsupported operation {!r}".format(name, operation))
        if not is_number(check["reported"]) or not is_number(check["tolerance_percentage_points"]):
            raise ValidationError("{} has nonnumeric comparison data".format(name))
        difference = abs(value - check["reported"])
        if difference > check["tolerance_percentage_points"] + 1e-12:
            raise ValidationError(
                "{} recomputes to {:.8g}%, outside {:.8g}% +/- {:.8g} percentage points".format(
                    name,
                    value,
                    check["reported"],
                    check["tolerance_percentage_points"],
                )
            )
        computed[name] = value

    values = document["cases"]["lid_driven_cavity"]["stage_a_normalizers_and_selected_weights"]["values"]
    formulas = {
        "lambda_x": values["gamma_s"] * values["J_A"] / (3.0 * values["R_A_x"]),
        "lambda_y": values["gamma_s"] * values["J_A"] / (3.0 * values["R_A_y"]),
        "lambda_c": values["gamma_s"] * values["J_A"] / (3.0 * values["R_A_c"]),
        "lambda_I_m": values["gamma_i"] * values["J_A"] / (3.0 * values["I_A_GL4_m"]),
        "lambda_I_x": values["gamma_i"] * values["J_A"] / (3.0 * values["I_A_GL4_x"]),
        "lambda_I_y": values["gamma_i"] * values["J_A"] / (3.0 * values["I_A_GL4_y"]),
    }
    for name, recomputed in formulas.items():
        if not math.isclose(recomputed, values[name], rel_tol=1e-12, abs_tol=1e-18):
            raise ValidationError("cavity normalized weight does not reproduce: {}".format(name))

    return {"metric_groups": len(groups), "derived_checks": len(computed)}


def numeric_leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = "{}.{}".format(prefix, key) if prefix else key
            yield from numeric_leaves(child, child_path)
        return
    if not is_number(value):
        raise ValidationError("reported-value crosswalk encountered a nonnumeric leaf")
    yield prefix, float(value)


def expected_reported_location(group_path, value_path):
    one_d_prefix = "cases.one_dimensional_convection_diffusion"
    burgers_prefix = "cases.two_dimensional_burgers"
    cavity_prefix = "cases.lid_driven_cavity"

    if group_path == one_d_prefix + ".primary_metrics":
        parts = value_path.split(".")
        if len(parts) == 3 and parts[0] == "models":
            model = ONE_D_MODEL_ALIASES.get(parts[1])
            metric = ONE_D_METRIC_ALIASES.get(parts[2])
            if model and metric:
                return "one_d", "metrics.models.{}.{}".format(model, metric)

    if group_path == one_d_prefix + ".analytical_reference_resolution":
        if value_path in {
            "doubled_resolution_linf_change",
            "doubled_resolution_relative_l2_change",
        }:
            return (
                "one_d",
                "metrics.analytical_reference_resolution.{}".format(value_path),
            )

    if group_path == burgers_prefix + ".primary_metrics":
        parts = value_path.split(".")
        if len(parts) == 3 and parts[0] == "models":
            model = BURGERS_MODEL_ALIASES.get(parts[1])
            template = BURGERS_METRIC_PATHS.get(parts[2])
            if model and template:
                return "burgers", "metrics." + template.format(model=model)

    if group_path == burgers_prefix + ".reference_refinement":
        refinement_paths = {
            "grid_40_to_80_u": "n40_to_n80_relative_l2_u",
            "grid_40_to_80_v": "n40_to_n80_relative_l2_v",
            "grid_80_to_160_u": "n80_to_n160_relative_l2_u",
            "grid_80_to_160_v": "n80_to_n160_relative_l2_v",
            "grid_160_to_200_u": "n160_to_n200_relative_l2_u",
            "grid_160_to_200_v": "n160_to_n200_relative_l2_v",
            "halved_time_step_max_change": "dt_5e4_to_2p5e4_max_relative_l2",
        }
        metric = refinement_paths.get(value_path)
        if metric:
            return "burgers", "metrics.reference_refinement_at_t2.{}".format(metric)

    if group_path == cavity_prefix + ".primary_metrics":
        parts = value_path.split(".")
        if len(parts) == 3 and parts[0] == "models":
            model = CAVITY_MODEL_ALIASES.get(parts[1])
            balance_paths = {
                "canonical_q4_gl8_mass_rms": "canonical_q4.mass",
                "canonical_q4_gl8_x_momentum_rms": "canonical_q4.x_momentum",
                "canonical_q4_gl8_y_momentum_rms": "canonical_q4.y_momentum",
                "shifted_q4_gl8_mass_rms": "shifted_q4.mass",
                "shifted_q4_gl8_x_momentum_rms": "shifted_q4.x_momentum",
                "shifted_q4_gl8_y_momentum_rms": "shifted_q4.y_momentum",
                "whole_domain_q1_mass_rms": "whole_domain_q1.mass",
                "whole_domain_q1_x_momentum_rms": "whole_domain_q1.x_momentum",
                "whole_domain_q1_y_momentum_rms": "whole_domain_q1.y_momentum",
            }
            if model and parts[2] in balance_paths:
                return (
                    "cavity",
                    "metrics.gl8_balance.{}.{}".format(
                        model, balance_paths[parts[2]]
                    ),
                )
            velocity_components = {"test_u_rmse": "u", "test_v_rmse": "v"}
            if model and parts[2] in velocity_components:
                return (
                    "cavity",
                    "metrics.field_diagnostics.{}.velocity_rmse.{}".format(
                        model, velocity_components[parts[2]]
                    ),
                )

    if group_path == cavity_prefix + ".advective_conservative_metrics":
        parts = value_path.split(".")
        model_aliases = {
            "matched_strong_control": "matched_strong_gamma_i0",
            "selected_gl4_hybrid": "selected_gl4_hybrid",
        }
        metric_aliases = {
            "advective_x_rms": "advective_x_rmse",
            "advective_y_rms": "advective_y_rmse",
            "conservative_x_rms": "conservative_x_rmse",
            "conservative_y_rms": "conservative_y_rmse",
        }
        if len(parts) == 2:
            model = model_aliases.get(parts[0])
            metric = metric_aliases.get(parts[1])
            if model and metric:
                return (
                    "cavity",
                    "metrics.advective_conservative.{}.{}".format(model, metric),
                )

    if group_path == cavity_prefix + ".pressure_metrics":
        parts = value_path.split(".")
        model_aliases = {
            "matched_strong_control": "matched_strong_gamma_i0",
            "selected_gl4_hybrid": "selected_gl4_hybrid",
        }
        if len(parts) == 2 and parts[1] in {
            "pressure_rmse",
            "pressure_relative_l2",
            "midpoint_gradient_relative_l2",
            "fine_gradient_relative_l2",
        }:
            model = model_aliases.get(parts[0])
            if model:
                return "cavity", "metrics.pressure.{}.{}".format(model, parts[1])

    if group_path == cavity_prefix + ".stage_a_normalizers_and_selected_weights":
        normalizer_paths = {
            "gamma_s": "gamma_s",
            "gamma_i": "gamma_i",
            "J_A": "stage_a_objective",
            "R_A_x": "stage_a_strong_mse.x_momentum",
            "R_A_y": "stage_a_strong_mse.y_momentum",
            "R_A_c": "stage_a_strong_mse.divergence",
            "I_A_GL4_m": "stage_a_gl4_mse.mass",
            "I_A_GL4_x": "stage_a_gl4_mse.x_momentum",
            "I_A_GL4_y": "stage_a_gl4_mse.y_momentum",
            "I_A_TH_m": "stage_a_trapezoid_mse.mass",
            "I_A_TH_x": "stage_a_trapezoid_mse.x_momentum",
            "I_A_TH_y": "stage_a_trapezoid_mse.y_momentum",
            "lambda_x": "selected_strong_weights.x_momentum",
            "lambda_y": "selected_strong_weights.y_momentum",
            "lambda_c": "selected_strong_weights.divergence",
            "lambda_I_m": "selected_gl4_weights.mass",
            "lambda_I_x": "selected_gl4_weights.x_momentum",
            "lambda_I_y": "selected_gl4_weights.y_momentum",
        }
        metric = normalizer_paths.get(value_path)
        if metric:
            return (
                "cavity",
                "metrics.stage_a_normalizers_and_selected_weights.{}".format(metric),
            )

    if group_path == "verification" and "." not in value_path:
        return "verification", "metrics.{}".format(value_path)

    raise ValidationError(
        "expected metric has no reported-value crosswalk: {}.values.{}".format(
            group_path, value_path
        )
    )


def load_reported_metric_documents(root):
    documents = {}
    for name, relative_name in REPORTED_METRIC_FILES.items():
        document = require_mapping(
            load_json(root / Path(relative_name)), relative_name
        )
        metrics = require_mapping(document.get("metrics"), "{}.metrics".format(relative_name))
        documents[name] = document
        if not metrics:
            raise ValidationError("{} has no reported metrics".format(relative_name))
    return documents


def validate_expected_reported_crosswalk(root, expected, reported_documents=None):
    documents = (
        load_reported_metric_documents(root)
        if reported_documents is None
        else reported_documents
    )
    comparisons = 0
    for group_path, group in metric_groups(expected):
        tolerance = group["tolerance"]
        for value_path, expected_value in numeric_leaves(group["values"]):
            document_name, reported_path = expected_reported_location(
                group_path, value_path
            )
            if document_name not in documents:
                raise ValidationError(
                    "missing reported-metric document: {}".format(document_name)
                )
            reported_value = lookup(documents[document_name], reported_path)
            if not math.isclose(
                reported_value,
                expected_value,
                rel_tol=tolerance["relative"],
                abs_tol=tolerance["absolute"],
            ):
                raise ValidationError(
                    "reported metric {} = {:.17g} does not match expected {:.17g}".format(
                        reported_path, reported_value, expected_value
                    )
                )
            comparisons += 1
    return comparisons


def validate_expected_model_ids(configs, expected_results):
    cases = require_mapping(expected_results.get("cases"), "expected_results.cases")
    for case_id, config in configs.items():
        case_results = require_mapping(cases.get(case_id), "expected_results.cases.{}".format(case_id))
        result_model_ids = set()
        for path, group in metric_groups(case_results):
            values = require_mapping(group["values"], "{}.values".format(path))
            if "models" not in values:
                continue
            models = require_mapping(values["models"], "{}.values.models".format(path))
            for model_id in models:
                if not isinstance(model_id, str) or not model_id.strip():
                    raise ValidationError("{}.values.models has an invalid model ID".format(path))
            result_model_ids.update(models)
        if not result_model_ids:
            raise ValidationError("{} has no metric group with values.models".format(case_id))
        config_model_ids = set(config["expected_model_ids"])
        if config_model_ids != result_model_ids:
            missing = sorted(config_model_ids - result_model_ids)
            extra = sorted(result_model_ids - config_model_ids)
            details = []
            if missing:
                details.append("missing from results: {}".format(", ".join(missing)))
            if extra:
                details.append("not declared by config: {}".format(", ".join(extra)))
            raise ValidationError("{} model IDs disagree ({})".format(case_id, "; ".join(details)))


def load_index_file(path, expected_count, label):
    try:
        stream = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValidationError("missing index file: {}".format(path)) from exc

    values = []
    with stream:
        for line_number, raw_line in enumerate(stream, 1):
            value = raw_line.strip()
            if not re.fullmatch(r"[0-9]+", value):
                raise ValidationError("{}:{} is not an unsigned integer".format(label, line_number))
            values.append(int(value))
    if len(values) != expected_count:
        raise ValidationError(
            "{} has {} lines; expected {}".format(label, len(values), expected_count)
        )
    if len(values) != len(set(values)):
        raise ValidationError("{} contains duplicate indices".format(label))
    return set(values)


def validate_integral_time_pair(case_root, fit_name, fit_count, validation_name, validation_count):
    fit_values = load_index_file(case_root / fit_name, fit_count, "{}/{}".format(case_root.name, fit_name))
    validation_values = load_index_file(
        case_root / validation_name,
        validation_count,
        "{}/{}".format(case_root.name, validation_name),
    )
    overlap = fit_values & validation_values
    if overlap:
        sample = ", ".join(str(value) for value in sorted(overlap)[:5])
        raise ValidationError("{} integral-time roles overlap at {}".format(case_root.name, sample))


def validate_numbered_series(directory, prefix, count, label):
    if not directory.is_dir():
        raise ValidationError("missing data directory: {}".format(directory))
    pattern = re.compile(r"^{}([0-9]{{4}})\.txt$".format(re.escape(prefix)))
    indices = {
        int(match.group(1))
        for path in directory.iterdir()
        if path.is_file()
        for match in [pattern.fullmatch(path.name)]
        if match is not None
    }
    expected = set(range(count))
    if indices != expected:
        missing = sorted(expected - indices)
        extra = sorted(indices - expected)
        details = []
        if missing:
            details.append("missing {}".format(", ".join(str(value) for value in missing[:5])))
        if extra:
            details.append("unexpected {}".format(", ".join(str(value) for value in extra[:5])))
        raise ValidationError("{} series is incomplete ({})".format(label, "; ".join(details)))


def validate_digest_map(directory, records, label):
    for relative_name, expected_digest in records.items():
        relative_path = PurePosixPath(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValidationError("{} has an unsafe path: {}".format(label, relative_name))
        if not isinstance(expected_digest, str) or not SHA256_PATTERN.fullmatch(expected_digest):
            raise ValidationError("{} has an invalid SHA256: {}".format(label, relative_name))
        candidate = directory / Path(*relative_path.parts)
        if not candidate.is_file():
            raise ValidationError("{} is missing {}".format(label, relative_name))
        if sha256(candidate) != expected_digest:
            raise ValidationError("{} SHA256 mismatch: {}".format(label, relative_name))


def validate_declared_digests(records, label):
    records = require_mapping(records, label)
    if not records:
        raise ValidationError("{} is empty".format(label))
    for name, digest in records.items():
        if not isinstance(name, str) or not name:
            raise ValidationError("{} has an invalid name".format(label))
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ValidationError("{} has an invalid SHA256: {}".format(label, name))


def validate_assets(root):
    for category in ("data", "point_sets", "results", "checkpoints"):
        for directory in ARTIFACT_DIRECTORIES.values():
            path = root / category / directory
            if not path.is_dir():
                raise ValidationError("missing artifact directory: {}".format(path))

    data_root = root / "data"
    validate_numbered_series(
        data_root / "one_d_convection_diffusion",
        "rho_T3_",
        400,
        "one-dimensional observations",
    )
    for role in ("observed", "reference"):
        validate_numbered_series(
            data_root / "two_d_burgers" / role,
            "uv_times_",
            201,
            "Burgers {} snapshots".format(role),
        )
    cavity_root = data_root / "lid_driven_cavity"
    validate_numbered_series(cavity_root / "snapshots", "coarse_", 1001, "cavity snapshots")
    validate_numbered_series(
        cavity_root / "dynamic_boundaries", "BCx_", 1001, "cavity x-boundary records"
    )
    validate_numbered_series(
        cavity_root / "dynamic_boundaries", "BCy_", 1001, "cavity y-boundary records"
    )

    one_d_manifest = load_json(
        data_root / "one_d_convection_diffusion" / "SELECTION_INPUT_MANIFEST.json"
    )
    one_d_records = {}
    for record in one_d_manifest["entries"]:
        candidate = data_root / "one_d_convection_diffusion" / record["path"]
        if not candidate.is_file():
            raise ValidationError("one-dimensional data is missing {}".format(record["path"]))
        if candidate.stat().st_size != record["bytes"]:
            raise ValidationError("one-dimensional data byte count mismatch: {}".format(record["path"]))
        one_d_records[record["path"]] = record["sha256"]
    validate_digest_map(
        data_root / "one_d_convection_diffusion", one_d_records, "one-dimensional data"
    )

    cavity_input_manifest = load_json(cavity_root / "selection_input_manifest_a2.json")
    dynamic_record = require_mapping(
        cavity_input_manifest.get("dynamic_boundaries"),
        "cavity selection-input dynamic boundaries",
    )
    if dynamic_record.get("source_manifest_path") != "dynamic_boundaries_manifest.json":
        raise ValidationError("cavity dynamic-boundary manifest path is not package-relative")
    validate_digest_map(
        cavity_root / "snapshots",
        cavity_input_manifest["snapshot_inventory"]["file_sha256"],
        "cavity snapshots",
    )
    dynamic_manifest = load_json(cavity_root / "dynamic_boundaries_manifest.json")
    validate_digest_map(
        cavity_root / "dynamic_boundaries",
        dynamic_manifest["file_sha256"],
        "cavity dynamic boundaries",
    )

    point_root = root / "point_sets"
    for relative_name, expected_count in INDEX_FILE_COUNTS.items():
        load_index_file(point_root / relative_name, expected_count, relative_name)
    for case, role_specification in INTEGRAL_TIME_ROLES.items():
        validate_integral_time_pair(point_root / case, *role_specification)

    burgers_point_root = point_root / "two_d_burgers"
    burgers_design_manifest = load_json(burgers_point_root / "manifest.json")
    if burgers_design_manifest.get("data_source") != "../../data/two_d_burgers/observed":
        raise ValidationError("Burgers point-set data source is not package-relative")
    if burgers_design_manifest.get("design_file") != "design.npz":
        raise ValidationError("Burgers point-set design path mismatch")
    burgers_design = burgers_point_root / "design.npz"
    if sha256(burgers_design) != burgers_design_manifest.get("design_file_sha256"):
        raise ValidationError("Burgers point-set design SHA256 mismatch")

    cavity_design = point_root / "lid_driven_cavity" / "design.npz"
    cavity_design_manifest = load_json(
        point_root / "lid_driven_cavity" / "manifest.json"
    )
    expected_design_digest = cavity_design_manifest["repackaging"][
        "packaged_design_sha256"
    ]
    if sha256(cavity_design) != expected_design_digest:
        raise ValidationError("cavity packaged design SHA256 mismatch")

    cavity_roles_root = point_root / "lid_driven_cavity" / "integral_time_roles"
    cavity_roles = load_json(cavity_roles_root / "role_manifest.json")
    for role, expected_name in (
        ("fit", "integral_time_fit_indices.txt"),
        ("validation", "integral_time_validation_indices.txt"),
    ):
        record = require_mapping(cavity_roles.get(role), "cavity {} role".format(role))
        if record.get("path") != expected_name:
            raise ValidationError("cavity {} role path is not package-relative".format(role))
        if sha256(cavity_roles_root / expected_name) != record.get("sha256"):
            raise ValidationError("cavity {} role SHA256 mismatch".format(role))

    reporting_manifest = load_json(
        cavity_root / "reporting_plan_manifest.json"
    )
    reporting_plan = cavity_root / reporting_manifest["npz"]["path"]
    if sha256(reporting_plan) != reporting_manifest["npz"]["sha256"]:
        raise ValidationError("cavity reporting-plan SHA256 mismatch")
    validate_declared_digests(
        reporting_manifest["generator_source_sha256"],
        "omitted cavity generator provenance",
    )
    validate_declared_digests(
        {
            "RAMPED_CAVITY_GL4_PROTOCOL_V1.md": reporting_manifest[
                "frozen_configuration"
            ]["protocol_sha256"],
            "RAMPED_CAVITY_GL4_PROTOCOL_V1_AMENDMENT_1.md": reporting_manifest[
                "frozen_configuration"
            ]["amendment_sha256"],
        },
        "omitted cavity protocol provenance",
    )

    return (
        len(ARTIFACT_DIRECTORIES) * 4
        + 6
        + len(INDEX_FILE_COUNTS)
        + len(INTEGRAL_TIME_ROLES)
        + 9
        + 2
        + 2
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_burgers_forcing_manifest(root):
    generation_root = root / "calculations" / "two_d_burgers" / "generation"
    manifest = require_mapping(
        load_json(generation_root / "forcing_realization_manifest.json"),
        "Burgers forcing manifest",
    )
    require_keys(
        manifest,
        [
            "schema_version",
            "artifact",
            "contract",
            "parameters",
            "npz",
            "array_contract_sha256",
            "published_generator_sha256",
        ],
        "Burgers forcing manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValidationError("Burgers forcing manifest has unsupported schema")
    npz = require_mapping(manifest["npz"], "Burgers forcing manifest npz")
    require_keys(npz, ["path", "sha256"], "Burgers forcing manifest npz")
    relative_path = PurePosixPath(npz["path"])
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.as_posix() != npz["path"]
        or len(relative_path.parts) != 1
    ):
        raise ValidationError("Burgers forcing manifest has an unsafe NPZ path")
    npz_path = generation_root / Path(*relative_path.parts)
    if not npz_path.is_file() or sha256(npz_path) != npz["sha256"]:
        raise ValidationError("Burgers forcing-realization NPZ SHA256 mismatch")

    array_hashes = require_mapping(
        manifest["array_contract_sha256"], "Burgers forcing array hashes"
    )
    if not array_hashes or any(
        not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest)
        for digest in array_hashes.values()
    ):
        raise ValidationError("Burgers forcing manifest has invalid array hashes")
    generator_digest = manifest["published_generator_sha256"]
    if not isinstance(generator_digest, str) or not SHA256_PATTERN.fullmatch(
        generator_digest
    ):
        raise ValidationError("Burgers forcing manifest has an invalid generator hash")
    archived_generator = (
        root
        / "calculations"
        / "two_d_burgers"
        / "archived_source"
        / "generate_burgers_observed.py"
    )
    if not archived_generator.is_file() or sha256(archived_generator) != generator_digest:
        raise ValidationError("published Burgers generator SHA256 mismatch")
    return 2 + len(array_hashes)


def validate_checkpoint_manifest(root, configs):
    checkpoint_root = root / "checkpoints"
    manifest_path = checkpoint_root / "manifest.json"
    document = require_mapping(load_json(manifest_path), "checkpoints/manifest.json")
    required_top_level = {"schema_version", "hash_algorithm", "checkpoints"}
    if set(document) != required_top_level:
        raise ValidationError("checkpoints/manifest.json has unexpected or missing top-level keys")
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValidationError("checkpoint manifest has unsupported schema version")
    if document["hash_algorithm"] != "sha256":
        raise ValidationError("checkpoint manifest must use sha256")
    records = document["checkpoints"]
    if not isinstance(records, list) or not records:
        raise ValidationError("checkpoint manifest must contain checkpoint records")

    config_by_directory = {
        ARTIFACT_DIRECTORIES[case_id]: config for case_id, config in configs.items()
    }
    record_keys = {"case", "model", "format", "path", "bytes", "sha256"}
    paths = set()
    case_models = set()
    seen_cases = set()
    for index, record in enumerate(records):
        label = "checkpoint record {}".format(index)
        record = require_mapping(record, label)
        if set(record) != record_keys:
            raise ValidationError("{} has unexpected or missing keys".format(label))

        case = record["case"]
        model = record["model"]
        if not isinstance(case, str) or not isinstance(model, str):
            raise ValidationError("{} has an invalid case or model".format(label))
        if case not in config_by_directory:
            raise ValidationError("{} has unknown case {!r}".format(label, case))
        allowed_models = CHECKPOINT_MODELS[case]
        if model not in allowed_models:
            raise ValidationError("{} has a model inconsistent with {}".format(label, case))
        linked_model = allowed_models[model]
        if linked_model is not None and linked_model not in config_by_directory[case]["expected_model_ids"]:
            raise ValidationError("{} links to an undeclared expected model".format(label))
        if (case, model) in case_models:
            raise ValidationError("duplicate checkpoint case/model: {}/{}".format(case, model))
        case_models.add((case, model))
        seen_cases.add(case)

        if not isinstance(record["format"], str) or record["format"] not in {
            "mfa",
            "plain_control_vector",
        }:
            raise ValidationError("{} has unsupported format".format(label))
        relative_name = record["path"]
        if not isinstance(relative_name, str) or not relative_name:
            raise ValidationError("{} has an invalid path".format(label))
        relative_path = PurePosixPath(relative_name)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative_name
            or not relative_path.parts
            or relative_path.parts[0] != case
        ):
            raise ValidationError("{} has an unsafe or case-inconsistent path".format(label))
        if relative_name in paths:
            raise ValidationError("duplicate checkpoint path: {}".format(relative_name))
        paths.add(relative_name)

        candidate = (checkpoint_root / Path(*relative_path.parts)).resolve()
        try:
            candidate.relative_to(checkpoint_root.resolve())
        except ValueError as exc:
            raise ValidationError("{} resolves outside checkpoints".format(label)) from exc
        if not candidate.is_file():
            raise ValidationError("missing checkpoint: {}".format(relative_name))
        expected_bytes = record["bytes"]
        if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes <= 0:
            raise ValidationError("{} has an invalid byte count".format(label))
        if candidate.stat().st_size != expected_bytes:
            raise ValidationError("checkpoint byte count mismatch: {}".format(relative_name))
        expected_digest = record["sha256"]
        if not isinstance(expected_digest, str) or not SHA256_PATTERN.fullmatch(expected_digest):
            raise ValidationError("{} has an invalid SHA256 digest".format(label))
        if sha256(candidate) != expected_digest:
            raise ValidationError("checkpoint SHA256 mismatch: {}".format(relative_name))

    if seen_cases != set(config_by_directory):
        missing = sorted(set(config_by_directory) - seen_cases)
        raise ValidationError("checkpoint manifest has no records for: {}".format(", ".join(missing)))
    return len(records)


def validate_calculation_records(root):
    source_count = 0
    for case_id, directory in CALCULATION_DIRECTORIES.items():
        calculation_root = root / "calculations" / directory
        for filename in ("README.md", "audit_results.py", "reported_metrics.json"):
            if not (calculation_root / filename).is_file():
                raise ValidationError(
                    "missing calculation record: {}/{}".format(directory, filename)
                )

        record = require_mapping(
            load_json(calculation_root / "reported_metrics.json"),
            "{}/reported_metrics.json".format(directory),
        )
        require_keys(
            record,
            ["schema_version", "case", "metrics", "packaged_sources"],
            "{}/reported_metrics.json".format(directory),
        )
        if record["schema_version"] != SCHEMA_VERSION:
            raise ValidationError("{} calculation record has unsupported schema".format(case_id))
        if record["case"] != directory:
            raise ValidationError("{} calculation record has the wrong case".format(case_id))
        require_mapping(record["metrics"], "{}.metrics".format(case_id))
        sources = require_mapping(record["packaged_sources"], "{}.packaged_sources".format(case_id))
        if not sources:
            raise ValidationError("{} calculation record has no packaged sources".format(case_id))
        for label, relative_name in sources.items():
            if not isinstance(label, str) or not label:
                raise ValidationError("{} calculation record has an invalid source label".format(case_id))
            if not isinstance(relative_name, str) or not relative_name:
                raise ValidationError("{} calculation record has an invalid source path".format(case_id))
            relative_path = PurePosixPath(relative_name)
            permitted_roots = {
                ("results", directory),
                ("point_sets", directory),
            }
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative_path.as_posix() != relative_name
                or tuple(relative_path.parts[:2]) not in permitted_roots
            ):
                raise ValidationError(
                    "{} calculation source is unsafe or case-inconsistent: {}".format(
                        case_id, relative_name
                    )
                )
            candidate = root / Path(*relative_path.parts)
            if not candidate.exists():
                raise ValidationError(
                    "{} calculation source is missing: {}".format(case_id, relative_name)
                )
            if candidate.is_dir() and not any(path.is_file() for path in candidate.rglob("*")):
                raise ValidationError(
                    "{} calculation source directory is empty: {}".format(
                        case_id, relative_name
                    )
                )
            source_count += 1

    verification_root = root / "calculations" / "verification"
    for filename in ("README.md", "audit_results.py", "reported_metrics.json"):
        if not (verification_root / filename).is_file():
            raise ValidationError("missing package-wide calculation record: {}".format(filename))
    verification = require_mapping(
        load_json(verification_root / "reported_metrics.json"),
        "verification/reported_metrics.json",
    )
    require_keys(
        verification,
        ["schema_version", "case", "metrics", "packaged_sources", "source_sha256"],
        "verification/reported_metrics.json",
    )
    if verification["schema_version"] != SCHEMA_VERSION:
        raise ValidationError("package-wide verification record has unsupported schema")
    if verification["case"] != "package_wide_verification":
        raise ValidationError("package-wide verification record has the wrong case")
    validate_numeric_tree(verification["metrics"], "verification.metrics")
    sources = require_mapping(
        verification["packaged_sources"], "verification.packaged_sources"
    )
    source_hashes = require_mapping(
        verification["source_sha256"], "verification.source_sha256"
    )
    if set(sources.values()) != set(source_hashes):
        raise ValidationError("package-wide verification source paths and hashes disagree")
    for label, relative_name in sources.items():
        if not isinstance(label, str) or not label:
            raise ValidationError("package-wide verification has an invalid source label")
        if not isinstance(relative_name, str) or not relative_name:
            raise ValidationError("package-wide verification has an invalid source path")
        relative_path = PurePosixPath(relative_name)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative_name
            or len(relative_path.parts) < 3
            or relative_path.parts[0] != "results"
            or relative_path.parts[1] not in RESULT_INVENTORY_DIRECTORIES
        ):
            raise ValidationError(
                "package-wide verification source is unsafe: {}".format(relative_name)
            )
        candidate = root / Path(*relative_path.parts)
        expected_digest = source_hashes[relative_name]
        if not candidate.is_file():
            raise ValidationError(
                "package-wide verification source is missing: {}".format(relative_name)
            )
        if not isinstance(expected_digest, str) or not SHA256_PATTERN.fullmatch(
            expected_digest
        ):
            raise ValidationError(
                "package-wide verification has an invalid SHA256: {}".format(
                    relative_name
                )
            )
        if sha256(candidate) != expected_digest:
            raise ValidationError(
                "package-wide verification SHA256 mismatch: {}".format(relative_name)
            )
        source_count += 1
    return {"cases": len(CALCULATION_DIRECTORIES) + 1, "sources": source_count}


def package_files(root, manifest):
    excluded_names = {"__pycache__", "work"}
    files = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root)
        if path.resolve() == manifest or relative_path.as_posix() == "SHA256SUMS":
            continue
        if any(part in excluded_names for part in relative_path.parts):
            continue
        if path.suffix == ".pyc":
            continue
        files.add(relative_path.as_posix())
    return files


def validate_manifest(root, manifest):
    root = root.resolve()
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise ValidationError("missing SHA256 manifest: {}".format(manifest))
    entries = {}
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = MANIFEST_PATTERN.fullmatch(line)
        if not match:
            raise ValidationError("{}:{} has invalid SHA256 syntax".format(manifest, line_number))
        expected_digest, relative_name = match.groups()
        relative_path = PurePosixPath(relative_name)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative_name
        ):
            raise ValidationError("{}:{} has an unsafe path".format(manifest, line_number))
        if relative_name in entries:
            raise ValidationError("{}:{} repeats {}".format(manifest, line_number, relative_name))
        candidate = (root / Path(*relative_path.parts)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValidationError("{}:{} resolves outside the package".format(manifest, line_number)) from exc
        if candidate == manifest:
            raise ValidationError("the SHA256 manifest may not list itself")
        if not candidate.is_file():
            raise ValidationError("manifest entry is missing: {}".format(relative_name))
        actual_digest = sha256(candidate)
        if actual_digest != expected_digest:
            raise ValidationError("SHA256 mismatch: {}".format(relative_name))
        entries[relative_name] = expected_digest
    if not entries:
        raise ValidationError("SHA256 manifest has no file entries")
    expected_files = package_files(root, manifest)
    listed_files = set(entries)
    missing = sorted(expected_files - listed_files)
    extra = sorted(listed_files - expected_files)
    if missing or extra:
        details = []
        if missing:
            details.append("missing entries: {}".format(", ".join(missing[:5])))
        if extra:
            details.append("extra entries: {}".format(", ".join(extra[:5])))
        raise ValidationError("SHA256 manifest is incomplete ({})".format("; ".join(details)))
    return len(entries)


def validate_result_subset_manifests(root):
    entry_count = 0
    for directory in RESULT_INVENTORY_DIRECTORIES:
        result_root = (root / "results" / directory).resolve()
        manifest = result_root / SUBSET_MANIFEST_NAME
        if not manifest.is_file():
            raise ValidationError("missing result inventory: {}".format(manifest))
        lines = manifest.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != SUBSET_MANIFEST_HEADER:
            raise ValidationError("invalid result-inventory header: {}".format(manifest))

        entries = set()
        for line_number, line in enumerate(lines[1:], 2):
            fields = line.split("\t")
            if len(fields) != 3:
                raise ValidationError(
                    "{}:{} has invalid tab-separated syntax".format(manifest, line_number)
                )
            relative_name, expected_digest, byte_text = fields
            relative_path = PurePosixPath(relative_name)
            if (
                not relative_name
                or relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative_path.as_posix() != relative_name
            ):
                raise ValidationError("{}:{} has an unsafe path".format(manifest, line_number))
            if relative_name in entries:
                raise ValidationError(
                    "{}:{} repeats {}".format(manifest, line_number, relative_name)
                )
            if not SHA256_PATTERN.fullmatch(expected_digest):
                raise ValidationError(
                    "{}:{} has an invalid SHA256 digest".format(manifest, line_number)
                )
            try:
                expected_bytes = int(byte_text)
            except ValueError as exc:
                raise ValidationError(
                    "{}:{} has an invalid byte count".format(manifest, line_number)
                ) from exc
            if expected_bytes < 0 or str(expected_bytes) != byte_text:
                raise ValidationError(
                    "{}:{} has an invalid byte count".format(manifest, line_number)
                )

            candidate = (result_root / Path(*relative_path.parts)).resolve()
            try:
                candidate.relative_to(result_root)
            except ValueError as exc:
                raise ValidationError(
                    "{}:{} resolves outside its result directory".format(manifest, line_number)
                ) from exc
            if candidate == manifest or not candidate.is_file():
                raise ValidationError(
                    "result-inventory entry is missing or self-referential: {}".format(
                        relative_name
                    )
                )
            if candidate.stat().st_size != expected_bytes:
                raise ValidationError("result-inventory byte count mismatch: {}".format(relative_name))
            if sha256(candidate) != expected_digest:
                raise ValidationError("result-inventory SHA256 mismatch: {}".format(relative_name))
            entries.add(relative_name)

        expected_files = {
            path.relative_to(result_root).as_posix()
            for path in result_root.rglob("*")
            if path.is_file()
            and path != manifest
            and path.suffix != ".pyc"
            and not any(part in {"__pycache__", "work"} for part in path.parts)
        }
        if entries != expected_files:
            missing = sorted(expected_files - entries)
            extra = sorted(entries - expected_files)
            details = []
            if missing:
                details.append("missing entries: {}".format(", ".join(missing[:5])))
            if extra:
                details.append("extra entries: {}".format(", ".join(extra[:5])))
            raise ValidationError(
                "result inventory is incomplete for {} ({})".format(
                    directory, "; ".join(details)
                )
            )
        if not entries:
            raise ValidationError("result inventory is empty for {}".format(directory))
        entry_count += len(entries)
    return {"directories": len(RESULT_INVENTORY_DIRECTORIES), "entries": entry_count}


def validate_package(root, manifest=None, require_manifest=False):
    root = root.resolve()
    configs = {}
    for case_id, filename in CONFIG_FILES.items():
        config = load_json(root / "configs" / filename)
        validate_config(config, case_id)
        configs[case_id] = config

    expected = load_json(root / "expected_results.json")
    expected_summary = validate_expected_results(expected)
    validate_expected_model_ids(configs, expected)
    reported_metric_comparisons = validate_expected_reported_crosswalk(root, expected)
    asset_checks = validate_assets(root)
    burgers_forcing_checks = validate_burgers_forcing_manifest(root)
    checkpoint_entries = validate_checkpoint_manifest(root, configs)
    calculation_summary = validate_calculation_records(root)
    result_inventory_summary = validate_result_subset_manifests(root)

    selected_manifest = manifest
    if selected_manifest is None:
        default_manifest = root / "SHA256SUMS"
        if default_manifest.exists():
            selected_manifest = default_manifest
    elif not selected_manifest.is_absolute():
        selected_manifest = root / selected_manifest

    manifest_entries = 0
    if selected_manifest is not None:
        manifest_entries = validate_manifest(root, selected_manifest)
    elif require_manifest:
        raise ValidationError("SHA256SUMS is required but was not found")

    return {
        "configs": len(configs),
        "metric_groups": expected_summary["metric_groups"],
        "derived_checks": expected_summary["derived_checks"],
        "reported_metric_comparisons": reported_metric_comparisons,
        "asset_checks": asset_checks,
        "burgers_forcing_checks": burgers_forcing_checks,
        "checkpoint_entries": checkpoint_entries,
        "calculation_cases": calculation_summary["cases"],
        "calculation_sources": calculation_summary["sources"],
        "result_inventory_directories": result_inventory_summary["directories"],
        "result_inventory_entries": result_inventory_summary["entries"],
        "manifest_entries": manifest_entries,
    }


def parse_args(argv=None):
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root, help="reproducibility directory")
    parser.add_argument("--manifest", type=Path, help="optional SHA256 manifest")
    parser.add_argument("--require-manifest", action="store_true", help="fail when no manifest is available")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        summary = validate_package(args.root, args.manifest, args.require_manifest)
    except ValidationError as exc:
        print("Validation failed: {}".format(exc), file=sys.stderr)
        return 1
    manifest_message = "{} manifest entries".format(summary["manifest_entries"])
    if summary["manifest_entries"] == 0:
        manifest_message = "no manifest present"
    print(
        "Validated {} configs, {} metric groups, {} derived checks, {} expected-to-reported "
        "metric comparisons, {} asset checks, "
        "{} checkpoints, {} calculation records ({} sources), {} curated-result "
        "inventories ({} entries), and {} Burgers "
        "forcing checks; {}.".format(
            summary["configs"],
            summary["metric_groups"],
            summary["derived_checks"],
            summary["reported_metric_comparisons"],
            summary["asset_checks"],
            summary["checkpoint_entries"],
            summary["calculation_cases"],
            summary["calculation_sources"],
            summary["result_inventory_directories"],
            summary["result_inventory_entries"],
            summary["burgers_forcing_checks"],
            manifest_message,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

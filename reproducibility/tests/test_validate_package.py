import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import validate_package


class PackageValidationTests(unittest.TestCase):
    def test_repository_metadata(self):
        summary = validate_package.validate_package(PACKAGE_ROOT)
        self.assertEqual(summary["configs"], 3)
        self.assertGreaterEqual(summary["metric_groups"], 8)
        self.assertGreaterEqual(summary["derived_checks"], 20)
        self.assertGreaterEqual(summary["asset_checks"], 30)
        self.assertEqual(summary["checkpoint_entries"], 10)
        self.assertEqual(summary["calculation_cases"], 4)
        self.assertEqual(summary["reported_metric_comparisons"], 124)
        self.assertEqual(summary["burgers_forcing_checks"], 12)

    def test_rejects_reported_metric_drift(self):
        expected = validate_package.load_json(PACKAGE_ROOT / "expected_results.json")
        reported = validate_package.load_reported_metric_documents(PACKAGE_ROOT)
        modified = copy.deepcopy(reported)
        modified["one_d"]["metrics"]["models"]["data_only"][
            "midpoint_data_mse"
        ] = 1.0
        with self.assertRaises(validate_package.ValidationError):
            validate_package.validate_expected_reported_crosswalk(
                PACKAGE_ROOT, expected, modified
            )

    def test_rejects_selected_weight_outside_candidate_family(self):
        path = PACKAGE_ROOT / "configs" / "two_dimensional_burgers.json"
        config = validate_package.load_json(path)
        modified = copy.deepcopy(config)
        modified["selection"]["integral_balance"]["selected_gamma"] = 0.05
        with self.assertRaises(validate_package.ValidationError):
            validate_package.validate_config(modified, "two_dimensional_burgers")

    def test_rejects_inconsistent_reported_change(self):
        expected = validate_package.load_json(PACKAGE_ROOT / "expected_results.json")
        modified = copy.deepcopy(expected)
        modified["derived_checks"][0]["reported"] = 80.0
        with self.assertRaises(validate_package.ValidationError):
            validate_package.validate_expected_results(modified)

    def test_rejects_expected_model_id_not_present_in_results(self):
        configs = {}
        for case_id, filename in validate_package.CONFIG_FILES.items():
            configs[case_id] = validate_package.load_json(PACKAGE_ROOT / "configs" / filename)
        expected = validate_package.load_json(PACKAGE_ROOT / "expected_results.json")
        modified = copy.deepcopy(configs)
        modified["two_dimensional_burgers"]["expected_model_ids"].append("unreported_model")
        with self.assertRaises(validate_package.ValidationError):
            validate_package.validate_expected_model_ids(modified, expected)

    def test_rejects_wrong_one_dimensional_midpoint_count(self):
        path = PACKAGE_ROOT / "configs" / "one_dimensional_convection_diffusion.json"
        config = validate_package.load_json(path)
        modified = copy.deepcopy(config)
        modified["point_roles"]["fitting"]["pde_rows"] = None
        with self.assertRaises(validate_package.ValidationError):
            validate_package.validate_config(modified, "one_dimensional_convection_diffusion")

    def test_rejects_burgers_weight_selection_times_as_optimizer_validation(self):
        path = PACKAGE_ROOT / "configs" / "two_dimensional_burgers.json"
        config = validate_package.load_json(path)
        modified = copy.deepcopy(config)
        modified["point_roles"]["selection"]["optimizer_balance_validation_times"] = 200
        with self.assertRaises(validate_package.ValidationError):
            validate_package.validate_config(modified, "two_dimensional_burgers")

    def test_checkpoint_manifest_checks_case_model_size_and_digest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = []
            configs = {}
            selected_models = {
                "one_d_convection_diffusion": (
                    "one_dimensional_convection_diffusion",
                    "matched_strong_gamma_0",
                    "pibsr_strong_beta_3e_5",
                ),
                "two_d_burgers": (
                    "two_dimensional_burgers",
                    "matched_strong_beta_0.01_kappa_10_gamma_0",
                    "pibsr_strong_beta_0_01_kappa_10",
                ),
                "lid_driven_cavity": (
                    "lid_driven_cavity",
                    "matched_strong_gamma_s_0.02_gamma_i_0",
                    "matched_strong_control",
                ),
            }
            for case, (case_id, model, expected_model) in selected_models.items():
                payload = root / "checkpoints" / case / "weights.txt"
                payload.parent.mkdir(parents=True)
                payload.write_text(case + "\n", encoding="utf-8")
                configs[case_id] = {"expected_model_ids": [expected_model]}
                records.append(
                    {
                        "case": case,
                        "model": model,
                        "format": "plain_control_vector",
                        "path": "{}/weights.txt".format(case),
                        "bytes": payload.stat().st_size,
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                    }
                )
            manifest = {
                "schema_version": 1,
                "hash_algorithm": "sha256",
                "checkpoints": records,
            }
            manifest_path = root / "checkpoints" / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validate_package.validate_checkpoint_manifest(root, configs), 3)

            manifest["checkpoints"][0]["bytes"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(validate_package.ValidationError):
                validate_package.validate_checkpoint_manifest(root, configs)

            manifest["checkpoints"][0]["bytes"] -= 1
            first_path = root / "checkpoints" / manifest["checkpoints"][0]["path"]
            first_path.write_bytes(b"x" * first_path.stat().st_size)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(validate_package.ValidationError):
                validate_package.validate_checkpoint_manifest(root, configs)

            manifest["checkpoints"][0]["sha256"] = hashlib.sha256(first_path.read_bytes()).hexdigest()
            manifest["checkpoints"][0]["model"] = "model_from_another_case"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(validate_package.ValidationError):
                validate_package.validate_checkpoint_manifest(root, configs)

    def test_numbered_series_and_integral_roles(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            series = root / "series"
            series.mkdir()
            for index in range(3):
                (series / "sample_{:04d}.txt".format(index)).write_text("x\n", encoding="utf-8")
            validate_package.validate_numbered_series(series, "sample_", 3, "test series")
            (series / "sample_0002.txt").unlink()
            with self.assertRaises(validate_package.ValidationError):
                validate_package.validate_numbered_series(series, "sample_", 3, "test series")

            roles = root / "roles"
            roles.mkdir()
            (roles / "fit.txt").write_text("1\n2\n", encoding="utf-8")
            (roles / "validation.txt").write_text("2\n3\n", encoding="utf-8")
            with self.assertRaises(validate_package.ValidationError):
                validate_package.validate_integral_time_pair(
                    roles, "fit.txt", 2, "validation.txt", 2
                )

    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = root / "payload.txt"
            payload.write_text("PI-BSR\n", encoding="utf-8")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            manifest = root / "SHA256SUMS"
            manifest.write_text("{}  payload.txt\n".format(digest), encoding="utf-8")
            self.assertEqual(validate_package.validate_manifest(root, manifest), 1)

    def test_result_subset_manifest_round_trip_and_digest_check(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifests = []
            for directory in validate_package.RESULT_INVENTORY_DIRECTORIES:
                result_root = root / "results" / directory
                result_root.mkdir(parents=True)
                payload = result_root / "reported.tsv"
                payload.write_text("metric\tvalue\nloss\t1.0\n", encoding="utf-8")
                digest = hashlib.sha256(payload.read_bytes()).hexdigest()
                manifest = result_root / validate_package.SUBSET_MANIFEST_NAME
                manifest.write_text(
                    validate_package.SUBSET_MANIFEST_HEADER
                    + "\nreported.tsv\t{}\t{}\n".format(digest, payload.stat().st_size),
                    encoding="utf-8",
                )
                manifests.append(manifest)

            summary = validate_package.validate_result_subset_manifests(root)
            self.assertEqual(summary, {"directories": 4, "entries": 4})

            lines = manifests[0].read_text(encoding="utf-8").splitlines()
            fields = lines[1].split("\t")
            fields[1] = "0" * 64
            manifests[0].write_text(lines[0] + "\n" + "\t".join(fields) + "\n", encoding="utf-8")
            with self.assertRaises(validate_package.ValidationError):
                validate_package.validate_result_subset_manifests(root)

    def test_manifest_rejects_missing_regular_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = root / "payload.txt"
            unlisted = root / "unlisted.txt"
            payload.write_text("PI-BSR\n", encoding="utf-8")
            unlisted.write_text("must be listed\n", encoding="utf-8")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            manifest = root / "SHA256SUMS"
            manifest.write_text("{}  payload.txt\n".format(digest), encoding="utf-8")
            with self.assertRaises(validate_package.ValidationError):
                validate_package.validate_manifest(root, manifest)

    def test_manifest_rejects_generated_work_entry(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = root / "payload.txt"
            work = root / "work" / "scratch.txt"
            work.parent.mkdir()
            payload.write_text("PI-BSR\n", encoding="utf-8")
            work.write_text("generated\n", encoding="utf-8")
            lines = []
            for path in (payload, work):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                lines.append("{}  {}".format(digest, path.relative_to(root).as_posix()))
            manifest = root / "SHA256SUMS"
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(validate_package.ValidationError):
                validate_package.validate_manifest(root, manifest)

    def test_manifest_rejects_parent_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "SHA256SUMS"
            manifest.write_text("{}  ../outside.txt\n".format("0" * 64), encoding="utf-8")
            with self.assertRaises(validate_package.ValidationError):
                validate_package.validate_manifest(root, manifest)

    def test_declared_provenance_digest_validation(self):
        validate_package.validate_declared_digests(
            {"omitted_source.py": "0" * 64}, "omitted source provenance"
        )
        with self.assertRaises(validate_package.ValidationError):
            validate_package.validate_declared_digests(
                {"omitted_source.py": "not-a-sha256"},
                "omitted source provenance",
            )


if __name__ == "__main__":
    unittest.main()

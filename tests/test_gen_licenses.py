"""Unit tests for the pure heuristics in ``scripts/gen_licenses.py``.

These are the file-level oracle for the license-artifact generator. The CI drift
gate only proves the committed artifacts match what the *current* script produces,
so a deterministic bug in the script would produce identically-wrong output on both
sides of the diff and pass the gate unnoticed. These tests exercise the heuristics
directly, with adversarial fixtures for the cases that have bitten us: basename
collisions, look-alike repo hosts, substring license matches, and non-ascii text.
"""

import pytest

import gen_licenses as gl


def _pkg(name="example", version="1.0.0", **overrides):
    """A package record in the shape the container extractor emits."""
    pkg = {
        "name": name,
        "version": version,
        "license_expression": "",
        "classifiers": [],
        "license_field": "",
        "home_page": "",
        "project_urls": [],
        "license_files": {"LICENSE": "license text"},
    }
    pkg.update(overrides)
    return pkg


class TestCanonicalDir:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("typing_extensions", "typing-extensions"),
            ("Foo.Bar_Baz", "foo-bar-baz"),
            ("zope.interface", "zope-interface"),
            ("already-canonical", "already-canonical"),
            ("multi___sep...name", "multi-sep-name"),
        ],
    )
    def test_normalizes(self, name, expected):
        assert gl.canonical_dir(name) == expected


class TestDeclaredLicense:
    def test_prefers_expression(self):
        assert gl.declared_license(_pkg(license_expression="MIT")) == "MIT"

    def test_classifiers_when_no_expression(self):
        pkg = _pkg(
            classifiers=[
                "License :: OSI Approved :: MIT License",
                "License :: OSI Approved :: BSD License",
            ]
        )
        assert gl.declared_license(pkg) == "MIT License, BSD License"

    def test_license_field_fallback(self):
        assert gl.declared_license(_pkg(license_field="Apache-2.0")) == "Apache-2.0"

    def test_default_when_nothing_declared(self):
        assert gl.declared_license(_pkg()) == "See bundled license"


class TestLicenseFamily:
    @pytest.mark.parametrize(
        "declared,family",
        [
            ("Apache License, Version 2.0", "Apache License, Version 2.0"),
            ("Apache-2.0", "Apache License, Version 2.0"),
            ("MPL 2.0", "Mozilla Public License 2.0"),
            ("Mozilla Public License 2.0", "Mozilla Public License 2.0"),
            ("Python Software Foundation License", "Python Software Foundation License"),
            ("BSD-3-Clause", "BSD License"),
            ("MIT License", "MIT License"),
        ],
    )
    def test_known_families(self, declared, family):
        assert gl.license_family(declared) == family

    @pytest.mark.parametrize(
        "declared",
        [
            "Permitted use only",  # "mit" inside "permitted"
            "Transmitted works license",  # "mit" inside "transmitted"
            "Some Proprietary EULA",
        ],
    )
    def test_substring_does_not_falsely_match(self, declared):
        # Word-boundary matching: a short token must not fire inside an unrelated
        # word, or the package gets silently mis-grouped.
        assert gl.license_family(declared) == "Other licenses"


class TestBestUrl:
    def test_home_page_wins(self):
        assert gl.best_url(_pkg(home_page="https://example.com")) == "https://example.com"

    def test_project_url_homepage_key(self):
        pkg = _pkg(project_urls=["Homepage, https://proj.example"])
        assert gl.best_url(pkg) == "https://proj.example"

    def test_priority_key_returned_verbatim(self):
        pkg = _pkg(project_urls=["Source, https://github.com/org/repo"])
        assert gl.best_url(pkg) == "https://github.com/org/repo"

    def test_repo_host_stripped_to_root(self):
        # A github/gitlab URL reached via the hostname fallback (non-priority
        # label) is trimmed of blob/releases/tree tails.
        pkg = _pkg(project_urls=["Changelog, https://github.com/org/repo/releases/tag/v1"])
        assert gl.best_url(pkg) == "https://github.com/org/repo"

    def test_lookalike_host_not_treated_as_repo(self):
        # hostname is evil.com, not github.com — must not be picked/stripped as a repo
        url = "https://evil.com/github.com/org/repo/blob/main/x"
        pkg = _pkg(project_urls=[f"Changelog, {url}"])
        assert gl.best_url(pkg) == url

    def test_no_urls_returns_empty(self):
        assert gl.best_url(_pkg()) == ""


class TestCopyrightLine:
    def test_returns_line_with_year(self):
        assert gl.copyright_line({"LICENSE": "Copyright 2021 Acme Inc."}) == (
            "Copyright 2021 Acme Inc."
        )

    def test_prefers_notice_over_license(self):
        files = {"LICENSE": "Copyright 2000 Old", "NOTICE": "Copyright 2022 New"}
        assert gl.copyright_line(files) == "Copyright 2022 New"

    def test_skips_apache_boilerplate(self):
        assert gl.copyright_line({"LICENSE": "Copyright [yyyy] [name of copyright owner]"}) == ""

    def test_requires_a_real_signal(self):
        # "copyright" with no year, (c), or © is boilerplate prose, not attribution
        assert gl.copyright_line({"LICENSE": "Copyright information follows"}) == ""

    def test_accepts_copyright_symbol(self):
        assert gl.copyright_line({"NOTICE": "Copyright © Acme"}) == "Copyright © Acme"

    def test_strips_trailing_comma(self):
        assert gl.copyright_line({"NOTICE": "Copyright 2021 Acme,"}) == "Copyright 2021 Acme"


class TestRenderNotice:
    def test_groups_and_references_artifacts(self):
        pkg = _pkg(
            name="Requests",
            version="2.0",
            license_expression="Apache-2.0",
            license_files={"licenses/LICENSE": "x"},
        )
        out = gl.render_notice("aws", [pkg], has_uv=False)
        assert "NOTICE" in out
        assert "Apache License, Version 2.0" in out
        assert "Requests 2.0" in out
        # artifact path uses canonical dir + the file's full relpath
        assert "LICENSES/requests/licenses/LICENSE" in out

    def test_uv_section_only_when_present(self):
        assert "Build tooling" not in gl.render_notice("aws", [_pkg()], has_uv=False)
        assert "Build tooling" in gl.render_notice("aws", [_pkg()], has_uv=True)


class TestBuildOutputs:
    def test_no_basename_collision(self):
        # Two license files sharing a basename must not overwrite each other.
        pkg = _pkg(
            name="pkg",
            license_files={"LICENSE": "root", "licenses/LICENSE": "nested"},
        )
        outputs = gl.build_outputs("aws", [pkg], has_uv=False)
        assert outputs["LICENSES/pkg/LICENSE"] == "root"
        assert outputs["LICENSES/pkg/licenses/LICENSE"] == "nested"
        assert "NOTICE" in outputs


class TestWriteAndDrift:
    def test_write_then_check_in_sync(self, tmp_path):
        pkg = _pkg(name="pkg", license_files={"licenses/LICENSE": "text ©"})
        outputs = gl.build_outputs("aws", [pkg], has_uv=False)
        gl.write_outputs(tmp_path, outputs)
        # utf-8 round trip through nested path
        written = (tmp_path / "LICENSES/pkg/licenses/LICENSE").read_text(encoding="utf-8")
        assert written == "text ©"
        assert gl.find_drift(tmp_path, outputs) == []

    def test_drift_detects_changed(self, tmp_path):
        outputs = {"NOTICE": "hello"}
        gl.write_outputs(tmp_path, outputs)
        (tmp_path / "NOTICE").write_text("changed", encoding="utf-8")
        assert any("changed" in d for d in gl.find_drift(tmp_path, outputs))

    def test_drift_detects_missing(self, tmp_path):
        outputs = {"NOTICE": "hi", "LICENSES/pkg/LICENSE": "x"}
        (tmp_path / "NOTICE").write_text("hi", encoding="utf-8")  # license file absent
        drift = gl.find_drift(tmp_path, outputs)
        assert any(d.startswith("missing") and "LICENSES/pkg/LICENSE" in d for d in drift)

    def test_drift_detects_stale(self, tmp_path):
        outputs = {"NOTICE": "hi"}
        gl.write_outputs(tmp_path, outputs)
        stale = tmp_path / "LICENSES/old/LICENSE"
        stale.parent.mkdir(parents=True)
        stale.write_text("old", encoding="utf-8")
        assert any(d.startswith("stale") for d in gl.find_drift(tmp_path, outputs))

    def test_write_cleans_removed_deps(self, tmp_path):
        gl.write_outputs(tmp_path, {"NOTICE": "n", "LICENSES/a/LICENSE": "a"})
        gl.write_outputs(tmp_path, {"NOTICE": "n"})
        assert not (tmp_path / "LICENSES/a/LICENSE").exists()

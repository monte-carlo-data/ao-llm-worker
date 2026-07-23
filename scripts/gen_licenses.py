#!/usr/bin/env python3
"""Generate per-cloud-platform NOTICE + LICENSES/ from a built worker image.

The authoritative dependency set (linux platform wheels + full transitive
closure) exists only inside the built image, not in a local dev venv, so this
extracts license artifacts directly from a container. Every wheel we ship
bundles its license text under `*.dist-info/licenses/`, so no manual upstream
fetching is needed.

For each cloud-platform variant it writes:
    licensing/<cloud>/NOTICE
    licensing/<cloud>/LICENSES/<package>/<license files>

The `uv` binary is copied into the image from the astral-sh/uv stage (not a pip
package), so it never appears in the venv inventory. Its license lives at
licensing/uv/ (maintained by hand) and the Dockerfile copies it into the image
directly; this script only records it in the generated NOTICE.

Usage:
    python scripts/gen_licenses.py --cloud aws --image montecarlodata/ao-llm-worker:0.0.0-latest-aws
    python scripts/gen_licenses.py --cloud gcp --image montecarlodata/ao-llm-worker:0.0.0-latest-gcp

CI runs this against the freshly built image, then `git add -A licensing/<cloud>`
followed by `git diff --cached --quiet` to fail if the committed artifacts drift
from what actually ships (the staged form also catches newly added, untracked
files).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent

# Runs inside the container against its venv interpreter. Prints one JSON blob:
# a list of {name, version, license fields, license_files: {relpath: text}}.
CONTAINER_EXTRACTOR = r"""
import importlib.metadata as im
import json

out = []
for d in im.distributions():
    md = d.metadata
    name = md["Name"]
    if name == "llm-worker":
        continue  # our own project
    files = {}
    for f in (d.files or []):
        s = str(f).replace("\\", "/")
        if ".dist-info/" not in s:
            continue
        rel = s.split(".dist-info/", 1)[1]  # "licenses/LICENSE", "LICENSE", "METADATA", ...
        base = rel.rsplit("/", 1)[-1]
        # PEP 639 puts license files under licenses/; older wheels drop LICENSE-ish
        # files in the dist-info root. Take both, keyed by basename (flattened).
        if rel.startswith("licenses/") or base.upper().startswith(
            ("LICENSE", "LICENCE", "COPYING", "NOTICE")
        ):
            try:
                files[base] = d.locate_file(f).read_text(errors="replace")
            except Exception:
                pass
    out.append({
        "name": name,
        "version": md["Version"],
        "license_expression": md.get("License-Expression") or "",
        "classifiers": [c for c in (md.get_all("Classifier") or []) if c.startswith("License")],
        "license_field": (md.get("License") or "").strip()[:80],
        "home_page": md.get("Home-page") or "",
        "project_urls": md.get_all("Project-URL") or [],
        "license_files": files,
    })
print(json.dumps(out))
"""

# Normalized license family for NOTICE grouping. Order matters: the first
# family whose token appears in the declared license wins the grouping (a
# package's full declared expression is still shown next to it).
LICENSE_FAMILIES = [
    ("Apache License, Version 2.0", ("apache",)),
    ("Mozilla Public License 2.0", ("mpl", "mozilla")),
    ("Python Software Foundation License", ("psf", "python software foundation")),
    ("BSD License", ("bsd",)),
    ("MIT License", ("mit",)),
]


def canonical_dir(name: str) -> str:
    """PEP 503-ish normalized directory name (lowercase, single hyphens)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_license(pkg: dict) -> str:
    if pkg["license_expression"]:
        return pkg["license_expression"]
    classifiers = [c.split("::")[-1].strip() for c in pkg["classifiers"]]
    if classifiers:
        return ", ".join(classifiers)
    return pkg["license_field"] or "See bundled license"


def license_family(declared: str) -> str:
    low = declared.lower()
    for family, tokens in LICENSE_FAMILIES:
        # Word-boundary match so a short token like "mit" doesn't fire inside an
        # unrelated word ("permitted", "transmit") in a free-text license field.
        if any(re.search(rf"\b{re.escape(t)}\b", low) for t in tokens):
            return family
    return "Other licenses"


def best_url(pkg: dict) -> str:
    if pkg["home_page"]:
        return pkg["home_page"]
    urls = {}
    for entry in pkg["project_urls"]:
        label, _, url = entry.partition(",")
        urls[label.strip().lower()] = url.strip()
    for key in ("homepage", "home", "repository", "source", "source code", "code"):
        if key in urls:
            return urls[key]
    # Prefer a repo URL over changelog/release/docs pages that other keys point to.
    # Match on the parsed hostname (not a substring) so a github.com/gitlab.com in a
    # path or a look-alike host can't masquerade as the repo URL.
    for url in urls.values():
        if (urlparse(url).hostname or "").lower() in ("github.com", "gitlab.com"):
            return re.sub(r"/(blob|releases|tree)/.*$", "", url)
    return next(iter(urls.values()), "")


# Fragments of the verbatim Apache-2.0 license body that start with "copyright"
# but are boilerplate, not a real attribution.
_COPYRIGHT_NOISE = (
    "[yyyy]",
    "[name of copyright owner]",
    "that is included in or attached",
    "copyright notice",
    "copyright, patent",
    "copyright ownership",
)


def copyright_line(files: dict[str, str]) -> str:
    """Best-effort copyright attribution from the bundled texts.

    Apache-licensed projects carry real attribution in their NOTICE file (the
    LICENSE is the verbatim Apache body), so scan NOTICE files first, and only
    accept a line carrying a real signal — a year, "(c)", or "©".
    """
    ordered = sorted(
        files.items(), key=lambda kv: 0 if "NOTICE" in kv[0].upper() else 1
    )
    for _, text in ordered:
        for raw in text.splitlines():
            line = raw.strip().rstrip(",")
            low = line.lower()
            if not low.startswith("copyright") or any(
                n in low for n in _COPYRIGHT_NOISE
            ):
                continue
            if re.search(r"\b\d{4}\b", line) or "(c)" in low or "©" in line:
                return line
    return ""


def extract(image: str) -> list[dict]:
    proc = subprocess.run(
        ["docker", "run", "-i", "--rm", image, "/app/.venv/bin/python", "-"],
        input=CONTAINER_EXTRACTOR,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"docker extraction failed:\n{proc.stderr}")
    return sorted(json.loads(proc.stdout), key=lambda p: p["name"].lower())


def render_notice(cloud: str, packages: list[dict], has_uv: bool) -> str:
    groups: dict[str, list[dict]] = {}
    for pkg in packages:
        groups.setdefault(license_family(declared_license(pkg)), []).append(pkg)

    lines = [
        "NOTICE",
        "======",
        "",
        "ao-llm-worker",
        "Copyright 2026 Monte Carlo AI, Inc.",
        "",
        "This product includes software developed at Monte Carlo AI, Inc.",
        "(https://montecarlo.ai).",
        "",
        "-" * 80,
        f"Third-party components ({cloud} image variant)",
        "-" * 80,
        "",
        "The ao-llm-worker container image bundles the following third-party",
        "components. Each is installed unmodified into the application virtual",
        "environment (/app/.venv) of the distributed image. A verbatim copy of each",
        "component's license is provided under the LICENSES/ directory distributed",
        "alongside this NOTICE (LICENSES/<project>/, mounted at /app/LICENSES in the",
        "image). Where an upstream project ships its own NOTICE file, that file is",
        "reproduced verbatim at LICENSES/<project>/NOTICE and its attribution notices",
        "are incorporated here by reference, as required by Apache License 2.0",
        "section 4(d).",
        "",
        "This file is generated by scripts/gen_licenses.py from the built image; the",
        "exact versions bundled in any given release are recorded in uv.lock.",
        "",
    ]

    for family, _ in LICENSE_FAMILIES + [("Other licenses", ())]:
        pkgs = groups.get(family)
        if not pkgs:
            continue
        lines.append(family)
        for i, pkg in enumerate(sorted(pkgs, key=lambda p: p["name"].lower()), 1):
            cdir = canonical_dir(pkg["name"])
            declared = declared_license(pkg)
            cr = copyright_line(pkg["license_files"])
            url = best_url(pkg)
            artifacts = ", ".join(
                f"LICENSES/{cdir}/{f}" for f in sorted(pkg["license_files"])
            )
            head = f"  {i:>2}. {pkg['name']} {pkg['version']}"
            if declared not in family:
                head += f"  [{declared}]"
            lines.append(head)
            if cr:
                lines.append(f"      {cr}")
            if url:
                lines.append(f"      {url}")
            if artifacts:
                lines.append(f"      {artifacts}")
        lines.append("")

    if has_uv:
        lines += [
            "Build tooling",
            "  uv (the Python package installer) is copied into the image from the",
            "  astral-sh/uv distribution and is dual-licensed Apache-2.0 OR MIT.",
            "  LICENSES/uv/LICENSE-APACHE, LICENSES/uv/LICENSE-MIT",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cloud", required=True, choices=["aws", "azure", "gcp"])
    ap.add_argument("--image", required=True, help="Built image ref to extract from")
    args = ap.parse_args()

    packages = extract(args.image)

    out_dir = REPO_ROOT / "licensing" / args.cloud
    licenses_dir = out_dir / "LICENSES"
    # Clean the LICENSES tree so removed deps don't linger.
    if licenses_dir.exists():
        for child in sorted(licenses_dir.rglob("*"), reverse=True):
            child.unlink() if child.is_file() else child.rmdir()
    licenses_dir.mkdir(parents=True, exist_ok=True)

    for pkg in packages:
        if not pkg["license_files"]:
            sys.exit(
                f"ERROR: {pkg['name']} ships no bundled license file — collect it manually."
            )
        pkg_dir = licenses_dir / canonical_dir(pkg["name"])
        pkg_dir.mkdir(parents=True, exist_ok=True)
        for rel, text in pkg["license_files"].items():
            dest = pkg_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)

    # uv ships as a copied binary (not a pip package), so it isn't in the venv
    # inventory. Its license lives at licensing/uv/ (shared, maintained by hand)
    # and the Dockerfile copies it into the image's LICENSES/ directly, so we only
    # note it in the NOTICE here — no duplication into the per-provider tree.
    has_uv = (REPO_ROOT / "licensing" / "uv").is_dir()

    (out_dir / "NOTICE").write_text(render_notice(args.cloud, packages, has_uv))
    print(
        f"{args.cloud}: {len(packages)} packages + {'uv' if has_uv else 'no uv'} -> {out_dir}"
    )


if __name__ == "__main__":
    main()

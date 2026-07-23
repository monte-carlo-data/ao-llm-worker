# Licensing artifacts

Each `licensing/<cloud>/` directory holds the third-party attribution for one
image variant:

- `NOTICE` — generated summary of every bundled dependency, grouped by license
  family, with copyright line, project URL, and the paths of its bundled license
  files.
- `LICENSES/<package>/…` — a verbatim copy of each dependency's bundled license
  text, laid out under the package's canonicalized name.

`licensing/uv/` is **maintained by hand** (the `uv` binary is copied into the
image from the astral-sh distribution, not installed as a pip package, so it
never appears in the venv inventory). Everything else is generated.

## When to regenerate

Regenerate whenever the bundled dependency set changes for a cloud — a new or
bumped dependency, a new extra, a Python version change. CI's drift gate
(`gen_licenses.py --check`, run per cloud in `build-and-push`/`publish-dev`) will
fail the build if the committed artifacts don't match what the freshly built
image actually ships, so a dependency change can't merge without its attribution
being updated.

## How to regenerate

The authoritative dependency set (linux platform wheels + full transitive
closure) only exists inside the built image, so the generator extracts license
files directly from a container — you need the image built **locally** (no push
or registry auth required):

```sh
# Build each cloud variant locally (tags are arbitrary — they only need to exist
# in your local Docker daemon).
docker build --build-arg CLOUD=aws   -t ao-llm-worker:regen-aws   .
docker build --build-arg CLOUD=azure -t ao-llm-worker:regen-azure .
docker build --build-arg CLOUD=gcp   -t ao-llm-worker:regen-gcp   .

# Regenerate the artifacts from each image.
python scripts/gen_licenses.py --cloud aws   --image ao-llm-worker:regen-aws
python scripts/gen_licenses.py --cloud azure --image ao-llm-worker:regen-azure
python scripts/gen_licenses.py --cloud gcp   --image ao-llm-worker:regen-gcp

git add licensing/ && git commit
```

To verify without writing (what CI does), add `--check` — it regenerates in
memory, compares against the committed artifacts, and exits non-zero on any
missing, changed, or stale file:

```sh
python scripts/gen_licenses.py --cloud aws --image ao-llm-worker:regen-aws --check
```

## Failure modes the generator enforces

- **No bundled license** — if a dependency ships no license file, generation
  aborts (collect the license manually before it can ship).
- **Unrecognized license family** — if a dependency's declared license doesn't
  map to a known family, generation aborts so a human vets it (extend
  `LICENSE_FAMILIES` in `scripts/gen_licenses.py`) rather than letting it ship in
  a catch-all bucket.

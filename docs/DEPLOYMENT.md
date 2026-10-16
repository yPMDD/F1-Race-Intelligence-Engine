# Production Deploy Runbook

How `dev` becomes production across **Neon**, **Hugging Face Spaces** and
**Vercel**.

---

## Topology

```
GitHub (main)
   │
   ├─ CI            pytest · vitest · docker build · secret scan     (no deploy)
   │
   └─ Deploy        ┌─ 1. migrate      → Neon        (direct endpoint)
                     └─ 2. deploy-backend → HF Space (git push, then poll build)
                                                └─ container start-up runs migrations again
   Vercel            auto-deploys from the same `main` push (own Git integration)
```

Three independent systems, one `main`. Each deploys only what it owns, so a
Vercel failure never blocks a database migration and vice versa.

---

## One-time setup

### 1. Neon

Create a project, then take **two** connection strings from
**Connect → Connection pooling**:

| Purpose | Hostname | Where it goes |
| --- | --- | --- |
| App runtime | `ep-xxx-**pooler**.region.aws.neon.tech` | Space secret `DATABASE_URL` |
| Migrations | `ep-xxx.region.aws.neon.tech` | GH secret `NEON_DATABASE_URL_DIRECT` |

Both end with `?sslmode=require`.

> The app **must** use the pooler. Alembic **must** use the direct connection:
> Neon runs PgBouncer in transaction mode, which does not support SQL-level
> `PREPARE`/`EXECUTE` or session `SET`.

### 2. Hugging Face

1. Create a **Docker** Space, e.g. `yPMDD/f1-race-intelligence-engine`.

   > **Which mode are you in?**
   >
   > - **Synced** — created with *Import GitHub repo*. HF rebuilds the Space
   >   automatically whenever the **default branch** moves, so merging to
   >   `main` is all that is needed. Leave `HF_TOKEN` / `HF_SPACE_REPO` unset;
   >   the deploy job detects this and skips the push.
   > - **Push** — created empty, with the deploy workflow rsyncing source into
   >   it. Set both `HF_TOKEN` and `HF_SPACE_REPO`.
   >
   > This distinction matters: in *synced* mode the Space builds `main`, not
   > `dev`, so **every fix is invisible until the PR is merged.**

2. **Settings → Variables and secrets**:

   | Kind | Key | Value |
   | --- | --- | --- |
   | Secret | `GROQ_API_KEY` | `gsk_…` |
   | Secret | `DATABASE_URL` | Neon **pooler** URL |
   | Secret | `CORS_ORIGINS` | your Vercel domain, e.g. `https://f1-dashboard.vercel.app` |
   | Variable | `GROQ_MODEL` | `qwen/qwen3.8-27b` |
   | Variable | `GROQ_MAX_TOKENS` | `800` |
   | Variable | `PORT` | `7860` |

   > ⚠️ **`CORS_ORIGINS` is not optional once a frontend is deployed.**
   > `main` currently runs `allow_origins=["*"]`, which is why a deployed
   > Vercel frontend works today. Merging this branch replaces that with a
   > strict allow-list that defaults to localhost dev ports only. **Set
   > `CORS_ORIGINS` to your Vercel domain BEFORE merging**, or the deployed
   > frontend will start failing every request with a CORS error.

3. Only for **push** mode: create a Hugging Face **write token**
   (Settings → Access Tokens → *Write*) and store it as the GitHub secret
   `HF_TOKEN`.

### 3. GitHub

Repository → **Settings → Environments**, create one named **`production`**
(add reviewers if you ever want a manual gate later).

**Settings → Secrets and variables → Actions**:

| Name | Kind | Value |
| --- | --- | --- |
| `NEON_DATABASE_URL_DIRECT` | Secret | Neon direct URL |
| `HF_TOKEN` | Secret | HF write token |
| `HF_SPACE_REPO` | **Variable** | `yPMDD/f1-race-intelligence-engine` |

### 4. Vercel

Import the repo → **Root Directory: `apps/dashboard`** (this is required;
Vercel defaults to the repo root and will fail to find `vite`).

| Environment | Variable | Value |
| --- | --- | --- |
| Production | `VITE_API_URL` | `https://<user>-<space>.hf.space` |
| Preview | `VITE_API_URL` | same, or omit to hit the deployed Space |

Vercel deploys automatically on every push to `main`. No workflow needed.

### 5. About the Docker image size

The image is **1.21 GB**, down from **3.59 GB**. `scripts/docker_install_deps.sh`
handles this, and the reasoning matters if you ever touch it:

| Approach | Result | Why |
| --- | --- | --- |
| `uv sync` as-is | 3.59 GB | torch's PyPI wheel is a CUDA build on Linux |
| `UV_TORCH_BACKEND=cpu` | 3.59 GB | ignored for a locked install; `uv.lock` pins torch to PyPI |
| `--no-install-package torch` | 3.25 GB | **removes** torch from the venv (verified) and still installs the `nvidia-*` wheels, which are separate lock entries |
| `[tool.uv.sources]` in `pyproject.toml` | no effect | a clean `uv lock` still resolves torch from `registry.pypi.org` |
| prune `nvidia-*` + reinstall CPU torch | **1.21 GB** | what the script does |

`UV_NO_CACHE=1` in the Dockerfile is also required: uv keeps every downloaded
wheel in `/root/.cache/uv`, so pruning alone still leaves ~2 GB of CUDA wheels
inside the image layer.

Two traps worth knowing:

- **Do not use `uv run` inside the container.** It re-syncs from `uv.lock` on
  every invocation, which silently reinstalls the CUDA torch and all 16
  `nvidia-*` packages on each cold start, undoing all of the above. The
  entrypoint calls `/app/.venv/bin/python` directly.
- An earlier revision of this build reported 766 MB — that image was *broken*:
  the `--no-install-package` approach had removed torch entirely, and only
  appeared to work because `uv run` reinstalled it at runtime. The build now
  asserts `torch.__version__` ends in `+cpu` and that no `nvidia*` directory
  exists, so that failure mode cannot ship again.

---

## Everyday flow

```bash
git checkout dev
git pull
# ... work ...
uv run pytest
npm --prefix apps/dashboard test
git commit -am "…"
git push origin dev
```

Open a PR `dev → main`. **CI must pass** (4 jobs) before it will merge.
Merging to `main` deploys production automatically.

---

## Local setup mirrors production

```bash
uv sync
cp .env.example .env          # fill in GROQ_API_KEY, DATABASE_URL
uv run alembic upgrade head    # schema is NOT auto-created any more
```

`create_all()` at import time was removed — it never alters existing tables, so
it would have silently hidden every future migration.

For a new machine that has never run the app before:

```bash
uv run alembic upgrade head   # creates everything on an empty database
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Space stuck on "Building" | Docker build downloading deps | Check the Space's build log; the vector-store rebuild step adds minutes |
| `FATAL: missing required environment variables` | Space secrets not set | §2 above |
| `FATAL: database unreachable after 30 attempts` | Neon compute asleep, or wrong host | Confirm the **pooler** URL and that the Neon project is not paused |
| `syntax error at or near ":"` in a migration | Migration pointed at the pooler | `NEON_DATABASE_URL_DIRECT` must be the non-pooler URL |
| Space runs, `/agent/chat` 429s | `GROQ_MAX_TOKENS` too high for the free tier | Keep ≤ 1000 (default 800) |
| Frontend shows "Backend unreachable" | `CORS_ORIGINS` missing the Vercel domain, or `VITE_API_URL` wrong | Check both; the origin is the **frontend** domain |
| CORS error but URL is right | Trailing slash in `CORS_ORIGINS` | Must be `https://x.vercel.app`, not `.../` |
| Vercel build fails instantly | Root Directory not set to `apps/dashboard` | §4 above |

### Rolling back

```bash
git revert <sha>            # or: git revert <range>
git push origin main        # reverts production via the same pipeline
```

A revert runs `alembic downgrade` only if you also write one; reverting code
alone leaves the schema ahead of the image, which the start-up migration makes
harmless.

### Inspecting production data

```bash
# Schema state
NEON_DATABASE_URL_DIRECT=... uv run alembic current

# Drift between models and the live schema
NEON_DATABASE_URL_DIRECT=... uv run alembic check
```

---

## Why migrations run twice

1. **CI, before the Space is pushed** — the normal path.
2. **Container start-up, via `docker-entrypoint.sh`** — belt and braces.

The ordering is deliberate. If the CI migration job fails, the Space is never
rebuilt, so production keeps running the old image against the old schema.
The reverse (new image, old schema) would break at query time. The start-up
migration guarantees the safe ordering either way, and the entrypoint refuses
to serve if migrations fail.

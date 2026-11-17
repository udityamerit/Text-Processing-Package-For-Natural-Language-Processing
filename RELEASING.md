# Releasing

How a new version of `nlp-text-preprocessing` gets to PyPI and GitHub.

## 1. Prepare

1. Update `__version__` in `nlp_text_preprocessing/__init__.py` (and the version assertion in `tests/test_infrastructure.py`).
2. Move the "Unreleased" notes in `CHANGELOG.md` under a new version heading.
3. Run the checks:

   ```bash
   pip install -e ".[dev]"
   ruff check .
   pytest -q
   python -m build
   twine check dist/*
   ```

4. Commit, tag and push:

   ```bash
   git commit -am "Release X.Y.Z"
   git tag vX.Y.Z
   git push origin main --tags
   ```

## 2. Publish to PyPI

### Option A: Trusted Publishing (recommended, no token needed)

`.github/workflows/publish.yml` builds the package and uploads it whenever a GitHub Release is published. PyPI trusts the workflow itself, so no API token is stored anywhere.

One-time setup on pypi.org:

1. Sign in and open the project: **nlp-text-preprocessing → Manage → Publishing**.
2. Under **Add a new publisher → GitHub**, enter:

   | Field | Value |
   |---|---|
   | Owner | `udityamerit` |
   | Repository name | `Text-Processing-Package-For-Natural-Language-Processing` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

3. In the GitHub repo, open **Settings → Environments** and create an environment called `pypi`. Optionally require your approval for deployments there.

4. In the GitHub repo open **Settings → Secrets and variables → Actions → Variables** and add a repository variable `TRUSTED_PUBLISHING` with the value `true`.

After that, publishing a GitHub Release (step 3 below) uploads the files to PyPI. The workflow uses `skip-existing: true`, so re-running it for a version that is already on PyPI does nothing instead of failing.

Until `TRUSTED_PUBLISHING` is set, the `publish` job is skipped (the `build` job still runs and checks the package). Without a registered publisher PyPI refuses the upload with `invalid-publisher`, which is why the job is gated. Use Option B until then.

### Option B: upload by hand with an API token

```bash
python -m build
twine check dist/nlp_text_preprocessing-X.Y.Z*
twine upload dist/nlp_text_preprocessing-X.Y.Z*
```

Use `__token__` as the username and a PyPI API token (scoped to this project) as the password. Never commit the token.

Check the result: `https://pypi.org/pypi/nlp-text-preprocessing/X.Y.Z/json`. The project's main JSON page can lag behind for a few minutes because of caching.

## 3. GitHub Release

Open **Releases → Draft a new release**, choose the tag `vX.Y.Z`, paste the notes from `CHANGELOG.md` and publish. Mark it as the latest release.

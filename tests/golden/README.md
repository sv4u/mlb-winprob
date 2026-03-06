# Golden API response files

Used to verify the HTTP gateway (after gRPC migration) returns byte-for-byte identical
JSON to the pre-migration API. Covers field naming (snake_case), null handling, float precision.

## Capturing golden files

From repo root, with data and models present (so the app is "ready"):

```bash
# Optional: deterministic git_commit for admin/version
export GIT_COMMIT=testabc
uv run python scripts/capture_golden_api.py
```

This writes `api_*.json` for each endpoint. Commit the files you want CI to assert on.

## Running the test

```bash
uv run pytest tests/integration/test_golden_api.py -v
```

The test calls the same endpoints (via TestClient) and asserts the response matches the
golden file. Endpoints that return variable data (e.g. git_commit, pipeline state) may
need normalized comparison or updated golden after capture.

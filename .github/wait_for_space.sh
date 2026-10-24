#!/usr/bin/env bash
# Poll the Hugging Face Space build status until it settles.
#
# HF does not return a build "result" from the push itself; the push only
# queues a rebuild. Without polling, CI would report success for a Space that
# then failed to build.
set -euo pipefail

SPACE="${SPACE:?SPACE must be set, e.g. user/space}"
TOKEN="${TOKEN:?TOKEN must be set}"

API="https://huggingface.co/api/spaces/${SPACE}"

for attempt in $(seq 1 60); do   # ~20 min at 20s intervals
  body=$(curl -fsS -H "Authorization: Bearer ${TOKEN}" "${API}" 2>/dev/null || echo "{}")

  stage=$(printf '%s' "$body" | grep -oE '"stage":"[A-Z_]+"' | head -1 | cut -d'"' -f4 || true)
  # BUILD_FAILED / BUILD_ERROR / RUNTIME_ERROR are terminal failures.
  if printf '%s' "$body" | grep -qE '"(BUILD_FAILED|BUILD_ERROR|RUNTIME_ERROR)"'; then
    echo "::error::Space build failed at stage '${stage:-unknown}'"
    printf '%s' "$body" | head -c 600
    echo ""
    exit 1
  fi

  if [ "$stage" = "RUNNING" ] || [ "$stage" = "BUILDING" ]; then
    echo "stage=${stage} (attempt ${attempt}/60)"
  fi

  if [ "$stage" = "RUNNING" ] || [ "$stage" = "SLEEPING" ]; then
    # Deployed and serving. Confirm it actually answers before declaring done.
    url="https://${SPACE//\//-}.hf.space/health"
    if curl -fsS --max-time 15 "$url" >/dev/null 2>&1; then
      echo "::notice::Space is live and healthy at ${url}"
      exit 0
    fi
  fi

  sleep 20
done

echo "::warning::timed out waiting for the Space build"
exit 1

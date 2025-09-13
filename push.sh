#!/usr/bin/env bash
set -euo pipefail

# Simple git push script with zero params.
# - Stages all changes, auto-commits with a timestamp message, and pushes.
# - Detects repo root, current branch, and remote.

err() { printf "Error: %s\n" "$*" >&2; }
info() { printf "%s\n" "$*"; }

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  err "Not a git repository."
  exit 1
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" = "HEAD" ]; then
  err "Detached HEAD state detected; please checkout a branch before pushing."
  exit 1
fi

# Determine remote: prefer upstream remote, else origin, else first remote
remote=""
if upstream=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null); then
  remote=${upstream%%/*}
fi
if [ -z "$remote" ]; then
  if git remote | grep -qx "origin"; then
    remote=origin
  else
    remote=$(git remote | head -n1 || true)
  fi
fi

if [ -z "$remote" ]; then
  err "No git remote found. Add one with: git remote add origin <url>"
  exit 1
fi

# Stage all changes
git add -A

# Commit if there are staged changes
if ! git diff --cached --quiet; then
  msg="Auto-commit: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  info "Committing staged changes: $msg"
  git commit -m "$msg"
else
  info "No changes to commit. Proceeding to push."
fi

# Ensure upstream is set; if not, set it on first push
if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
  info "Pushing to upstream of '$branch'..."
  git push
else
  info "Setting upstream and pushing to '$remote' '$branch'..."
  git push -u "$remote" "$branch"
fi

info "Push complete."


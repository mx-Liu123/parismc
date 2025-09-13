#!/usr/bin/env bash
set -euo pipefail

# Simple git pull script with zero params.
# - Detects repo root, current branch, and remote.
# - Uses rebase + autostash for a cleaner history.

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
  err "Detached HEAD state detected; please checkout a branch."
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

info "Pulling latest changes for branch '$branch' from '$remote' (rebase, autostash)..."

# Ensure remote branch exists and set upstream if missing
if ! git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
  git fetch "$remote"
  if git show-ref --verify --quiet "refs/remotes/$remote/$branch"; then
    git branch --set-upstream-to="$remote/$branch" "$branch" >/dev/null 2>&1 || true
  fi
fi

# Perform the pull
git pull --rebase --autostash "$remote" "$branch"

info "Pull complete."


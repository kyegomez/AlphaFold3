#!/bin/bash

# Check if we are inside a Git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Must be run inside a Git repository."
    exit 1
fi

# Fetch all open pull requests
echo "Fetching open PRs..."
prs=$(gh pr list --state open --json number --jq '.[].number')

# Check if there are PRs to merge
if [ -z "$prs" ]; then
    echo "No open PRs to merge."
    exit 0
fi

echo "Found PRs: $prs"

# Loop through each pull request number and merge it
for pr in $prs; do
    echo "Attempting to merge PR #$pr"
    merge_output=$(gh pr merge $pr --auto --merge)
    merge_status=$?
    if [ $merge_status -ne 0 ]; then
        echo "Failed to merge PR #$pr. Error: $merge_output"
    else
        echo "Successfully merged PR #$pr"
    fi
done

echo "Processing complete."

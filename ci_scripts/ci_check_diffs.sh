#!/usr/bin/env bash
set -euo pipefail

hash_cmd="md5sum"   # change to sha256sum if you prefer

# List test names (directories ending with *_dump) in a run, without the _dump suffix.
list_tests() {
  local run_dir="$1"
  ( cd "$run_dir" && find . -maxdepth 1 -mindepth 1 -type d -name '*_dump' -printf '%P\n' \
      | sed 's/_dump$//' | LC_ALL=C sort )
}

# Read PASS/FAIL from run/<test>_dump/status (or print MISSING if not present).
read_status() {
  local run_dir="$1" test="$2"
  local f="$run_dir/${test}_dump/status"
  if [[ -f "$f" ]]; then
    # Trim to first word just in case
    read -r s < "$f" || s="MISSING"
    printf '%s' "$s"
  else
    printf '%s' "MISSING"
  fi
}

# Build manifest for a single test: lines "<hash>␠␠<relative/path>"
# Only includes files under ${test}_dump/triton{,_cache}.
make_manifest_for_test() {
  local run_dir="$1" test="$2"
  (
    cd "$run_dir" || exit 0

    local dump="${test}_dump"
    local targets=()
    [[ -d "$dump/triton"       ]] && targets+=("$dump/triton")
    [[ -d "$dump/triton_cache" ]] && targets+=("$dump/triton_cache")
    [[ ${#targets[@]} -gt 0 ]] || exit 0
    LC_ALL=C find "${targets[@]}" -type f -print0 \
      | LC_ALL=C sort -z \
      | while IFS= read -r -d '' f; do
          rel="${f#./}"
          "$hash_cmd" -- "$rel" | awk -v rel="$rel" '{print $1"  "rel}'
        done
  )
}

# Diff two manifests: prints ADDED / REMOVED / CHANGED (sorted)
diff_manifests() {
  awk 'BEGIN{ FS="  " }
       FNR==NR { new[$2]=$1; next }
               { old[$2]=$1 }
       END {
         for (p in new) if (!(p in old))                  print "ADDED  " p;
         for (p in old) if (!(p in new))                  print "REMOVED  " p;
         for (p in new) if ((p in old) && new[p]!=old[p]) print "CHANGED  " p;
       }' "$1" "$2" | LC_ALL=C sort
}

# Main flow
# Usage: ./script.sh NEW_RUN_DIR OLD_RUN_DIR [OLD_RUN_DIR ...]
main() {
  if [[ $# -lt 2 ]]; then
    echo "Usage: $0 NEW_RUN_DIR OLD_RUN_DIR [OLD_RUN_DIR ...]" >&2
    exit 1
  fi
  local new_dir="$1"; shift
  [[ -d "$new_dir" ]] || { echo "ERROR: $new_dir is not a directory" >&2; exit 2; }

  cat $new_dir/cann_version.log

  # Pre-list tests in new run (once)
  mapfile -t new_tests < <(list_tests "$new_dir")

  for old_dir in "$@"; do
    echo "$(basename "$new_dir") vs $(basename "$old_dir")"
    echo

    if [[ ! -d "$old_dir" ]]; then
      echo "ERROR: old run not a directory"
      echo
      continue
    fi

    # Union of test names: new ∪ old
    mapfile -t old_tests < <(list_tests "$old_dir")
    mapfile -t all_tests < <(printf "%s\n" "${new_tests[@]}" "${old_tests[@]}" | LC_ALL=C sort -u)

    for test in "${all_tests[@]}"; do
      echo "$test"
      echo "$(read_status "$new_dir" "$test") vs $(read_status "$old_dir" "$test")"

      changes="$(diff_manifests \
        <(make_manifest_for_test "$new_dir" "$test") \
        <(make_manifest_for_test "$old_dir" "$test") || echo "no changes found")"

      if [[ -n "$changes" ]]; then
        printf '%s\n' "$changes"
      fi
      echo
    done
  done
}

main "$@"


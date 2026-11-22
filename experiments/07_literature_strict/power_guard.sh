#!/bin/bash
# AC power by default. An explicit user instruction can allow battery above 40%.
ensure_power() {
  local battery percent
  battery=$(pmset -g batt)
  if printf '%s\n' "$battery" | grep -q 'AC Power'; then
    return 0
  fi
  percent=$(printf '%s\n' "$battery" | sed -nE 's/.*[[:space:]]([0-9]+)%;.*/\1/p' | head -1)
  if [[ "${EXPERIMENT_ALLOW_BATTERY:-0}" == 1 && "$percent" =~ ^[0-9]+$ ]] && (( percent >= 40 )); then
    printf 'Running on battery with explicit permission; charge %s%%.\n' "$percent" >&2
    return 0
  fi
  echo 'Stopped before next job: external power is absent or battery is below 40%.' >&2
  return 2
}

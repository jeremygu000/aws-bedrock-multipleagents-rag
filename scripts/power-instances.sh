#!/usr/bin/env bash
set -euo pipefail

# Print help text for human operators.
usage() {
  cat <<'USAGE'
Usage:
  scripts/power-instances.sh <start|stop|status> <neo4j|grafana|phoenix|all> [--wait]

Examples:
  scripts/power-instances.sh stop neo4j --wait
  scripts/power-instances.sh start grafana
  scripts/power-instances.sh start phoenix --wait
  scripts/power-instances.sh status all

Environment:
  AWS_PROFILE         Optional profile name (for example: my-dev)
  AWS_REGION          Optional region override
  AWS_DEFAULT_REGION  Used when AWS_REGION is not set
USAGE
}

ACTION="${1:-}"
TARGET="${2:-}"
WAIT_FLAG="${3:-}"

# Require at least action + target.
if [[ -z "$ACTION" || -z "$TARGET" ]]; then
  usage
  exit 1
fi

# Validate supported actions early.
case "$ACTION" in
  start | stop | status) ;;
  *)
    echo "Invalid action: $ACTION"
    usage
    exit 1
    ;;
esac

# Validate supported targets early.
case "$TARGET" in
  neo4j | grafana | phoenix | all) ;;
  *)
    echo "Invalid target: $TARGET"
    usage
    exit 1
    ;;
esac

# Optional --wait blocks until the instance reaches the target state.
WAIT_FOR_STATE=false
if [[ "$WAIT_FLAG" == "--wait" ]]; then
  WAIT_FOR_STATE=true
elif [[ -n "$WAIT_FLAG" ]]; then
  echo "Unknown option: $WAIT_FLAG"
  usage
  exit 1
fi

# Build a reusable AWS CLI command with resolved region/profile.
AWS_REGION_EFFECTIVE="${AWS_REGION:-${AWS_DEFAULT_REGION:-ap-southeast-2}}"
AWS_PROFILE_EFFECTIVE="${AWS_PROFILE:-}"
AWS_CMD=(aws --region "$AWS_REGION_EFFECTIVE")
if [[ -n "$AWS_PROFILE_EFFECTIVE" ]]; then
  AWS_CMD+=(--profile "$AWS_PROFILE_EFFECTIVE")
fi

echo "Using region: $AWS_REGION_EFFECTIVE"
if [[ -n "$AWS_PROFILE_EFFECTIVE" ]]; then
  echo "Using profile: $AWS_PROFILE_EFFECTIVE"
fi

# Resolve an EC2 instance id from a CloudFormation stack output.
get_instance_id() {
  local stack_name="$1"
  local output_key="$2"
  local instance_id

  instance_id="$("${AWS_CMD[@]}" cloudformation describe-stacks \
    --stack-name "$stack_name" \
    --query "Stacks[0].Outputs[?OutputKey=='$output_key'].OutputValue | [0]" \
    --output text 2>/dev/null || true)"

  if [[ -z "$instance_id" || "$instance_id" == "None" ]]; then
    echo "Could not resolve instance id from stack '$stack_name' output '$output_key'." >&2
    echo "Deploy the stack first or confirm stack name/output key." >&2
    exit 1
  fi

  echo "$instance_id"
}

# Read the current EC2 lifecycle state (running/stopped/etc).
get_state() {
  local instance_id="$1"

  "${AWS_CMD[@]}" ec2 describe-instances \
    --instance-ids "$instance_id" \
    --query "Reservations[0].Instances[0].State.Name" \
    --output text
}

# Apply start/stop/status behavior with defensive state checks.
apply_action() {
  local label="$1"
  local instance_id="$2"
  local state
  state="$(get_state "$instance_id")"

  if [[ "$ACTION" == "status" ]]; then
    echo "$label ($instance_id): $state"
    return
  fi

  # Stop only when the instance is in a stoppable state.
  if [[ "$ACTION" == "stop" ]]; then
    if [[ "$state" == "stopped" || "$state" == "stopping" ]]; then
      echo "$label ($instance_id): already $state, skip."
      return
    fi
    if [[ "$state" == "terminated" ]]; then
      echo "$label ($instance_id): already terminated, cannot stop." >&2
      return
    fi

    echo "Stopping $label ($instance_id)..."
    "${AWS_CMD[@]}" ec2 stop-instances --instance-ids "$instance_id" >/dev/null
    if [[ "$WAIT_FOR_STATE" == "true" ]]; then
      "${AWS_CMD[@]}" ec2 wait instance-stopped --instance-ids "$instance_id"
      echo "$label ($instance_id): stopped."
    fi
    return
  fi

  # Start only when the instance is in a startable state.
  if [[ "$ACTION" == "start" ]]; then
    if [[ "$state" == "running" || "$state" == "pending" ]]; then
      echo "$label ($instance_id): already $state, skip."
      return
    fi
    if [[ "$state" == "terminated" ]]; then
      echo "$label ($instance_id): terminated, cannot start." >&2
      return
    fi
    if [[ "$state" == "stopping" ]]; then
      echo "$label ($instance_id): currently stopping, wait and retry start." >&2
      return
    fi

    echo "Starting $label ($instance_id)..."
    "${AWS_CMD[@]}" ec2 start-instances --instance-ids "$instance_id" >/dev/null
    if [[ "$WAIT_FOR_STATE" == "true" ]]; then
      "${AWS_CMD[@]}" ec2 wait instance-running --instance-ids "$instance_id"
      echo "$label ($instance_id): running."
    fi
  fi
}

# Look up stack-managed instance ids once, then execute target action(s).
NEO4J_INSTANCE_ID="$(get_instance_id "Neo4jDataStack" "Neo4jInstanceId")"
GRAFANA_INSTANCE_ID="$(get_instance_id "MonitoringEc2Stack" "MonitoringInstanceId")"
PHOENIX_INSTANCE_ID="$(get_instance_id "PhoenixEc2Stack" "PhoenixInstanceId")"

if [[ "$TARGET" == "neo4j" || "$TARGET" == "all" ]]; then
  apply_action "neo4j" "$NEO4J_INSTANCE_ID"
fi

if [[ "$TARGET" == "grafana" || "$TARGET" == "all" ]]; then
  apply_action "grafana" "$GRAFANA_INSTANCE_ID"
fi

if [[ "$TARGET" == "phoenix" || "$TARGET" == "all" ]]; then
  apply_action "phoenix" "$PHOENIX_INSTANCE_ID"
fi

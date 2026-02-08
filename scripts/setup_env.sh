#!/usr/bin/env bash
# Setup conda activate/deactivate hooks to auto-load .env
#
# Usage:
#   bash scripts/setup_env.sh
#
# After setup, environment variables are automatically loaded/unloaded
# when you run: conda activate moremouse / conda deactivate

set -euo pipefail

CONDA_ENV_NAME="moremouse"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"

# Find conda env directory
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    CONDA_ENV_DIR="$(conda env list | grep "^${CONDA_ENV_NAME} " | awk '{print $NF}')"
elif [[ -n "${CONDA_PREFIX:-}" && "$(basename "$CONDA_PREFIX")" == "$CONDA_ENV_NAME" ]]; then
    CONDA_ENV_DIR="$CONDA_PREFIX"
else
    echo "Error: conda env '$CONDA_ENV_NAME' not found"
    echo "Create it first: conda env create -f environment.yml"
    exit 1
fi

# Create .env from template if it doesn't exist
if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f "$PROJECT_DIR/.env.example" ]]; then
        cp "$PROJECT_DIR/.env.example" "$ENV_FILE"
        echo "Created .env from .env.example — please verify paths"
    else
        echo "Error: .env.example not found"
        exit 1
    fi
fi

# Create activate.d / deactivate.d directories
mkdir -p "$CONDA_ENV_DIR/etc/conda/activate.d"
mkdir -p "$CONDA_ENV_DIR/etc/conda/deactivate.d"

# Write activate hook
cat > "$CONDA_ENV_DIR/etc/conda/activate.d/moremouse_env.sh" << 'ACTIVATE_EOF'
#!/usr/bin/env bash
# Auto-load MoReMouse .env on conda activate

_MOREMOUSE_ENV_VARS=()

_load_moremouse_env() {
    local env_file=""

    # Search order: MOREMOUSE_PROJECT_DIR > git root > CWD
    if [[ -n "${MOREMOUSE_PROJECT_DIR:-}" && -f "$MOREMOUSE_PROJECT_DIR/.env" ]]; then
        env_file="$MOREMOUSE_PROJECT_DIR/.env"
    elif git rev-parse --show-toplevel &>/dev/null; then
        local git_root
        git_root="$(git rev-parse --show-toplevel)"
        if [[ -f "$git_root/.env" ]]; then
            env_file="$git_root/.env"
        fi
    elif [[ -f ".env" ]]; then
        env_file=".env"
    fi

    if [[ -z "$env_file" ]]; then
        return 0
    fi

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// /}" ]] && continue

        local key="${line%%=*}"
        local value="${line#*=}"

        # Remove surrounding quotes
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"

        export "$key=$value"
        _MOREMOUSE_ENV_VARS+=("$key")
    done < "$env_file"

    echo "moremouse: loaded ${#_MOREMOUSE_ENV_VARS[@]} env vars from $env_file"
}

_load_moremouse_env
ACTIVATE_EOF

# Write deactivate hook
cat > "$CONDA_ENV_DIR/etc/conda/deactivate.d/moremouse_env.sh" << 'DEACTIVATE_EOF'
#!/usr/bin/env bash
# Unset MoReMouse env vars on conda deactivate

if [[ -n "${_MOREMOUSE_ENV_VARS+x}" ]]; then
    for var in "${_MOREMOUSE_ENV_VARS[@]}"; do
        unset "$var"
    done
    unset _MOREMOUSE_ENV_VARS
fi
DEACTIVATE_EOF

chmod +x "$CONDA_ENV_DIR/etc/conda/activate.d/moremouse_env.sh"
chmod +x "$CONDA_ENV_DIR/etc/conda/deactivate.d/moremouse_env.sh"

echo ""
echo "Setup complete!"
echo ""
echo "  Activate hook: $CONDA_ENV_DIR/etc/conda/activate.d/moremouse_env.sh"
echo "  Deactivate hook: $CONDA_ENV_DIR/etc/conda/deactivate.d/moremouse_env.sh"
echo "  Env file: $ENV_FILE"
echo ""
echo "Usage:"
echo "  1. Edit .env with your actual paths"
echo "  2. conda activate $CONDA_ENV_NAME"
echo "  3. Environment variables are auto-loaded"
echo ""
echo "Tip: Set MOREMOUSE_PROJECT_DIR so .env is found regardless of CWD:"
echo "  echo 'export MOREMOUSE_PROJECT_DIR=$PROJECT_DIR' >> ~/.bashrc.local"

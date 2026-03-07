#!/usr/bin/env bash
# =============================================================================
# Amber-G1-Manipulation-RL Installer
# =============================================================================
# Installs all dependencies needed to TEST the RL policy on a real Unitree G1
# humanoid robot. Run this on the G1's onboard computer after cloning the repo.
#
# What this installs:
#   - System dependencies (build tools, Python dev headers)
#   - Python virtual environment with all runtime packages
#   - Unitree SDK2 Python bindings (robot communication)
#   - PyTorch (CPU, for policy inference)
#   - g1_tasks package (this repo)
#   - (Optional) Isaac Lab for sim verification before hardware deployment
#
# Usage:
#   chmod +x install.sh
#   ./install.sh              # Standard install (deploy to real robot)
#   ./install.sh --with-sim   # Also install Isaac Lab for sim verification
#
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
VENV_DIR=".venv"
PYTHON_MIN_VERSION="3.10"
UNITREE_SDK2_REPO="https://github.com/unitreerobotics/unitree_sdk2_python.git"
UNITREE_SDK2_BRANCH="main"

# --- Color helpers -----------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*"; }

# --- Parse arguments ---------------------------------------------------------
WITH_SIM=false
for arg in "$@"; do
    case "$arg" in
        --with-sim) WITH_SIM=true ;;
        --help|-h)
            echo "Usage: $0 [--with-sim]"
            echo ""
            echo "Options:"
            echo "  --with-sim   Also install Isaac Lab for sim verification"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# --- Pre-flight checks -------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "  Amber-G1-Manipulation-RL Installer"
echo "============================================================"
echo ""
info "Repository root: $SCRIPT_DIR"
info "Install sim tools: $WITH_SIM"
echo ""

# Check OS
if [[ ! -f /etc/os-release ]]; then
    error "Cannot detect OS. This installer targets Ubuntu 22.04/24.04."
    exit 1
fi
source /etc/os-release
info "Detected OS: $PRETTY_NAME"

# Check Python
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        PY_VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PY_MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)")
        PY_MINOR=$("$cmd" -c "import sys; print(sys.version_info.minor)")
        if [[ "$PY_MAJOR" -ge 3 ]] && [[ "$PY_MINOR" -ge 10 ]]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    error "Python >= $PYTHON_MIN_VERSION not found."
    error "Install it with: sudo apt install python3.10 python3.10-venv python3.10-dev"
    exit 1
fi

info "Using Python: $PYTHON_CMD ($PY_VER)"

# --- Step 1: System dependencies --------------------------------------------
echo ""
info "=== Step 1/5: System dependencies ==="

SYSTEM_PACKAGES=(
    build-essential
    cmake
    git
    curl
    "python${PY_VER}-venv"
    "python${PY_VER}-dev"
    libhdf5-dev
    libeigen3-dev
)

MISSING_PKGS=()
for pkg in "${SYSTEM_PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" &>/dev/null 2>&1; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    info "Installing missing system packages: ${MISSING_PKGS[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y -qq "${MISSING_PKGS[@]}"
    success "System packages installed"
else
    success "All system packages already installed"
fi

# --- Step 2: Python virtual environment --------------------------------------
echo ""
info "=== Step 2/5: Python virtual environment ==="

if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment already exists at $VENV_DIR"
    info "Reusing existing environment. Delete it and re-run to start fresh."
else
    info "Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    success "Virtual environment created at $VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated virtual environment"

# Upgrade pip
pip install --upgrade pip setuptools wheel -q
success "pip/setuptools/wheel upgraded"

# --- Step 3: Python packages -------------------------------------------------
echo ""
info "=== Step 3/5: Python packages ==="

# Detect architecture for PyTorch
ARCH=$(uname -m)
info "Architecture: $ARCH"

# Install PyTorch CPU (sufficient for inference on the onboard computer)
info "Installing PyTorch (CPU)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
success "PyTorch (CPU) installed"

# Core dependencies
info "Installing core dependencies..."
pip install \
    numpy \
    scipy \
    pyyaml \
    tensorboard \
    -q
success "Core dependencies installed"

# --- Step 4: Unitree SDK2 Python ---------------------------------------------
echo ""
info "=== Step 4/5: Unitree SDK2 Python ==="

# Check if already installed
if python -c "import unitree_sdk2py" 2>/dev/null; then
    success "unitree_sdk2py already installed"
else
    info "Installing unitree_sdk2py..."

    # Clone and install
    TEMP_SDK_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_SDK_DIR" EXIT

    git clone --depth 1 --branch "$UNITREE_SDK2_BRANCH" "$UNITREE_SDK2_REPO" "$TEMP_SDK_DIR/unitree_sdk2_python"
    pip install "$TEMP_SDK_DIR/unitree_sdk2_python" -q

    # Verify
    if python -c "import unitree_sdk2py" 2>/dev/null; then
        success "unitree_sdk2py installed and verified"
    else
        error "unitree_sdk2py installation failed. You may need to install it manually."
        error "See: https://github.com/unitreerobotics/unitree_sdk2_python"
    fi
fi

# --- Step 5: Install this package --------------------------------------------
echo ""
info "=== Step 5/5: g1_tasks package ==="

cd "$SCRIPT_DIR/g1_upper_body_tasks"
pip install -e . -q
cd "$SCRIPT_DIR"
success "g1_tasks package installed in editable mode"

# --- Optional: Isaac Lab for sim verification --------------------------------
if [[ "$WITH_SIM" == true ]]; then
    echo ""
    info "=== Optional: Isaac Lab (sim verification) ==="
    warn "Isaac Lab requires NVIDIA GPU + Isaac Sim. Skipping auto-install."
    warn ""
    warn "To install Isaac Lab manually:"
    warn "  1. Install Isaac Sim 4.5+: https://developer.nvidia.com/isaac-sim"
    warn "  2. Install Isaac Lab 1.0+: https://isaac-sim.github.io/IsaacLab/"
    warn "  3. Install rsl_rl:  pip install rsl_rl"
    warn "  4. Re-run:  pip install -e g1_upper_body_tasks/"
    warn ""
    warn "Then verify in sim with:"
    warn "  python g1_upper_body_tasks/scripts/play_g1_upper.py \\"
    warn "    --task Isaac-Reach-G1-Upper-v0 --num_envs 32"
fi

# --- Summary -----------------------------------------------------------------
echo ""
echo "============================================================"
echo -e "  ${GREEN}Installation complete!${NC}"
echo "============================================================"
echo ""
echo "  Activate the environment:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Test policy on real G1 (IK-based):"
echo "    python g1_upper_body_tasks/scripts/deploy_g1_real.py \\"
echo "      --checkpoint checkpoints/reach_policy_v1.pt \\"
echo "      --mode keyboard"
echo ""
echo "  Test policy on real G1 (joint-space):"
echo "    python g1_upper_body_tasks/scripts/deploy_g1_joint_space.py \\"
echo "      --checkpoint checkpoints/reach_policy_joint_space_v1.pt \\"
echo "      --mode keyboard"
echo ""
echo "  E-STOP: Press SPACEBAR during deployment to trigger emergency stop."
echo "          The robot will dampen to its current position and halt."
echo ""
echo "============================================================"

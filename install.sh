#!/usr/bin/env bash
# =============================================================================
# Amber-G1-Manipulation-RL Installer
# =============================================================================
# Installs all dependencies needed to TEST the RL policy on a real Unitree G1
# humanoid robot. Run this on the G1's onboard computer after cloning the repo.
#
# What this installs:
#   - System dependencies (build tools, dev headers)
#   - Conda environment (jonas_g1_env) with all runtime packages
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
CONDA_ENV_NAME="jonas_g1_env"
PYTHON_MIN_VERSION="3.10"
UNITREE_SDK2_REPO="https://github.com/unitreerobotics/unitree_sdk2_python.git"
CYCLONEDDS_REPO="https://github.com/eclipse-cyclonedds/cyclonedds"
CYCLONEDDS_BRANCH="releases/0.10.x"

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

# Check conda
if ! command -v conda &>/dev/null; then
    error "conda not found. Please install Miniconda or Anaconda first."
    error "See: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
info "Found conda: $(conda --version)"

# --- Step 1: System dependencies --------------------------------------------
echo ""
info "=== Step 1/5: System dependencies ==="

SYSTEM_PACKAGES=(
    build-essential
    cmake
    git
    curl
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

# --- Step 2: Conda environment -----------------------------------------------
echo ""
info "=== Step 2/5: Conda environment ($CONDA_ENV_NAME) ==="

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    info "Conda environment '$CONDA_ENV_NAME' already exists, activating..."
    conda activate "$CONDA_ENV_NAME"
else
    info "Creating conda environment '$CONDA_ENV_NAME' with Python >= $PYTHON_MIN_VERSION..."
    conda create -n "$CONDA_ENV_NAME" python=3.11 -y
    conda activate "$CONDA_ENV_NAME"
    success "Conda environment created"
fi

# Verify Python version
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
if [[ "$PY_MINOR" -lt 10 ]]; then
    error "Python in '$CONDA_ENV_NAME' is $PY_VER but >= $PYTHON_MIN_VERSION is required."
    error "Recreate with: conda create -n $CONDA_ENV_NAME python=3.11 -y"
    exit 1
fi
info "Using Python $PY_VER in conda env '$CONDA_ENV_NAME'"

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
    mujoco \
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

    # Clone unitree_sdk2_python to home directory
    SDK_DIR="$HOME/unitree_sdk2_python"
    if [[ -d "$SDK_DIR" ]]; then
        info "unitree_sdk2_python repo already cloned at $SDK_DIR"
    else
        git clone "$UNITREE_SDK2_REPO" "$SDK_DIR"
    fi

    # Build cyclonedds first (required dependency)
    CYCLONE_DIR="$HOME/cyclonedds"
    if [[ -d "$CYCLONE_DIR/install" ]]; then
        info "cyclonedds already built at $CYCLONE_DIR/install"
    else
        info "Building cyclonedds..."
        if [[ -d "$CYCLONE_DIR" ]]; then
            info "cyclonedds repo already cloned at $CYCLONE_DIR"
        else
            git clone "$CYCLONEDDS_REPO" -b "$CYCLONEDDS_BRANCH" "$CYCLONE_DIR"
        fi

        mkdir -p "$CYCLONE_DIR/build" "$CYCLONE_DIR/install"
        cd "$CYCLONE_DIR/build"
        cmake .. -DCMAKE_INSTALL_PREFIX="$CYCLONE_DIR/install"
        cmake --build . --target install
        cd "$SCRIPT_DIR"
        success "cyclonedds built"
    fi

    export CYCLONEDDS_HOME="$CYCLONE_DIR/install"
    info "Set CYCLONEDDS_HOME=$CYCLONEDDS_HOME"

    # Install unitree_sdk2_python into the active conda env
    pip install -e "$SDK_DIR" -q

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
echo "    conda activate $CONDA_ENV_NAME"
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

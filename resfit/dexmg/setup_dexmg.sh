# Get the directory of this script and navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPS_DIR="$REPO_ROOT/deps"

# Create deps directory if it doesn't exist
mkdir -p "$DEPS_DIR"

# Git clone robosuite into deps directory
git clone https://github.com/ARISE-Initiative/robosuite "$DEPS_DIR/robosuite"
git -C "$DEPS_DIR/robosuite" checkout 77a4751233c29456a5381209e30dd0dbf39a6557

# Install robosuite
python -m pip install -e "$DEPS_DIR/robosuite"

# Git clone dexmimicgen into deps directory
git clone https://github.com/NVlabs/dexmimicgen.git "$DEPS_DIR/dexmimicgen"
git -C "$DEPS_DIR/dexmimicgen" checkout e606f36a38b1d4ba8f56d06d6c0cd059b20ebbaf

# Install dexmimicgen
python -m pip install -e "$DEPS_DIR/dexmimicgen"

# Install a couple of dependencies
python -m pip install mink==0.0.7 gymnasium==1.1.1

# Install PyOpenGL-accelerate
python -m pip install PyOpenGL-accelerate

# Original additional installs
python -m pip install ipdb serial deepdiff

python -m pip install -U "numba>=0.59" "llvmlite>=0.42"

python -m pip install tabulate

python -m pip install torchrl==0.8.0 tensordict==0.8.2 torchcodec==0.4.0

python -m pip install mujoco==3.3.2 protobuf==3.20.3 diffusers==0.33.1 llvmlite==0.42.0 multidict==6.0.5 numba==0.59.1

git clone https://github.com/NVlabs/mimicgen.git "$DEPS_DIR/mimicgen"
git -C "$DEPS_DIR/mimicgen" checkout main

# Install MimicGen
python -m pip install -e "$DEPS_DIR/mimicgen"


micromamba install -n residual -c conda-forge "ffmpeg>=6,<8" -y

# Upgrade to a Numba that supports NumPy 2.x (and its llvmlite)
pip install --upgrade --no-cache-dir "numba>=0.60" "llvmlite>=0.44"

pip install draccus==0.10.0
 
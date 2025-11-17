# Get the directory of this script and navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPS_DIR="$REPO_ROOT/deps"

# Create deps directory if it doesn't exist
mkdir -p "$DEPS_DIR"

# Git clone lerobot into deps directory
git clone https://github.com/huggingface/lerobot.git "$DEPS_DIR/lerobot"
git -C "$DEPS_DIR/lerobot" checkout 69901b9b6a2300914ca3de0ea14b6fa6e0203bd4

# Install lerobot
python -m pip install -e "$DEPS_DIR/lerobot" --no-deps

# Install a couple of dependencies
python -m pip install -r resfit/lerobot/lerobot_requirements.txt
python -m pip install --upgrade torch torchvision torchcodec
python -m pip install datasets==3.6.0
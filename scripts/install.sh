#!/bin/bash
# gpuemu installer script
# Usage: curl -fsSL https://gpuemu.dev/install.sh | sh
#    or: curl -fsSL https://github.com/example/gpuemu/releases/latest/download/install.sh | sh

set -e

# Configuration
GITHUB_REPO="example/gpuemu"
INSTALL_DIR="${GPUEMU_INSTALL_DIR:-$HOME/.gpuemu}"
BIN_DIR="$INSTALL_DIR/bin"

# Colors (disabled in non-interactive mode)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

info() {
    printf "${BLUE}info:${NC} %s\n" "$1"
}

success() {
    printf "${GREEN}success:${NC} %s\n" "$1"
}

warn() {
    printf "${YELLOW}warning:${NC} %s\n" "$1"
}

error() {
    printf "${RED}error:${NC} %s\n" "$1" >&2
    exit 1
}

# Detect OS and architecture
detect_platform() {
    local os arch

    case "$(uname -s)" in
        Linux)
            os="linux"
            ;;
        Darwin)
            os="darwin"
            ;;
        MINGW*|MSYS*|CYGWIN*|Windows_NT)
            os="windows"
            ;;
        *)
            error "Unsupported operating system: $(uname -s)"
            ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)
            arch="x86_64"
            ;;
        aarch64|arm64)
            arch="aarch64"
            ;;
        *)
            error "Unsupported architecture: $(uname -m)"
            ;;
    esac

    echo "${os}-${arch}"
}

# Get the latest release version
get_latest_version() {
    local latest_url="https://api.github.com/repos/${GITHUB_REPO}/releases/latest"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$latest_url" | grep '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/'
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$latest_url" | grep '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/'
    else
        error "curl or wget is required for installation"
    fi
}

# Download file
download() {
    local url="$1"
    local output="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$output"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$url" -O "$output"
    else
        error "curl or wget is required for installation"
    fi
}

# Main installation
main() {
    info "gpuemu installer"

    # Detect platform
    local platform
    platform=$(detect_platform)
    info "Detected platform: $platform"

    # Get version
    local version="${GPUEMU_VERSION:-$(get_latest_version)}"
    if [ -z "$version" ]; then
        error "Could not determine latest version. Set GPUEMU_VERSION environment variable."
    fi
    info "Installing version: $version"

    # Construct download URL
    local artifact_name
    case "$platform" in
        linux-x86_64)
            artifact_name="gpuemu-linux-x86_64.tar.gz"
            ;;
        linux-aarch64)
            artifact_name="gpuemu-linux-aarch64.tar.gz"
            ;;
        darwin-x86_64)
            artifact_name="gpuemu-darwin-x86_64.tar.gz"
            ;;
        darwin-aarch64)
            artifact_name="gpuemu-darwin-aarch64.tar.gz"
            ;;
        windows-x86_64)
            artifact_name="gpuemu-windows-x86_64.zip"
            ;;
        *)
            error "No pre-built binary available for $platform"
            ;;
    esac

    local download_url="https://github.com/${GITHUB_REPO}/releases/download/${version}/${artifact_name}"

    # Create directories
    mkdir -p "$BIN_DIR"

    # Download and extract
    local tmp_dir
    tmp_dir=$(mktemp -d)
    trap "rm -rf $tmp_dir" EXIT

    info "Downloading from $download_url"
    download "$download_url" "$tmp_dir/$artifact_name"

    info "Extracting..."
    case "$artifact_name" in
        *.tar.gz)
            tar -xzf "$tmp_dir/$artifact_name" -C "$tmp_dir"
            ;;
        *.zip)
            unzip -q "$tmp_dir/$artifact_name" -d "$tmp_dir"
            ;;
    esac

    # Install binaries
    info "Installing to $BIN_DIR"
    if [ "$platform" = "windows-x86_64" ]; then
        cp "$tmp_dir/gpuemu.exe" "$BIN_DIR/"
        cp "$tmp_dir/gpuemu-daemon.exe" "$BIN_DIR/"
    else
        cp "$tmp_dir/gpuemu" "$BIN_DIR/"
        cp "$tmp_dir/gpuemu-daemon" "$BIN_DIR/"
        chmod +x "$BIN_DIR/gpuemu" "$BIN_DIR/gpuemu-daemon"
    fi

    success "gpuemu installed successfully!"

    # Check if BIN_DIR is in PATH
    case ":$PATH:" in
        *":$BIN_DIR:"*)
            ;;
        *)
            echo ""
            warn "Add gpuemu to your PATH:"
            echo ""
            echo "  For bash/zsh, add to ~/.bashrc or ~/.zshrc:"
            echo "    export PATH=\"\$HOME/.gpuemu/bin:\$PATH\""
            echo ""
            echo "  For fish, run:"
            echo "    fish_add_path $BIN_DIR"
            echo ""
            ;;
    esac

    # Verify installation
    if [ -x "$BIN_DIR/gpuemu" ]; then
        echo ""
        info "Verify installation:"
        echo "  $BIN_DIR/gpuemu --version"
    fi
}

main "$@"

# Homebrew formula for gpuemu (CLI + daemon).
#
# Tap install (until this lands in homebrew-core):
#   brew tap skelf-research/gpuemu https://github.com/Skelf-Research/homebrew-gpuemu
#   brew install gpuemu
#
# Or install from the formula directly (no tap):
#   brew install --formula https://raw.githubusercontent.com/Skelf-Research/gpuemu/main/packaging/homebrew/gpuemu.rb

class Gpuemu < Formula
  desc "Hardware-free GPU kernel correctness validation"
  homepage "https://docs.skelfresearch.com/gpuemu"
  license any_of: ["MIT", "Apache-2.0"]
  version "0.1.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/Skelf-Research/gpuemu/releases/download/v#{version}/gpuemu-darwin-aarch64.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000" # filled at release
    else
      url "https://github.com/Skelf-Research/gpuemu/releases/download/v#{version}/gpuemu-darwin-x86_64.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/Skelf-Research/gpuemu/releases/download/v#{version}/gpuemu-linux-aarch64.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    else
      url "https://github.com/Skelf-Research/gpuemu/releases/download/v#{version}/gpuemu-linux-x86_64.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    end
  end

  def install
    bin.install "gpuemu"
    bin.install "gpuemu-daemon"
  end

  test do
    assert_match "gpuemu", shell_output("#{bin}/gpuemu version")
  end
end

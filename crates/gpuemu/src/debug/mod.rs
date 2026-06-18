//! Interactive debugging mode for gpuemu.

mod repl;

use anyhow::Result;

pub use repl::start_repl;

/// Configuration for debug mode.
#[derive(Debug, Clone)]
pub struct DebugConfig {
    /// Specific seed to investigate
    pub seed: Option<u64>,
    /// Filter by op name
    pub op: Option<String>,
    /// Use REPL mode (vs TUI)
    #[allow(dead_code)]
    pub repl_mode: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            seed: None,
            op: None,
            repl_mode: false,
        }
    }
}

/// Start the interactive debug session.
pub fn start_debug(config: DebugConfig) -> Result<()> {
    // For now, always use REPL mode
    // TUI mode requires ratatui which we'll add later
    start_repl(config)
}

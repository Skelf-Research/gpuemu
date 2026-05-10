//! Project initialization and scaffolding.

mod templates;

use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;

pub use templates::{get_reference_template, render_template};

/// Configuration for project initialization.
#[derive(Debug)]
pub struct InitConfig {
    /// Project name
    pub name: String,
    /// Target framework (pytorch, jax, tensorflow)
    pub framework: String,
    /// Include example ops
    pub with_examples: bool,
    /// CI platform to set up (github, gitlab)
    pub ci: Option<String>,
    /// Target directory (defaults to current directory)
    pub target_dir: PathBuf,
}

impl Default for InitConfig {
    fn default() -> Self {
        Self {
            name: "my-project".to_string(),
            framework: "pytorch".to_string(),
            with_examples: false,
            ci: None,
            target_dir: PathBuf::from("."),
        }
    }
}

/// Initialize a new gpuemu project.
pub fn init_project(config: &InitConfig) -> Result<InitResult> {
    let mut result = InitResult::default();

    // Check if gpuemu.toml already exists
    let config_path = config.target_dir.join("gpuemu.toml");
    if config_path.exists() {
        anyhow::bail!(
            "gpuemu.toml already exists at {:?}. Use --force to overwrite.",
            config_path
        );
    }

    // Create directory structure
    create_directory_structure(config, &mut result)?;

    // Create gpuemu.toml
    create_config_file(config, &mut result)?;

    // Create scripts directory with __init__.py
    create_scripts_directory(config, &mut result)?;

    // Create .gitignore
    create_gitignore(config, &mut result)?;

    // Create example files if requested
    if config.with_examples {
        create_example_files(config, &mut result)?;
    }

    // Create CI configuration if requested
    if let Some(ref ci_platform) = config.ci {
        create_ci_config(config, ci_platform, &mut result)?;
    }

    Ok(result)
}

/// Result of project initialization.
#[derive(Debug, Default)]
pub struct InitResult {
    /// Files created
    pub files_created: Vec<PathBuf>,
    /// Directories created
    pub dirs_created: Vec<PathBuf>,
    /// Warnings
    pub warnings: Vec<String>,
}

impl InitResult {
    /// Print summary of initialization.
    pub fn print_summary(&self) {
        println!("Created gpuemu project:");
        println!();

        if !self.dirs_created.is_empty() {
            println!("Directories:");
            for dir in &self.dirs_created {
                println!("  {}/", dir.display());
            }
        }

        if !self.files_created.is_empty() {
            println!("Files:");
            for file in &self.files_created {
                println!("  {}", file.display());
            }
        }

        if !self.warnings.is_empty() {
            println!();
            println!("Warnings:");
            for warning in &self.warnings {
                println!("  - {}", warning);
            }
        }
    }

    /// Print next steps.
    pub fn print_next_steps(&self, config: &InitConfig) {
        println!();
        println!("Next steps:");
        println!("  1. Edit gpuemu.toml to configure your ops/kernels");
        println!("  2. Create reference scripts in scripts/");
        println!("  3. Run 'gpuemu daemon start' to start the daemon");
        println!("  4. Run 'gpuemu test' to validate");

        if config.ci.is_some() {
            println!();
            println!("CI setup:");
            println!("  - Commit the generated workflow file");
            println!("  - CI will run validation on push/PR");
        }
    }
}

/// Create the directory structure.
fn create_directory_structure(config: &InitConfig, result: &mut InitResult) -> Result<()> {
    let dirs = vec![
        config.target_dir.join("scripts"),
        config.target_dir.join(".gpuemu"),
    ];

    for dir in dirs {
        if !dir.exists() {
            fs::create_dir_all(&dir)
                .with_context(|| format!("Failed to create directory: {:?}", dir))?;
            result.dirs_created.push(dir);
        }
    }

    // Create .gitkeep in .gpuemu
    let gitkeep = config.target_dir.join(".gpuemu/.gitkeep");
    if !gitkeep.exists() {
        fs::write(&gitkeep, "")?;
        result.files_created.push(gitkeep);
    }

    Ok(())
}

/// Create the gpuemu.toml configuration file.
fn create_config_file(config: &InitConfig, result: &mut InitResult) -> Result<()> {
    let config_path = config.target_dir.join("gpuemu.toml");

    let content = render_template(
        templates::GPUEMU_TOML,
        &[
            ("project_name", &config.name),
            ("framework", &config.framework),
        ],
    );

    fs::write(&config_path, content)
        .with_context(|| format!("Failed to write {:?}", config_path))?;

    result.files_created.push(config_path);
    Ok(())
}

/// Create the scripts directory with __init__.py.
fn create_scripts_directory(config: &InitConfig, result: &mut InitResult) -> Result<()> {
    let init_path = config.target_dir.join("scripts/__init__.py");

    fs::write(&init_path, templates::SCRIPTS_INIT)
        .with_context(|| format!("Failed to write {:?}", init_path))?;

    result.files_created.push(init_path);
    Ok(())
}

/// Create .gitignore file.
fn create_gitignore(config: &InitConfig, result: &mut InitResult) -> Result<()> {
    let gitignore_path = config.target_dir.join(".gitignore");

    // Don't overwrite existing .gitignore
    if gitignore_path.exists() {
        // Append gpuemu-specific ignores
        let existing = fs::read_to_string(&gitignore_path)?;
        if !existing.contains(".gpuemu/") {
            let mut content = existing;
            if !content.ends_with('\n') {
                content.push('\n');
            }
            content.push_str("\n# gpuemu\n.gpuemu/\n");
            fs::write(&gitignore_path, content)?;
            result.warnings.push(format!(
                "Appended gpuemu entries to existing {:?}",
                gitignore_path
            ));
        }
    } else {
        fs::write(&gitignore_path, templates::GITIGNORE)
            .with_context(|| format!("Failed to write {:?}", gitignore_path))?;
        result.files_created.push(gitignore_path);
    }

    Ok(())
}

/// Create example files.
fn create_example_files(config: &InitConfig, result: &mut InitResult) -> Result<()> {
    // Create example reference script
    let ref_template = get_reference_template(&config.framework);
    let ref_content = render_template(ref_template, &[("op_name", "example_op")]);

    let ref_path = config.target_dir.join("scripts/ref_example_op.py");
    fs::write(&ref_path, ref_content).with_context(|| format!("Failed to write {:?}", ref_path))?;
    result.files_created.push(ref_path);

    // Create tests directory and example test
    let tests_dir = config.target_dir.join("tests");
    if !tests_dir.exists() {
        fs::create_dir_all(&tests_dir)?;
        result.dirs_created.push(tests_dir.clone());
    }

    let test_path = tests_dir.join("test_ops.py");
    fs::write(&test_path, templates::EXAMPLE_TEST)
        .with_context(|| format!("Failed to write {:?}", test_path))?;
    result.files_created.push(test_path);

    // Update gpuemu.toml to include example op
    let config_path = config.target_dir.join("gpuemu.toml");
    let mut content = fs::read_to_string(&config_path)?;

    // Add example op configuration
    let example_op = r#"

# Example op (created with --with-examples)
[[ops]]
name = "example_op"
module = "example_module.example_op"
reference = "scripts/ref_example_op.py"

[ops.tolerances]
float32 = 1e-5
float16 = 1e-3

[ops.invariants]
no_nan = true
no_inf = true
"#;

    content.push_str(example_op);
    fs::write(&config_path, content)?;

    Ok(())
}

/// Create CI configuration files.
fn create_ci_config(config: &InitConfig, ci_platform: &str, result: &mut InitResult) -> Result<()> {
    match ci_platform.to_lowercase().as_str() {
        "github" => create_github_actions(config, result),
        "gitlab" => create_gitlab_ci(config, result),
        other => {
            result.warnings.push(format!(
                "Unknown CI platform '{}'. Supported: github, gitlab",
                other
            ));
            Ok(())
        }
    }
}

/// Create GitHub Actions workflow.
fn create_github_actions(config: &InitConfig, result: &mut InitResult) -> Result<()> {
    let workflows_dir = config.target_dir.join(".github/workflows");
    if !workflows_dir.exists() {
        fs::create_dir_all(&workflows_dir)?;
        result.dirs_created.push(workflows_dir.clone());
    }

    // Read the template from the templates directory
    let template_path = find_template_file("github-actions.yml")?;
    let content = fs::read_to_string(&template_path)
        .with_context(|| format!("Failed to read template: {:?}", template_path))?;

    let workflow_path = workflows_dir.join("gpuemu.yml");
    fs::write(&workflow_path, content)
        .with_context(|| format!("Failed to write {:?}", workflow_path))?;

    result.files_created.push(workflow_path);
    Ok(())
}

/// Create GitLab CI configuration.
fn create_gitlab_ci(config: &InitConfig, result: &mut InitResult) -> Result<()> {
    // Read the template from the templates directory
    let template_path = find_template_file("gitlab-ci.yml")?;
    let content = fs::read_to_string(&template_path)
        .with_context(|| format!("Failed to read template: {:?}", template_path))?;

    let ci_path = config.target_dir.join(".gitlab-ci.yml");
    fs::write(&ci_path, content).with_context(|| format!("Failed to write {:?}", ci_path))?;

    result.files_created.push(ci_path);
    Ok(())
}

/// Find template file in various locations.
fn find_template_file(name: &str) -> Result<PathBuf> {
    // Try relative to executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            // Check ../templates
            let path = parent.join("../templates").join(name);
            if path.exists() {
                return Ok(path);
            }

            // Check ../../templates (for cargo run)
            let path = parent.join("../../templates").join(name);
            if path.exists() {
                return Ok(path);
            }
        }
    }

    // Try relative to current directory (for development)
    let local_paths = [
        PathBuf::from("templates").join(name),
        PathBuf::from("../templates").join(name),
        PathBuf::from("../../templates").join(name),
    ];

    for path in &local_paths {
        if path.exists() {
            return Ok(path.clone());
        }
    }

    anyhow::bail!(
        "Template file '{}' not found. Looked in: {:?}",
        name,
        local_paths
    )
}

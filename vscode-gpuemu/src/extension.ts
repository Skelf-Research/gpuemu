import * as vscode from 'vscode';
import { GpuemuStatusBar } from './providers/statusBar';
import { FailuresTreeProvider } from './providers/failuresTree';
import { DiagnosticManager } from './providers/diagnostics';
import { GpuemuCodeActionProvider } from './providers/codeActions';
import { GpuemuTestController } from './providers/testController';
import { ValidationWatcher } from './providers/validationWatcher';
import { ConfigValidator } from './providers/configValidator';
import { registerCommands } from './commands';
import { GpuemuRunner } from './runner';

let statusBar: GpuemuStatusBar | undefined;
let failuresProvider: FailuresTreeProvider | undefined;
let runner: GpuemuRunner | undefined;
let diagnostics: DiagnosticManager | undefined;
let testController: GpuemuTestController | undefined;
let validationWatcher: ValidationWatcher | undefined;
let configValidator: ConfigValidator | undefined;

export async function activate(context: vscode.ExtensionContext) {
    console.log('gpuemu extension is activating');

    // Initialize runner
    runner = new GpuemuRunner();

    // Initialize status bar
    const config = vscode.workspace.getConfiguration('gpuemu');
    if (config.get('showStatusBar', true)) {
        statusBar = new GpuemuStatusBar();
        context.subscriptions.push(statusBar);
    }

    // Initialize failures tree view
    failuresProvider = new FailuresTreeProvider(runner);
    vscode.window.registerTreeDataProvider('gpuemuFailures', failuresProvider);

    // Initialize diagnostics (pseudo-LSP core)
    diagnostics = new DiagnosticManager(runner);
    context.subscriptions.push(diagnostics);

    // Initialize config validator (gpuemu.toml linting)
    configValidator = new ConfigValidator();
    context.subscriptions.push(configValidator);

    // Validate gpuemu.toml on open and save
    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument(async (doc) => {
            if (doc.uri.fsPath.endsWith('gpuemu.toml')) {
                const diags = await configValidator!.validateDocument(doc);
                configValidator!.setDiagnostics(doc.uri, diags);
            }
        }),
    );
    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument(async (doc) => {
            if (doc.uri.fsPath.endsWith('gpuemu.toml') && configValidator) {
                const diags = await configValidator.validateDocument(doc);
                configValidator.setDiagnostics(doc.uri, diags);
            }
        }),
    );

    // Initialize code actions (right-click → reproduce, minimize, fuzz)
    const codeActionProvider = new GpuemuCodeActionProvider(runner);
    context.subscriptions.push(
        vscode.languages.registerCodeActionsProvider(
            { scheme: 'file' },
            codeActionProvider,
            { providedCodeActionKinds: GpuemuCodeActionProvider.providedCodeActionKinds },
        ),
    );

    // Initialize test controller (Testing sidebar)
    testController = new GpuemuTestController(runner);
    context.subscriptions.push(testController);

    // Register test run profile
    testController.testController.createRunProfile(
        'Validate',
        vscode.TestRunProfileKind.Run,
        async (request, token) => {
            await testController!.runTests(request, token);
        },
        true,
    );

    // Initialize on-save validation watcher
    validationWatcher = new ValidationWatcher(runner, diagnostics);
    context.subscriptions.push(validationWatcher);

    // Register commands (including new diagnostic-aware ones)
    registerCommands(context, runner, statusBar, failuresProvider, diagnostics);

    // Register the minimize command
    context.subscriptions.push(
        vscode.commands.registerCommand('gpuemu.minimize', async (seed?: number) => {
            if (!seed) {
                const input = await vscode.window.showInputBox({
                    prompt: 'Enter seed to minimize',
                    validateInput: (value) => {
                        const num = parseInt(value);
                        return isNaN(num) ? 'Enter a valid seed number' : undefined;
                    },
                });
                if (!input) {
                    return;
                }
                seed = parseInt(input);
            }

            const terminal = vscode.window.createTerminal('gpuemu minimize');
            terminal.show();
            terminal.sendText(`gpuemu minimize ${seed}`);
        }),
    );

    // Register the refresh diagnostics command
    context.subscriptions.push(
        vscode.commands.registerCommand('gpuemu.refreshDiagnostics', async () => {
            if (diagnostics) {
                await diagnostics.refresh();
            }
        }),
    );

    // Check for gpuemu.toml and set context
    const hasConfig = await checkForConfig();
    vscode.commands.executeCommand('setContext', 'gpuemu.hasConfig', hasConfig);

    // Auto-start daemon if configured
    if (hasConfig && config.get('autoStartDaemon', true)) {
        try {
            await runner.startDaemon();
            statusBar?.setRunning(true);
        } catch (error) {
            console.error('Failed to auto-start daemon:', error);
        }
    }

    // Watch for config file changes
    const configWatcher = vscode.workspace.createFileSystemWatcher('**/gpuemu.toml');
    configWatcher.onDidCreate(() => {
        vscode.commands.executeCommand('setContext', 'gpuemu.hasConfig', true);
    });
    configWatcher.onDidDelete(() => {
        vscode.commands.executeCommand('setContext', 'gpuemu.hasConfig', false);
    });
    context.subscriptions.push(configWatcher);

    // Start status polling if daemon might be running
    if (statusBar) {
        startStatusPolling(statusBar, runner);
    }

    // Initial diagnostic load
    if (hasConfig) {
        diagnostics.refresh().catch(() => {});
    }

    console.log('gpuemu extension activated');
}

export function deactivate() {
    console.log('gpuemu extension is deactivating');
}

async function checkForConfig(): Promise<boolean> {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
        return false;
    }

    for (const folder of workspaceFolders) {
        const configPath = vscode.Uri.joinPath(folder.uri, 'gpuemu.toml');
        try {
            await vscode.workspace.fs.stat(configPath);
            return true;
        } catch {
            // File doesn't exist
        }
    }
    return false;
}

function startStatusPolling(statusBar: GpuemuStatusBar, runner: GpuemuRunner) {
    setInterval(async () => {
        try {
            const status = await runner.checkStatus();
            statusBar.setRunning(status.running);
            if (status.version) {
                statusBar.setVersion(status.version);
            }
        } catch {
            statusBar.setRunning(false);
        }
    }, 5000);
}
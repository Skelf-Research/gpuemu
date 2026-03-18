import * as vscode from 'vscode';
import { GpuemuStatusBar } from './providers/statusBar';
import { FailuresTreeProvider } from './providers/failuresTree';
import { registerCommands } from './commands';
import { GpuemuRunner } from './runner';

let statusBar: GpuemuStatusBar | undefined;
let failuresProvider: FailuresTreeProvider | undefined;
let runner: GpuemuRunner | undefined;

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

    // Register commands
    registerCommands(context, runner, statusBar, failuresProvider);

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

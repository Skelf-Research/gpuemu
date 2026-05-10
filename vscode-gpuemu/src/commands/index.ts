import * as vscode from 'vscode';
import { GpuemuRunner } from '../runner';
import { GpuemuStatusBar } from '../providers/statusBar';
import { FailuresTreeProvider } from '../providers/failuresTree';
import { DiagnosticManager } from '../providers/diagnostics';

export function registerCommands(
    context: vscode.ExtensionContext,
    runner: GpuemuRunner,
    statusBar: GpuemuStatusBar | undefined,
    failuresProvider: FailuresTreeProvider,
    diagnostics: DiagnosticManager | undefined,
) {
    // Init command
    context.subscriptions.push(
        vscode.commands.registerCommand('gpuemu.init', async () => {
            const name = await vscode.window.showInputBox({
                prompt: 'Project name',
                value: vscode.workspace.name || 'my-project',
            });

            if (!name) {
                return;
            }

            const framework = await vscode.window.showQuickPick(
                ['pytorch', 'jax', 'tensorflow'],
                { placeHolder: 'Select framework' }
            );

            if (!framework) {
                return;
            }

            try {
                await runner.init({ name, framework });
                vscode.window.showInformationMessage(`gpuemu project initialized: ${name}`);
                vscode.commands.executeCommand('setContext', 'gpuemu.hasConfig', true);
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to initialize: ${error}`);
            }
        })
    );

    // Start daemon command
    context.subscriptions.push(
        vscode.commands.registerCommand('gpuemu.startDaemon', async () => {
            try {
                await vscode.window.withProgress(
                    {
                        location: vscode.ProgressLocation.Notification,
                        title: 'Starting gpuemu daemon...',
                    },
                    async () => {
                        await runner.startDaemon();
                    }
                );
                statusBar?.setRunning(true);
                vscode.window.showInformationMessage('gpuemu daemon started');
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to start daemon: ${error}`);
            }
        })
    );

    // Stop daemon command
    context.subscriptions.push(
        vscode.commands.registerCommand('gpuemu.stopDaemon', async () => {
            try {
                await runner.stopDaemon();
                statusBar?.setRunning(false);
                vscode.window.showInformationMessage('gpuemu daemon stopped');
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to stop daemon: ${error}`);
            }
        })
    );

    // Run tests command
    context.subscriptions.push(
        vscode.commands.registerCommand('gpuemu.runTests', async () => {
            const terminal = vscode.window.createTerminal('gpuemu test');
            terminal.show();
            terminal.sendText('gpuemu test');
            // Refresh diagnostics after tests complete (give time for daemon to store results)
            setTimeout(async () => {
                await diagnostics?.refresh();
                await failuresProvider.refresh();
            }, 5000);
        })
    );

    // Run quick tests command
    context.subscriptions.push(
        vscode.commands.registerCommand('gpuemu.runQuickTests', async () => {
            const terminal = vscode.window.createTerminal('gpuemu test');
            terminal.show();
            terminal.sendText('gpuemu test --quick');
            setTimeout(async () => {
                await diagnostics?.refresh();
                await failuresProvider.refresh();
            }, 3000);
        })
    );

    // Fuzz command
    context.subscriptions.push(
        vscode.commands.registerCommand('gpuemu.fuzz', async (options?: { op?: string; iterations?: number }) => {
            let iterations = options?.iterations?.toString();
            let opName = options?.op;

            if (!iterations) {
                iterations = await vscode.window.showInputBox({
                    prompt: 'Number of iterations',
                    value: '100',
                    validateInput: (value) => {
                        const num = parseInt(value);
                        return isNaN(num) || num <= 0 ? 'Enter a positive number' : undefined;
                    },
                });
            }

            if (!iterations) {
                return;
            }

            const terminal = vscode.window.createTerminal('gpuemu fuzz');
            terminal.show();
            const opArg = opName ? ` --op ${opName}` : '';
            terminal.sendText(`gpuemu fuzz --iterations ${iterations}${opArg}`);
            setTimeout(async () => {
                await diagnostics?.refresh();
                await failuresProvider.refresh();
            }, parseInt(iterations) * 100);
        })
    );

    // Reproduce command
    context.subscriptions.push(
        vscode.commands.registerCommand('gpuemu.reproduce', async (seed?: number) => {
            if (!seed) {
                const input = await vscode.window.showInputBox({
                    prompt: 'Enter seed to reproduce',
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

            try {
                const output = await runner.reproduce(seed);
                const doc = await vscode.workspace.openTextDocument({
                    content: output,
                    language: 'text',
                });
                vscode.window.showTextDocument(doc);
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to reproduce: ${error}`);
            }
        })
    );

    // Show failures command
    context.subscriptions.push(
        vscode.commands.registerCommand('gpuemu.showFailures', async () => {
            await failuresProvider.refresh();
            vscode.commands.executeCommand('gpuemuFailures.focus');
        })
    );
}

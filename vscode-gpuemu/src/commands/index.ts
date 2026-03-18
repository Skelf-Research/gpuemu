import * as vscode from 'vscode';
import { GpuemuRunner } from '../runner';
import { GpuemuStatusBar } from '../providers/statusBar';
import { FailuresTreeProvider } from '../providers/failuresTree';

export function registerCommands(
    context: vscode.ExtensionContext,
    runner: GpuemuRunner,
    statusBar: GpuemuStatusBar | undefined,
    failuresProvider: FailuresTreeProvider
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
        })
    );

    // Run quick tests command
    context.subscriptions.push(
        vscode.commands.registerCommand('gpuemu.runQuickTests', async () => {
            const terminal = vscode.window.createTerminal('gpuemu test');
            terminal.show();
            terminal.sendText('gpuemu test --quick');
        })
    );

    // Fuzz command
    context.subscriptions.push(
        vscode.commands.registerCommand('gpuemu.fuzz', async () => {
            const iterations = await vscode.window.showInputBox({
                prompt: 'Number of iterations',
                value: '100',
                validateInput: (value) => {
                    const num = parseInt(value);
                    return isNaN(num) || num <= 0 ? 'Enter a positive number' : undefined;
                },
            });

            if (!iterations) {
                return;
            }

            const terminal = vscode.window.createTerminal('gpuemu fuzz');
            terminal.show();
            terminal.sendText(`gpuemu fuzz --iterations ${iterations}`);
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

import * as vscode from 'vscode';
import { GpuemuRunner } from '../runner';
import { DiagnosticManager } from './diagnostics';

/**
 * Watches for file saves that should trigger validation:
 *
 * - .py files in the scripts/ directory → revalidate the corresponding op
 * - .cu / .cuh files → lint artifacts if available
 * - gpuemu.toml → refresh daemon config
 */
export class ValidationWatcher implements vscode.Disposable {
    private disposables: vscode.Disposable[] = [];

    constructor(
        private runner: GpuemuRunner,
        private diagnostics: DiagnosticManager,
    ) {
        // Watch for file saves
        this.disposables.push(
            vscode.workspace.onDidSaveTextDocument(async (doc) => {
                await this.onSave(doc);
            }),
        );

        // Watch for gpuemu.toml changes
        this.disposables.push(
            vscode.workspace.createFileSystemWatcher('**/gpuemu.toml').onDidChange(async () => {
                vscode.commands.executeCommand('gpuemu.refreshDiagnostics');
            }),
        );
    }

    private async onSave(doc: vscode.TextDocument): Promise<void> {
        const filePath = doc.uri.fsPath;

        // Reference scripts (*.py in scripts/ directory)
        if (filePath.endsWith('.py') && filePath.includes('scripts/')) {
            const opName = this.inferOpName(filePath);
            if (opName) {
                vscode.window.setStatusBarMessage(
                    `$(sync~spin) gpuemu: validating ${opName}...`,
                    3000,
                );
                try {
                    const output = await this.runner.runTests(true);
                    await this.diagnostics.refresh();
                    vscode.window.setStatusBarMessage(
                        `$(check) gpuemu: ${opName} validated`,
                        3000,
                    );
                } catch {
                    vscode.window.setStatusBarMessage(
                        `$(error) gpuemu: validation failed`,
                        3000,
                    );
                }
            }
        }

        // CUDA kernel source files
        if (filePath.endsWith('.cu') || filePath.endsWith('.cuh')) {
            try {
                const output = await this.runner.exec(['lint', filePath]);
                vscode.window.setStatusBarMessage(
                    `$(check) gpuemu: kernel linted`,
                    3000,
                );
            } catch {
                // Linting may fail if cuobjdump not available
            }
        }
    }

    /** Infer the op name from a reference script path.
     *
     * Convention: scripts/ref_<op_name>.py → op_name
     */
    private inferOpName(filePath: string): string | null {
        const filename = filePath.split('/').pop() || '';
        const match = filename.match(/^ref_(.+)\.py$/);
        return match ? match[1] : null;
    }

    dispose(): void {
        for (const d of this.disposables) {
            d.dispose();
        }
    }
}
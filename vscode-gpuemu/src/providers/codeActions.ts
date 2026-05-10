import * as vscode from 'vscode';
import { GpuemuRunner } from '../runner';

/**
 * Provides code actions for gpuemu diagnostics.
 *
 * Right-clicking a validation failure diagnostic offers:
 * - "Reproduce failure" — runs `gpuemu reproduce <seed>` to get exact repro info
 * - "Minimize test case" — runs `gpuemu minimize <seed>` to find the smallest failing case
 * - "Run fuzz tests" — runs `gpuemu fuzz --op <op-name>` for more testing
 */
export class GpuemuCodeActionProvider implements vscode.CodeActionProvider {
    static readonly providedCodeActionKinds = [
        vscode.CodeActionKind.QuickFix,
        vscode.CodeActionKind.Refactor,
    ];

    constructor(private runner: GpuemuRunner) {}

    async provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range | vscode.Selection,
        context: vscode.CodeActionContext,
        _token: vscode.CancellationToken,
    ): Promise<vscode.CodeAction[]> {
        const actions: vscode.CodeAction[] = [];

        // Find gpuemu diagnostics in the context
        const gpuemuDiagnostics = context.diagnostics.filter(
            d => d.source === 'gpuemu'
        );

        if (gpuemuDiagnostics.length === 0) {
            return actions;
        }

        for (const diagnostic of gpuemuDiagnostics) {
            // Extract op name and seed from the diagnostic message
            const opName = this.extractOpName(diagnostic.message);
            const seed = this.extractSeed(diagnostic.message);

            // Action: Reproduce failure
            if (seed !== undefined) {
                const reproduceAction = new vscode.CodeAction(
                    'Reproduce failure (seed: ' + seed + ')',
                    vscode.CodeActionKind.QuickFix,
                );
                reproduceAction.diagnostics = [diagnostic];
                reproduceAction.command = {
                    command: 'gpuemu.reproduce',
                    title: 'Reproduce Failure',
                    arguments: [seed],
                };
                reproduceAction.isPreferred = true;
                actions.push(reproduceAction);
            }

            // Action: Minimize test case
            if (seed !== undefined) {
                const minimizeAction = new vscode.CodeAction(
                    'Minimize test case (seed: ' + seed + ')',
                    vscode.CodeActionKind.QuickFix,
                );
                minimizeAction.diagnostics = [diagnostic];
                minimizeAction.command = {
                    command: 'gpuemu.minimize',
                    title: 'Minimize Test Case',
                    arguments: [seed],
                };
                actions.push(minimizeAction);
            }

            // Action: Fuzz this op
            if (opName) {
                const fuzzAction = new vscode.CodeAction(
                    'Fuzz ' + opName + ' (50 iterations)',
                    vscode.CodeActionKind.Refactor,
                );
                fuzzAction.diagnostics = [diagnostic];
                fuzzAction.command = {
                    command: 'gpuemu.fuzz',
                    title: 'Fuzz Op',
                    arguments: [{ op: opName, iterations: 50 }],
                };
                actions.push(fuzzAction);
            }
        }

        return actions;
    }

    /** Extract the op name from a diagnostic message like "[flash_attn] Tolerance exceeded..." */
    private extractOpName(message: string): string | undefined {
        const match = message.match(/^\[([^\]]+)\]/);
        return match ? match[1] : undefined;
    }

    /** Extract the seed from a diagnostic message or related information. */
    private extractSeed(message: string): number | undefined {
        // Try to find "seed: N" in the message
        const seedMatch = message.match(/seed:\s*(\d+)/);
        if (seedMatch) {
            return parseInt(seedMatch[1], 10);
        }
        return undefined;
    }
}
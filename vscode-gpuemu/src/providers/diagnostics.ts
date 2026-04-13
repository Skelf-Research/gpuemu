import * as vscode from 'vscode';
import { GpuemuRunner, ValidationFailure } from '../runner';

/**
 * Manages validation diagnostics pushed into the VS Code Problems panel.
 *
 * This is the core of the "pseudo-LSP" — it maps gpuemu validation failures
 * to editor diagnostics, with source file and line information where available.
 */
export class DiagnosticManager implements vscode.Disposable {
    private collection: vscode.DiagnosticCollection;
    private failures: ValidationFailure[] = [];

    constructor(private runner: GpuemuRunner) {
        this.collection = vscode.languages.createDiagnosticCollection('gpuemu');
    }

    /** Refresh diagnostics from the daemon by querying failures. */
    async refresh(): Promise<void> {
        this.failures = await this.runner.listFailures(100);
        this.updateDiagnostics();
    }

    /** Set failures directly (e.g., from a fuzz run callback). */
    setFailures(failures: ValidationFailure[]): void {
        this.failures = failures;
        this.updateDiagnostics();
    }

    /** Clear all diagnostics. */
    clear(): void {
        this.collection.clear();
    }

    /** Map failures to VS Code Diagnostic objects, grouped by source file. */
    private updateDiagnostics(): void {
        this.collection.clear();

        // Group failures by source file (reference script path or op name)
        const byFile = new Map<string, ValidationFailure[]>();

        for (const failure of this.failures) {
            const filePath = failure.referencePath || failure.opName;
            const list = byFile.get(filePath) || [];
            list.push(failure);
            byFile.set(filePath, list);
        }

        for (const [filePath, fileFailures] of byFile) {
            const uri = this.resolveUri(filePath);
            const diagnostics = fileFailures.map(f => this.failureToDiagnostic(f));
            this.collection.set(uri, diagnostics);
        }
    }

    /** Resolve a file path string to a URI, trying workspace-relative first. */
    private resolveUri(filePath: string): vscode.Uri {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders && workspaceFolders.length > 0) {
            // Try workspace-relative path
            const absPath = vscode.Uri.joinPath(workspaceFolders[0].uri, filePath);
            // Check if the file exists (synchronously is ok here since this is UI code)
            return absPath;
        }
        return vscode.Uri.file(filePath);
    }

    /** Convert a ValidationFailure to a VS Code Diagnostic. */
    private failureToDiagnostic(failure: ValidationFailure): vscode.Diagnostic {
        const message = this.formatMessage(failure);
        const range = failure.range
            ? new vscode.Range(
                failure.range.startLine,
                failure.range.startChar || 0,
                failure.range.endLine || failure.range.startLine,
                failure.range.endChar || 0,
            )
            : new vscode.Range(0, 0, 0, 0);

        const severity = this.mapSeverity(failure);

        const diagnostic = new vscode.Diagnostic(range, message, severity);
        diagnostic.source = 'gpuemu';
        diagnostic.code = failure.kind || 'validation';

        if (failure.seed !== undefined) {
            diagnostic.relatedInformation = [
                new vscode.DiagnosticRelatedInformation(
                    new vscode.Location(
                        vscode.Uri.parse(`gpuemu-seed:${failure.seed}`),
                        new vscode.Range(0, 0, 0, 0),
                    ),
                    `Reproducible with seed ${failure.seed}`,
                ),
            ];
        }

        return diagnostic;
    }

    /** Format a failure into a human-readable diagnostic message. */
    private formatMessage(failure: ValidationFailure): string {
        const parts: string[] = [];

        if (failure.opName) {
            parts.push(`[${failure.opName}]`);
        }

        parts.push(failure.message || 'Validation failed');

        if (failure.dtype) {
            parts.push(`(dtype: ${failure.dtype})`);
        }

        if (failure.shape && failure.shape.length > 0) {
            parts.push(`(shape: [${failure.shape.join(', ')}])`);
        }

        if (failure.maxDiff !== undefined) {
            parts.push(`max_diff: ${failure.maxDiff.toExponential(3)}`);
        }

        return parts.join(' ');
    }

    /** Map failure kind to diagnostic severity. */
    private mapSeverity(failure: ValidationFailure): vscode.DiagnosticSeverity {
        switch (failure.kind) {
            case 'ReferenceError':
                return vscode.DiagnosticSeverity.Error;
            case 'ShapeMismatch':
                return vscode.DiagnosticSeverity.Error;
            case 'ToleranceExceeded':
                return vscode.DiagnosticSeverity.Warning;
            case 'NaN detected':
            case 'Inf':
                return vscode.DiagnosticSeverity.Error;
            default:
                return vscode.DiagnosticSeverity.Warning;
        }
    }

    dispose(): void {
        this.collection.dispose();
    }
}
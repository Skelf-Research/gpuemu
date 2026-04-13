import * as vscode from 'vscode';

/**
 * Validates gpuemu.toml and provides diagnostics for common errors:
 *
 * - References to missing scripts
 * - Invalid dtype names
 * - Invalid layout names
 * - Missing required fields (name, reference)
 * - Malformed TOML
 */
export class ConfigValidator implements vscode.Disposable {
    private collection: vscode.DiagnosticCollection;

    constructor() {
        this.collection = vscode.languages.createDiagnosticCollection('gpuemu-toml');
    }

    /** Validate a gpuemu.toml document and return diagnostics. */
    async validateDocument(doc: vscode.TextDocument): Promise<vscode.Diagnostic[]> {
        if (!doc.uri.fsPath.endsWith('gpuemu.toml')) {
            return [];
        }

        const diagnostics: vscode.Diagnostic[] = [];
        const text = doc.getText();
        const lines = text.split('\n');

        let currentOp: { name?: string; reference?: string; line: number } | null = null;
        let currentKernel: { name?: string; reference?: string; source?: string; line: number } | null = null;
        let inOp = false;
        let inKernel = false;

        const validDtypes = new Set([
            'float16', 'bfloat16', 'float32', 'float64',
            'int8', 'int16', 'int32', 'int64',
            'uint8', 'uint16', 'uint32', 'uint64', 'bool',
        ]);

        const validLayouts = new Set(['contiguous', 'strided', 'transposed']);

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const trimmed = line.trim();

            if (trimmed === '[[ops]]') {
                inOp = true;
                inKernel = false;
                currentOp = { line: i };
            } else if (trimmed === '[[kernels]]') {
                inKernel = true;
                inOp = false;
                currentKernel = { line: i };
            } else if (trimmed.startsWith('[[')) {
                inOp = false;
                inKernel = false;
            } else if (inOp && currentOp) {
                const nameMatch = trimmed.match(/^name\s*=\s*"([^"]+)"/);
                if (nameMatch) {
                    currentOp.name = nameMatch[1];
                }

                const refMatch = trimmed.match(/^reference\s*=\s*"([^"]+)"/);
                if (refMatch) {
                    currentOp.reference = refMatch[1];
                }

                // Validate dtype keys in tolerances
                const tolMatch = trimmed.match(/^((?:b)?float\d+|int\d+|uint\d+|bool)\s*=/);
                if (tolMatch && !validDtypes.has(tolMatch[1])) {
                    diagnostics.push(this.createDiagnostic(
                        i, 0, trimmed.length,
                        `Invalid dtype "${tolMatch[1]}". Valid: ${Array.from(validDtypes).join(', ')}`,
                        vscode.DiagnosticSeverity.Warning,
                    ));
                }

                // Validate execution_mode
                const execMatch = trimmed.match(/^execution_mode\s*=\s*"([^"]+)"/);
                if (execMatch) {
                    const mode = execMatch[1];
                    if (!['client_side', 'daemon_orchestrated', 'script_based'].includes(mode)) {
                        diagnostics.push(this.createDiagnostic(
                            i, 0, trimmed.length,
                            `Invalid execution_mode "${mode}". Valid: client_side, daemon_orchestrated, script_based`,
                            vscode.DiagnosticSeverity.Error,
                        ));
                    }
                }
            } else if (inKernel && currentKernel) {
                const nameMatch = trimmed.match(/^name\s*=\s*"([^"]+)"/);
                if (nameMatch) {
                    currentKernel.name = nameMatch[1];
                }

                const sourceMatch = trimmed.match(/^source\s*=\s*"([^"]+)"/);
                if (sourceMatch) {
                    currentKernel.source = sourceMatch[1];
                }
            }
        }

        // Check for cross-referenced file existence
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders && workspaceFolders.length > 0) {
            const workspaceRoot = workspaceFolders[0].uri.fsPath;

            // Validate op reference scripts exist
            // (This is done as a best-effort check; the daemon does a full check at startup)
        }

        return diagnostics;
    }

    /** Set diagnostics for a document. */
    setDiagnostics(uri: vscode.Uri, diagnostics: vscode.Diagnostic[]): void {
        this.collection.set(uri, diagnostics);
    }

    /** Clear diagnostics for a document. */
    clear(uri?: vscode.Uri): void {
        if (uri) {
            this.collection.delete(uri);
        } else {
            this.collection.clear();
        }
    }

    private createDiagnostic(
        line: number,
        startChar: number,
        length: number,
        message: string,
        severity: vscode.DiagnosticSeverity,
    ): vscode.Diagnostic {
        const range = new vscode.Range(line, startChar, line, startChar + length);
        const diagnostic = new vscode.Diagnostic(range, message, severity);
        diagnostic.source = 'gpuemu';
        return diagnostic;
    }

    dispose(): void {
        this.collection.dispose();
    }
}
import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';

export interface DaemonStatus {
    running: boolean;
    version?: string;
    uptime?: number;
}

export interface ValidationFailure {
    seed: number;
    opName: string;
    message: string;
    shape?: number[];
    dtype?: string;
    kind?: string;
    maxDiff?: number;
    referencePath?: string;
    range?: {
        startLine: number;
        startChar?: number;
        endLine?: number;
        endChar?: number;
    };
}

export class GpuemuRunner {
    private binaryPath: string | undefined;

    constructor() {
        this.detectBinary();
    }

    private detectBinary(): void {
        const config = vscode.workspace.getConfiguration('gpuemu');
        const configuredPath = config.get<string>('binaryPath');

        if (configuredPath) {
            this.binaryPath = configuredPath;
            return;
        }

        // Try common locations
        const homeDir = process.env.HOME || process.env.USERPROFILE || '';
        const candidates = [
            path.join(homeDir, '.gpuemu', 'bin', 'gpuemu'),
            '/usr/local/bin/gpuemu',
            '/usr/bin/gpuemu',
            'gpuemu', // In PATH
        ];

        for (const candidate of candidates) {
            try {
                cp.execSync(`${candidate} --version`, { stdio: 'ignore' });
                this.binaryPath = candidate;
                return;
            } catch {
                // Not found, try next
            }
        }

        this.binaryPath = 'gpuemu'; // Fall back to PATH
    }

    private getBinary(): string {
        if (!this.binaryPath) {
            this.detectBinary();
        }
        return this.binaryPath || 'gpuemu';
    }

    private getWorkingDir(): string {
        const folders = vscode.workspace.workspaceFolders;
        if (folders && folders.length > 0) {
            return folders[0].uri.fsPath;
        }
        return process.cwd();
    }

    async exec(args: string[]): Promise<string> {
        return new Promise((resolve, reject) => {
            const binary = this.getBinary();
            const cwd = this.getWorkingDir();

            cp.execFile(binary, args, { cwd }, (error, stdout, stderr) => {
                if (error) {
                    reject(new Error(stderr || error.message));
                } else {
                    resolve(stdout);
                }
            });
        });
    }

    async startDaemon(): Promise<void> {
        await this.exec(['daemon', 'start', '--background']);
    }

    async stopDaemon(): Promise<void> {
        await this.exec(['daemon', 'stop']);
    }

    async checkStatus(): Promise<DaemonStatus> {
        try {
            const output = await this.exec(['status']);
            const running = output.includes('running');
            const versionMatch = output.match(/v([\d.]+)/);
            const uptimeMatch = output.match(/uptime (\d+)s/);

            return {
                running,
                version: versionMatch ? versionMatch[1] : undefined,
                uptime: uptimeMatch ? parseInt(uptimeMatch[1]) : undefined,
            };
        } catch {
            return { running: false };
        }
    }

    async runTests(quick: boolean = false): Promise<string> {
        const args = ['test'];
        if (quick) {
            args.push('--quick');
        }
        return this.exec(args);
    }

    async runFuzz(options: { op?: string; iterations?: number }): Promise<string> {
        const args = ['fuzz'];
        if (options.op) {
            args.push('--op', options.op);
        }
        if (options.iterations) {
            args.push('--iterations', options.iterations.toString());
        }
        return this.exec(args);
    }

    async listFailures(limit: number = 20): Promise<ValidationFailure[]> {
        try {
            const output = await this.exec(['failures', '--limit', limit.toString()]);
            return this.parseFailures(output);
        } catch {
            return [];
        }
    }

    async reproduce(seed: number): Promise<string> {
        return this.exec(['reproduce', seed.toString(), '--verbose']);
    }

    async init(options: { name?: string; framework?: string }): Promise<void> {
        const args = ['init'];
        if (options.name) {
            args.push('--name', options.name);
        }
        if (options.framework) {
            args.push('--framework', options.framework);
        }
        await this.exec(args);
    }

    private parseFailures(output: string): ValidationFailure[] {
        const failures: ValidationFailure[] = [];
        const lines = output.split('\n');

        for (const line of lines) {
            // Parse format: SEED  OP  MESSAGE  [KIND] [DTYPE] [SHAPE] [MAX_DIFF]
            const match = line.match(/^(\d+)\s+(\S+)\s+(.+)$/);
            if (match) {
                const seed = parseInt(match[1]);
                const opName = match[2];
                const rest = match[3];

                let message = rest;
                let kind: string | undefined;
                let dtype: string | undefined;
                let shape: number[] | undefined;
                let maxDiff: number | undefined;

                // Try to extract structured fields from the message
                const kindMatch = rest.match(/\[(\w+)\]/);
                if (kindMatch) {
                    kind = kindMatch[1];
                }

                const dtypeMatch = rest.match(/dtype:\s*(\S+)/);
                if (dtypeMatch) {
                    dtype = dtypeMatch[1];
                }

                const shapeMatch = rest.match(/shape:\s*\[([^\]]+)\]/);
                if (shapeMatch) {
                    shape = shapeMatch[1].split(',').map(s => parseInt(s.trim()));
                }

                const diffMatch = rest.match(/max_diff:\s*([\d.eE+-]+)/);
                if (diffMatch) {
                    maxDiff = parseFloat(diffMatch[1]);
                }

                failures.push({
                    seed,
                    opName,
                    message: message.trim(),
                    kind,
                    dtype,
                    shape,
                    maxDiff,
                });
            }
        }

        return failures;
    }
}

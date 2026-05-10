import * as vscode from 'vscode';
import { GpuemuRunner } from '../runner';

/**
 * Integrates gpuemu validation results into VS Code's Test Explorer.
 *
 * Ops registered in gpuemu.toml appear as test suites. Each fuzz iteration
 * appears as an individual test case. Results are mapped to pass/fail/error
 * states in the Testing sidebar.
 */
export class GpuemuTestController implements vscode.Disposable {
    private controller: vscode.TestController;
    private ops: string[] = [];

    constructor(private runner: GpuemuRunner) {
        this.controller = vscode.tests.createTestController(
            'gpuemu-tests',
            'gpuemu Validation',
        );

        this.controller.refreshHandler = async () => {
            await this.refresh();
        };

        // Resolve handler for lazy-loading test details
        this.controller.resolveHandler = async (item) => {
            if (!item) {
                await this.refresh();
            }
        };
    }

    /** Expose createRunProfile externally. */
    get testController(): vscode.TestController {
        return this.controller;
    }

    /** Discover ops from gpuemu.toml and populate the test tree. */
    async refresh(): Promise<void> {
        try {
            const config = await this.loadConfig();
            this.ops = config.ops || [];
            this.populateTestTree();
        } catch {
            // Config not found or invalid — keep existing tree
        }
    }

    /** Run tests via the daemon and update results. */
    async runTests(
        request: vscode.TestRunRequest,
        cancellation: vscode.CancellationToken,
    ): Promise<void> {
        const run = this.controller.createTestRun(request);

        try {
            const testsToRun = this.getTestsToRun(request);

            for (const test of testsToRun) {
                if (cancellation.isCancellationRequested) {
                    run.skipped(test);
                    continue;
                }

                run.started(test);

                try {
                    // Extract op name from the test item's ID
                    const opName = this.opNameFromTestId(test.id);
                    const output = await this.runner.runTests(false);

                    // Parse whether this specific test passed
                    if (this.parseTestResult(output, opName)) {
                        run.passed(test, 0);
                    } else {
                        run.failed(test, new vscode.TestMessage(output));
                    }
                } catch (error) {
                    run.errored(test, new vscode.TestMessage(String(error)));
                }
            }
        } finally {
            run.end();
        }
    }

    /** Create test items from the registered ops. */
    private populateTestTree(): void {
        this.controller.items.replace([]);

        for (const opName of this.ops) {
            const testItem = this.controller.createTestItem(
                `gpuemu:${opName}`,
                opName,
            );
            testItem.description = 'validation test';
            testItem.tags = [new vscode.TestTag('gpuemu'), new vscode.TestTag('op')];
            this.controller.items.add(testItem);
        }
    }

    /** Try to load the gpuemu.toml config to discover ops. */
    private async loadConfig(): Promise<{ ops: string[] }> {
        try {
            const folders = vscode.workspace.workspaceFolders;
            if (!folders || folders.length === 0) {
                return { ops: [] };
            }

            const configPath = vscode.Uri.joinPath(folders[0].uri, 'gpuemu.toml');
            const content = await vscode.workspace.fs.readFile(configPath);
            const text = new TextDecoder().decode(content);

            // Simple TOML parsing for op names (look for [[ops]] sections)
            const ops: string[] = [];
            const lines = text.split('\n');
            let inOps = false;

            for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed === '[[ops]]') {
                    inOps = true;
                } else if (trimmed.startsWith('[[')) {
                    inOps = false;
                } else if (inOps) {
                    const nameMatch = trimmed.match(/^name\s*=\s*"([^"]+)"/);
                    if (nameMatch) {
                        ops.push(nameMatch[1]);
                    }
                }
            }

            return { ops };
        } catch {
            return { ops: [] };
        }
    }

    /** Get test items that should be run based on the request. */
    private getTestsToRun(request: vscode.TestRunRequest): vscode.TestItem[] {
        if (request.include) {
            return [...request.include];
        }

        const all: vscode.TestItem[] = [];
        this.controller.items.forEach(item => all.push(item));
        return all;
    }

    /** Parse test output to determine if a specific op passed. */
    private parseTestResult(output: string, opName: string): boolean {
        // Look for the op in the output — if it appears in a failure line, it failed
        const lines = output.split('\n');
        for (const line of lines) {
            if (line.includes(opName) && (line.includes('FAIL') || line.includes('failed'))) {
                return false;
            }
        }
        return true;
    }

    /** Extract op name from test item ID like "gpuemu:flash_attention". */
    private opNameFromTestId(testId: string): string {
        return testId.replace(/^gpuemu:/, '');
    }

    dispose(): void {
        this.controller.dispose();
    }
}
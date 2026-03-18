import * as vscode from 'vscode';
import { GpuemuRunner, ValidationFailure } from '../runner';

export class FailuresTreeProvider implements vscode.TreeDataProvider<FailureItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<FailureItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private failures: ValidationFailure[] = [];

    constructor(private runner: GpuemuRunner) {}

    async refresh(): Promise<void> {
        this.failures = await this.runner.listFailures();
        this._onDidChangeTreeData.fire(undefined);
    }

    getTreeItem(element: FailureItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: FailureItem): Promise<FailureItem[]> {
        if (element) {
            return [];
        }

        if (this.failures.length === 0) {
            await this.refresh();
        }

        return this.failures.map((failure) => new FailureItem(failure));
    }
}

class FailureItem extends vscode.TreeItem {
    constructor(public readonly failure: ValidationFailure) {
        super(`${failure.opName} (seed: ${failure.seed})`, vscode.TreeItemCollapsibleState.None);

        this.tooltip = `${failure.message}`;
        this.description = failure.message.slice(0, 40);

        this.iconPath = new vscode.ThemeIcon('error', new vscode.ThemeColor('errorForeground'));

        this.command = {
            command: 'gpuemu.reproduce',
            title: 'Reproduce Failure',
            arguments: [failure.seed],
        };

        this.contextValue = 'failure';
    }
}

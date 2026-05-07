import * as vscode from 'vscode';

export class GpuemuStatusBar implements vscode.Disposable {
    private statusBarItem: vscode.StatusBarItem;
    private running: boolean = false;
    private version: string | undefined;

    constructor() {
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left,
            100
        );
        this.statusBarItem.command = 'gpuemu.startDaemon';
        this.update();
        this.statusBarItem.show();
    }

    setRunning(running: boolean): void {
        this.running = running;
        this.statusBarItem.command = running ? 'gpuemu.stopDaemon' : 'gpuemu.startDaemon';
        this.update();
    }

    setVersion(version: string): void {
        this.version = version;
        this.update();
    }

    private update(): void {
        if (this.running) {
            const versionText = this.version ? ` v${this.version}` : '';
            this.statusBarItem.text = `$(check) gpuemu${versionText}`;
            this.statusBarItem.tooltip = 'gpuemu daemon is running. Click to stop.';
            this.statusBarItem.backgroundColor = undefined;
        } else {
            this.statusBarItem.text = '$(circle-slash) gpuemu';
            this.statusBarItem.tooltip = 'gpuemu daemon is not running. Click to start.';
            this.statusBarItem.backgroundColor = new vscode.ThemeColor(
                'statusBarItem.warningBackground'
            );
        }
    }

    dispose(): void {
        this.statusBarItem.dispose();
    }
}

import { AssetStore, type AssetKind, type FolderMap } from '$lib/storage/asset-store.js';

/**
 * Shared reactive state for the asset editor.
 * Used by both asset-editor (center) and context-sidebar (right panel).
 *
 * Blob URL lifecycle is managed here — no effect loops in components.
 */
class AssetEditorState {
	store = $state<AssetStore | null>(null);
	dbName = $state('');
	folders = $state<FolderMap>({});
	folderFiles = $state<Record<string, string[]>>({});
	selectedFolder = $state<string | null>(null);
	loading = $state(true);

	/** Blob URLs for image previews, keyed by asset path. */
	previewUrls = $state<Record<string, string>>({});

	async open(dbName: string) {
		this.dbName = dbName;
		this.loading = true;
		this.revokeAll();
		this.selectedFolder = null;
		this.store = await AssetStore.open(dbName);
		await this.reload();
		this.loading = false;
	}

	async selectFolder(name: string) {
		this.selectedFolder = name;
		await this.loadPreviews();
	}

	async reload() {
		if (!this.store) return;
		this.folders = await this.store.getFolders();
		const files: Record<string, string[]> = {};
		for (const name of Object.keys(this.folders)) {
			files[name] = await this.store.list(name + '/');
		}
		this.folderFiles = files;
	}

	async createFolder(name: string, kind: AssetKind) {
		if (!this.store) return;
		await this.store.createFolder(name, kind);
		await this.reload();
		await this.selectFolder(name);
	}

	async deleteFolder(name: string) {
		if (!this.store) return;
		this.revokeFolderUrls(name);
		await this.store.deleteFolder(name);
		if (this.selectedFolder === name) this.selectedFolder = null;
		await this.reload();
	}

	async uploadFiles(files: FileList | null) {
		if (!this.store || !files || !this.selectedFolder) return;
		for (const file of files) {
			const path = `${this.selectedFolder}/${file.name}`;
			const data = new Uint8Array(await file.arrayBuffer());
			await this.store.put(path, data);
		}
		await this.reload();
		await this.loadPreviews();
	}

	async deleteFile(path: string) {
		if (!this.store) return;
		this.revokeUrl(path);
		await this.store.delete(path);
		await this.reload();
	}

	get selectedFiles(): string[] {
		return this.selectedFolder ? (this.folderFiles[this.selectedFolder] ?? []) : [];
	}

	get selectedKind(): AssetKind {
		return this.selectedFolder ? (this.folders[this.selectedFolder] ?? 'other') : 'other';
	}

	// --- Blob URL management (no reactive loops) ---

	private async loadPreviews() {
		if (!this.store || this.selectedKind !== 'image') return;
		const urls = { ...this.previewUrls };
		for (const path of this.selectedFiles) {
			if (urls[path]) continue;
			const data = await this.store.get(path);
			if (data) {
				const blob = new Blob([data.buffer as ArrayBuffer]);
				urls[path] = URL.createObjectURL(blob);
			}
		}
		this.previewUrls = urls;
	}

	private revokeUrl(path: string) {
		if (this.previewUrls[path]) {
			URL.revokeObjectURL(this.previewUrls[path]);
			const urls = { ...this.previewUrls };
			delete urls[path];
			this.previewUrls = urls;
		}
	}

	private revokeFolderUrls(folderName: string) {
		const prefix = folderName + '/';
		const urls = { ...this.previewUrls };
		for (const key of Object.keys(urls)) {
			if (key.startsWith(prefix)) {
				URL.revokeObjectURL(urls[key]);
				delete urls[key];
			}
		}
		this.previewUrls = urls;
	}

	revokeAll() {
		for (const url of Object.values(this.previewUrls)) URL.revokeObjectURL(url);
		this.previewUrls = {};
	}
}

export const assetEditorState = new AssetEditorState();

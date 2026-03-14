export type AssetKind = 'image' | 'other';

/** Folder metadata: name → kind mapping */
export type FolderMap = Record<string, AssetKind>;

function idb<T>(request: IDBRequest<T>): Promise<T> {
	return new Promise((resolve, reject) => {
		request.onsuccess = () => resolve(request.result);
		request.onerror = () => reject(request.error);
	});
}

function txDone(tx: IDBTransaction): Promise<void> {
	return new Promise((resolve, reject) => {
		tx.oncomplete = () => resolve();
		tx.onerror = () => reject(tx.error);
		tx.onabort = () => reject(tx.error);
	});
}

const ASSETS = 'assets';
const META = 'meta';

function openAssetDB(dbName: string): Promise<IDBDatabase> {
	return new Promise((resolve, reject) => {
		const r = indexedDB.open(dbName, 1);
		r.onupgradeneeded = () => {
			const db = r.result;
			if (!db.objectStoreNames.contains(ASSETS)) db.createObjectStore(ASSETS);
			if (!db.objectStoreNames.contains(META)) db.createObjectStore(META);
		};
		r.onsuccess = () => resolve(r.result);
		r.onerror = () => reject(r.error);
	});
}

export class AssetStore {
	private constructor(private db: IDBDatabase) {}

	static async open(dbName: string): Promise<AssetStore> {
		return new AssetStore(await openAssetDB(dbName));
	}

	/** Get raw asset bytes by full path (e.g. "portraits/alice.png"). */
	async get(path: string): Promise<Uint8Array | null> {
		const result = await idb<Uint8Array | undefined>(
			this.db.transaction(ASSETS, 'readonly').objectStore(ASSETS).get(path),
		);
		return result ?? null;
	}

	/** Store an asset at the given path. Increments version. */
	async put(path: string, data: Uint8Array): Promise<void> {
		const tx = this.db.transaction([ASSETS, META], 'readwrite');
		tx.objectStore(ASSETS).put(data, path);
		const metaStore = tx.objectStore(META);
		const getReq = metaStore.get('version');
		getReq.onsuccess = () => {
			const current = (getReq.result as number | undefined) ?? 0;
			metaStore.put(current + 1, 'version');
		};
		await txDone(tx);
	}

	/** Delete an asset by path. Increments version. */
	async delete(path: string): Promise<void> {
		const tx = this.db.transaction([ASSETS, META], 'readwrite');
		tx.objectStore(ASSETS).delete(path);
		const metaStore = tx.objectStore(META);
		const getReq = metaStore.get('version');
		getReq.onsuccess = () => {
			const current = (getReq.result as number | undefined) ?? 0;
			metaStore.put(current + 1, 'version');
		};
		await txDone(tx);
	}

	/** List asset paths matching a prefix. Empty string returns all. */
	async list(prefix: string = ''): Promise<string[]> {
		const all = await idb<IDBValidKey[]>(
			this.db.transaction(ASSETS, 'readonly').objectStore(ASSETS).getAllKeys(),
		);
		const paths = all.filter((k): k is string => typeof k === 'string');
		if (!prefix) return paths;
		return paths.filter((p) => p.startsWith(prefix));
	}

	/** Read the version counter. */
	async version(): Promise<number> {
		const result = await idb<number | undefined>(
			this.db.transaction(META, 'readonly').objectStore(META).get('version'),
		);
		return result ?? 0;
	}

	// --- Folder management ---

	/** Get all folders with their kinds. */
	async getFolders(): Promise<FolderMap> {
		const result = await idb<FolderMap | undefined>(
			this.db.transaction(META, 'readonly').objectStore(META).get('folders'),
		);
		return result ?? {};
	}

	/** Create a folder with the given kind. Increments version. */
	async createFolder(name: string, kind: AssetKind): Promise<void> {
		const tx = this.db.transaction(META, 'readwrite');
		const store = tx.objectStore(META);
		const getReq = store.get('folders');
		getReq.onsuccess = () => {
			const folders: FolderMap = (getReq.result as FolderMap | undefined) ?? {};
			folders[name] = kind;
			store.put(folders, 'folders');
			// Increment version
			const verReq = store.get('version');
			verReq.onsuccess = () => {
				const current = (verReq.result as number | undefined) ?? 0;
				store.put(current + 1, 'version');
			};
		};
		await txDone(tx);
	}

	/** Delete a folder and all its assets. Increments version. */
	async deleteFolder(name: string): Promise<void> {
		// First list all assets in this folder
		const paths = await this.list(name + '/');

		const tx = this.db.transaction([ASSETS, META], 'readwrite');
		const assets = tx.objectStore(ASSETS);
		const meta = tx.objectStore(META);

		// Delete all assets in the folder
		for (const path of paths) {
			assets.delete(path);
		}

		// Remove folder from map
		const getReq = meta.get('folders');
		getReq.onsuccess = () => {
			const folders: FolderMap = (getReq.result as FolderMap | undefined) ?? {};
			delete folders[name];
			meta.put(folders, 'folders');
			// Increment version
			const verReq = meta.get('version');
			verReq.onsuccess = () => {
				const current = (verReq.result as number | undefined) ?? 0;
				meta.put(current + 1, 'version');
			};
		};
		await txDone(tx);
	}
}

export type AssetKind = 'image' | 'other';
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

async function hashBytes(data: Uint8Array): Promise<string> {
	const digest = await crypto.subtle.digest('SHA-256', data.buffer as ArrayBuffer);
	return Array.from(new Uint8Array(digest))
		.map((b) => b.toString(16).padStart(2, '0'))
		.join('');
}

const ASSETS = 'assets';
const META = 'meta';
const BLOBS = 'blobs';
const BLOB_DB_NAME = 'asset_blobs';

type BlobEntry = { data: Uint8Array; rc: number };

function openDB(dbName: string, stores: string[]): Promise<IDBDatabase> {
	return new Promise((resolve, reject) => {
		const r = indexedDB.open(dbName, 1);
		r.onupgradeneeded = () => {
			const db = r.result;
			for (const name of stores) {
				if (!db.objectStoreNames.contains(name)) db.createObjectStore(name);
			}
		};
		r.onsuccess = () => resolve(r.result);
		r.onerror = () => reject(r.error);
	});
}

export class AssetStore {
	private constructor(
		private db: IDBDatabase,
		private blobDb: IDBDatabase,
	) {}

	static async open(dbName: string): Promise<AssetStore> {
		const [db, blobDb] = await Promise.all([
			openDB(dbName, [ASSETS, META]),
			openDB(BLOB_DB_NAME, [BLOBS]),
		]);
		return new AssetStore(db, blobDb);
	}

	/** Get raw asset bytes by path. Resolves hash → blob. */
	async get(path: string): Promise<Uint8Array | null> {
		const hash = await idb<string | undefined>(
			this.db.transaction(ASSETS, 'readonly').objectStore(ASSETS).get(path),
		);
		if (!hash) return null;
		const entry = await idb<BlobEntry | undefined>(
			this.blobDb.transaction(BLOBS, 'readonly').objectStore(BLOBS).get(hash),
		);
		return entry?.data ?? null;
	}

	/** Store asset. Hashes data, dedup in shared blob store, saves hash in per-entity store. */
	async put(path: string, data: Uint8Array): Promise<void> {
		const hash = await hashBytes(data);

		// Upsert blob with refcount increment
		const blobTx = this.blobDb.transaction(BLOBS, 'readwrite');
		const blobStore = blobTx.objectStore(BLOBS);
		const getReq = blobStore.get(hash);
		getReq.onsuccess = () => {
			const existing = getReq.result as BlobEntry | undefined;
			if (existing) {
				blobStore.put({ data: existing.data, rc: existing.rc + 1 }, hash);
			} else {
				blobStore.put({ data, rc: 1 }, hash);
			}
		};
		await txDone(blobTx);

		// Check if replacing an existing asset (need to decrement old blob)
		const oldHash = await idb<string | undefined>(
			this.db.transaction(ASSETS, 'readonly').objectStore(ASSETS).get(path),
		);
		if (oldHash && oldHash !== hash) {
			await this.decrementBlob(oldHash);
		}

		// Save hash pointer + increment version
		const tx = this.db.transaction([ASSETS, META], 'readwrite');
		tx.objectStore(ASSETS).put(hash, path);
		const metaStore = tx.objectStore(META);
		const verReq = metaStore.get('version');
		verReq.onsuccess = () => {
			const current = (verReq.result as number | undefined) ?? 0;
			metaStore.put(current + 1, 'version');
		};
		await txDone(tx);
	}

	/** Delete asset. Decrements blob refcount. */
	async delete(path: string): Promise<void> {
		// Read hash before deleting
		const hash = await idb<string | undefined>(
			this.db.transaction(ASSETS, 'readonly').objectStore(ASSETS).get(path),
		);

		// Delete from asset store + increment version
		const tx = this.db.transaction([ASSETS, META], 'readwrite');
		tx.objectStore(ASSETS).delete(path);
		const metaStore = tx.objectStore(META);
		const verReq = metaStore.get('version');
		verReq.onsuccess = () => {
			const current = (verReq.result as number | undefined) ?? 0;
			metaStore.put(current + 1, 'version');
		};
		await txDone(tx);

		// Decrement blob refcount
		if (hash) await this.decrementBlob(hash);
	}

	/** List asset paths matching prefix. */
	async list(prefix: string = ''): Promise<string[]> {
		const all = await idb<IDBValidKey[]>(
			this.db.transaction(ASSETS, 'readonly').objectStore(ASSETS).getAllKeys(),
		);
		const paths = all.filter((k): k is string => typeof k === 'string');
		if (!prefix) return paths;
		return paths.filter((p) => p.startsWith(prefix));
	}

	async version(): Promise<number> {
		const result = await idb<number | undefined>(
			this.db.transaction(META, 'readonly').objectStore(META).get('version'),
		);
		return result ?? 0;
	}

	// --- Folder management ---

	async getFolders(): Promise<FolderMap> {
		const result = await idb<FolderMap | undefined>(
			this.db.transaction(META, 'readonly').objectStore(META).get('folders'),
		);
		return result ?? {};
	}

	async createFolder(name: string, kind: AssetKind): Promise<void> {
		const tx = this.db.transaction(META, 'readwrite');
		const store = tx.objectStore(META);
		const getReq = store.get('folders');
		getReq.onsuccess = () => {
			const folders: FolderMap = (getReq.result as FolderMap | undefined) ?? {};
			folders[name] = kind;
			store.put(folders, 'folders');
			const verReq = store.get('version');
			verReq.onsuccess = () => {
				const current = (verReq.result as number | undefined) ?? 0;
				store.put(current + 1, 'version');
			};
		};
		await txDone(tx);
	}

	async deleteFolder(name: string): Promise<void> {
		const paths = await this.list(name + '/');

		// Collect hashes before deletion for blob cleanup
		const hashes: string[] = [];
		const assetTx = this.db.transaction(ASSETS, 'readonly');
		const assetStore = assetTx.objectStore(ASSETS);
		for (const path of paths) {
			const hash = await idb<string | undefined>(assetStore.get(path));
			if (hash) hashes.push(hash);
		}

		// Delete assets + folder + increment version
		const tx = this.db.transaction([ASSETS, META], 'readwrite');
		const assets = tx.objectStore(ASSETS);
		const meta = tx.objectStore(META);
		for (const path of paths) assets.delete(path);
		const getReq = meta.get('folders');
		getReq.onsuccess = () => {
			const folders: FolderMap = (getReq.result as FolderMap | undefined) ?? {};
			delete folders[name];
			meta.put(folders, 'folders');
			const verReq = meta.get('version');
			verReq.onsuccess = () => {
				const current = (verReq.result as number | undefined) ?? 0;
				meta.put(current + 1, 'version');
			};
		};
		await txDone(tx);

		// Decrement blob refcounts
		for (const hash of hashes) await this.decrementBlob(hash);
	}

	// --- Blob refcount management ---

	private async decrementBlob(hash: string): Promise<void> {
		const tx = this.blobDb.transaction(BLOBS, 'readwrite');
		const store = tx.objectStore(BLOBS);
		const getReq = store.get(hash);
		getReq.onsuccess = () => {
			const entry = getReq.result as BlobEntry | undefined;
			if (!entry) return;
			if (entry.rc <= 1) {
				store.delete(hash);
			} else {
				store.put({ data: entry.data, rc: entry.rc - 1 }, hash);
			}
		};
		await txDone(tx);
	}
}

import { computeHash, type Backend, type BlobStore, type CmpxchgOp, type CollectedBlob, type Hash, type RefStore } from './merkle.js';

const DB_NAME = 'acvus';
const DB_VERSION = 1;
const BLOBS = 'blobs';
const REFS = 'refs';

type BlobEntry = { data: Uint8Array; rc: number };

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

function openDB(): Promise<IDBDatabase> {
	return new Promise((resolve, reject) => {
		const r = indexedDB.open(DB_NAME, DB_VERSION);
		r.onupgradeneeded = () => {
			const db = r.result;
			if (!db.objectStoreNames.contains(BLOBS)) db.createObjectStore(BLOBS);
			if (!db.objectStoreNames.contains(REFS)) db.createObjectStore(REFS);
		};
		r.onsuccess = () => resolve(r.result);
		r.onerror = () => reject(r.error);
	});
}

class IDBBlobStore implements BlobStore {
	constructor(private db: IDBDatabase) {}

	async get(hash: Hash): Promise<Uint8Array | null> {
		const entry = await idb<BlobEntry | undefined>(
			this.db.transaction(BLOBS, 'readonly').objectStore(BLOBS).get(hash),
		);
		return entry?.data ?? null;
	}

	async getMany(hashes: Hash[]): Promise<(Uint8Array | null)[]> {
		if (hashes.length === 0) return [];
		const tx = this.db.transaction(BLOBS, 'readonly');
		const store = tx.objectStore(BLOBS);
		const requests = hashes.map((h) => store.get(h));
		await txDone(tx);
		return requests.map((r) => (r.result as BlobEntry | undefined)?.data ?? null);
	}

	async put(data: Uint8Array): Promise<Hash> {
		const hash = await computeHash(data);
		const tx = this.db.transaction(BLOBS, 'readwrite');
		const store = tx.objectStore(BLOBS);
		const getReq = store.get(hash);
		getReq.onsuccess = () => {
			if (!getReq.result) store.put({ data, rc: 0 } satisfies BlobEntry, hash);
		};
		await txDone(tx);
		return hash;
	}

	async putMany(blobs: CollectedBlob[]): Promise<void> {
		if (blobs.length === 0) return;
		const tx = this.db.transaction(BLOBS, 'readwrite');
		const store = tx.objectStore(BLOBS);
		for (const { hash, data } of blobs) {
			const getReq = store.get(hash);
			getReq.onsuccess = () => {
				if (!getReq.result) store.put({ data, rc: 0 } satisfies BlobEntry, hash);
			};
		}
		await txDone(tx);
	}

	async delete(hash: Hash): Promise<void> {
		await idb(this.db.transaction(BLOBS, 'readwrite').objectStore(BLOBS).delete(hash));
	}

	async deleteMany(hashes: Hash[]): Promise<void> {
		if (hashes.length === 0) return;
		const tx = this.db.transaction(BLOBS, 'readwrite');
		const store = tx.objectStore(BLOBS);
		for (const hash of hashes) store.delete(hash);
		await txDone(tx);
	}

	async getRefCount(hash: Hash): Promise<number> {
		const entry = await idb<BlobEntry | undefined>(
			this.db.transaction(BLOBS, 'readonly').objectStore(BLOBS).get(hash),
		);
		return entry?.rc ?? 0;
	}

	async setRefCount(hash: Hash, count: number): Promise<void> {
		const tx = this.db.transaction(BLOBS, 'readwrite');
		const store = tx.objectStore(BLOBS);
		const getReq = store.get(hash);
		getReq.onsuccess = () => {
			const entry = getReq.result as BlobEntry | undefined;
			if (entry) store.put({ data: entry.data, rc: count } satisfies BlobEntry, hash);
		};
		await txDone(tx);
	}

	async listAll(): Promise<Hash[]> {
		return idb(this.db.transaction(BLOBS, 'readonly').objectStore(BLOBS).getAllKeys()) as Promise<Hash[]>;
	}
}

class IDBRefStore implements RefStore {
	constructor(private db: IDBDatabase) {}

	async get(key: string): Promise<Hash | null> {
		const val = await idb<Hash | undefined>(
			this.db.transaction(REFS, 'readonly').objectStore(REFS).get(key),
		);
		return val ?? null;
	}

	async set(key: string, hash: Hash): Promise<void> {
		await idb(this.db.transaction(REFS, 'readwrite').objectStore(REFS).put(hash, key));
	}

	cmpxchg(key: string, expected: Hash | null, desired: Hash): Promise<boolean> {
		return new Promise((resolve, reject) => {
			const tx = this.db.transaction(REFS, 'readwrite');
			const store = tx.objectStore(REFS);
			const getReq = store.get(key);
			getReq.onsuccess = () => {
				if ((getReq.result ?? null) !== expected) { resolve(false); return; }
				const putReq = store.put(desired, key);
				putReq.onsuccess = () => resolve(true);
				putReq.onerror = () => reject(putReq.error);
			};
			getReq.onerror = () => reject(getReq.error);
		});
	}

	cmpxchgMany(ops: CmpxchgOp[]): Promise<boolean> {
		if (ops.length === 0) return Promise.resolve(true);
		return new Promise((resolve, reject) => {
			const tx = this.db.transaction(REFS, 'readwrite');
			const store = tx.objectStore(REFS);
			let checked = 0;

			for (const op of ops) {
				const getReq = store.get(op.key);
				getReq.onsuccess = () => {
					if ((getReq.result ?? null) !== op.expected) {
						resolve(false);
						try { tx.abort(); } catch { /* already done */ }
						return;
					}
					checked++;
					if (checked === ops.length) {
						for (const o of ops) store.put(o.desired, o.key);
						tx.oncomplete = () => resolve(true);
					}
				};
				getReq.onerror = () => reject(getReq.error);
			}
		});
	}

	async list(): Promise<string[]> {
		return idb(this.db.transaction(REFS, 'readonly').objectStore(REFS).getAllKeys()) as Promise<string[]>;
	}
}

export class IndexedDBBackend implements Backend {
	readonly blobs: BlobStore;
	readonly refs: RefStore;

	private constructor(db: IDBDatabase) {
		this.blobs = new IDBBlobStore(db);
		this.refs = new IDBRefStore(db);
	}

	static async open(): Promise<IndexedDBBackend> {
		return new IndexedDBBackend(await openDB());
	}
}

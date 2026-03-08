import { computeHash, type Backend, type BlobStore, type CmpxchgOp, type CollectedBlob, type Hash, type RefStore } from './merkle.js';

class MemoryBlobStore implements BlobStore {
	private store = new Map<Hash, { data: Uint8Array; rc: number }>();

	async get(hash: Hash) { return this.store.get(hash)?.data ?? null; }
	async getMany(hashes: Hash[]) { return hashes.map((h) => this.store.get(h)?.data ?? null); }

	async put(data: Uint8Array) {
		const hash = await computeHash(data);
		if (!this.store.has(hash)) this.store.set(hash, { data, rc: 0 });
		return hash;
	}

	async putMany(blobs: CollectedBlob[]) {
		for (const { hash, data } of blobs) {
			if (!this.store.has(hash)) this.store.set(hash, { data, rc: 0 });
		}
	}

	async delete(hash: Hash) { this.store.delete(hash); }
	async deleteMany(hashes: Hash[]) { for (const h of hashes) this.store.delete(h); }

	async getRefCount(hash: Hash) { return this.store.get(hash)?.rc ?? 0; }
	async setRefCount(hash: Hash, count: number) {
		const entry = this.store.get(hash);
		if (entry) entry.rc = count;
	}

	async listAll() { return [...this.store.keys()]; }

	get size() { return this.store.size; }
}

class MemoryRefStore implements RefStore {
	private store = new Map<string, Hash>();

	async get(key: string) { return this.store.get(key) ?? null; }
	async set(key: string, hash: Hash) { this.store.set(key, hash); }

	async cmpxchg(key: string, expected: Hash | null, desired: Hash) {
		if ((this.store.get(key) ?? null) !== expected) return false;
		this.store.set(key, desired);
		return true;
	}

	async cmpxchgMany(ops: CmpxchgOp[]) {
		for (const op of ops) {
			if ((this.store.get(op.key) ?? null) !== op.expected) return false;
		}
		for (const op of ops) this.store.set(op.key, op.desired);
		return true;
	}

	async list() { return [...this.store.keys()]; }
}

export class MemoryBackend implements Backend {
	readonly blobs = new MemoryBlobStore();
	readonly refs = new MemoryRefStore();
}

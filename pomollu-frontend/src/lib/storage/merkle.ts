const encoder = new TextEncoder();
const decoder = new TextDecoder();

// --- Hash ---

export type Hash = string;

export async function computeHash(data: Uint8Array): Promise<Hash> {
	const buf = await crypto.subtle.digest('SHA-256', data as ArrayBufferView<ArrayBuffer>);
	return Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, '0')).join('');
}

// --- Blob format ---

// Every merkle blob stores payload + child hashes.
// Self-describing: refcount walks don't need the schema.
type BlobData = { d: unknown; c: Hash[] };

async function makeBlob(data: unknown, children: Hash[]): Promise<CollectedBlob> {
	const bytes = encoder.encode(JSON.stringify({ d: data, c: children }));
	const hash = await computeHash(bytes);
	return { hash, data: bytes };
}

function decodeBlob(bytes: Uint8Array): BlobData {
	return JSON.parse(decoder.decode(bytes));
}

// --- Collected blob (produced by collect, consumed by putMany) ---

export type CollectedBlob = { hash: Hash; data: Uint8Array };

// --- Backend interfaces ---

export interface BlobStore {
	get(hash: Hash): Promise<Uint8Array | null>;
	getMany(hashes: Hash[]): Promise<(Uint8Array | null)[]>;
	put(data: Uint8Array): Promise<Hash>;
	putMany(blobs: CollectedBlob[]): Promise<void>;
	delete(hash: Hash): Promise<void>;
	deleteMany(hashes: Hash[]): Promise<void>;
	getRefCount(hash: Hash): Promise<number>;
	setRefCount(hash: Hash, count: number): Promise<void>;
	listAll(): Promise<Hash[]>;
}

export type CmpxchgOp = { key: string; expected: Hash | null; desired: Hash };

export interface RefStore {
	get(key: string): Promise<Hash | null>;
	set(key: string, hash: Hash): Promise<void>;
	cmpxchg(key: string, expected: Hash | null, desired: Hash): Promise<boolean>;
	cmpxchgMany(ops: CmpxchgOp[]): Promise<boolean>;
	list(): Promise<string[]>;
}

export interface Backend {
	blobs: BlobStore;
	refs: RefStore;
}

// --- Merkle Schema ---

export type Collected = { root: Hash; blobs: CollectedBlob[] };

export interface MerkleSchema<T> {
	collect(value: T): Promise<Collected>;
	load(blobs: BlobStore, hash: Hash): Promise<T>;
	join(local: T, remote: T): T;
}

// --- Schema implementations ---

// Leaf: stores value directly as a blob. Join = replace (remote wins).
export function leaf<T>(): MerkleSchema<T> {
	return {
		async collect(value) {
			const blob = await makeBlob(value, []);
			return { root: blob.hash, blobs: [blob] };
		},
		async load(blobs, hash) {
			const bytes = await blobs.get(hash);
			if (!bytes) throw new Error(`blob not found: ${hash}`);
			return decodeBlob(bytes).d as T;
		},
		join(_local, remote) {
			return remote;
		},
	};
}

// Map: stores { [id]: childHash }. Join = union + child join on conflicts.
export function map<T>(child: MerkleSchema<T>): MerkleSchema<Record<string, T>> {
	return {
		async collect(value) {
			const entries = Object.entries(value);
			const collected = await Promise.all(entries.map(([, v]) => child.collect(v)));
			const allBlobs: CollectedBlob[] = [];
			const hashes: Record<string, Hash> = {};
			entries.forEach(([id], i) => {
				hashes[id] = collected[i].root;
				allBlobs.push(...collected[i].blobs);
			});
			const self = await makeBlob(hashes, Object.values(hashes));
			allBlobs.push(self);
			return { root: self.hash, blobs: allBlobs };
		},
		async load(blobs, hash) {
			const bytes = await blobs.get(hash);
			if (!bytes) throw new Error(`blob not found: ${hash}`);
			const hashes = decodeBlob(bytes).d as Record<string, Hash>;
			const entries = Object.entries(hashes);
			const result: Record<string, T> = {};
			await Promise.all(entries.map(async ([id, h]) => {
				result[id] = await child.load(blobs, h);
			}));
			return result;
		},
		join(local, remote) {
			const result: Record<string, T> = { ...remote };
			for (const [id, v] of Object.entries(local)) {
				result[id] = id in remote ? child.join(v, remote[id]) : v;
			}
			return result;
		},
	};
}

// Record: fixed-field schema. Each field has its own schema.
type SchemaRecord<T> = { [K in keyof T]: MerkleSchema<T[K]> };

export function record<T extends Record<string, unknown>>(schemas: SchemaRecord<T>): MerkleSchema<T> {
	const keys = Object.keys(schemas) as (keyof T & string)[];
	return {
		async collect(value) {
			const collected = await Promise.all(keys.map((k) => schemas[k].collect(value[k])));
			const allBlobs: CollectedBlob[] = [];
			const hashes: Record<string, Hash> = {};
			keys.forEach((k, i) => {
				hashes[k] = collected[i].root;
				allBlobs.push(...collected[i].blobs);
			});
			const self = await makeBlob(hashes, Object.values(hashes));
			allBlobs.push(self);
			return { root: self.hash, blobs: allBlobs };
		},
		async load(blobs, hash) {
			const bytes = await blobs.get(hash);
			if (!bytes) throw new Error(`blob not found: ${hash}`);
			const hashes = decodeBlob(bytes).d as Record<string, Hash>;
			const values = await Promise.all(keys.map((k) => schemas[k].load(blobs, hashes[k])));
			const result = {} as T;
			keys.forEach((k, i) => { result[k] = values[i]; });
			return result;
		},
		join(local, remote) {
			const result = {} as T;
			for (const k of keys) {
				result[k] = schemas[k].join(local[k], remote[k]);
			}
			return result;
		},
	};
}

// EntityMap: bridges T[] (app) ↔ Record<id, T> (merkle tree).
export function entityMap<T extends { id: string }>(child: MerkleSchema<T>): MerkleSchema<T[]> {
	const inner = map(child);
	function toRecord(arr: T[]): Record<string, T> {
		const rec: Record<string, T> = {};
		for (const item of arr) rec[item.id] = item;
		return rec;
	}
	return {
		collect: (value) => inner.collect(toRecord(value)),
		load: async (blobs, hash) => Object.values(await inner.load(blobs, hash)),
		join: (local, remote) => Object.values(inner.join(toRecord(local), toRecord(remote))),
	};
}

// --- MerkleStore ---

export type SaveResult<T> =
	| { status: 'ok'; hash: Hash }
	| { status: 'conflict'; remote: T };

export class MerkleStore {
	constructor(private backend: Backend) {}

	private static dedup(collected: CollectedBlob[]): CollectedBlob[] {
		const unique = new Map<Hash, Uint8Array>();
		for (const b of collected) unique.set(b.hash, b.data);
		return [...unique].map(([hash, data]) => ({ hash, data }));
	}

	async save<T>(ref: string, schema: MerkleSchema<T>, value: T): Promise<SaveResult<T>> {
		const { blobs, refs } = this.backend;
		const [oldHash, { root, blobs: collected }] = await Promise.all([
			refs.get(ref),
			schema.collect(value),
		]);

		if (oldHash === root) return { status: 'ok', hash: root };

		await blobs.putMany(MerkleStore.dedup(collected));

		const ok = await refs.cmpxchg(ref, oldHash, root);
		if (!ok) {
			const remoteHash = await refs.get(ref);
			if (!remoteHash) throw new Error(`ref disappeared: ${ref}`);
			const remote = await schema.load(blobs, remoteHash);
			return { status: 'conflict', remote };
		}

		await this.incRefTree(root);
		if (oldHash) await this.decRefTree(oldHash);

		return { status: 'ok', hash: root };
	}

	async forceSave<T>(ref: string, schema: MerkleSchema<T>, value: T): Promise<Hash> {
		const { blobs, refs } = this.backend;
		const [oldHash, { root, blobs: collected }] = await Promise.all([
			refs.get(ref),
			schema.collect(value),
		]);

		if (oldHash === root) return root;

		await blobs.putMany(MerkleStore.dedup(collected));

		await refs.set(ref, root);
		await this.incRefTree(root);
		if (oldHash) await this.decRefTree(oldHash);

		return root;
	}

	async load<T>(ref: string, schema: MerkleSchema<T>): Promise<T | null> {
		const hash = await this.backend.refs.get(ref);
		if (!hash) return null;
		return schema.load(this.backend.blobs, hash);
	}

	// Mark-and-sweep GC (safety net for refcount leaks)
	async gc(): Promise<number> {
		const { blobs, refs } = this.backend;
		const keys = await refs.list();
		const hashes = await Promise.all(keys.map((k) => refs.get(k)));
		const reachable = new Set<Hash>();
		await Promise.all(hashes.map((h) => h ? this.mark(h, reachable) : undefined));
		const all = await blobs.listAll();
		const unreachable = all.filter((h) => !reachable.has(h));
		if (unreachable.length > 0) await blobs.deleteMany(unreachable);
		return unreachable.length;
	}

	private async incRefTree(hash: Hash, visited = new Set<Hash>()): Promise<void> {
		if (visited.has(hash)) return;
		visited.add(hash);
		const { blobs } = this.backend;
		const [count, bytes] = await Promise.all([blobs.getRefCount(hash), blobs.get(hash)]);
		await blobs.setRefCount(hash, count + 1);
		if (bytes) await Promise.all(decodeBlob(bytes).c.map((c) => this.incRefTree(c, visited)));
	}

	private async decRefTree(hash: Hash, visited = new Set<Hash>()): Promise<void> {
		if (visited.has(hash)) return;
		visited.add(hash);
		const { blobs } = this.backend;
		const [count, bytes] = await Promise.all([blobs.getRefCount(hash), blobs.get(hash)]);
		if (count <= 1) {
			if (bytes) await Promise.all(decodeBlob(bytes).c.map((c) => this.decRefTree(c, visited)));
			await blobs.delete(hash);
		} else {
			await blobs.setRefCount(hash, count - 1);
		}
	}

	private async mark(hash: Hash, reachable: Set<Hash>): Promise<void> {
		if (reachable.has(hash)) return;
		reachable.add(hash);
		const bytes = await this.backend.blobs.get(hash);
		if (!bytes) return;
		await Promise.all(decodeBlob(bytes).c.map((c) => this.mark(c, reachable)));
	}
}

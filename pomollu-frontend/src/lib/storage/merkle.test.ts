import { describe, it, expect, beforeEach } from 'vitest';
import { leaf, map, record, entityMap, MerkleStore, type MerkleSchema } from './merkle.js';
import { MemoryBackend } from './memory.js';

// --- Schema tests ---

describe('leaf', () => {
	const schema = leaf<{ name: string; value: number }>();
	let backend: MemoryBackend;

	beforeEach(() => { backend = new MemoryBackend(); });

	it('collect and load roundtrip', async () => {
		const value = { name: 'hello', value: 42 };
		const { root, blobs } = await schema.collect(value);

		expect(root).toBeTruthy();
		expect(blobs).toHaveLength(1);
		expect(blobs[0].hash).toBe(root);

		await backend.blobs.putMany(blobs);
		const loaded = await schema.load(backend.blobs, root);
		expect(loaded).toEqual(value);
	});

	it('same value produces same hash', async () => {
		const a = await schema.collect({ name: 'x', value: 1 });
		const b = await schema.collect({ name: 'x', value: 1 });
		expect(a.root).toBe(b.root);
	});

	it('different values produce different hashes', async () => {
		const a = await schema.collect({ name: 'a', value: 1 });
		const b = await schema.collect({ name: 'b', value: 2 });
		expect(a.root).not.toBe(b.root);
	});

	it('join returns remote (replace)', () => {
		const local = { name: 'local', value: 1 };
		const remote = { name: 'remote', value: 2 };
		expect(schema.join(local, remote)).toEqual(remote);
	});
});

describe('map', () => {
	const schema = map(leaf<string>());
	let backend: MemoryBackend;

	beforeEach(() => { backend = new MemoryBackend(); });

	it('collect and load roundtrip', async () => {
		const value = { a: 'hello', b: 'world' };
		const { root, blobs } = await schema.collect(value);

		// 2 leaf blobs + 1 map blob
		expect(blobs).toHaveLength(3);

		await backend.blobs.putMany(blobs);
		const loaded = await schema.load(backend.blobs, root);
		expect(loaded).toEqual(value);
	});

	it('join merges both sides (union)', () => {
		const local = { a: 'one', c: 'three' };
		const remote = { a: 'ONE', b: 'two' };
		const joined = schema.join(local, remote);
		// remote wins for 'a' (leaf replace), local-only 'c' kept, remote-only 'b' kept
		expect(joined).toEqual({ a: 'ONE', b: 'two', c: 'three' });
	});

	it('shared children produce same hash (dedup)', async () => {
		const value = { a: 'same', b: 'same' };
		const { blobs } = await schema.collect(value);
		const uniqueHashes = new Set(blobs.map((b) => b.hash));
		// 1 unique leaf blob (shared) + 1 map blob = 2 unique
		expect(uniqueHashes.size).toBe(2);
	});
});

describe('record', () => {
	type Data = { name: string; items: Record<string, number> };
	const schema: MerkleSchema<Data> = record<Data>({
		name: leaf<string>(),
		items: map(leaf<number>()),
	});
	let backend: MemoryBackend;

	beforeEach(() => { backend = new MemoryBackend(); });

	it('collect and load roundtrip', async () => {
		const value: Data = { name: 'test', items: { x: 1, y: 2 } };
		const { root, blobs } = await schema.collect(value);

		await backend.blobs.putMany(blobs);
		const loaded = await schema.load(backend.blobs, root);
		expect(loaded).toEqual(value);
	});

	it('join delegates to field schemas', () => {
		const local: Data = { name: 'local', items: { a: 1 } };
		const remote: Data = { name: 'remote', items: { b: 2 } };
		const joined = schema.join(local, remote);
		expect(joined.name).toBe('remote'); // leaf replace
		expect(joined.items).toEqual({ a: 1, b: 2 }); // map union
	});
});

describe('entityMap', () => {
	type Entity = { id: string; name: string };
	const schema = entityMap(leaf<Entity>());
	let backend: MemoryBackend;

	beforeEach(() => { backend = new MemoryBackend(); });

	it('collect and load roundtrip', async () => {
		const value: Entity[] = [
			{ id: '1', name: 'Alice' },
			{ id: '2', name: 'Bob' },
		];
		const { root, blobs } = await schema.collect(value);

		await backend.blobs.putMany(blobs);
		const loaded = await schema.load(backend.blobs, root);
		// Order may differ since Record keys are unordered
		expect(loaded).toHaveLength(2);
		expect(loaded.find((e) => e.id === '1')).toEqual({ id: '1', name: 'Alice' });
		expect(loaded.find((e) => e.id === '2')).toEqual({ id: '2', name: 'Bob' });
	});

	it('join merges by id', () => {
		const local: Entity[] = [
			{ id: '1', name: 'Alice-local' },
			{ id: '3', name: 'Charlie' },
		];
		const remote: Entity[] = [
			{ id: '1', name: 'Alice-remote' },
			{ id: '2', name: 'Bob' },
		];
		const joined = schema.join(local, remote);
		expect(joined.find((e) => e.id === '1')?.name).toBe('Alice-remote'); // replace
		expect(joined.find((e) => e.id === '2')?.name).toBe('Bob'); // remote only
		expect(joined.find((e) => e.id === '3')?.name).toBe('Charlie'); // local only
	});
});

// --- MerkleStore tests ---

describe('MerkleStore', () => {
	type Entity = { id: string; value: number };
	const schema = entityMap(leaf<Entity>());
	let backend: MemoryBackend;
	let store: MerkleStore;

	beforeEach(() => {
		backend = new MemoryBackend();
		store = new MerkleStore(backend);
	});

	it('save and load', async () => {
		const data: Entity[] = [{ id: 'a', value: 1 }];
		const result = await store.save('test', schema, data);
		expect(result.status).toBe('ok');

		const loaded = await store.load('test', schema);
		expect(loaded).toHaveLength(1);
		expect(loaded![0]).toEqual({ id: 'a', value: 1 });
	});

	it('load returns null when ref does not exist', async () => {
		const loaded = await store.load('nonexistent', schema);
		expect(loaded).toBeNull();
	});

	it('save with no changes returns same hash', async () => {
		const data: Entity[] = [{ id: 'a', value: 1 }];
		const r1 = await store.save('test', schema, data);
		const r2 = await store.save('test', schema, data);
		expect(r1.status).toBe('ok');
		expect(r2.status).toBe('ok');
		if (r1.status === 'ok' && r2.status === 'ok') {
			expect(r1.hash).toBe(r2.hash);
		}
	});

	it('detects conflict when ref changed externally', async () => {
		await store.save('test', schema, [{ id: 'a', value: 1 }]);

		// Intercept cmpxchg to simulate concurrent external change
		const originalCmpxchg = backend.refs.cmpxchg.bind(backend.refs);
		let intercepted = false;
		backend.refs.cmpxchg = async (key, expected, desired) => {
			if (!intercepted) {
				intercepted = true;
				// Simulate another client saving different data
				const ext = await schema.collect([{ id: 'a', value: 999 }]);
				await backend.blobs.putMany(ext.blobs);
				await backend.refs.set('test', ext.root);
				return false;
			}
			return originalCmpxchg(key, expected, desired);
		};

		const result = await store.save('test', schema, [{ id: 'a', value: 2 }]);
		expect(result.status).toBe('conflict');
		if (result.status === 'conflict') {
			expect(result.remote[0].value).toBe(999);
		}

		// Restore
		backend.refs.cmpxchg = originalCmpxchg;
	});

	it('forceSave overwrites regardless', async () => {
		await store.save('test', schema, [{ id: 'a', value: 1 }]);

		// External change
		const store2 = new MerkleStore(backend);
		await store2.save('test', schema, [{ id: 'a', value: 999 }]);

		// Force save overwrites
		await store.forceSave('test', schema, [{ id: 'a', value: 2 }]);
		const loaded = await store.load('test', schema);
		expect(loaded![0].value).toBe(2);
	});

	it('refcounts are set after save', async () => {
		await store.save('test', schema, [{ id: 'a', value: 1 }]);

		const allHashes = await backend.blobs.listAll();
		for (const hash of allHashes) {
			const rc = await backend.blobs.getRefCount(hash);
			expect(rc).toBeGreaterThan(0);
		}
	});

	it('gc cleans orphaned blobs', async () => {
		await store.save('test', schema, [{ id: 'a', value: 1 }]);

		// Manually insert orphan blob
		const orphanHash = await backend.blobs.put(new Uint8Array([1, 2, 3]));
		const beforeCount = (await backend.blobs.listAll()).length;

		const deleted = await store.gc();
		expect(deleted).toBe(1);

		const afterCount = (await backend.blobs.listAll()).length;
		expect(afterCount).toBe(beforeCount - 1);
		expect(await backend.blobs.get(orphanHash)).toBeNull();
	});

	it('old blobs cleaned up after update', async () => {
		await store.save('test', schema, [{ id: 'a', value: 1 }]);
		const countAfterFirst = (await backend.blobs.listAll()).length;

		await store.save('test', schema, [{ id: 'a', value: 2 }]);

		// After update, old blobs with rc=0 should have been deleted by decRefTree
		// But shared blobs (like the map wrapper with same structure) might still exist
		const countAfterSecond = (await backend.blobs.listAll()).length;

		// Run GC to verify no orphans remain
		const orphans = await store.gc();
		expect(orphans).toBe(0);
	});

	it('multiple saves and loads preserve data integrity', async () => {
		const versions = [
			[{ id: 'a', value: 1 }],
			[{ id: 'a', value: 2 }, { id: 'b', value: 3 }],
			[{ id: 'b', value: 3 }],
			[{ id: 'c', value: 4 }, { id: 'd', value: 5 }, { id: 'e', value: 6 }],
		];

		for (const data of versions) {
			await store.save('test', schema, data);
			const loaded = await store.load('test', schema);
			expect(loaded!.sort((a, b) => a.id.localeCompare(b.id)))
				.toEqual(data.sort((a, b) => a.id.localeCompare(b.id)));
		}

		// No orphans after all updates
		expect(await store.gc()).toBe(0);
	});
});

// --- cmpxchgMany test ---

describe('MemoryBackend cmpxchgMany', () => {
	it('atomic: all succeed or all fail', async () => {
		const backend = new MemoryBackend();
		await backend.refs.set('a', 'hash1');
		await backend.refs.set('b', 'hash2');

		// One wrong expectation → all fail
		const result = await backend.refs.cmpxchgMany([
			{ key: 'a', expected: 'hash1', desired: 'new1' },
			{ key: 'b', expected: 'WRONG', desired: 'new2' },
		]);
		expect(result).toBe(false);
		expect(await backend.refs.get('a')).toBe('hash1'); // unchanged
		expect(await backend.refs.get('b')).toBe('hash2'); // unchanged

		// All correct → all succeed
		const result2 = await backend.refs.cmpxchgMany([
			{ key: 'a', expected: 'hash1', desired: 'new1' },
			{ key: 'b', expected: 'hash2', desired: 'new2' },
		]);
		expect(result2).toBe(true);
		expect(await backend.refs.get('a')).toBe('new1');
		expect(await backend.refs.get('b')).toBe('new2');
	});
});

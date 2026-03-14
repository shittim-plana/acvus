import type { Prompt, Profile, Bot } from '$lib/types.js';
import { createId } from '$lib/stores.svelte.js';
import { zipSync, unzipSync, strToU8, strFromU8 } from 'fflate';
import { AssetStore, type FolderMap } from '$lib/storage/asset-store.js';

/** Replace top-level `id` with a fresh one, return the new entity. */
export function withNewId<T extends { id: string }>(data: T): T {
	return { ...data, id: createId() };
}

function isObj(v: unknown): v is Record<string, unknown> {
	return typeof v === 'object' && v !== null && !Array.isArray(v);
}

export function validatePrompt(data: unknown): data is Prompt {
	if (!isObj(data)) return false;
	return typeof data.name === 'string'
		&& Array.isArray(data.children)
		&& Array.isArray(data.contextBindings);
}

export function validateProfile(data: unknown): data is Profile {
	if (!isObj(data)) return false;
	return typeof data.name === 'string'
		&& Array.isArray(data.children);
}

export function validateBot(data: unknown): data is Bot {
	if (!isObj(data)) return false;
	return typeof data.name === 'string'
		&& typeof data.promptId === 'string'
		&& typeof data.profileId === 'string'
		&& Array.isArray(data.children);
}

// ---------------------------------------------------------------------------
// ZIP export/import — entity JSON + assets
// ---------------------------------------------------------------------------

function downloadBlob(data: Uint8Array, filename: string, mime: string) {
	const blob = new Blob([data.buffer as ArrayBuffer], { type: mime });
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	a.click();
	URL.revokeObjectURL(url);
}

/**
 * Export an entity (Bot/Prompt/Profile) + its assets as a ZIP file.
 *
 * ZIP structure:
 *   entity.json
 *   assets/meta.json        ← { folders: FolderMap, version: number }
 *   assets/<folder>/<file>  ← raw binary data
 */
export async function exportEntityZip(entity: { id: string; name: string }, dbName: string): Promise<void> {
	const files: Record<string, Uint8Array> = {};

	// 1. Entity JSON
	files['entity.json'] = strToU8(JSON.stringify(entity, null, 2));

	// 2. Assets from IndexedDB
	const store = await AssetStore.open(dbName);
	const folders = await store.getFolders();
	const version = await store.version();

	files['assets/meta.json'] = strToU8(JSON.stringify({ folders, version }));

	const allPaths = await store.list();
	for (const path of allPaths) {
		const data = await store.get(path);
		if (data) {
			files[`assets/${path}`] = data;
		}
	}

	// 3. Create ZIP and download
	const zipped = zipSync(files);
	downloadBlob(zipped, `${entity.name}.zip`, 'application/zip');
}

type ImportedEntity = { entity: unknown; dbName: string };

function pickZipFile(): Promise<Uint8Array> {
	return new Promise((resolve, reject) => {
		const input = document.createElement('input');
		input.type = 'file';
		input.accept = '.zip';
		input.onchange = () => {
			const file = input.files?.[0];
			if (!file) { reject(new Error('no file selected')); return; }
			const reader = new FileReader();
			reader.onload = () => {
				resolve(new Uint8Array(reader.result as ArrayBuffer));
			};
			reader.onerror = () => reject(reader.error);
			reader.readAsArrayBuffer(file);
		};
		input.click();
	});
}

/**
 * Import a ZIP file → entity JSON + assets restored to IndexedDB.
 *
 * Returns the parsed entity data (caller must validate and store it).
 * The new entity gets a fresh ID, and assets are stored under `asset_{newId}`.
 */
export async function importEntityZip(): Promise<ImportedEntity> {
	const zipData = await pickZipFile();
	const files = unzipSync(zipData);

	// 1. Parse entity JSON
	const entityBytes = files['entity.json'];
	if (!entityBytes) throw new Error('ZIP missing entity.json');
	const entity = JSON.parse(strFromU8(entityBytes));

	// Assign new ID
	const newId = createId();
	entity.id = newId;
	const dbName = `asset_${newId}`;

	// 2. Parse asset metadata
	const metaBytes = files['assets/meta.json'];
	let folders: FolderMap = {};
	if (metaBytes) {
		const meta = JSON.parse(strFromU8(metaBytes));
		folders = meta.folders ?? {};
	}

	// 3. Restore assets to IndexedDB
	const store = await AssetStore.open(dbName);

	// Create folders
	for (const [name, kind] of Object.entries(folders)) {
		await store.createFolder(name, kind);
	}

	// Store asset files
	const assetPrefix = 'assets/';
	for (const [path, data] of Object.entries(files)) {
		if (path.startsWith(assetPrefix) && path !== 'assets/meta.json') {
			const assetPath = path.slice(assetPrefix.length);
			if (assetPath) {
				await store.put(assetPath, data);
			}
		}
	}

	return { entity, dbName };
}

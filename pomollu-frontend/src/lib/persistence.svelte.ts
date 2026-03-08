import { MerkleStore, record, entityMap, leaf, type Backend, type MerkleSchema } from './storage/merkle.js';
import { exportData, importData, uiState, type StoreData } from './stores.svelte.js';
import type { Provider, Prompt, Profile, Bot, Session } from './types.js';

const DATA_REF = 'data';
const UI_KEY = 'acvus-ui';
const SAVE_DEBOUNCE_MS = 1000;

export const dataSchema: MerkleSchema<StoreData> = record<StoreData>({
	providers: entityMap(leaf<Provider>()),
	prompts: entityMap(leaf<Prompt>()),
	profiles: entityMap(leaf<Profile>()),
	bots: entityMap(leaf<Bot>()),
	sessions: entityMap(leaf<Session>()),
});

// --- Sync status ---

export type SyncStatus = 'idle' | 'syncing' | 'error';

class SyncState {
	status = $state<SyncStatus>('idle');
	message = $state('');
}

export const syncState = new SyncState();

let store: MerkleStore | null = null;
let saveTimer: ReturnType<typeof setTimeout> | undefined;
let hasLoadedOnce = false;

async function saveData() {
	if (!store) return;
	const data = exportData();
	// Guard: don't save if all stores are empty (likely HMR reset)
	if (
		data.providers.length === 0 &&
		data.prompts.length === 0 &&
		data.profiles.length === 0 &&
		data.bots.length === 0 &&
		data.sessions.length === 0 &&
		hasLoadedOnce
	) return;
	syncState.status = 'syncing';
	syncState.message = '';
	try {
		const result = await store.save(DATA_REF, dataSchema, data);
		if (result.status === 'conflict') {
			await store.forceSave(DATA_REF, dataSchema, data);
		}
		syncState.status = 'idle';
	} catch (e) {
		syncState.status = 'error';
		syncState.message = e instanceof Error ? e.message : 'unknown error';
	}
}

function scheduleSave() {
	clearTimeout(saveTimer);
	saveTimer = setTimeout(saveData, SAVE_DEBOUNCE_MS);
}

function saveUI() {
	try {
		localStorage.setItem(UI_KEY, JSON.stringify(uiState.toJSON()));
	} catch { /* localStorage full or unavailable */ }
}

function loadUI() {
	try {
		const raw = localStorage.getItem(UI_KEY);
		if (raw) uiState.loadJSON(JSON.parse(raw));
	} catch { /* corrupted, use defaults */ }
}

export async function initPersistence(backend: Backend): Promise<() => void> {
	store = new MerkleStore(backend);

	syncState.status = 'syncing';
	try {
		const data = await store.load(DATA_REF, dataSchema);
		if (data) importData(data);
		hasLoadedOnce = true;
		syncState.status = 'idle';
	} catch (e) {
		syncState.status = 'error';
		syncState.message = e instanceof Error ? e.message : 'failed to load';
	}

	// Load UI from localStorage + reconcile
	loadUI();
	uiState.reconcile();

	// Auto-save effects
	const cleanup = $effect.root(() => {
		$effect(() => {
			exportData(); // track all store state as dependencies
			scheduleSave();
		});

		$effect(() => {
			uiState.toJSON(); // track UIState as dependency
			saveUI();
		});
	});

	return () => {
		clearTimeout(saveTimer);
		cleanup();
	};
}

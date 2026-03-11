/**
 * Ephemeral state registry — module-scoped Map that lives for the browser tab's lifetime.
 *
 * Each entry is sole-owned: only the code that created it should access it.
 * Entries survive component remounts but vanish on page reload.
 */

interface EphemeralEntry<T> {
	value: T;
	onDispose?: (value: T) => void;
}

const store = new Map<string, EphemeralEntry<any>>();

/**
 * Acquire or create an ephemeral state entry.
 * Same key always returns the same instance (stable across remounts).
 *
 * @param onDispose — called when `disposeEphemeral(key)` is invoked (e.g. session deletion).
 *   Only registered on first creation; subsequent calls with the same key ignore this parameter.
 */
export function ephemeral<T extends object>(key: string, create: () => T, onDispose?: (value: T) => void): T {
	let entry = store.get(key) as EphemeralEntry<T> | undefined;
	if (!entry) {
		entry = { value: create(), onDispose };
		store.set(key, entry);
	}
	return entry.value;
}

/**
 * Release an ephemeral state entry (e.g. when the owning session is deleted).
 * Invokes the `onDispose` callback if one was registered, then removes the entry.
 */
export function disposeEphemeral(key: string): void {
	const entry = store.get(key);
	if (entry) {
		entry.onDispose?.(entry.value);
		store.delete(key);
	}
}

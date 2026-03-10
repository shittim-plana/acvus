/**
 * Ephemeral state registry — module-scoped Map that lives for the browser tab's lifetime.
 *
 * Each entry is sole-owned: only the code that created it should access it.
 * Entries survive component remounts but vanish on page reload.
 */

const store = new Map<string, object>();

/**
 * Acquire or create an ephemeral state entry.
 * Same key always returns the same instance (stable across remounts).
 */
export function ephemeral<T extends object>(key: string, create: () => T): T {
	let s = store.get(key) as T | undefined;
	if (!s) {
		s = create();
		store.set(key, s);
	}
	return s;
}

/**
 * Release an ephemeral state entry (e.g. when the owning session is deleted).
 * The caller may run cleanup (free WASM objects, etc.) before calling this.
 */
export function disposeEphemeral(key: string): void {
	store.delete(key);
}

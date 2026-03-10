<script lang="ts">
	import type { EntityRef } from '$lib/entity-versions.svelte.js';
	import { entityVersions } from '$lib/entity-versions.svelte.js';
	import { uiState } from '$lib/stores.svelte.js';
	import type { Snippet } from 'svelte';

	let {
		children,
		deps,
		onConfigChange,
		debounceMs = 0,
		lockable = true,
	}: {
		children: Snippet;
		deps: EntityRef[];
		onConfigChange: () => void;
		debounceMs?: number;
		/** Show lock UI when deps are locked. Default true. Set false for pages that manage their own lock state (e.g. chat). */
		lockable?: boolean;
	} = $props();

	let locked = $derived(lockable && uiState.isAnyLocked(deps));
	let depsVersion = $derived(entityVersions.depsVersion(deps));
	let lastVersion = -1;
	let pendingWhileLocked = false;
	let timer: ReturnType<typeof setTimeout> | null = null;

	// Version change → fire callback (or mark pending if locked).
	// First mount (lastVersion === -1) always fires — lock only defers subsequent changes.
	$effect(() => {
		const ver = depsVersion;
		if (ver === lastVersion) return;
		const isFirstRun = lastVersion === -1;
		lastVersion = ver;
		if (locked && !isFirstRun) {
			pendingWhileLocked = true;
			return;
		}
		fireCallback();
		return () => { if (timer) clearTimeout(timer); };
	});

	// Unlock → replay pending callback.
	$effect(() => {
		if (!locked && pendingWhileLocked) {
			pendingWhileLocked = false;
			fireCallback();
		}
		return () => { if (timer) clearTimeout(timer); };
	});

	function fireCallback() {
		if (debounceMs <= 0) {
			onConfigChange();
		} else {
			if (timer) clearTimeout(timer);
			timer = setTimeout(() => {
				timer = null;
				onConfigChange();
			}, debounceMs);
		}
	}
</script>

<div class="flex h-full flex-col">
	{#if locked}
		<div class="shrink-0 border-b bg-amber-500/10 px-4 py-1.5 text-xs text-amber-700 dark:text-amber-400">
			Turn in progress — editing locked
		</div>
	{/if}
	<div class="flex-1 min-h-0 flex flex-col" class:basepage-locked={locked}>
		{@render children()}
	</div>
</div>

<style>
	/* Lock: disable interactive elements but allow scrolling/reading. */
	.basepage-locked :global(input),
	.basepage-locked :global(textarea),
	.basepage-locked :global(button),
	.basepage-locked :global(select),
	.basepage-locked :global([contenteditable]) {
		pointer-events: none !important;
		opacity: 0.5;
	}
</style>

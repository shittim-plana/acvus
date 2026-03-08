<script lang="ts">
	import { syncState } from '$lib/persistence.svelte.js';
</script>

{#if syncState.status !== 'idle'}
	<div
		class="sync-indicator"
		class:sync-error={syncState.status === 'error'}
		title={syncState.status === 'error' ? syncState.message : 'Syncing...'}
	>
		{#if syncState.status === 'syncing'}
			<svg class="sync-spinner" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
				<circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="2" stroke-dasharray="28" stroke-dashoffset="8" stroke-linecap="round" />
			</svg>
		{:else}
			<svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
				<circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="1.5" />
				<path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" />
			</svg>
		{/if}
	</div>
{/if}

<style>
	.sync-indicator {
		position: fixed;
		top: 0.5rem;
		right: 0.5rem;
		z-index: 50;
		width: 1.25rem;
		height: 1.25rem;
		color: var(--color-muted-foreground);
		opacity: 0.6;
	}
	.sync-indicator svg {
		width: 100%;
		height: 100%;
	}
	.sync-error {
		color: var(--color-destructive);
		opacity: 1;
	}
	.sync-spinner {
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to { transform: rotate(360deg); }
	}
</style>

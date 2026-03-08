<script lang="ts">
	import type { Snippet } from 'svelte';
	import { longpress } from '$lib/actions/longpress.js';

	let {
		active = false,
		onselect,
		ondelete,
		children
	}: {
		active?: boolean;
		onselect: () => void;
		ondelete?: () => void;
		children: Snippet;
	} = $props();
</script>

<div class="sidebar-item" use:longpress>
	<button
		class="flex-1 rounded-md px-2.5 py-1.5 text-left text-sm transition-colors truncate
			{active ? 'bg-accent text-accent-foreground' : 'text-muted-foreground hover:bg-accent/50'}"
		onclick={onselect}
	>
		{@render children()}
	</button>
	{#if ondelete}
		<button
			class="delete-btn"
			onclick={ondelete}
			title="Delete"
		>&times;</button>
	{/if}
</div>

<style>
	.sidebar-item {
		display: flex;
		align-items: center;
		gap: 2px;
		position: relative;
	}
	.delete-btn {
		flex-shrink: 0;
		padding: 4px;
		border-radius: 4px;
		border: none;
		background: transparent;
		color: var(--color-muted-foreground);
		opacity: 0;
		pointer-events: none;
		transition: opacity 0.15s;
		cursor: pointer;
	}
	.delete-btn:hover {
		color: var(--color-destructive);
	}
	@media (hover: hover) {
		.sidebar-item:hover > .delete-btn {
			opacity: 1;
			pointer-events: auto;
		}
	}
	:global([data-long-pressed] > .delete-btn) {
		opacity: 1;
		pointer-events: auto;
	}
</style>

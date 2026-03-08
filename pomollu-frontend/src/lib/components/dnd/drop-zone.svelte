<script lang="ts">
	import { useDroppable } from '@dnd-kit-svelte/svelte';
	import type { Snippet } from 'svelte';

	let {
		id,
		active = false,
		children,
	}: {
		id: string;
		active?: boolean;
		children?: Snippet;
	} = $props();

	const droppable = useDroppable({ id: () => id });
</script>

<div {@attach droppable.ref} class="drop-zone" class:active>
	{#if children}
		{@render children()}
	{/if}
</div>

<style>
	.drop-zone {
		min-height: 8px;
		border-radius: 4px;
		font-size: 11px;
		color: var(--color-muted-foreground);
		text-align: center;
		transition: min-height 200ms ease, background 200ms ease;
	}
	.drop-zone.active {
		min-height: 36px;
		background: oklch(from var(--color-primary) l c h / 0.15);
		border: 1px dashed var(--color-primary);
		display: flex;
		align-items: center;
		justify-content: center;
	}
</style>

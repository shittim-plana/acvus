<script lang="ts">
	import { useDraggable, useDroppable } from '@dnd-kit-svelte/svelte';
	import type { Snippet } from 'svelte';

	let {
		id,
		children,
		style = '',
		isDragSource = false,
		isDropTarget = false,
	}: {
		id: string;
		children: Snippet;
		style?: string;
		isDragSource?: boolean;
		isDropTarget?: boolean;
	} = $props();

	const draggable = useDraggable({ id: () => id });
	const droppable = useDroppable({ id: () => id });
</script>

<div
	{@attach draggable.ref}
	{@attach droppable.ref}
	class="drag-item"
	class:drag-source={isDragSource}
	class:drop-target={isDropTarget}
	{style}
>
	{@render children()}
</div>

<style>
	.drag-item {
		transition: transform 200ms ease, opacity 200ms ease;
	}
	.drag-source {
		opacity: 0.3;
	}
	.drop-target {
		box-shadow: 0 -2px 0 0 var(--color-primary);
	}
</style>

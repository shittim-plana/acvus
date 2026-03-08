<script lang="ts">
	import type { BlockNode } from '$lib/types.js';
	import type { BlockOwner } from '$lib/stores.svelte.js';
	import { DragDropProvider } from '@dnd-kit-svelte/svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import { Button } from '$lib/components/ui/button';
	import { moveNode } from '$lib/block-tree.js';
	import BlockTree from './block-tree.svelte';

	let {
		children,
		owner,
		onchildrenchange,
		onadd,
		onaddfolder,
		onaddnode
	}: {
		children: BlockNode[];
		owner: BlockOwner;
		onchildrenchange: (children: BlockNode[]) => void;
		onadd: () => void;
		onaddfolder: () => void;
		onaddnode?: () => void;
	} = $props();

	let dragSourceId: string | null = $state(null);
	let dropTargetId: string | null = $state(null);
	let collapsedIds = new SvelteSet<string>();

	function toggleFolder(id: string) {
		if (collapsedIds.has(id)) collapsedIds.delete(id);
		else collapsedIds.add(id);
	}

	function handleDragStart(event: any) {
		const { source } = event.operation;
		if (source) dragSourceId = String(source.id);
	}

	function handleDragOver(event: any) {
		const { source, target } = event.operation;
		if (target && source && String(target.id) !== String(source.id)) {
			dropTargetId = String(target.id);
		}
	}

	function handleDragEnd(_event: any) {
		const sid = dragSourceId;
		const tid = dropTargetId;
		dragSourceId = null;
		dropTargetId = null;

		if (!sid || !tid || sid === tid) return;
		onchildrenchange(moveNode(children, sid, tid));
	}
</script>

<div class="space-y-1">
	<DragDropProvider onDragStart={handleDragStart} onDragOver={handleDragOver} onDragEnd={handleDragEnd}>
		<BlockTree {children} {owner} {dragSourceId} {dropTargetId} {collapsedIds} ontogglefolder={toggleFolder} />
	</DragDropProvider>
	<div class="flex items-center justify-center gap-1">
		<Button variant="ghost" size="icon-sm" onclick={onadd} title="Add block">+</Button>
		<Button variant="ghost" size="icon-sm" onclick={onaddfolder} title="Add folder">
			<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 10v6"/><path d="M9 13h6"/><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/></svg>
		</Button>
		{#if onaddnode}
			<Button variant="ghost" size="icon-sm" onclick={onaddnode} title="Add node">
				<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 8v8"/><path d="M8 12h8"/></svg>
			</Button>
		{/if}
	</div>
</div>

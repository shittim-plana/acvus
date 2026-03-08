<script lang="ts">
	import type { SidebarNode } from '$lib/types.js';
	import { SvelteSet } from 'svelte/reactivity';
	import { sidebarNodeId, sidebarDropId, moveSidebarNode, addSidebarNode, removeSidebarNode } from '$lib/sidebar-tree.js';
	import { longpress } from '$lib/actions/longpress.js';
	import { createId, generateName } from '$lib/stores.svelte.js';
	import { DragDropProvider } from '@dnd-kit-svelte/svelte';
	import DragItem from './dnd/drag-item.svelte';
	import DropZone from './dnd/drop-zone.svelte';

	let {
		nodes,
		onnodeschange,
		onselect,
		ondelete,
		isActive,
		itemLabel,
		onadd,
	}: {
		nodes: SidebarNode[];
		onnodeschange: (nodes: SidebarNode[]) => void;
		onselect: (id: string) => void;
		ondelete?: (id: string) => void;
		isActive: (id: string) => boolean;
		itemLabel: (id: string) => string;
		onadd: () => void;
	} = $props();

	let collapsedIds = new SvelteSet<string>();
	let dragSourceId: string | null = $state(null);
	let dropTargetId: string | null = $state(null);

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

	function handleDragEnd() {
		const sid = dragSourceId;
		const tid = dropTargetId;
		dragSourceId = null;
		dropTargetId = null;
		if (!sid || !tid || sid === tid) return;
		onnodeschange(moveSidebarNode(nodes, sid, tid));
	}

	function collectFolderNames(list: SidebarNode[]): string[] {
		const out: string[] = [];
		for (const n of list) {
			if (n.kind === 'folder') { out.push(n.name); out.push(...collectFolderNames(n.children)); }
		}
		return out;
	}

	function addFolder() {
		const name = generateName('New Folder', collectFolderNames(nodes));
		onnodeschange(addSidebarNode(nodes, { kind: 'folder', id: createId(), name, children: [] }));
	}

	function removeNode(id: string) {
		onnodeschange(removeSidebarNode(nodes, id));
	}
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
{#snippet renderNodes(list: SidebarNode[], depth: number)}
	{#each list as node (sidebarNodeId(node))}
		{#if node.kind === 'item'}
			<div class="sl-row-wrap" use:longpress>
				<div onclick={() => onselect(node.id)}>
					<DragItem id={node.id} isDragSource={dragSourceId === node.id} isDropTarget={dropTargetId === node.id}
						style={isActive(node.id) ? 'border-color: var(--color-primary);' : ''}>
						<span class="sl-item" class:active={isActive(node.id)} style:padding-left="{0.5 + depth * 0.75}rem">
							{itemLabel(node.id)}
						</span>
					</DragItem>
				</div>
				{#if ondelete}
					<button class="sl-delete" onclick={() => ondelete(node.id)} title="Delete">&times;</button>
				{/if}
			</div>
		{:else}
			<div class="sl-row-wrap" use:longpress>
				<div onclick={() => toggleFolder(node.id)}>
					<DragItem id={node.id} isDragSource={dragSourceId === node.id} isDropTarget={dropTargetId === node.id}
						style="background: var(--color-secondary);">
						<span class="sl-folder" style:padding-left="{0.25 + depth * 0.75}rem">
							<span class="sl-folder-name">{node.name}</span>
							{#if collapsedIds.has(node.id) && node.children.length > 0}
								<span class="sl-child-count">{node.children.length}</span>
							{/if}
						</span>
					</DragItem>
				</div>
				<button class="sl-delete" onclick={() => removeNode(node.id)} title="Delete folder">&times;</button>
			</div>
			{#if !collapsedIds.has(node.id)}
				<div class="sl-indent" style:margin-left="{0.625 + depth * 0.75}rem">
					{@render renderNodes(node.children, depth + 1)}
					<DropZone id={sidebarDropId(node.id)} active={dropTargetId === sidebarDropId(node.id)} />
				</div>
			{/if}
		{/if}
	{/each}
{/snippet}

<div class="sl-root">
	<DragDropProvider onDragStart={handleDragStart} onDragOver={handleDragOver} onDragEnd={handleDragEnd}>
		{@render renderNodes(nodes, 0)}
	</DragDropProvider>
	<div class="sl-actions">
		<button class="sl-action-btn" onclick={onadd} title="Add item">+</button>
		<button class="sl-action-btn" onclick={addFolder} title="Add folder">
			<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 10v6"/><path d="M9 13h6"/><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/></svg>
		</button>
	</div>
</div>

<style>
	/* Override block-tree's global .drag-item styles */
	.sl-root :global(.drag-item) {
		padding: 0.3rem 0.5rem;
		border-radius: 0.375rem;
		font-size: 0.8125rem;
		cursor: grab;
		touch-action: none;
		background: var(--color-card);
		border: 1px solid var(--color-border);
		transition: transform 200ms ease, opacity 200ms ease, border-color 0.15s;
	}
	.sl-root :global(.drag-item:hover) {
		border-color: color-mix(in srgb, var(--color-border) 100%, var(--color-foreground) 20%);
	}

	.sl-root {
		display: flex;
		flex-direction: column;
		gap: 0.1875rem;
		padding: 0.25rem;
	}

	/* Row wrapper — matches block-tree's .tree-item pattern */
	.sl-row-wrap {
		position: relative;
	}

	/* Item */
	.sl-item {
		display: block;
		color: var(--color-muted-foreground);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.sl-item.active {
		color: var(--color-foreground);
	}

	/* Folder */
	.sl-folder {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		color: var(--color-muted-foreground);
	}
	.sl-folder-name {
		font-size: 0.6875rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.03em;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.sl-child-count {
		margin-left: auto;
		font-size: 0.625rem;
		opacity: 0.5;
	}

	/* Delete — round, absolute positioned like bot-sidebar */
	.sl-delete {
		position: absolute;
		right: -3px;
		top: -3px;
		opacity: 0;
		pointer-events: none;
		transition: opacity 0.15s;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 14px;
		height: 14px;
		border-radius: 9999px;
		border: none;
		background: var(--color-destructive);
		color: var(--color-destructive-foreground);
		font-size: 9px;
		line-height: 1;
		cursor: pointer;
		z-index: 1;
	}
	@media (hover: hover) {
		.sl-row-wrap:hover > .sl-delete {
			opacity: 1;
			pointer-events: auto;
		}
	}
	:global([data-long-pressed] > .sl-delete) {
		opacity: 1;
		pointer-events: auto;
	}

	/* Indent guide for folder children */
	.sl-indent {
		display: flex;
		flex-direction: column;
		gap: 0.1875rem;
		margin-top: 0.1875rem;
		border-left: 2px solid var(--color-border);
		padding-left: 0.375rem;
	}
	/* Shrink DropZone inside folder when there are siblings */
	.sl-indent :global(.drop-zone) {
		min-height: 4px;
	}

	/* Actions */
	.sl-actions {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.25rem;
		padding-top: 0.375rem;
	}
	.sl-action-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 1.5rem;
		height: 1.5rem;
		border-radius: 0.25rem;
		border: none;
		background: transparent;
		color: var(--color-muted-foreground);
		cursor: pointer;
		font-size: 0.875rem;
	}
	.sl-action-btn:hover {
		background: var(--color-accent);
		color: var(--color-foreground);
	}
</style>

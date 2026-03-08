import type { SidebarNode } from './types.js';

export function sidebarNodeId(node: SidebarNode): string {
	return node.kind === 'item' ? node.id : node.id;
}

export function findSidebarNode(nodes: SidebarNode[], id: string): SidebarNode | undefined {
	for (const node of nodes) {
		if (sidebarNodeId(node) === id) return node;
		if (node.kind === 'folder') {
			const found = findSidebarNode(node.children, id);
			if (found) return found;
		}
	}
	return undefined;
}

export function addSidebarNode(nodes: SidebarNode[], child: SidebarNode): SidebarNode[] {
	return [...nodes, child];
}

export function removeSidebarNode(nodes: SidebarNode[], id: string): SidebarNode[] {
	const result: SidebarNode[] = [];
	for (const node of nodes) {
		if (sidebarNodeId(node) === id) continue;
		if (node.kind === 'folder') {
			result.push({ ...node, children: removeSidebarNode(node.children, id) });
		} else {
			result.push(node);
		}
	}
	return result;
}

export function collectSidebarItemIds(nodes: SidebarNode[], out: string[] = []): string[] {
	for (const node of nodes) {
		if (node.kind === 'item') out.push(node.id);
		else collectSidebarItemIds(node.children, out);
	}
	return out;
}

export function moveSidebarNode(nodes: SidebarNode[], sourceId: string, targetId: string): SidebarNode[] {
	const sourceNode = findSidebarNode(nodes, sourceId);
	if (!sourceNode) return nodes;
	const without = removeSidebarNode(nodes, sourceId);
	return insertSidebarNode(without, sourceId, targetId, sourceNode);
}

const FOLDER_DROP_PREFIX = 'sidebar-folder-drop:';

export function sidebarDropId(folderId: string): string {
	return FOLDER_DROP_PREFIX + folderId;
}

function insertSidebarNode(nodes: SidebarNode[], sourceId: string, targetId: string, sourceNode: SidebarNode): SidebarNode[] {
	// Drop into folder
	if (targetId.startsWith(FOLDER_DROP_PREFIX)) {
		const folderId = targetId.slice(FOLDER_DROP_PREFIX.length);
		return nodes.map((n) => {
			if (n.kind === 'folder' && n.id === folderId) {
				return { ...n, children: [...n.children, sourceNode] };
			}
			if (n.kind === 'folder') {
				return { ...n, children: insertSidebarNode(n.children, sourceId, targetId, sourceNode) };
			}
			return n;
		});
	}

	// Insert before target
	const result: SidebarNode[] = [];
	for (const node of nodes) {
		if (sidebarNodeId(node) === targetId) {
			result.push(sourceNode);
		}
		if (node.kind === 'folder') {
			result.push({ ...node, children: insertSidebarNode(node.children, sourceId, targetId, sourceNode) });
		} else {
			result.push(node);
		}
	}
	return result;
}

/** Sync tree with actual item IDs: remove stale, append new */
export function syncSidebarTree(tree: SidebarNode[], itemIds: string[]): SidebarNode[] {
	const idSet = new Set(itemIds);
	const existing = new Set(collectSidebarItemIds(tree));

	// Remove items no longer in store
	let synced = pruneStale(tree, idSet);

	// Append new items not in tree
	for (const id of itemIds) {
		if (!existing.has(id)) {
			synced = [...synced, { kind: 'item', id }];
		}
	}
	return synced;
}

function pruneStale(nodes: SidebarNode[], validIds: Set<string>): SidebarNode[] {
	const result: SidebarNode[] = [];
	for (const node of nodes) {
		if (node.kind === 'item') {
			if (validIds.has(node.id)) result.push(node);
		} else {
			const children = pruneStale(node.children, validIds);
			result.push({ ...node, children });
		}
	}
	return result;
}

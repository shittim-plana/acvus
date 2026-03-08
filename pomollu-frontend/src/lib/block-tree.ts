import type { Block, BlockNode, Node } from './types.js';
import { blockLabel } from './types.js';

export const DROP_PREFIX = 'folder-drop:';

function childrenOf(node: BlockNode): BlockNode[] | undefined {
	if (node.kind === 'folder') return node.folder.children;
	return undefined;
}

function withUpdatedChildren(node: BlockNode, children: BlockNode[]): BlockNode {
	if (node.kind === 'folder') return { kind: 'folder', folder: { ...node.folder, children } };
	return node;
}

export function nodeId(node: BlockNode): string {
	switch (node.kind) {
		case 'block': return node.block.id;
		case 'folder': return node.folder.id;
		case 'node': return node.node.id;
	}
}

export function findTreeNode(nodes: BlockNode[], id: string): BlockNode | undefined {
	for (const node of nodes) {
		if (nodeId(node) === id) return node;
		const sub = childrenOf(node);
		if (sub) {
			const found = findTreeNode(sub, id);
			if (found) return found;
		}
	}
	return undefined;
}

export function findBlock(nodes: BlockNode[], id: string): Block | undefined {
	const node = findTreeNode(nodes, id);
	return node?.kind === 'block' ? node.block : undefined;
}

export function collectBlocks(nodes: BlockNode[], out: Block[] = []): Block[] {
	for (const node of nodes) {
		if (node.kind === 'block') out.push(node.block);
		const sub = childrenOf(node);
		if (sub) collectBlocks(sub, out);
	}
	return out;
}

export function collectNodes(nodes: BlockNode[], out: Node[] = []): Node[] {
	for (const node of nodes) {
		if (node.kind === 'node') out.push(node.node);
		const sub = childrenOf(node);
		if (sub) collectNodes(sub, out);
	}
	return out;
}

export function findNodeItem(nodes: BlockNode[], id: string): Node | undefined {
	const node = findTreeNode(nodes, id);
	return node?.kind === 'node' ? node.node : undefined;
}

export function updateTreeNode(nodes: BlockNode[], id: string, updater: (n: BlockNode) => BlockNode): BlockNode[] {
	return nodes.map((node) => {
		if (nodeId(node) === id) return updater(node);
		const sub = childrenOf(node);
		if (!sub) return node;
		const updated = updateTreeNode(sub, id, updater);
		if (updated === sub) return node;
		return withUpdatedChildren(node, updated);
	});
}

export function updateBlock(nodes: BlockNode[], id: string, updater: (b: Block) => Block): BlockNode[] {
	return updateTreeNode(nodes, id, (n) => {
		if (n.kind !== 'block') return n;
		return { kind: 'block', block: updater(n.block) };
	});
}

export function updateNodeItem(nodes: BlockNode[], id: string, updater: (n: Node) => Node): BlockNode[] {
	return updateTreeNode(nodes, id, (n) => {
		if (n.kind !== 'node') return n;
		return { kind: 'node', node: updater(n.node) };
	});
}

export function collectAllIds(nodes: BlockNode[], out: string[] = []): string[] {
	for (const node of nodes) {
		out.push(nodeId(node));
		const sub = childrenOf(node);
		if (sub) collectAllIds(sub, out);
	}
	return out;
}

export function collectAllNames(nodes: BlockNode[], out: string[] = []): string[] {
	for (const node of nodes) {
		switch (node.kind) {
			case 'block': out.push(blockLabel(node.block)); break;
			case 'folder': out.push(node.folder.name); break;
			case 'node': out.push(node.node.name); break;
		}
		const sub = childrenOf(node);
		if (sub) collectAllNames(sub, out);
	}
	return out;
}

export function addNode(nodes: BlockNode[], node: BlockNode): BlockNode[] {
	return [...nodes, node];
}

// --- DnD tree operations ---

export function findAndRemove(nodes: BlockNode[], id: string): { remaining: BlockNode[]; removed: BlockNode | null } {
	const idx = nodes.findIndex((n) => nodeId(n) === id);
	if (idx >= 0) {
		return { remaining: [...nodes.slice(0, idx), ...nodes.slice(idx + 1)], removed: nodes[idx] };
	}
	for (let i = 0; i < nodes.length; i++) {
		const n = nodes[i];
		const sub = childrenOf(n);
		if (sub) {
			const result = findAndRemove(sub, id);
			if (result.removed) {
				const updated = [...nodes];
				updated[i] = withUpdatedChildren(n, result.remaining);
				return { remaining: updated, removed: result.removed };
			}
		}
	}
	return { remaining: nodes, removed: null };
}

export function removeTreeNode(nodes: BlockNode[], id: string): BlockNode[] {
	return findAndRemove(nodes, id).remaining;
}

export function insertBefore(nodes: BlockNode[], targetId: string, node: BlockNode): BlockNode[] | null {
	const idx = nodes.findIndex((n) => nodeId(n) === targetId);
	if (idx >= 0) {
		return [...nodes.slice(0, idx), node, ...nodes.slice(idx)];
	}
	for (let i = 0; i < nodes.length; i++) {
		const n = nodes[i];
		const sub = childrenOf(n);
		if (sub) {
			const result = insertBefore(sub, targetId, node);
			if (result) {
				const updated = [...nodes];
				updated[i] = withUpdatedChildren(n, result);
				return updated;
			}
		}
	}
	return null;
}

export function appendToFolder(nodes: BlockNode[], folderId: string, node: BlockNode): BlockNode[] | null {
	for (let i = 0; i < nodes.length; i++) {
		const n = nodes[i];
		const sub = childrenOf(n);
		if (sub && nodeId(n) === folderId) {
			const updated = [...nodes];
			updated[i] = withUpdatedChildren(n, [...sub, node]);
			return updated;
		}
		if (sub) {
			const result = appendToFolder(sub, folderId, node);
			if (result) {
				const updated = [...nodes];
				updated[i] = withUpdatedChildren(n, result);
				return updated;
			}
		}
	}
	return null;
}

/** Check if `descendantId` is a descendant of the node with `ancestorId`. */
function isDescendant(nodes: BlockNode[], ancestorId: string, descendantId: string): boolean {
	const ancestor = findTreeNode(nodes, ancestorId);
	if (!ancestor) return false;
	const sub = childrenOf(ancestor);
	if (!sub) return false;
	return !!findTreeNode(sub, descendantId);
}

export function moveNode(nodes: BlockNode[], sourceId: string, targetId: string): BlockNode[] {
	if (sourceId === targetId) return nodes;

	// Prevent circular: don't move a folder into its own descendant
	const actualTargetId = targetId.startsWith(DROP_PREFIX)
		? targetId.slice(DROP_PREFIX.length)
		: targetId;
	if (isDescendant(nodes, sourceId, actualTargetId)) return nodes;

	const { remaining, removed } = findAndRemove(nodes, sourceId);
	if (!removed) return nodes;

	if (targetId.startsWith(DROP_PREFIX)) {
		const folderId = targetId.slice(DROP_PREFIX.length);
		return appendToFolder(remaining, folderId, removed) ?? remaining;
	}

	return insertBefore(remaining, targetId, removed) ?? remaining;
}

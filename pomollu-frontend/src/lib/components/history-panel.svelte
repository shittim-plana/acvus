<script lang="ts">
	import type { TurnNode } from '$lib/engine.js';
	import { tick } from 'svelte';

	let {
		nodes,
		cursor,
		onGoto,
		disabled = false,
	}: {
		nodes: TurnNode[];
		cursor: string;
		onGoto: (id: string) => void;
		disabled?: boolean;
	} = $props();

	let childrenMap = $derived.by(() => {
		const map = new Map<string, string[]>();
		for (const node of nodes) {
			const key = node.parent ?? '__root__';
			const arr = map.get(key);
			if (arr) arr.push(node.uuid);
			else map.set(key, [node.uuid]);
		}
		return map;
	});

	let nodeMap = $derived.by(() => {
		const map = new Map<string, TurnNode>();
		for (const node of nodes) map.set(node.uuid, node);
		return map;
	});

	let ancestorSet = $derived.by(() => {
		const set = new Set<string>();
		let cur: string | null = cursor;
		while (cur) {
			set.add(cur);
			cur = nodeMap.get(cur)?.parent ?? null;
		}
		return set;
	});

	const BRANCH_COLORS = [
		'oklch(0.637 0.237 25.331)',  // red
		'oklch(0.606 0.220 292.717)', // purple
		'oklch(0.655 0.196 248.514)', // blue
		'oklch(0.648 0.18 163)',      // teal
		'oklch(0.705 0.213 47.604)',  // orange
		'oklch(0.723 0.219 149.579)', // green
		'oklch(0.682 0.177 320)',     // magenta
	];

	type LayoutItem = {
		uuid: string;
		col: number;
		depth: number;
		colorIdx: number;
		isAncestor: boolean;
		isCursor: boolean;
		parentUuid: string | null;
	};

	let layout = $derived.by(() => {
		const items: LayoutItem[] = [];
		const colOf = new Map<string, number>();
		const depthOf = new Map<string, number>();
		let maxCol = -1;
		let nextColor = 0;

		function walk(uuid: string, col: number, depth: number, colorIdx: number) {
			if (colOf.has(uuid)) return;
			colOf.set(uuid, col);
			depthOf.set(uuid, depth);
			if (col > maxCol) maxCol = col;

			const node = nodeMap.get(uuid);
			if (!node) return;

			items.push({
				uuid,
				col,
				depth,
				colorIdx,
				isAncestor: ancestorSet.has(uuid),
				isCursor: uuid === cursor,
				parentUuid: node.parent,
			});

			const children = childrenMap.get(uuid) ?? [];
			if (children.length === 0) return;
			const [first, ...rest] = children;
			walk(first, col, depth + 1, colorIdx);
			for (const id of rest) {
				maxCol++;
				walk(id, maxCol, depth + 1, nextColor++);
			}
		}

		const roots = nodes.filter((n) => n.parent == null);
		for (const root of roots) {
			if (maxCol >= 0) maxCol++;
			walk(root.uuid, Math.max(0, maxCol), 0, -1);
		}

		let maxDepth = 0;
		for (const d of depthOf.values()) {
			if (d > maxDepth) maxDepth = d;
		}

		return { items, colOf, depthOf, maxCol: Math.max(0, maxCol), maxDepth };
	});

	const CELL_W = 44;
	const ROW_H = 32;
	const NODE_W = 38;
	const NODE_H = 20;
	const ROUND = 4;

	function midX(col: number) { return col * CELL_W + CELL_W / 2; }
	function topEdge(depth: number) { return depth * ROW_H + (ROW_H - NODE_H) / 2; }
	function botEdge(depth: number) { return depth * ROW_H + (ROW_H + NODE_H) / 2; }

	function branchColor(idx: number): string {
		return idx < 0 ? 'var(--color-primary)' : BRANCH_COLORS[idx % BRANCH_COLORS.length];
	}

	function nodeStyle(item: LayoutItem): string {
		const c = branchColor(item.colorIdx);
		const x = item.col * CELL_W + (CELL_W - NODE_W) / 2;
		const y = item.depth * ROW_H + (ROW_H - NODE_H) / 2;
		let s = `left:${x}px;top:${y}px;width:${NODE_W}px;height:${NODE_H}px;border-radius:${ROUND}px;`;
		if (item.isCursor) {
			s += `background:${c};color:white;font-weight:600;`;
		} else if (item.isAncestor) {
			s += `background:color-mix(in oklch,${c} 15%,var(--color-background));border:1.5px solid ${c};color:${c};`;
		} else {
			s += `background:color-mix(in oklch,${c} 8%,var(--color-background));border:1px solid color-mix(in oklch,${c} 25%,var(--color-background));color:color-mix(in oklch,${c} 50%,var(--color-muted-foreground));`;
		}
		return s;
	}

	let scrollEl: HTMLDivElement;

	$effect(() => {
		void nodes.length;
		void cursor;
		tick().then(() => {
			if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
		});
	});
</script>

<div class="flex h-full flex-col bg-background">
	<div class="flex items-center gap-1.5 border-b px-3 py-2">
		<span class="text-xs font-medium">Tree</span>
		<span class="text-[10px] text-muted-foreground">{nodes.length} nodes</span>
	</div>

	<div class="flex-1 overflow-auto" bind:this={scrollEl}>
		{#if nodes.length === 0}
			<div class="px-2 py-4 text-center text-xs text-muted-foreground">No history yet.</div>
		{:else}
			{@const totalW = (layout.maxCol + 1) * CELL_W}
			{@const totalH = (layout.maxDepth + 1) * ROW_H}
			<div class="flex justify-center">
			<div class="relative" style="width:{totalW + 16}px;min-height:{totalH + 16}px;padding:8px;">
				<!-- Connection lines -->
				<svg
					class="absolute pointer-events-none z-0"
					style="left:8px;top:8px;"
					width={totalW}
					height={totalH}
				>
					{#each layout.items as item}
						{#if item.parentUuid != null}
							{@const pCol = layout.colOf.get(item.parentUuid)}
							{@const pDepth = layout.depthOf.get(item.parentUuid)}
							{#if pCol != null && pDepth != null}
								{@const x1 = midX(pCol)}
								{@const y1 = botEdge(pDepth)}
								{@const x2 = midX(item.col)}
								{@const y2 = topEdge(item.depth)}
								{@const c = branchColor(item.colorIdx)}
								{@const op = item.isAncestor ? 0.6 : 0.2}
								{#if x1 === x2}
									<line {x1} {y1} {x2} {y2} stroke={c} stroke-opacity={op} stroke-width="1.5" />
								{:else}
									{@const my = (y1 + y2) / 2}
									<path
										d="M {x1} {y1} C {x1} {my},{x2} {my},{x2} {y2}"
										fill="none"
										stroke={c}
										stroke-opacity={op}
										stroke-width="1.5"
									/>
								{/if}
							{/if}
						{/if}
					{/each}
				</svg>

				<!-- Nodes -->
				{#each layout.items as item (item.uuid)}
					<button
						class="absolute z-10 flex items-center justify-center transition-all
							{disabled ? 'opacity-50 cursor-not-allowed' : 'hover:brightness-110 hover:scale-105 active:scale-95'}"
						style={nodeStyle(item)}
						onclick={() => onGoto(item.uuid)}
						{disabled}
						title={item.uuid}
					>
						<span class="text-[9px] font-mono leading-none">
							{item.uuid.slice(0, 5)}
						</span>
					</button>
				{/each}
			</div>
			</div>
		{/if}
	</div>
</div>

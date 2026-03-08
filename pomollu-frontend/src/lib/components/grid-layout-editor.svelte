<script lang="ts">
	import type { GridLayout, DisplayRegion } from '$lib/types.js';
	import { GRID_HISTORY, gridStyle, cumulativeBoundaries } from '$lib/types.js';
	import { onDestroy } from 'svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Select from '$lib/components/ui/select';
	import { SplitSquareHorizontal, SplitSquareVertical } from 'lucide-svelte';

	let {
		layout,
		regions,
		onupdate
	}: {
		layout: GridLayout;
		regions: DisplayRegion[];
		onupdate: (layout: GridLayout) => void;
	} = $props();

	let containerEl = $state<HTMLElement>();
	let selectedCell = $state<{ row: number; col: number } | null>(null);

	let cols = $derived(layout.colSizes.length);
	let rows = $derived(layout.rowSizes.length);

	let selectedValue = $derived.by(() => {
		if (!selectedCell) return '';
		return layout.areas[selectedCell.row]?.[selectedCell.col] ?? '';
	});

	const hues = [200, 140, 30, 280, 340, 60];

	// Precomputed region lookup
	let regionMap = $derived(new Map(regions.map((r, i) => [r.id, { name: r.name || 'Untitled', hue: hues[i % hues.length] }])));

	let cellOptions = $derived([
		{ value: GRID_HISTORY, label: 'History' },
		{ value: '', label: 'Empty' },
		...regions.map((r) => ({ value: r.id, label: r.name || 'Untitled' }))
	]);

	function cellLabel(value: string): string {
		if (value === GRID_HISTORY) return 'History';
		if (value === '') return '';
		return regionMap.get(value)?.name ?? 'Untitled';
	}

	function cellColor(value: string): string {
		if (value === GRID_HISTORY) return 'var(--color-primary)';
		if (value === '') return 'transparent';
		const info = regionMap.get(value);
		if (!info) return 'transparent';
		return `hsl(${info.hue} 60% 55%)`;
	}

	let colBoundaries = $derived(cumulativeBoundaries(layout.colSizes));
	let rowBoundaries = $derived(cumulativeBoundaries(layout.rowSizes));

	// --- Editor placements (merges adjacent same-value cells, including empty) ---

	type EditorPlacement = {
		value: string;
		row: number; col: number;
		rowSpan: number; colSpan: number;
		colStart: number; colEnd: number;
		rowStart: number; rowEnd: number;
	};

	let editorPlacements = $derived.by(() => {
		const rowCount = layout.areas.length;
		const colCount = layout.colSizes.length;
		const visited = new Set<number>();
		const result: EditorPlacement[] = [];

		for (let r = 0; r < rowCount; r++) {
			for (let c = 0; c < colCount; c++) {
				if (visited.has(r * colCount + c)) continue;
				const value = layout.areas[r]?.[c] ?? '';

				let maxCol = c;
				while (maxCol + 1 < colCount && !visited.has(r * colCount + (maxCol + 1)) && layout.areas[r][maxCol + 1] === value) maxCol++;

				let maxRow = r;
				outer: while (maxRow + 1 < rowCount) {
					for (let cc = c; cc <= maxCol; cc++) {
						if (visited.has((maxRow + 1) * colCount + cc) || layout.areas[maxRow + 1]?.[cc] !== value) break outer;
					}
					maxRow++;
				}

				for (let rr = r; rr <= maxRow; rr++) {
					for (let cc = c; cc <= maxCol; cc++) {
						visited.add(rr * colCount + cc);
					}
				}

				result.push({
					value,
					row: r, col: c,
					rowSpan: maxRow - r + 1, colSpan: maxCol - c + 1,
					colStart: c + 1, colEnd: maxCol + 2,
					rowStart: r + 1, rowEnd: maxRow + 2,
				});
			}
		}
		return result;
	});

	function sumSizes(sizes: number[], from: number, to: number): number {
		let sum = 0;
		for (let i = from; i < to; i++) sum += sizes[i];
		return sum;
	}

	function selectCell(row: number, col: number) {
		if (selectedCell?.row === row && selectedCell?.col === col) {
			selectedCell = null;
		} else {
			selectedCell = { row, col };
		}
	}

	function findPlacement(row: number, col: number): EditorPlacement | undefined {
		return editorPlacements.find((p) => p.row === row && p.col === col);
	}

	function assignCell(value: string) {
		if (!selectedCell) return;
		const p = findPlacement(selectedCell.row, selectedCell.col);
		if (!p) return;
		const newAreas = layout.areas.map((row) => [...row]);
		for (let r = p.row; r < p.row + p.rowSpan; r++) {
			for (let c = p.col; c < p.col + p.colSpan; c++) {
				newAreas[r][c] = value;
			}
		}
		onupdate({ ...layout, areas: newAreas });
	}

	function splitVertical() {
		if (cols >= 6) return;
		const idx = selectedCell?.col ?? largestIndex(layout.colSizes);
		const selectedRow = selectedCell?.row;
		const half = layout.colSizes[idx] / 2;
		const newColSizes = [...layout.colSizes];
		newColSizes.splice(idx, 1, half, half);
		const newAreas = layout.areas.map((row, ri) => {
			const newRow = [...row];
			if (selectedRow != null && ri !== selectedRow) {
				// Not the selected row: duplicate value to maintain span
				newRow.splice(idx + 1, 0, row[idx]);
			} else {
				// Selected row (or no selection): insert empty
				newRow.splice(idx + 1, 0, '');
			}
			return newRow;
		});
		onupdate({ ...layout, colSizes: newColSizes, areas: newAreas });
		selectedCell = null;
	}

	function splitHorizontal() {
		if (rows >= 6) return;
		const idx = selectedCell?.row ?? largestIndex(layout.rowSizes);
		const selectedCol = selectedCell?.col;
		const half = layout.rowSizes[idx] / 2;
		const newRowSizes = [...layout.rowSizes];
		newRowSizes.splice(idx, 1, half, half);
		const newAreas = [...layout.areas];
		const newRow = layout.areas[idx].map((cell, ci) => {
			if (selectedCol != null && ci === selectedCol) return '';
			return selectedCol != null ? cell : '';
		});
		newAreas.splice(idx + 1, 0, newRow);
		onupdate({ ...layout, rowSizes: newRowSizes, areas: newAreas });
		selectedCell = null;
	}

	function largestIndex(sizes: number[]): number {
		let idx = 0;
		for (let i = 1; i < sizes.length; i++) {
			if (sizes[i] > sizes[idx]) idx = i;
		}
		return idx;
	}

	function removeDivider(axis: 'col' | 'row', index: number) {
		if (axis === 'col') {
			if (cols <= 1) return;
			const newColSizes = [...layout.colSizes];
			newColSizes[index - 1] += newColSizes[index];
			newColSizes.splice(index, 1);
			const newAreas = layout.areas.map((row) => {
				const newRow = [...row];
				newRow.splice(index, 1);
				return newRow;
			});
			onupdate({ ...layout, colSizes: newColSizes, areas: newAreas });
		} else {
			if (rows <= 1) return;
			const newRowSizes = [...layout.rowSizes];
			newRowSizes[index - 1] += newRowSizes[index];
			newRowSizes.splice(index, 1);
			const newAreas = [...layout.areas];
			newAreas.splice(index, 1);
			onupdate({ ...layout, rowSizes: newRowSizes, areas: newAreas });
		}
		selectedCell = null;
	}

	// --- Dragging dividers ---

	type DragState = {
		axis: 'col' | 'row';
		boundaryIndex: number;
		startPos: number;
		startBoundary: number;
		containerSize: number;
		minPct: number;
		maxPct: number;
	};

	let drag = $state<DragState | null>(null);

	function handleDragStart(axis: 'col' | 'row', boundaryIndex: number, e: MouseEvent) {
		e.preventDefault();
		if (!containerEl) return;
		const rect = containerEl.getBoundingClientRect();
		const boundaries = axis === 'col' ? colBoundaries : rowBoundaries;
		const containerSize = axis === 'col' ? rect.width : rect.height;

		const minPct = boundaries[boundaryIndex - 1] + 5;
		const maxPct = boundaries[boundaryIndex + 1] - 5;

		drag = {
			axis,
			boundaryIndex,
			startPos: axis === 'col' ? e.clientX : e.clientY,
			startBoundary: boundaries[boundaryIndex],
			containerSize,
			minPct,
			maxPct
		};
		window.addEventListener('mousemove', handleDragMove);
		window.addEventListener('mouseup', handleDragEnd);
	}

	function handleDragMove(e: MouseEvent) {
		if (!drag) return;
		const pos = drag.axis === 'col' ? e.clientX : e.clientY;
		const deltaPx = pos - drag.startPos;
		const deltaPct = (deltaPx / drag.containerSize) * 100;
		const newBoundary = Math.min(drag.maxPct, Math.max(drag.minPct, drag.startBoundary + deltaPct));

		const sizes = drag.axis === 'col' ? [...layout.colSizes] : [...layout.rowSizes];
		const bi = drag.boundaryIndex;
		const boundaries = drag.axis === 'col' ? [...colBoundaries] : [...rowBoundaries];
		boundaries[bi] = newBoundary;
		sizes[bi - 1] = boundaries[bi] - boundaries[bi - 1];
		sizes[bi] = boundaries[bi + 1] - boundaries[bi];

		if (drag.axis === 'col') {
			onupdate({ ...layout, colSizes: sizes });
		} else {
			onupdate({ ...layout, rowSizes: sizes });
		}
	}

	function handleDragEnd() {
		drag = null;
		window.removeEventListener('mousemove', handleDragMove);
		window.removeEventListener('mouseup', handleDragEnd);
	}

	onDestroy(() => {
		if (drag) handleDragEnd();
	});

	let computedGridStyle = $derived(gridStyle(layout));

	const aspectPresets = [
		{ value: '0', label: 'Auto' },
		{ value: '1', label: '1:1' },
		{ value: String(4 / 3), label: '4:3' },
		{ value: String(16 / 9), label: '16:9' },
		{ value: String(9 / 16), label: '9:16' },
		{ value: String(3 / 4), label: '3:4' },
	];

	let aspectLabel = $derived.by(() => {
		const a = layout.aspect ?? 0;
		if (a === 0) return 'Auto';
		const preset = aspectPresets.find((p) => Math.abs(Number(p.value) - a) < 0.01);
		return preset?.label ?? a.toFixed(2);
	});

	let canvasAspect = $derived((layout.aspect ?? 0) > 0 ? layout.aspect : 16 / 10);
</script>

<div class="space-y-2">
	<div class="flex items-center gap-1 flex-wrap">
		<Button variant="outline" size="sm" class="h-7 text-xs gap-1" disabled={cols >= 6} onclick={splitVertical}>
			<SplitSquareHorizontal class="h-3.5 w-3.5" /> Split V
		</Button>
		<Button variant="outline" size="sm" class="h-7 text-xs gap-1" disabled={rows >= 6} onclick={splitHorizontal}>
			<SplitSquareVertical class="h-3.5 w-3.5" /> Split H
		</Button>
		<div class="ml-auto">
			<Select.Root
				type="single"
				value={String(layout.aspect ?? 0)}
				onValueChange={(v) => { if (v !== undefined) onupdate({ ...layout, aspect: Number(v) }); }}
			>
				<Select.Trigger class="h-7 text-xs w-24">
					{aspectLabel}
				</Select.Trigger>
				<Select.Content>
					{#each aspectPresets as preset}
						<Select.Item value={preset.value}>{preset.label}</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
		</div>
	</div>

	<!-- Canvas -->
	<div class="canvas-wrapper">
		<div
			bind:this={containerEl}
			class="grid-canvas"
			style="{computedGridStyle} aspect-ratio: {canvasAspect};"
		>
		{#each editorPlacements as p}
			{@const isSelected = selectedCell?.row === p.row && selectedCell?.col === p.col}
			{@const w = sumSizes(layout.colSizes, p.col, p.col + p.colSpan)}
			{@const h = sumSizes(layout.rowSizes, p.row, p.row + p.rowSpan)}
			<button
				type="button"
				class="grid-cell"
				class:selected={isSelected}
				style="--cell-color: {cellColor(p.value)}; grid-column: {p.colStart} / {p.colEnd}; grid-row: {p.rowStart} / {p.rowEnd};"
				onclick={() => selectCell(p.row, p.col)}
			>
				{#if w > 12 && h > 15}
					<span class="cell-label">{cellLabel(p.value)}</span>
				{/if}
				<span class="cell-size">{Math.round(w)}×{Math.round(h)}</span>
			</button>
		{/each}

		<!-- Column dividers -->
		{#each { length: cols - 1 } as _, i}
			{@const leftPct = colBoundaries[i + 1]}
			<div
				class="divider divider-col"
				class:active={drag?.axis === 'col' && drag.boundaryIndex === i + 1}
				style="left: {leftPct}%;"
				onmousedown={(e) => handleDragStart('col', i + 1, e)}
				ondblclick={() => removeDivider('col', i + 1)}
				role="separator"
				aria-orientation="vertical"
				tabindex="-1"
				title="Drag to resize, double-click to remove"
			></div>
		{/each}

		<!-- Row dividers -->
		{#each { length: rows - 1 } as _, i}
			{@const topPct = rowBoundaries[i + 1]}
			<div
				class="divider divider-row"
				class:active={drag?.axis === 'row' && drag.boundaryIndex === i + 1}
				style="top: {topPct}%;"
				onmousedown={(e) => handleDragStart('row', i + 1, e)}
				ondblclick={() => removeDivider('row', i + 1)}
				role="separator"
				aria-orientation="horizontal"
				tabindex="-1"
				title="Drag to resize, double-click to remove"
			></div>
		{/each}
	</div>
	</div>

	<!-- Cell assignment -->
	{#if selectedCell}
		<div class="flex items-center gap-2">
			<span class="text-xs text-muted-foreground shrink-0">Assign:</span>
			<Select.Root
				type="single"
				value={selectedValue}
				onValueChange={(v) => { if (v !== undefined) assignCell(v); }}
			>
				<Select.Trigger class="flex-1 h-7 text-xs">
					{cellLabel(selectedValue) || 'Empty'}
				</Select.Trigger>
				<Select.Content>
					{#each cellOptions as opt}
						<Select.Item value={opt.value}>{opt.label}</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
		</div>
	{:else}
		<p class="text-xs text-muted-foreground">Click a cell to assign. Drag dividers to resize. Double-click a divider to remove it.</p>
	{/if}
</div>

<style>
	.canvas-wrapper {
		display: flex;
		align-items: center;
		justify-content: center;
		max-height: 20rem;
	}

	.grid-canvas {
		display: grid;
		position: relative;
		min-width: 16rem;
		max-width: 100%;
		max-height: 20rem;
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		overflow: hidden;
		user-select: none;
	}

	.grid-cell {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 0.125rem;
		border: 1px solid color-mix(in srgb, var(--color-border) 60%, transparent);
		background: color-mix(in srgb, var(--cell-color) 10%, transparent);
		cursor: pointer;
		transition: background 0.12s;
		padding: 0.125rem;
		margin: -0.5px;
		overflow: hidden;
	}

	.grid-cell:hover {
		background: color-mix(in srgb, var(--cell-color) 20%, transparent);
	}

	.grid-cell.selected {
		outline: 2px solid var(--color-primary);
		outline-offset: -2px;
		background: color-mix(in srgb, var(--cell-color) 22%, transparent);
	}

	.cell-label {
		font-size: 0.625rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--color-muted-foreground);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		max-width: 100%;
	}

	.cell-size {
		font-size: 0.5rem;
		font-family: monospace;
		color: color-mix(in srgb, var(--color-muted-foreground) 60%, transparent);
	}

	.divider {
		position: absolute;
		z-index: 10;
		transition: background 0.1s;
	}

	.divider-col {
		top: 0;
		bottom: 0;
		width: 6px;
		margin-left: -3px;
		cursor: col-resize;
	}

	.divider-row {
		left: 0;
		right: 0;
		height: 6px;
		margin-top: -3px;
		cursor: row-resize;
	}

	.divider:hover,
	.divider.active {
		background: color-mix(in srgb, var(--color-primary) 40%, transparent);
	}
</style>

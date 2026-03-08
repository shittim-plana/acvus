<script lang="ts">
	import type { Session } from '$lib/types.js';
	import { sessionStore } from '$lib/stores.svelte.js';
	import { ScrollArea } from '$lib/components/ui/scroll-area';

	let session = $derived(
		sessionStore.sessions.find((s) => s.id === sessionStore.activeSessionId)
	);

	let storageEntries = $derived.by(() => {
		if (!session?.storage) return [];
		if (typeof session.storage !== 'object' || session.storage === null) return [];
		return Object.entries(session.storage as Record<string, unknown>);
	});

	function formatValue(value: unknown, depth: number = 0): string {
		if (value === null || value === undefined) return 'null';
		if (typeof value === 'string') return value.length > 120 ? value.slice(0, 120) + '...' : value;
		if (typeof value === 'number' || typeof value === 'boolean') return String(value);
		if (Array.isArray(value)) return `List[${value.length}]`;
		if (typeof value === 'object') {
			const keys = Object.keys(value);
			if (keys.length === 0) return '{}';
			return `{${keys.join(', ')}}`;
		}
		return String(value);
	}

	function isExpandable(value: unknown): boolean {
		if (Array.isArray(value)) return value.length > 0;
		if (typeof value === 'object' && value !== null) return Object.keys(value).length > 0;
		return false;
	}

	let expanded = $state<Set<string>>(new Set());

	function toggleExpand(key: string) {
		if (expanded.has(key)) {
			expanded = new Set([...expanded].filter((k) => k !== key));
		} else {
			expanded = new Set([...expanded, key]);
		}
	}
</script>

<div class="flex h-full flex-col">
	<div class="shrink-0 border-b px-3 py-2">
		<div class="text-xs font-medium text-muted-foreground">SESSION STORAGE</div>
	</div>
	<ScrollArea class="flex-1">
		<div class="p-2">
			{#if storageEntries.length === 0}
				<div class="px-2 py-4 text-xs text-muted-foreground/60 text-center">
					No storage yet. Run a turn to see data.
				</div>
			{:else}
				<div class="storage-tree">
					{#each storageEntries as [key, value] (key)}
						{@const expandable = isExpandable(value)}
						<div class="storage-node">
							<!-- svelte-ignore a11y_click_events_have_key_events -->
							<!-- svelte-ignore a11y_no_static_element_interactions -->
							<div class="storage-row" class:expandable onclick={() => expandable && toggleExpand(key)}>
								{#if expandable}
									<span class="chevron" class:open={expanded.has(key)}>▸</span>
								{:else}
									<span class="chevron-space"></span>
								{/if}
								<span class="storage-key">{key}</span>
								{#if !expanded.has(key)}
									<span class="storage-value">{formatValue(value)}</span>
								{/if}
							</div>
							{#if expanded.has(key) && expandable}
								<div class="storage-children">
									{#if Array.isArray(value)}
										{#each value as item, i}
											<div class="storage-row leaf">
												<span class="storage-index">[{i}]</span>
												<span class="storage-value">{formatValue(item)}</span>
											</div>
										{/each}
									{:else if typeof value === 'object' && value !== null}
										{#each Object.entries(value) as [k, v]}
											<div class="storage-row leaf">
												<span class="storage-key">{k}</span>
												<span class="storage-value">{formatValue(v)}</span>
											</div>
										{/each}
									{/if}
								</div>
							{/if}
						</div>
					{/each}
				</div>
			{/if}
		</div>
	</ScrollArea>
</div>

<style>
	.storage-tree {
		display: flex;
		flex-direction: column;
		gap: 1px;
	}
	.storage-node {
		display: flex;
		flex-direction: column;
	}
	.storage-row {
		display: flex;
		align-items: baseline;
		gap: 0.375rem;
		padding: 0.25rem 0.375rem;
		border-radius: 4px;
		font-size: 0.6875rem;
		line-height: 1.4;
		font-family: var(--font-mono, ui-monospace, monospace);
		min-height: 1.5rem;
	}
	.storage-row.expandable {
		cursor: pointer;
	}
	.storage-row.expandable:hover {
		background: var(--color-accent);
	}
	.chevron {
		flex-shrink: 0;
		width: 0.625rem;
		font-size: 0.5rem;
		color: var(--color-muted-foreground);
		transition: transform 0.1s;
		display: inline-block;
	}
	.chevron.open {
		transform: rotate(90deg);
	}
	.chevron-space {
		width: 0.625rem;
		flex-shrink: 0;
	}
	.storage-key {
		color: hsl(158 60% 55%);
		flex-shrink: 0;
		white-space: nowrap;
	}
	.storage-key::after {
		content: ':';
		color: var(--color-muted-foreground);
	}
	.storage-index {
		color: hsl(96 40% 65%);
		flex-shrink: 0;
		white-space: nowrap;
	}
	.storage-value {
		color: var(--color-muted-foreground);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.storage-children {
		margin-left: 1rem;
		border-left: 1px solid var(--color-border);
		padding-left: 0.375rem;
	}
	.leaf {
		padding: 0.125rem 0.375rem;
	}
</style>

<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import * as Select from '$lib/components/ui/select';
	import { Plus, X, Folder, Trash2 } from 'lucide-svelte';
	import { assetEditorState } from '$lib/asset-state.svelte.js';
	import type { AssetKind } from '$lib/storage/asset-store.js';

	let showNewFolder = $state(false);
	let newFolderName = $state('');
	let newFolderKind = $state<AssetKind>('image');

	async function createFolder() {
		if (!newFolderName.trim()) return;
		await assetEditorState.createFolder(newFolderName.trim(), newFolderKind);
		showNewFolder = false;
		newFolderName = '';
		newFolderKind = 'image';
	}
</script>

<div class="flex h-full flex-col">
	<div class="flex items-center justify-between shrink-0 border-b px-3 py-2">
		<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Folders</span>
		<Button variant="ghost" size="icon-sm" class="text-muted-foreground" onclick={() => { showNewFolder = true; }} title="New folder">
			<Plus class="h-3.5 w-3.5" />
		</Button>
	</div>

	<ScrollArea class="flex-1">
		<div class="p-1.5 space-y-0.5">
			{#if showNewFolder}
				<div class="rounded-md border p-2 space-y-1.5 mb-1">
					<Input
						class="h-7 text-xs"
						placeholder="Folder name..."
						value={newFolderName}
						oninput={(e) => { newFolderName = e.currentTarget.value; }}
						onkeydown={(e) => { if (e.key === 'Enter') createFolder(); if (e.key === 'Escape') showNewFolder = false; }}
					/>
					<div class="flex items-center gap-1">
						<Select.Root
							type="single"
							value={newFolderKind}
							onValueChange={(v) => { if (v) newFolderKind = v as AssetKind; }}
						>
							<Select.Trigger class="flex-1 h-7 text-xs">
								{newFolderKind === 'image' ? 'Image' : 'Other'}
							</Select.Trigger>
							<Select.Content>
								<Select.Item value="image">Image</Select.Item>
								<Select.Item value="other">Other</Select.Item>
							</Select.Content>
						</Select.Root>
						<Button size="sm" class="h-7 text-xs" onclick={createFolder}>OK</Button>
						<Button variant="ghost" size="icon-sm" class="h-7 w-7" onclick={() => { showNewFolder = false; }}>
							<X class="h-3 w-3" />
						</Button>
					</div>
				</div>
			{/if}

			{#each Object.entries(assetEditorState.folders) as [folderName, kind] (folderName)}
				{@const count = (assetEditorState.folderFiles[folderName] ?? []).length}
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					role="button"
					tabindex="0"
					class="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors group cursor-pointer
						{assetEditorState.selectedFolder === folderName
						? 'bg-accent text-accent-foreground'
						: 'text-foreground hover:bg-muted/50'}"
					onclick={() => { assetEditorState.selectFolder(folderName); }}
					onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') assetEditorState.selectFolder(folderName); }}
				>
					<Folder class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
					<span class="flex-1 truncate text-xs">{folderName}</span>
					<span class="text-[0.6rem] px-1 rounded bg-muted text-muted-foreground">{kind}</span>
					<span class="text-[0.6rem] text-muted-foreground">{count}</span>
					<button
						class="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive shrink-0 p-0.5 rounded"
						onclick={(e) => { e.stopPropagation(); assetEditorState.deleteFolder(folderName); }}
						title="Delete folder"
					>
						<Trash2 class="h-3 w-3" />
					</button>
				</div>
			{/each}

			{#if Object.keys(assetEditorState.folders).length === 0 && !showNewFolder && !assetEditorState.loading}
				<div class="text-center py-8 text-muted-foreground">
					<Folder class="h-6 w-6 mx-auto mb-1.5 opacity-30" />
					<p class="text-[0.65rem]">No folders yet</p>
				</div>
			{/if}
		</div>
	</ScrollArea>
</div>

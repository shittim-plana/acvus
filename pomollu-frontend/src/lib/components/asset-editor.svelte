<script lang="ts">
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Button } from '$lib/components/ui/button';
	import { Upload, X, File, Folder } from 'lucide-svelte';
	import { onMount, onDestroy } from 'svelte';
	import { assetEditorState } from '$lib/asset-state.svelte.js';

	let { dbName, entityName }: { dbName: string; entityName: string } = $props();

	let fileInput: HTMLInputElement | undefined = $state();

	onMount(() => { assetEditorState.open(dbName); });
	onDestroy(() => { assetEditorState.revokeAll(); });

	function fileName(path: string): string {
		return path.split('/').pop() ?? path;
	}
</script>

<div class="flex h-full flex-col">
	{#if assetEditorState.loading}
		<div class="flex flex-1 items-center justify-center text-sm text-muted-foreground">Loading...</div>
	{:else if !assetEditorState.selectedFolder}
		<div class="flex flex-1 items-center justify-center text-muted-foreground">
			<div class="text-center">
				<Folder class="h-10 w-10 mx-auto mb-2 opacity-20" />
				<p class="text-sm">Select a folder from the sidebar</p>
			</div>
		</div>
	{:else}
		<div class="flex items-center justify-between shrink-0 border-b px-4 py-2">
			<div class="flex items-center gap-2">
				<span class="text-sm font-medium">{assetEditorState.selectedFolder}</span>
				<span class="text-[0.65rem] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground">{assetEditorState.selectedKind}</span>
				<span class="text-xs text-muted-foreground">{assetEditorState.selectedFiles.length} files</span>
			</div>
			<Button variant="outline" size="sm" onclick={() => fileInput?.click()}>
				<Upload class="h-3 w-3 mr-1" /> Upload
			</Button>
			<input
				bind:this={fileInput}
				type="file"
				multiple
				class="hidden"
				onchange={() => { assetEditorState.uploadFiles(fileInput?.files ?? null); if (fileInput) fileInput.value = ''; }}
			/>
		</div>

		<ScrollArea class="flex-1">
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<div
				class="p-4 min-h-full"
				ondrop={(e) => { e.preventDefault(); assetEditorState.uploadFiles(e.dataTransfer?.files ?? null); }}
				ondragover={(e) => e.preventDefault()}
			>
				{#if assetEditorState.selectedFiles.length === 0}
					<div
						role="button"
						tabindex="0"
						class="rounded-lg border-2 border-dashed p-12 text-center text-muted-foreground cursor-pointer hover:border-foreground/30 transition-colors"
						onclick={() => fileInput?.click()}
						onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') fileInput?.click(); }}
					>
						<Upload class="h-8 w-8 mx-auto mb-2 opacity-30" />
						<p class="text-sm">Drop files here or click to upload</p>
					</div>
				{:else if assetEditorState.selectedKind === 'image'}
					<div class="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-3">
						{#each assetEditorState.selectedFiles as path (path)}
							<div class="group relative rounded-lg border overflow-hidden bg-muted/30 hover:border-foreground/20 transition-colors">
								{#if assetEditorState.previewUrls[path]}
									<div class="aspect-square">
										<img
											src={assetEditorState.previewUrls[path]}
											alt={fileName(path)}
											class="h-full w-full object-cover"
										/>
									</div>
								{:else}
									<div class="aspect-square flex items-center justify-center">
										<File class="h-8 w-8 text-muted-foreground/30" />
									</div>
								{/if}
								<div class="px-2 py-1.5 border-t bg-background">
									<p class="text-[0.65rem] truncate text-muted-foreground" title={fileName(path)}>{fileName(path)}</p>
								</div>
								<button
									class="absolute top-1 right-1 rounded-full bg-background/80 p-1 opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive transition-opacity"
									onclick={() => assetEditorState.deleteFile(path)}
									title="Delete"
								>
									<X class="h-3 w-3" />
								</button>
							</div>
						{/each}
					</div>
				{:else}
					<div class="space-y-1">
						{#each assetEditorState.selectedFiles as path (path)}
							<div class="flex items-center gap-2 rounded-md px-3 py-2 hover:bg-muted/50 group">
								<File class="h-4 w-4 shrink-0 text-muted-foreground" />
								<span class="flex-1 truncate text-sm">{fileName(path)}</span>
								<button
									class="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive shrink-0 p-1 rounded"
									onclick={() => assetEditorState.deleteFile(path)}
									title="Delete"
								>
									<X class="h-3 w-3" />
								</button>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		</ScrollArea>
	{/if}
</div>

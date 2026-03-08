<script lang="ts">
	import type { ContextBlock } from '$lib/types.js';
	import { blockTypeRegistry } from '$lib/stores.svelte.js';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Textarea } from '$lib/components/ui/textarea';
	import { Label } from '$lib/components/ui/label';
	import * as Select from '$lib/components/ui/select';
	import { Switch } from '$lib/components/ui/switch';
	import { Separator } from '$lib/components/ui/separator';
	import { slide } from 'svelte/transition';
	import { Plus, X } from 'lucide-svelte';

	let {
		block,
		onupdate
	}: {
		block: ContextBlock;
		onupdate: (updater: (b: ContextBlock) => ContextBlock) => void;
	} = $props();

	let customType = $state('');
	let isCustomType = $derived(!blockTypeRegistry.types.includes(block.info.type));

	const predefinedInfoKeys = new Set(['type', 'tags', 'description']);
	let arbitraryInfoEntries = $derived(
		Object.entries(block.info).filter(([k]) => !predefinedInfoKeys.has(k))
	);

	function updateInfo(key: string, value: unknown) {
		onupdate((b) => ({ ...b, info: { ...b.info, [key]: value } }));
	}

	function removeInfoField(key: string) {
		onupdate((b) => {
			const { [key]: _, ...rest } = b.info;
			return { ...b, info: rest as ContextBlock['info'] };
		});
	}

	function setType(value: string) {
		if (value === '__custom__') return;
		updateInfo('type', value);
		blockTypeRegistry.add(value);
	}

	function toggleEnabled() {
		onupdate((b) => ({ ...b, enabled: !b.enabled }));
	}

	function updateTag(key: string, value: string) {
		onupdate((b) => ({ ...b, info: { ...b.info, tags: { ...b.info.tags, [key]: value } } }));
	}

	function removeTag(key: string) {
		onupdate((b) => {
			const tags = { ...b.info.tags };
			delete tags[key];
			return { ...b, info: { ...b.info, tags } };
		});
	}

	let newTagKey = $state('');
	let newTagValue = $state('');
	function addTag() {
		if (!newTagKey.trim()) return;
		updateTag(newTagKey.trim(), newTagValue.trim());
		newTagKey = '';
		newTagValue = '';
	}

	let newFieldKey = $state('');
	let newFieldValue = $state('');
	function addInfoField() {
		if (!newFieldKey.trim() || predefinedInfoKeys.has(newFieldKey.trim())) return;
		updateInfo(newFieldKey.trim(), newFieldValue.trim());
		newFieldKey = '';
		newFieldValue = '';
	}

	function updateContentPart(index: number, value: string) {
		onupdate((b) => {
			const content = [...b.content];
			content[index] = { ...content[index], value };
			return { ...b, content };
		});
	}

	function addContentPart(content_type: string) {
		onupdate((b) => ({ ...b, content: [...b.content, { id: crypto.randomUUID(), content_type, value: '' }] }));
	}

	function removeContentPart(index: number) {
		onupdate((b) => ({ ...b, content: b.content.filter((_, i) => i !== index) }));
	}
</script>

<div class="transition-opacity" class:opacity-50={!block.enabled}>
		<div class="space-y-4">
			<div class="flex items-end gap-2">
				<div class="flex-1 space-y-1">
					<Label>Type</Label>
					<Select.Root
						type="single"
						value={isCustomType ? '__custom__' : block.info.type}
						onValueChange={(v) => { if (v && v !== '__custom__') setType(v); else if (v === '__custom__') updateInfo('type', ''); }}
					>
						<Select.Trigger class="w-full">{block.info.type}</Select.Trigger>
						<Select.Content>
							{#each blockTypeRegistry.types as t}
								<Select.Item value={t}>{t}</Select.Item>
							{/each}
							<Select.Item value="__custom__">custom...</Select.Item>
						</Select.Content>
					</Select.Root>
					{#if isCustomType}
						<Input
							class="h-7 text-xs"
							placeholder="Custom type name..."
							bind:value={customType}
							onkeydown={(e) => { if (e.key === ' ') e.preventDefault(); if (e.key === 'Enter') setType(customType); }}
							oninput={(e) => { customType = e.currentTarget.value.replace(/[^a-zA-Z0-9_-]/g, ''); }}
							onblur={() => { if (customType.trim()) setType(customType.trim()); }}
						/>
					{/if}
				</div>
				<div class="flex items-center gap-2 pb-7">
					<Switch checked={block.enabled} onCheckedChange={toggleEnabled} />
				</div>
			</div>

			<div class="space-y-1">
				<Label>Description</Label>
				<Input value={block.info.description} oninput={(e) => updateInfo('description', e.currentTarget.value)} placeholder="Short description..." />
			</div>

			<div class="space-y-1">
				<Label>Priority</Label>
				<Input type="number" value={String(block.priority)} oninput={(e) => onupdate((b) => ({ ...b, priority: Number(e.currentTarget.value) || 0 }))} />
			</div>

			<div class="space-y-2">
				<Label>Tags</Label>
				{#each Object.entries(block.info.tags) as [key, value]}
					<div class="flex items-center gap-1" transition:slide={{ duration: 150 }}>
						<span class="w-20 truncate text-xs text-muted-foreground">{key}</span>
						<Input class="h-8 flex-1 text-xs" {value} oninput={(e) => updateTag(key, e.currentTarget.value)} />
						<Button variant="ghost" size="icon-sm" class="h-8 w-8 text-muted-foreground hover:text-destructive" onclick={() => removeTag(key)}>
							<X class="h-3 w-3" />
						</Button>
					</div>
				{/each}
				<div class="flex items-center gap-1">
					<Input class="h-8 w-20 text-xs" placeholder="key" bind:value={newTagKey} onkeydown={(e) => { if (e.key === 'Enter') addTag(); }} />
					<Input class="h-8 flex-1 text-xs" placeholder="value" bind:value={newTagValue} onkeydown={(e) => { if (e.key === 'Enter') addTag(); }} />
					<Button variant="outline" size="icon-sm" class="h-8 w-8 text-primary shadow-sm" onclick={addTag}>
						<Plus class="h-4 w-4" />
					</Button>
				</div>
			</div>

			<div class="space-y-2">
				<Label>Custom Fields</Label>
				{#each arbitraryInfoEntries as [key, value]}
					<div class="flex items-center gap-1" transition:slide={{ duration: 150 }}>
						<span class="w-20 truncate text-xs text-muted-foreground">{key}</span>
						<Input class="h-8 flex-1 text-xs" value={String(value)} oninput={(e) => updateInfo(key, e.currentTarget.value)} />
						<Button variant="ghost" size="icon-sm" class="h-8 w-8 text-muted-foreground hover:text-destructive" onclick={() => removeInfoField(key)}>
							<X class="h-3 w-3" />
						</Button>
					</div>
				{/each}
				<div class="flex items-center gap-1">
					<Input class="h-8 w-20 text-xs" placeholder="key" bind:value={newFieldKey} onkeydown={(e) => { if (e.key === 'Enter') addInfoField(); }} />
					<Input class="h-8 flex-1 text-xs" placeholder="value" bind:value={newFieldValue} onkeydown={(e) => { if (e.key === 'Enter') addInfoField(); }} />
					<Button variant="outline" size="icon-sm" class="h-8 w-8 text-primary shadow-sm" onclick={addInfoField}>
						<Plus class="h-4 w-4" />
					</Button>
				</div>
			</div>

			<Separator />

			<div class="space-y-2">
				<Label>Content</Label>
				{#each block.content as part, i (part.id)}
					<div class="space-y-1" transition:slide={{ duration: 200 }}>
						<div class="flex items-center justify-between">
							<span class="text-xs font-medium text-muted-foreground bg-secondary/50 px-2 py-0.5 rounded-sm">{part.content_type}</span>
							{#if block.content.length > 1}
								<Button variant="ghost" size="icon-sm" class="h-6 w-6 text-muted-foreground hover:text-destructive" onclick={() => removeContentPart(i)}>
									<X class="h-3 w-3" />
								</Button>
							{/if}
						</div>
						{#if part.content_type === 'text'}
							<Textarea class="resize-none shadow-sm" value={part.value} oninput={(e) => updateContentPart(i, e.currentTarget.value)} placeholder="Content..." rows={4} />
						{:else}
							<div class="rounded border border-dashed py-4 text-center text-xs text-muted-foreground bg-muted/20">
								{part.content_type} — upload not yet implemented
							</div>
						{/if}
					</div>
				{/each}
				<div class="flex gap-2 pt-2">
					<Button variant="outline" size="sm" class="flex-1 border-dashed text-muted-foreground hover:text-primary hover:border-primary transition-colors hover:bg-transparent" onclick={() => addContentPart('text')}>
						<Plus class="h-3 w-3 mr-1" /> Text Array
					</Button>
					<Button variant="outline" size="sm" class="flex-1 border-dashed text-muted-foreground hover:text-primary hover:border-primary transition-colors hover:bg-transparent" onclick={() => addContentPart('image/png')}>
						<Plus class="h-3 w-3 mr-1" /> Image Array
					</Button>
				</div>
			</div>
		</div>
</div>

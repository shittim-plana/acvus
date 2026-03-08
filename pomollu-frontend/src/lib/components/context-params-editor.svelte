<script lang="ts">
	import type { ContextParam } from '$lib/types.js';
	import { Input } from '$lib/components/ui/input';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import { slide } from 'svelte/transition';
	import AcvusEngineField from './acvus-engine-field.svelte';

	let {
		params,
		onupdate,
		onTypeChange,
		contextTypes = {},
	}: {
		params: ContextParam[];
		onupdate: (params: ContextParam[]) => void;
		onTypeChange: (name: string, type: string) => void;
		contextTypes?: Record<string, string>;
	} = $props();

	function setResolution(index: number, kind: 'static' | 'dynamic' | 'unresolved') {
		const updated = params.map((p, i) => {
			if (i !== index) return p;
			if (kind === 'static') return { ...p, resolution: { kind: 'static' as const, value: '' } };
			if (kind === 'dynamic') return { ...p, resolution: { kind: 'dynamic' as const } };
			return { ...p, resolution: { kind: 'unresolved' as const } };
		});
		onupdate(updated);
	}

	function setStaticValue(index: number, value: string) {
		const updated = params.map((p, i) => {
			if (i !== index) return p;
			return { ...p, resolution: { kind: 'static' as const, value } };
		});
		onupdate(updated);
	}

	function setUserType(index: number, type: string) {
		const updated = params.map((p, i) => {
			if (i !== index) return p;
			return { ...p, userType: type || undefined };
		});
		onupdate(updated);
		const param = params[index];
		if (param && type) {
			onTypeChange(param.name, type);
		}
	}

	function resolvedTailType(param: ContextParam): string | undefined {
		return param.userType || (param.inferredType !== '?' ? param.inferredType : undefined);
	}
</script>

{#if params.length === 0}
	<p class="text-xs text-muted-foreground italic">No unresolved parameters.</p>
{:else}
	<div class="space-y-2">
		{#each params as param, i (param.name)}
			<div class="rounded-md border p-3 space-y-2" transition:slide={{ duration: 150 }}>
				<div class="flex items-center gap-2">
					<code class="text-xs font-semibold text-foreground">@{param.name}</code>
					{#if param.inferredType !== '?'}
						<Badge variant="secondary" class="text-[0.625rem]">{param.inferredType}</Badge>
					{:else if param.userType}
						<Badge variant="outline" class="text-[0.625rem]">{param.userType}</Badge>
					{:else}
						<Badge variant="destructive" class="text-[0.625rem]">?</Badge>
					{/if}
					<div class="flex-1"></div>
					<div class="flex gap-1">
						<Button
							variant={param.resolution.kind === 'static' ? 'default' : 'outline'}
							size="sm"
							class="h-6 text-[0.625rem] px-2"
							onclick={() => setResolution(i, 'static')}
						>Static</Button>
						<Button
							variant={param.resolution.kind === 'dynamic' ? 'default' : 'outline'}
							size="sm"
							class="h-6 text-[0.625rem] px-2"
							onclick={() => setResolution(i, 'dynamic')}
						>Dynamic</Button>
					</div>
				</div>

				{#if param.inferredType === '?' && !param.userType}
					<div class="space-y-1">
						<span class="text-[0.625rem] text-muted-foreground">Type hint</span>
						<Input
							class="text-xs h-7"
							placeholder="e.g. String, Int, List<String>..."
							value={param.userType ?? ''}
							oninput={(e) => setUserType(i, e.currentTarget.value)}
						/>
					</div>
				{/if}

				{#if param.resolution.kind === 'static'}
					<AcvusEngineField
						mode="script"
						value={param.resolution.value}
						oninput={(v) => setStaticValue(i, v)}
						placeholder="static value expression..."
						{contextTypes}
						expectedTailType={resolvedTailType(param)}
					/>
					{#if !param.resolution.value.trim()}
						<p class="text-[0.625rem] text-destructive">Static value is required.</p>
					{/if}
				{:else if param.resolution.kind === 'dynamic'}
					<p class="text-[0.625rem] text-muted-foreground italic">Provided at each turn input.</p>
				{/if}
			</div>
		{/each}
	</div>
{/if}

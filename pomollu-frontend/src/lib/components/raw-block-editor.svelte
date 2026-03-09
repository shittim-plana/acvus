<script lang="ts">
	import type { RawBlock } from '$lib/types.js';
	import AcvusEngineField from './acvus-engine-field.svelte';

	let {
		block,
		onupdate,
		contextTypes = {},
		analysisErrors = []
	}: {
		block: RawBlock;
		onupdate: (updater: (b: RawBlock) => RawBlock) => void;
		contextTypes?: Record<string, import('$lib/type-parser.js').TypeDesc>;
		analysisErrors?: string[];
	} = $props();
</script>

<div class="raw-editor">
	<AcvusEngineField
		mode={block.mode}
		value={block.text}
		oninput={(v) => onupdate((b) => ({ ...b, text: v }))}
		placeholder="Enter text..."
		unlimited
		{contextTypes}
		{analysisErrors}
	/>
</div>

<style>
	.raw-editor {
		display: flex;
		flex-direction: column;
		flex: 1;
		min-height: 0;
	}
</style>

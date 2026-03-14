<script lang="ts">
	import type { Bot } from '$lib/types.js';
	import { botStore, promptStore, profileStore, uiState, generateName } from '$lib/stores.svelte.js';
	import { Button } from '$lib/components/ui/button';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { longpress } from '$lib/actions/longpress.js';
	import { importEntityZip, validateBot } from '$lib/io.js';
	import { confirmDelete } from '$lib/confirm-dialog.svelte.js';

	async function importBot() {
		try {
			const { entity } = await importEntityZip();
			if (!validateBot(entity)) { alert('Invalid bot file'); return; }
			botStore.import(entity);
			uiState.selectBot(entity.id);
		} catch (e) {
			if ((e as Error).message !== 'no file selected') alert(`Import failed: ${e}`);
		}
	}

	function addBot() {
		let prompt = promptStore.prompts[0];
		if (!prompt) {
			prompt = promptStore.create('Default Prompt');
		}
		let profile = profileStore.profiles[0];
		if (!profile) {
			profile = profileStore.create('Default Profile');
		}
		const name = generateName('New Bot', botStore.bots.map(b => b.name));
		const bot = botStore.create(name, prompt.id, profile.id);
		uiState.selectBot(bot.id);
		uiState.createSession(bot.id, 'Chat 1');
	}
</script>

<div class="flex h-full w-16 flex-col border-r bg-background">
	<ScrollArea class="flex-1">
		<div class="flex flex-col items-center gap-1.5 p-2 pt-3">
			{#each botStore.bots as bot (bot.id)}
				<div class="bot-item" use:longpress>
					<button
						class="flex h-10 w-10 items-center justify-center rounded-lg text-xs font-medium transition-colors
							{uiState.selectedBotId === bot.id
							? 'bg-primary text-primary-foreground'
							: 'bg-secondary text-secondary-foreground hover:bg-accent'}"
						onclick={() => uiState.selectBot(bot.id)}
						title={bot.name}
					>
						{bot.name.slice(0, 2).toUpperCase()}
					</button>
					<button
						class="bot-delete"
						onclick={async (e) => { e.stopPropagation(); if (await confirmDelete('Delete this bot? This will also delete all its sessions.')) uiState.removeBot(bot.id); }}
						title="Delete bot"
					>&times;</button>
				</div>
			{/each}
			<Button variant="ghost" size="icon" onclick={addBot} title="New bot">+</Button>
			<Button variant="ghost" size="icon" onclick={importBot} title="Import bot">
				<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
			</Button>
		</div>
	</ScrollArea>
	<div class="flex flex-col items-center gap-1.5 border-t p-2">
		<Button
			variant={uiState.isSettings ? 'secondary' : 'ghost'}
			size="icon"
			onclick={() => uiState.selectSettings()}
			title="Settings"
		>
			<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
		</Button>
	</div>
</div>

<style>
	.bot-item {
		position: relative;
	}
	.bot-delete {
		position: absolute;
		right: -4px;
		top: -4px;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 16px;
		height: 16px;
		border-radius: 9999px;
		border: none;
		background: var(--color-destructive);
		color: var(--color-destructive-foreground);
		font-size: 10px;
		line-height: 1;
		cursor: pointer;
		opacity: 0;
		pointer-events: none;
		transition: opacity 0.15s;
	}
	@media (hover: hover) {
		.bot-item:hover > .bot-delete {
			opacity: 1;
			pointer-events: auto;
		}
	}
	:global([data-long-pressed] > .bot-delete) {
		opacity: 1;
		pointer-events: auto;
	}
</style>

<script lang="ts">
	import * as Resizable from '$lib/components/ui/resizable';
	import BotSidebar from '$lib/components/bot-sidebar.svelte';
	import SessionSidebar from '$lib/components/session-sidebar.svelte';
	import SettingsSidebar from '$lib/components/settings-sidebar.svelte';
	import ContextSidebar from '$lib/components/context-sidebar.svelte';
	import ChatPanel from '$lib/components/chat-panel.svelte';
	import BotSettings from '$lib/components/bot-settings.svelte';
	import PromptSettings from '$lib/components/prompt-settings.svelte';
	import ProfileSettings from '$lib/components/profile-settings.svelte';
	import ProviderSettings from '$lib/components/provider-settings.svelte';
	import NodeSettings from '$lib/components/node-settings.svelte';
	import BlockEditorPage from '$lib/components/block-editor-page.svelte';
	import TabBar from '$lib/components/tab-bar.svelte';
	import SyncIndicator from '$lib/components/sync-indicator.svelte';
	import { sessionStore, botStore, promptStore, profileStore, providerStore, uiState } from '$lib/stores.svelte.js';
	import type { BlockOwner } from '$lib/stores.svelte.js';
	import { initPersistence } from '$lib/persistence.svelte.js';
	import { IndexedDBBackend } from '$lib/storage/indexeddb.js';
	import {
		buildInjectedTypes, computeExternalContextEnv, collectUnresolvedParams,
		collectScriptsFromBindings, collectScriptsFromTree, collectNodeNames,
	} from '$lib/param-resolver.js';
	import type { ContextEnvResult } from '$lib/param-resolver.js';
	import type { BlockNode } from '$lib/types.js';
	import { onMount } from 'svelte';

	let activeTab = $derived(uiState.activeTab);

	// --- Owner env: inline discovery + orchestration typecheck ---

	const EMPTY_ENV: ContextEnvResult = { contextTypes: {}, nodeLocals: {}, nodeErrors: {} };
	const ENV_DEBOUNCE_MS = 300;

	let ownerEnv = $state<ContextEnvResult>(EMPTY_ENV);
	let envTimer: ReturnType<typeof setTimeout> | null = null;
	let lastOwnerKey: string | null = null;

	/** Full env computation with inline type discovery. No store side effects. */
	function computeFullEnv(owner: BlockOwner): ContextEnvResult {
		const getApi = (pid: string) => providerStore.get(pid)?.api ?? 'openai';

		switch (owner.kind) {
			case 'prompt': {
				const prompt = promptStore.get(owner.promptId);
				if (!prompt) return EMPTY_ENV;
				const injected = buildInjectedTypes(prompt.contextParams);
				const scripts = [
					...collectScriptsFromBindings(prompt.contextBindings),
					...collectScriptsFromTree(prompt.children),
				];
				const discovered = collectUnresolvedParams({
					scripts,
					nodeNames: collectNodeNames(prompt.children),
					providedKeys: new Set(prompt.contextBindings.map((b) => b.name).filter((n) => n)),
					contextTypes: injected,
				});
				for (const k of discovered) {
					if (k.type !== '?' && !(k.name in injected)) injected[k.name] = k.type;
				}
				return computeExternalContextEnv(prompt.children, injected, getApi);
			}
			case 'profile': {
				const profile = profileStore.get(owner.profileId);
				if (!profile) return EMPTY_ENV;
				const injected = buildInjectedTypes(profile.contextParams);
				const discovered = collectUnresolvedParams({
					scripts: collectScriptsFromTree(profile.children),
					nodeNames: collectNodeNames(profile.children),
					providedKeys: new Set(),
					contextTypes: injected,
				});
				for (const k of discovered) {
					if (k.type !== '?' && !(k.name in injected)) injected[k.name] = k.type;
				}
				return computeExternalContextEnv(profile.children, injected, getApi);
			}
			case 'bot': {
				const bot = botStore.get(owner.botId);
				if (!bot) return EMPTY_ENV;
				const prompt = promptStore.get(bot.promptId);
				const profile = profileStore.get(bot.profileId);
				const allParams = [
					...(prompt?.contextParams ?? []),
					...(profile?.contextParams ?? []),
					...bot.contextParams,
				];
				const injected = buildInjectedTypes(allParams);

				const providedKeys = new Set<string>();
				if (prompt) {
					for (const b of prompt.contextBindings) if (b.name) providedKeys.add(b.name);
					for (const p of prompt.contextParams) providedKeys.add(p.name);
				}
				if (profile) {
					for (const p of profile.contextParams) providedKeys.add(p.name);
				}

				const nodeNames = new Set<string>();
				if (prompt) for (const n of collectNodeNames(prompt.children)) nodeNames.add(n);
				if (profile) for (const n of collectNodeNames(profile.children)) nodeNames.add(n);
				for (const n of collectNodeNames(bot.children)) nodeNames.add(n);

				const scripts = [
					...(prompt ? collectScriptsFromBindings(prompt.contextBindings) : []),
					...(prompt ? collectScriptsFromTree(prompt.children) : []),
					...(profile ? collectScriptsFromTree(profile.children) : []),
					...collectScriptsFromTree(bot.children),
				];

				const allChildren: BlockNode[] = [
					...(prompt?.children ?? []),
					...(profile?.children ?? []),
					...bot.children,
				];

				const discovered = collectUnresolvedParams({
					scripts, nodeNames, providedKeys, contextTypes: injected,
				});
				for (const k of discovered) {
					if (k.type !== '?' && !(k.name in injected)) injected[k.name] = k.type;
				}
				return computeExternalContextEnv(allChildren, injected, getApi);
			}
		}
	}

	// Trigger key: captures all data that should cause a recomputation
	let envTriggerKey = $derived.by(() => {
		if (!activeTab) return null;
		if (activeTab.kind !== 'block' && activeTab.kind !== 'node') return null;
		const owner = activeTab.owner;
		switch (owner.kind) {
			case 'prompt': {
				const p = promptStore.get(owner.promptId);
				return JSON.stringify(['prompt', owner.promptId, p?.contextBindings, p?.children, p?.contextParams]);
			}
			case 'profile': {
				const p = profileStore.get(owner.profileId);
				return JSON.stringify(['profile', owner.profileId, p?.children, p?.contextParams]);
			}
			case 'bot': {
				const bot = botStore.get(owner.botId);
				const p = promptStore.get(bot?.promptId ?? '');
				const pr = profileStore.get(bot?.profileId ?? '');
				return JSON.stringify(['bot', owner.botId,
					bot?.children, bot?.display, bot?.regions, bot?.contextParams,
					p?.contextBindings, p?.children, p?.contextParams,
					pr?.children, pr?.contextParams,
				]);
			}
		}
	});

	// First load (tab switch) = immediate. Content changes = debounced. Both compute from scratch.
	$effect(() => {
		const key = envTriggerKey;

		if (key === null) {
			ownerEnv = EMPTY_ENV;
			lastOwnerKey = null;
			return;
		}
		if (!activeTab || (activeTab.kind !== 'block' && activeTab.kind !== 'node')) return;
		const owner = activeTab.owner;
		const ownerKey = JSON.stringify(owner);

		if (ownerKey !== lastOwnerKey) {
			lastOwnerKey = ownerKey;
			if (envTimer) clearTimeout(envTimer);
			ownerEnv = computeFullEnv(owner);
		} else {
			if (envTimer) clearTimeout(envTimer);
			envTimer = setTimeout(() => {
				envTimer = null;
				ownerEnv = computeFullEnv(owner);
			}, ENV_DEBOUNCE_MS);
		}

		return () => { if (envTimer) clearTimeout(envTimer); };
	});

	let cleanup: (() => void) | null = null;

	onMount(() => {
		IndexedDBBackend.open().then(async (backend) => {
			cleanup = await initPersistence(backend);
		});
		return () => cleanup?.();
	});

	$effect(() => {
		if (activeTab?.kind === 'chat') {
			sessionStore.activeSessionId = activeTab.sessionId;
		}
	});
</script>

<SyncIndicator />
<div class="flex h-screen bg-background text-foreground">
	<BotSidebar />

	{#if uiState.isSettings}
		<SettingsSidebar />
	{:else if uiState.selectedBotId}
		<SessionSidebar />
	{/if}

	<Resizable.PaneGroup direction="horizontal" class="flex-1">
		<Resizable.Pane defaultSize={75} minSize={40}>
			<div class="flex h-full flex-col">
				{#if uiState.tabs.length > 0}
					<TabBar />
				{/if}
				<div class="flex-1 overflow-hidden">
					{#if activeTab?.kind === 'block'}
						<BlockEditorPage blockId={activeTab.blockId} owner={activeTab.owner} contextTypes={ownerEnv.contextTypes} />
					{:else if activeTab?.kind === 'prompt'}
						<PromptSettings promptId={activeTab.promptId} />
					{:else if activeTab?.kind === 'profile'}
						<ProfileSettings profileId={activeTab.profileId} />
					{:else if activeTab?.kind === 'provider'}
						<ProviderSettings providerId={activeTab.providerId} />
					{:else if activeTab?.kind === 'node'}
						<NodeSettings nodeId={activeTab.nodeId} owner={activeTab.owner} contextTypes={ownerEnv.contextTypes} nodeLocals={ownerEnv.nodeLocals} nodeErrors={ownerEnv.nodeErrors} />
					{:else if activeTab?.kind === 'bot-settings'}
						<BotSettings botId={activeTab.botId} />
					{:else if activeTab?.kind === 'chat'}
						{@const chatSession = sessionStore.sessions.find((s) => s.id === activeTab.sessionId)}
						{@const chatBot = chatSession ? botStore.get(chatSession.botId) : undefined}
						{#if chatSession && chatBot}
							<ChatPanel session={chatSession} bot={chatBot} />
						{:else}
							<div class="flex h-full items-center justify-center text-sm text-muted-foreground">Session or bot not found.</div>
						{/if}
					{:else}
						<div class="flex h-full flex-col items-center justify-center gap-3 text-muted-foreground">
							<h1 class="text-2xl font-semibold text-foreground">Welcome to Pomollu!</h1>
							<a href="https://github.com/ArtBlnd/acvus" target="_blank" rel="noopener noreferrer" class="text-sm underline hover:text-foreground transition-colors">Source</a>
						</div>
					{/if}
				</div>
			</div>
		</Resizable.Pane>

		<Resizable.Handle withHandle />
		<Resizable.Pane defaultSize={25} minSize={10} maxSize={35}>
			<ContextSidebar />
		</Resizable.Pane>
	</Resizable.PaneGroup>
</div>

import type { EntityRef } from './entity-versions.svelte.js';
import type { Bot, Prompt, Profile } from './types.js';
import type { BlockOwner } from './stores.svelte.js';
import { collectNodes } from './block-tree.js';
import { botStore, promptStore, profileStore } from './stores.svelte.js';

export function collectPromptDeps(prompt: Prompt): EntityRef[] {
	const refs: EntityRef[] = [{ kind: 'prompt', id: prompt.id }];
	for (const node of collectNodes(prompt.children)) {
		if (node.kind === 'llm' && node.providerId)
			refs.push({ kind: 'provider', id: node.providerId });
	}
	return dedupe(refs);
}

export function collectProfileDeps(profile: Profile): EntityRef[] {
	const refs: EntityRef[] = [{ kind: 'profile', id: profile.id }];
	for (const node of collectNodes(profile.children)) {
		if (node.kind === 'llm' && node.providerId)
			refs.push({ kind: 'provider', id: node.providerId });
	}
	return dedupe(refs);
}

export function collectBotDeps(bot: Bot, prompt: Prompt, profile: Profile): EntityRef[] {
	const refs: EntityRef[] = [
		{ kind: 'bot', id: bot.id },
		...collectPromptDeps(prompt),
		...collectProfileDeps(profile),
	];
	for (const node of collectNodes(bot.children)) {
		if (node.kind === 'llm' && node.providerId)
			refs.push({ kind: 'provider', id: node.providerId });
	}
	return dedupe(refs);
}

export function collectOwnerDeps(owner: BlockOwner): EntityRef[] {
	switch (owner.kind) {
		case 'prompt': {
			const p = promptStore.get(owner.promptId);
			return p ? collectPromptDeps(p) : [];
		}
		case 'profile': {
			const p = profileStore.get(owner.profileId);
			return p ? collectProfileDeps(p) : [];
		}
		case 'bot': {
			const b = botStore.get(owner.botId);
			if (!b) return [];
			const prompt = promptStore.get(b.promptId);
			const profile = profileStore.get(b.profileId);
			if (!prompt || !profile) return [{ kind: 'bot', id: b.id }];
			return collectBotDeps(b, prompt, profile);
		}
	}
}

function dedupe(refs: EntityRef[]): EntityRef[] {
	const seen = new Set<string>();
	return refs.filter(r => {
		const key = `${r.kind}:${r.id}`;
		if (seen.has(key)) return false;
		seen.add(key);
		return true;
	});
}

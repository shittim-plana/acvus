import type { Prompt, Profile, Bot } from '$lib/types.js';
import { createId } from '$lib/stores.svelte.js';

export function downloadJson(data: unknown, filename: string) {
	const json = JSON.stringify(data, null, 2);
	const blob = new Blob([json], { type: 'application/json' });
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	a.click();
	URL.revokeObjectURL(url);
}

export function pickJsonFile(): Promise<unknown> {
	return new Promise((resolve, reject) => {
		const input = document.createElement('input');
		input.type = 'file';
		input.accept = '.json';
		input.onchange = () => {
			const file = input.files?.[0];
			if (!file) { reject(new Error('no file selected')); return; }
			const reader = new FileReader();
			reader.onload = () => {
				try {
					resolve(JSON.parse(reader.result as string));
				} catch (e) {
					reject(e);
				}
			};
			reader.readAsText(file);
		};
		input.click();
	});
}

/** Replace top-level `id` with a fresh one, return the new entity. */
export function withNewId<T extends { id: string }>(data: T): T {
	return { ...data, id: createId() };
}

function isObj(v: unknown): v is Record<string, unknown> {
	return typeof v === 'object' && v !== null && !Array.isArray(v);
}

export function validatePrompt(data: unknown): data is Prompt {
	if (!isObj(data)) return false;
	return typeof data.name === 'string'
		&& Array.isArray(data.children)
		&& Array.isArray(data.contextBindings);
}

export function validateProfile(data: unknown): data is Profile {
	if (!isObj(data)) return false;
	return typeof data.name === 'string'
		&& Array.isArray(data.children);
}

export function validateBot(data: unknown): data is Bot {
	if (!isObj(data)) return false;
	return typeof data.name === 'string'
		&& typeof data.promptId === 'string'
		&& typeof data.profileId === 'string'
		&& Array.isArray(data.children);
}

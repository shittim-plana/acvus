/**
 * Tauri bridge for the Pomollu mobile app (pomollu-tauri backend).
 *
 * In the GeckoView build, invoke() is transported over the WebExtension IPC
 * bridge (window.ipc → native messaging → Rust.ipc). Outside Tauri (plain
 * web dev), commands resolve to null so the UI degrades gracefully.
 *
 * Streaming: chat commands buffer chunks backend-side; poll with
 * pollStreamChunks() (~100ms) — Tauri events may not reach a GeckoView page
 * reliably, the buffer is the source of truth.
 */

export function isTauri(): boolean {
	return typeof window !== 'undefined' && ('__TAURI__' in window || '__TAURI_INTERNALS__' in window);
}

export async function invoke<T = unknown>(cmd: string, args?: Record<string, unknown>): Promise<T | null> {
	if (!isTauri()) {
		console.warn(`[mock] invoke: ${cmd}`, args);
		return null;
	}
	const { invoke: tauriInvoke } = await import('@tauri-apps/api/core');
	return tauriInvoke<T>(cmd, args);
}

// ── Chat streaming (per stream_id — concurrent streams are isolated) ─

export type ChatApiMessage = { role: string; content: string };

/** Fresh stream id. Pass a workspace id instead to scope one stream per workspace. */
export function newStreamId(): string {
	return globalThis.crypto?.randomUUID?.() ?? `s-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export async function pollStreamChunks(streamId: string): Promise<string[]> {
	return (await invoke<string[]>('poll_stream_chunks', { streamId })) ?? [];
}

export async function cancelChat(streamId: string): Promise<void> {
	await invoke('cancel_chat', { streamId });
}

/**
 * Drive a streaming chat command, polling that stream's buffer until it
 * resolves. Each call uses its own `stream_id`, so several can run at once
 * (e.g. Vertex in one workspace, GCA in another) without interfering.
 *
 * Returns `{ streamId, text }`; `streamId` lets the caller cancel mid-flight.
 */
export function startChat(
	cmd: 'chat_mistral' | 'chat_vertex' | 'chat_gca',
	args: Record<string, unknown>,
	onChunk: (text: string) => void,
	pollMs = 100,
): { streamId: string; text: Promise<string> } {
	const streamId = (args.streamId as string) ?? newStreamId();
	const done = invoke<string>(cmd, { ...args, streamId });
	const text = (async () => {
		let finished = false;
		void done.finally(() => (finished = true));
		while (!finished) {
			for (const chunk of await pollStreamChunks(streamId)) onChunk(chunk);
			await new Promise((r) => setTimeout(r, pollMs));
		}
		// Drain anything emitted between the last poll and resolution.
		for (const chunk of await pollStreamChunks(streamId)) onChunk(chunk);
		return (await done) ?? '';
	})();
	return { streamId, text };
}

// ── Vertex OAuth ────────────────────────────────────────────────────

export type OAuthStatus = { connected: boolean; expired: boolean };

export const vertexOAuth = {
	start: () => invoke<string>('vertex_oauth_start'),
	callback: (code: string) => invoke<string>('vertex_oauth_callback', { code }),
	status: () => invoke<OAuthStatus>('vertex_oauth_status'),
	disconnect: () => invoke<string>('vertex_oauth_disconnect'),
	listProjects: () => invoke<{ projectId: string; name: string }[]>('vertex_list_projects'),
	pending: () => invoke<{ code?: string }>('get_pending_oauth'),
};

// ── Gemini Code Assist OAuth (client_secret flow, free-tier Gemini) ──

export const gcaOAuth = {
	start: () => invoke<string>('gca_oauth_start'),
	callback: (code: string) => invoke<string>('gca_oauth_callback', { code }),
	status: () => invoke<OAuthStatus>('gca_oauth_status'),
	disconnect: () => invoke<string>('gca_oauth_disconnect'),
	/** loadCodeAssist → projectId; also opts out of free-tier data collection. */
	loadProject: () => invoke<string>('gca_load_project'),
	listModels: () => invoke<string[]>('gca_list_models'),
};

// ── Models — each provider has its own (partly different) catalog ────
//
// The sources differ in kind, not just contents:
//  - Mistral: live `/v1/models`, filtered to chat-capable + deduped to latest
//  - Vertex AI: live `publishers/google/models` for the signed-in region
//  - GCA: a static catalog (the v1internal surface has no list endpoint)
// So a model offered by one is not necessarily offered by another — query the
// active provider's list, don't assume a shared set.

export const models = {
	mistral: (apiKey: string) => invoke<{ id: string }[]>('mistral_list_models', { apiKey }),
	vertex: (region: string) => invoke<string[]>('vertex_list_models', { region }),
	gca: () => invoke<string[]>('gca_list_models'),
};

// ── Settings / session / workspaces ─────────────────────────────────

export const settings = {
	save: (value: unknown) => invoke('cmd_save_settings', { settings: value }),
	load: <T = Record<string, unknown>>() => invoke<T>('cmd_load_settings'),
};

export const session = {
	save: (value: unknown) => invoke('cmd_save_session', { session: value }),
	load: <T = unknown>() => invoke<T>('cmd_load_session'),
};

export const workspaces = {
	create: (name: string) => invoke<string>('cmd_workspace_create', { name }),
	list: () => invoke<Record<string, unknown>[]>('cmd_workspace_list'),
	load: (id: string) => invoke<Record<string, unknown>>('cmd_workspace_load', { id }),
	update: (id: string, data: unknown) => invoke('cmd_workspace_update', { id, data }),
	remove: (id: string) => invoke('cmd_workspace_delete', { id }),
	saveSession: (id: string, value: unknown) => invoke('cmd_workspace_save_session', { id, session: value }),
	loadSession: <T = unknown>(id: string) => invoke<T>('cmd_workspace_load_session', { id }),
};

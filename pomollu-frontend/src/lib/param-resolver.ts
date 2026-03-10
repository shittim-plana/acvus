import type { ContextKeyInfo, WebNode } from './engine.js';
import type { TypeDesc } from './type-parser.js';
import { isUnknownType } from './type-parser.js';
import type { Node, BlockNode, ContextBinding, DisplayEntry, DisplayRegion, ContextParam, Prompt, Profile, Bot } from './types.js';
import { isRawBlock, isScriptBlock, CONTEXT_TYPE } from './types.js';
import { collectBlocks, collectNodes } from './block-tree.js';
import { analyzeWithTypes, analyzeWithKnown, typecheckNodes } from './engine.js';

/**
 * Builtin context refs injected automatically by the WASM engine.
 * These are NOT user-defined params — excluded from param discovery.
 *
 * - turn: current turn info (engine internal struct)
 * - raw/self/content: node-internal variables (types provided by typecheckNodes)
 * - context: @context object (injected via CONTEXT_TYPE)
 * - item/index: iterator loop variables (injected by typecheckNodes / render_display)
 *
 * DO NOT move this to another file — internal to param-resolver.
 */
const BUILTIN_CONTEXT_REFS = new Set(['turn', 'raw', 'self', 'content', 'context', 'item', 'index']);

export type ScriptEntry = { source: string; mode: 'script' | 'template' };

export function collectScriptsFromBindings(bindings: ContextBinding[]): ScriptEntry[] {
	return bindings
		.filter((b) => b.script.trim())
		.map((b) => ({ source: b.script, mode: 'script' as const }));
}

export function collectScriptsFromTree(children: BlockNode[]): ScriptEntry[] {
	const scripts: ScriptEntry[] = [];
	for (const block of collectBlocks(children)) {
		if (isRawBlock(block)) {
			scripts.push({ source: block.text, mode: block.mode });
		} else if (isScriptBlock(block)) {
			scripts.push({ source: block.text, mode: 'script' });
		}
	}
	for (const node of collectNodes(children)) {
		collectScriptsFromNode(node, scripts);
	}
	return scripts;
}

function collectScriptsFromNode(node: Node, out: ScriptEntry[]) {
	if (node.selfSpec.initialValue.trim()) out.push({ source: node.selfSpec.initialValue, mode: 'script' });
	if (node.assert.trim() && node.assert !== 'true') out.push({ source: node.assert, mode: 'script' });
	if (node.strategy.mode === 'history' && node.strategy.historyBind.trim()) {
		out.push({ source: node.strategy.historyBind, mode: 'script' });
	}
	if (node.strategy.mode === 'if-modified' && node.strategy.key.trim()) {
		out.push({ source: node.strategy.key, mode: 'script' });
	}
	for (const msg of node.messages) {
		if (msg.kind === 'block' && msg.source.type === 'inline') {
			out.push({ source: msg.source.template, mode: 'template' });
		}
		if (msg.kind === 'iterator' && msg.iterator.trim()) {
			out.push({ source: msg.iterator, mode: 'script' });
		}
		// Do NOT collect iterator templates here.
		// They use @item/@index loop variables — handled by render_display.
	}
}

export function collectScriptsFromDisplay(entries: DisplayEntry[]): ScriptEntry[] {
	const scripts: ScriptEntry[] = [];
	for (const entry of entries) {
		if (entry.condition.trim()) scripts.push({ source: entry.condition, mode: 'script' });
		if (entry.template.trim()) scripts.push({ source: entry.template, mode: 'template' });
	}
	return scripts;
}

export function collectScriptsFromRegions(regions: DisplayRegion[]): ScriptEntry[] {
	const scripts: ScriptEntry[] = [];
	for (const region of regions) {
		if (region.kind === 'static' && region.template.trim()) {
			scripts.push({ source: region.template, mode: 'template' });
		} else if (region.kind === 'iterable') {
			if (region.iterator.trim()) scripts.push({ source: region.iterator, mode: 'script' });
			scripts.push(...collectScriptsFromDisplay(region.entries));
		}
	}
	return scripts;
}

export function collectNodeNames(children: BlockNode[]): Set<string> {
	return new Set(collectNodes(children).map((n) => n.name).filter((n) => n));
}

/**
 * Result of collectUnresolvedParams.
 *
 * `hasSkippedScripts`: true if any script failed to analyze (syntax error, etc.)
 * and was skipped. This is critical for Sanitization — see twoPassAnalysis.
 */
type CollectParamsResult = { keys: ContextKeyInfo[]; hasSkippedScripts: boolean };

function collectUnresolvedParams(opts: {
	scripts: { source: string; mode: 'script' | 'template' }[];
	nodeNames: Set<string>;
	providedKeys: Set<string>;
	contextTypes: Record<string, TypeDesc>;
	knownScripts?: Record<string, string>;
}): CollectParamsResult {
	const seen = new Map<string, { type: TypeDesc; status: 'eager' | 'lazy' | 'pruned' }>();
	let hasSkippedScripts = false;

	const knownEntries = opts.knownScripts ? Object.entries(opts.knownScripts) : [];
	for (const { source, mode } of opts.scripts) {
		if (!source.trim()) continue;
		const useKnown = knownEntries.length > 0;
		const result = useKnown
			? analyzeWithKnown(source, mode, opts.contextTypes, opts.knownScripts!)
			: analyzeWithTypes(source, mode, opts.contextTypes);
		if (!result.ok) {
			console.warn('[collectUnresolvedParams] skipping failed script:', mode, source.slice(0, 80), result.errors);
			hasSkippedScripts = true;
			continue;
		}
		const filteredKeys = result.context_keys.filter(
			(k) => !BUILTIN_CONTEXT_REFS.has(k.name) && !opts.nodeNames.has(k.name) && !opts.providedKeys.has(k.name)
		);
		for (const key of filteredKeys) {
			const existing = seen.get(key.name);
			if (!existing || isUnknownType(existing.type)) {
				seen.set(key.name, { type: key.type, status: key.status });
			} else if (existing.status === 'pruned' && key.status !== 'pruned') {
				// Pruned key found on a live path in another script → upgrade
				seen.set(key.name, { type: existing.type, status: key.status });
			} else if (existing.status === 'lazy' && key.status === 'eager') {
				seen.set(key.name, { type: existing.type, status: 'eager' });
			}
		}
	}

	const keys = Array.from(seen.entries())
		.map(([name, { type, status }]) => ({ name, type, status }))
		.sort((a, b) => a.name.localeCompare(b.name));
	return { keys, hasSkippedScripts };
}

/**
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * twoPassAnalysis — the ONLY 2-phase analysis function.
 * ALL WASM analysis/typecheck MUST go through this function.
 * No separate WASM calls outside of this function.
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 *
 * ── Phase 1: Pure Type Discovery ──────────────────────────────────
 * Analyzes all scripts WITHOUT known values.
 * All branches are live → all @context refs are discovered with correct types.
 * If all scripts succeed, keys not found here are orphans → deleted.
 * If any script fails, existing params are preserved (see Sanitization).
 *
 * ── Phase 2: Typecheck + Pruning ──────────────────────────────────
 * Typechecks with Phase 1's full type set (compiles including dead branches).
 * Re-analyzes with known values to classify dead-branch keys as pruned.
 * Pruned keys → active: false (hidden in UI, settings preserved).
 *
 * ── Sanitization ──────────────────────────────────────────────────
 * Merges Phase 1 discovered keys with existingParams.
 * Behavior depends on whether Phase 1 had skipped (failed) scripts:
 *
 *   ALL scripts succeeded (hasSkippedScripts = false):
 *     Phase 1 result is AUTHORITATIVE — keys not found are truly gone.
 *     → Only Phase 1 keys survive. Orphan params are deleted.
 *
 *   Some scripts failed (hasSkippedScripts = true):
 *     Phase 1 result is INCOMPLETE — failed scripts may have contributed keys
 *     that we can't see right now.
 *     → Existing params are PRESERVED to protect user settings (resolution,
 *       userType, editorMode) from being destroyed by a transient syntax error.
 *     → Phase 1 keys are merged in (new discoveries added, types updated).
 *
 * This distinction prevents the scenario where a temporary typo wipes out
 * all param settings, which then don't come back even after fixing the typo
 * (because the store was already overwritten with empty params).
 *
 * IMPORTANT: Phase 1 MUST run WITHOUT known values.
 * If known values are provided, the engine's partition logic
 * omits dead-branch known keys entirely, causing Phase 2 typecheck to fail.
 *
 * Each level (Prompt/Profile/Bot) calls this with its OWN params only.
 * Other levels' params are passed as providedKeys (already resolved).
 */
export type TwoPassResult = {
	env: ContextEnvResult;
	params: ContextParam[];
	activeParams: ContextParam[];
	ownParams: ContextParam[];
};

export function twoPassAnalysis(opts: {
	scripts: ScriptEntry[];
	nodeNames: Set<string>;
	providedKeys: Set<string>;
	existingParams: ContextParam[];
	baseTypes?: Record<string, TypeDesc>;
	children: BlockNode[];
	getApi: (providerId: string) => string;
}): TwoPassResult {
	const typesFromParams: Record<string, TypeDesc> = { ...(opts.baseTypes ?? {}) };
	for (const p of opts.existingParams) {
		if (p.userType) typesFromParams[p.name] = p.userType;
	}

	const knownScripts: Record<string, string> = {};
	for (const p of opts.existingParams) {
		if (p.resolution.kind === 'static' && p.resolution.value.trim()) {
			knownScripts[p.name] = p.resolution.value;
		}
	}

	const allScripts: ScriptEntry[] = [...opts.scripts];
	for (const p of opts.existingParams) {
		if (p.resolution.kind === 'static' && p.resolution.value.trim()) {
			allScripts.push({ source: p.resolution.value, mode: 'script' });
		}
	}

	// ── Phase 1: Pure type discovery (no known values) ──
	const phase1 = collectUnresolvedParams({
		scripts: allScripts,
		nodeNames: opts.nodeNames,
		providedKeys: opts.providedKeys,
		contextTypes: typesFromParams,
	});

	// Full type set from Phase 1.
	const fullTypes: Record<string, TypeDesc> = { ...typesFromParams };
	for (const k of phase1.keys) {
		if (!fullTypes[k.name]) fullTypes[k.name] = k.type;
	}

	// ── Phase 2: Typecheck + Pruning ──
	const webNodes = collectNodes(opts.children)
		.filter((n) => n.name)
		.map((n) => toWebNode(n, opts.getApi(n.providerId)));
	const typecheckResult = typecheckNodes(webNodes, fullTypes);
	const EMPTY_ENV: ContextEnvResult = { contextTypes: {}, nodeLocals: {}, nodeErrors: {} };
	const env: ContextEnvResult = 'error' in typecheckResult ? EMPTY_ENV : typecheckResult;

	// Pruning with known values.
	const phase1Names = new Set(phase1.keys.map((k) => k.name));
	const survivingNames = new Set(phase1Names);

	if (Object.keys(knownScripts).length > 0) {
		const phase2 = collectUnresolvedParams({
			scripts: allScripts,
			nodeNames: opts.nodeNames,
			providedKeys: opts.providedKeys,
			contextTypes: fullTypes,
			knownScripts,
		});
		for (const k of phase2.keys) {
			if (k.status === 'pruned') survivingNames.delete(k.name);
		}
	}

	// ── Sanitization ──
	// See docstring above for the full explanation of the two modes.
	const existingMap = new Map(opts.existingParams.map((p) => [p.name, p]));
	const phase1KeySet = new Set(phase1.keys.map((k) => k.name));

	// Build params from Phase 1 discovered keys (always included).
	const params: ContextParam[] = phase1.keys.map((k) => {
		const prev = existingMap.get(k.name);
		return {
			name: k.name,
			inferredType: k.type,
			resolution: prev?.resolution ?? { kind: 'unresolved' as const },
			userType: prev?.userType,
			editorMode: prev?.editorMode,
			active: survivingNames.has(k.name),
		};
	});

	// When scripts were skipped, preserve existing params that Phase 1 didn't see.
	// These may belong to the failed scripts — deleting them would destroy user settings.
	if (phase1.hasSkippedScripts) {
		for (const prev of opts.existingParams) {
			if (!phase1KeySet.has(prev.name)) {
				params.push(prev);
			}
		}
		// Re-sort to maintain stable ordering after merge.
		params.sort((a, b) => a.name.localeCompare(b.name));
	}

	const activeParams = params.filter((p) => p.active);
	return { env, params, activeParams, ownParams: params };
}

type GetApi = (providerId: string) => string;

export function analyzePrompt(prompt: Prompt, getApi: GetApi): TwoPassResult {
	const nodeNames = collectNodeNames(prompt.children);
	nodeNames.add('context');
	return twoPassAnalysis({
		scripts: [
			...collectScriptsFromBindings(prompt.contextBindings),
			...collectScriptsFromTree(prompt.children),
		],
		nodeNames,
		providedKeys: new Set(prompt.contextBindings.map((b) => b.name).filter((n) => n)),
		existingParams: prompt.contextParams,
		baseTypes: { context: CONTEXT_TYPE },
		children: prompt.children,
		getApi,
	});
}

export function analyzeProfile(profile: Profile, getApi: GetApi): TwoPassResult {
	const nodeNames = collectNodeNames(profile.children);
	nodeNames.add('context');
	return twoPassAnalysis({
		scripts: collectScriptsFromTree(profile.children),
		nodeNames,
		providedKeys: new Set(),
		existingParams: profile.contextParams,
		baseTypes: { context: CONTEXT_TYPE },
		children: profile.children,
		getApi,
	});
}

/**
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * analyzeBot — Bot-level param analysis with full hierarchy context.
 *
 * Each level manages its OWN params (locality):
 * - twoPassAnalysis discovers only bot-level params
 * - Prompt/Profile params are already resolved at their levels → providedKeys
 * - Prompt/Profile types → baseTypes (so typecheckNodes can compile)
 *
 * Returns ALL params from all 3 levels merged:
 * - prompt.contextParams + profile.contextParams + bot's discovered params
 * - Callers (e.g. buildSessionConfig) get the full picture from this single call.
 *
 * DO NOT change this structure.
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 */
export function analyzeBot(bot: Bot, prompt: Prompt, profile: Profile, getApi: GetApi): TwoPassResult {
	// providedKeys: bindings + prompt/profile params (already resolved at their levels).
	const providedKeys = new Set<string>();
	for (const b of prompt.contextBindings) if (b.name) providedKeys.add(b.name);
	for (const p of prompt.contextParams) providedKeys.add(p.name);
	for (const p of profile.contextParams) providedKeys.add(p.name);

	// nodeNames: all nodes across 3 levels.
	const nodeNames = new Set<string>();
	for (const n of collectNodeNames(prompt.children)) nodeNames.add(n);
	for (const n of collectNodeNames(profile.children)) nodeNames.add(n);
	for (const n of collectNodeNames(bot.children)) nodeNames.add(n);
	nodeNames.add('context');

	// scripts: all 3 levels + bot display/region iterators only.
	// Display entry condition/template use @item/@index loop vars — NOT param analysis targets.
	const scripts = [
		...collectScriptsFromBindings(prompt.contextBindings),
		...collectScriptsFromTree(prompt.children),
		...collectScriptsFromTree(profile.children),
		...collectScriptsFromTree(bot.children),
	];
	if (bot.display.iterator.trim()) {
		scripts.push({ source: bot.display.iterator, mode: 'script' as const });
	}
	for (const region of bot.regions) {
		if (region.kind === 'iterable' && region.iterator.trim()) {
			scripts.push({ source: region.iterator, mode: 'script' as const });
		} else if (region.kind === 'static' && region.template.trim()) {
			scripts.push({ source: region.template, mode: 'template' as const });
		}
	}

	// baseTypes: prompt/profile param types + CONTEXT_TYPE.
	// typecheckNodes needs full hierarchy types to compile all nodes.
	const baseTypes: Record<string, TypeDesc> = { context: CONTEXT_TYPE };
	for (const p of prompt.contextParams) {
		if (p.userType) baseTypes[p.name] = p.userType;
		else if (p.inferredType) baseTypes[p.name] = p.inferredType;
	}
	for (const p of profile.contextParams) {
		if (p.userType) baseTypes[p.name] = p.userType;
		else if (p.inferredType) baseTypes[p.name] = p.inferredType;
	}

	// twoPassAnalysis discovers bot-level params only.
	const result = twoPassAnalysis({
		scripts,
		nodeNames,
		providedKeys,
		existingParams: bot.contextParams,
		baseTypes,
		children: [...prompt.children, ...profile.children, ...bot.children],
		getApi,
	});

	// Merge all 3 levels' params into the result.
	// Bot analysis discovers bot-only params; prompt/profile params are already resolved.
	const ownParams = result.params;
	const allParams = [
		...prompt.contextParams,
		...profile.contextParams,
		...ownParams,
	];
	const allActiveParams = allParams.filter((p) => p.active !== false);

	return { env: result.env, params: allParams, activeParams: allActiveParams, ownParams };
}

/**
 * Convert a UI Node to the WebNode format expected by the WASM engine.
 * Message block references are resolved to inline templates.
 */
export function toWebNode(node: Node, api: string): WebNode {
	return {
		name: node.name,
		kind: node.kind,
		api,
		model: node.model,
		temperature: node.temperature,
		maxTokens: node.maxTokens,
		selfSpec: node.selfSpec,
		strategy: node.strategy,
		retry: node.retry,
		assert: node.assert,
		messages: node.messages.map((m) => {
			if (m.kind === 'block') {
				const template = m.source.type === 'inline' ? m.source.template : '';
				return { kind: 'block' as const, role: m.role, template };
			}
			return {
				kind: 'iterator' as const,
				iterator: m.iterator,
				role: m.role,
				slice: m.slice,
				tokenBudget: m.tokenBudget,
			};
		}),
		tools: node.tools.map((t) => ({
			name: t.name,
			description: t.description,
			node: t.nodeId,
			params: t.params,
		})),
	};
}

export type ContextEnvResult = {
	contextTypes: Record<string, TypeDesc>;
	nodeLocals: Record<string, { raw: TypeDesc; self: TypeDesc }>;
	nodeErrors: Record<string, Record<string, string>>;
};


/** Acvus template / script syntax highlighting — ported from acvus-playground */

type Rule = [RegExp, string | null];

function esc(s: string): string {
	return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function tokenize(text: string, rules: Rule[]): string {
	let result = '';
	let rest = text;
	while (rest.length > 0) {
		let matched = false;
		for (const [re, cls] of rules) {
			const m = rest.match(re);
			if (m) {
				const tok = esc(m[0]);
				result += cls ? `<span class="${cls}">${tok}</span>` : tok;
				rest = rest.slice(m[0].length);
				matched = true;
				break;
			}
		}
		if (!matched) {
			result += esc(rest[0]);
			rest = rest.slice(1);
		}
	}
	return result;
}

const ID = /[\w\p{L}]+/u;

const scriptRules: Rule[] = [
	[/^"(?:[^"\\]|\\.)*"/, 'hl-str'],
	[new RegExp(`^@${ID.source}`, 'u'), 'hl-ctx'],
	[new RegExp(`^\\$${ID.source}`, 'u'), 'hl-ctx'],
	[/^(?:in|match|true|false)\b/, 'hl-kw'],
	[/^_\b/, 'hl-kw'],
	[/^=>/, 'hl-kw'],
	[/^\|/, 'hl-pipe'],
	[/^\.\.=?/, 'hl-op'],
	[/^=\.\./, 'hl-op'],
	[/^[=!<>]=/, 'hl-op'],
	[/^[+\-*\/<>!]/, 'hl-op'],
	[/^\d+(?:\.\d+)?/, 'hl-num'],
	[new RegExp(`^${ID.source}`, 'u'), 'hl-fn'],
	[/^\s+/, null],
	[/^./, null],
];

const tmplInnerRules: Rule[] = [
	[/^\//, 'hl-delim'],
	...scriptRules,
];

// -- Markdown highlighting (for raw text outside {{ }}) --

const mdLineStartRules: Rule[] = [
	[/^#{1,6} [^\n]*/, 'hl-md-h'],
	[/^[-*+] /, 'hl-md-li'],
	[/^\d+\. /, 'hl-md-li'],
	[/^> /, 'hl-md-bq'],
	[/^---+(?=\n|$)/, 'hl-md-hr'],
];

const mdInlineRules: Rule[] = [
	[/^`[^`\n]+`/, 'hl-md-code'],
	[/^\*\*[^*\n]+?\*\*/, 'hl-md-b'],
	[/^\*(?!\*)[^*\n]+?\*(?!\*)/, 'hl-md-i'],
	[/^\[[^\]\n]+\]\([^)\n]+\)/, 'hl-md-link'],
	[/^[^*`\[\n#\->0-9]+/, 'hl-raw'],
	[/^\n/, null],
	[/^./, 'hl-raw'],
];

function highlightMarkdown(text: string): string {
	let result = '';
	let rest = text;
	let lineStart = true;
	while (rest.length > 0) {
		let matched = false;
		const rules: Rule[] = lineStart ? [...mdLineStartRules, ...mdInlineRules] : mdInlineRules;
		for (const [re, cls] of rules) {
			const m = rest.match(re);
			if (m) {
				const tok = esc(m[0]);
				result += cls ? `<span class="${cls}">${tok}</span>` : tok;
				lineStart = m[0][m[0].length - 1] === '\n';
				rest = rest.slice(m[0].length);
				matched = true;
				break;
			}
		}
		if (!matched) {
			result += '<span class="hl-raw">' + esc(rest[0]) + '</span>';
			lineStart = rest[0] === '\n';
			rest = rest.slice(1);
		}
	}
	return result;
}

// -- Public API --

export function highlightTemplate(text: string): string {
	let result = '';
	let i = 0;
	while (i < text.length) {
		let openDelim: string | null = null;
		if (text[i] === '{' && text[i + 1] === '-' && text[i + 2] === '{') {
			openDelim = '{-{';
		} else if (text[i] === '{' && text[i + 1] === '{') {
			openDelim = '{{';
		}
		if (openDelim) {
			result += '<span class="hl-delim">' + esc(openDelim) + '</span>';
			i += openDelim.length;
			let inner = '';
			let closed = false;
			while (i < text.length) {
				if (text[i] === '}' && text[i + 1] === '-' && text[i + 2] === '}') {
					result += tokenize(inner, tmplInnerRules);
					result += '<span class="hl-delim">}-}</span>';
					i += 3;
					closed = true;
					break;
				} else if (text[i] === '}' && text[i + 1] === '}') {
					result += tokenize(inner, tmplInnerRules);
					result += '<span class="hl-delim">}}</span>';
					i += 2;
					closed = true;
					break;
				}
				inner += text[i];
				i++;
			}
			if (!closed && inner) {
				result += tokenize(inner, tmplInnerRules);
			}
		} else {
			let raw = '';
			while (i < text.length) {
				if (text[i] === '{' && (text[i + 1] === '{' || (text[i + 1] === '-' && text[i + 2] === '{'))) break;
				raw += text[i];
				i++;
			}
			result += highlightMarkdown(raw);
		}
	}
	return result + '\n';
}

export function highlightScript(text: string): string {
	return tokenize(text, scriptRules) + '\n';
}

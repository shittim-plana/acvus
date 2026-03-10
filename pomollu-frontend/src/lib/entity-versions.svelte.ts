export type EntityKind = 'provider' | 'prompt' | 'profile' | 'bot';
export type EntityRef = { kind: EntityKind; id: string };

class EntityVersions {
	private versions = $state<Map<string, number>>(new Map());

	bump(kind: EntityKind, id: string): void {
		const key = `${kind}:${id}`;
		this.versions.set(key, (this.versions.get(key) ?? 0) + 1);
	}

	get(kind: EntityKind, id: string): number {
		return this.versions.get(`${kind}:${id}`) ?? 0;
	}

	depsVersion(deps: EntityRef[]): number {
		let v = 0;
		for (const d of deps) v += (this.versions.get(`${d.kind}:${d.id}`) ?? 0);
		return v;
	}
}

export const entityVersions = new EntityVersions();

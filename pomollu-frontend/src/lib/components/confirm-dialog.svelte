<script lang="ts">
	import { getPending, respond } from '$lib/confirm-dialog.svelte.js';
	import { Button } from '$lib/components/ui/button';

	let pending = $derived(getPending());

	function handleKeydown(e: KeyboardEvent) {
		if (!pending) return;
		if (e.key === 'Escape') respond(false);
		if (e.key === 'Enter') respond(true);
	}
</script>

<svelte:window onkeydown={handleKeydown} />

{#if pending}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div class="confirm-backdrop" onclick={() => respond(false)}>
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div class="confirm-card" onclick={(e) => e.stopPropagation()}>
			<p class="confirm-msg">{pending.message}</p>
			<div class="confirm-actions">
				<Button variant="outline" size="sm" onclick={() => respond(false)}>Cancel</Button>
				<Button variant={pending.variant} size="sm" onclick={() => respond(true)}>{pending.confirmLabel}</Button>
			</div>
		</div>
	</div>
{/if}

<style>
	.confirm-backdrop {
		position: fixed;
		inset: 0;
		z-index: 100;
		display: flex;
		align-items: center;
		justify-content: center;
		background: rgba(0, 0, 0, 0.5);
	}
	.confirm-card {
		background: var(--color-background);
		border: 1px solid var(--color-border);
		border-radius: 0.75rem;
		padding: 1.5rem;
		width: 22rem;
		max-width: 90vw;
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
		display: flex;
		flex-direction: column;
		gap: 1.25rem;
	}
	.confirm-msg {
		font-size: 0.875rem;
		color: var(--color-foreground);
		line-height: 1.5;
		margin: 0;
	}
	.confirm-actions {
		display: flex;
		justify-content: flex-end;
		gap: 0.5rem;
	}
</style>

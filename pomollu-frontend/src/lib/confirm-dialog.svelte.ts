/**
 * Programmatic confirm dialog.
 *
 * Usage:
 *   const ok = await confirmDelete('Delete this bot?');
 *   if (ok) doDelete();
 *
 * The ConfirmDialog.svelte component must be mounted (once, in +page.svelte)
 * for this to work. It subscribes to the pending request via $state.
 */

type PendingConfirm = {
	message: string;
	confirmLabel: string;
	variant: 'destructive' | 'default';
	resolve: (ok: boolean) => void;
};

let _pending = $state<PendingConfirm | null>(null);

export function getPending(): PendingConfirm | null {
	return _pending;
}

export function confirmDelete(message: string): Promise<boolean> {
	return confirm(message, 'Delete', 'destructive');
}

export function confirmAction(message: string, confirmLabel: string = 'Confirm'): Promise<boolean> {
	return confirm(message, confirmLabel, 'default');
}

function confirm(message: string, confirmLabel: string, variant: 'destructive' | 'default'): Promise<boolean> {
	// If already showing, resolve previous as false
	if (_pending) _pending.resolve(false);

	return new Promise<boolean>((resolve) => {
		_pending = { message, confirmLabel, variant, resolve };
	});
}

export function respond(ok: boolean) {
	if (_pending) {
		_pending.resolve(ok);
		_pending = null;
	}
}

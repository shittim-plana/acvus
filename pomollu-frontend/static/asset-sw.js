// Asset Service Worker — serves files from IndexedDB via URL
//
// URL format: /asset/{dbName}/{folder}/{file}
// Example:    /asset/asset_abc123/portraits/alice.webp
//
// The SW intercepts fetch requests matching this pattern,
// reads the binary data from IndexedDB, and returns a proper Response.

const ASSET_PREFIX = '/asset/';

const MIME_TYPES = {
	// Images
	webp: 'image/webp',
	png: 'image/png',
	jpg: 'image/jpeg',
	jpeg: 'image/jpeg',
	gif: 'image/gif',
	svg: 'image/svg+xml',
	ico: 'image/x-icon',
	bmp: 'image/bmp',
	avif: 'image/avif',
	// Audio
	mp3: 'audio/mpeg',
	wav: 'audio/wav',
	ogg: 'audio/ogg',
	// Video
	mp4: 'video/mp4',
	webm: 'video/webm',
	// Other
	pdf: 'application/pdf',
	json: 'application/json',
	txt: 'text/plain',
	csv: 'text/csv',
};

function mimeFromPath(path) {
	const ext = path.split('.').pop()?.toLowerCase() ?? '';
	return MIME_TYPES[ext] ?? 'application/octet-stream';
}

// ── IndexedDB helpers ──────────────────────────────────────────────

function idb(request) {
	return new Promise((resolve, reject) => {
		request.onsuccess = () => resolve(request.result);
		request.onerror = () => reject(request.error);
	});
}

const BLOB_DB_NAME = 'asset_blobs';

function openDB(dbName) {
	return new Promise((resolve, reject) => {
		const r = indexedDB.open(dbName, 1);
		r.onupgradeneeded = () => {
			const db = r.result;
			if (!db.objectStoreNames.contains('assets')) db.createObjectStore('assets');
			if (!db.objectStoreNames.contains('meta')) db.createObjectStore('meta');
		};
		r.onsuccess = () => resolve(r.result);
		r.onerror = () => reject(r.error);
	});
}

function openBlobDB() {
	return new Promise((resolve, reject) => {
		const r = indexedDB.open(BLOB_DB_NAME, 1);
		r.onupgradeneeded = () => {
			const db = r.result;
			if (!db.objectStoreNames.contains('blobs')) db.createObjectStore('blobs');
		};
		r.onsuccess = () => resolve(r.result);
		r.onerror = () => reject(r.error);
	});
}

async function readAsset(dbName, assetPath) {
	// Step 1: read hash from per-entity DB
	const entityDb = await openDB(dbName);
	let hash;
	try {
		hash = await idb(
			entityDb.transaction('assets', 'readonly').objectStore('assets').get(assetPath),
		);
	} finally {
		entityDb.close();
	}
	if (!hash) return null;

	// Step 2: read blob from shared blob DB
	const blobDb = await openBlobDB();
	try {
		const entry = await idb(
			blobDb.transaction('blobs', 'readonly').objectStore('blobs').get(hash),
		);
		return entry?.data ?? null;
	} finally {
		blobDb.close();
	}
}

// ── Fetch handler ──────────────────────────────────────────────────

self.addEventListener('fetch', (event) => {
	const url = new URL(event.request.url);
	if (!url.pathname.startsWith(ASSET_PREFIX)) return;

	// Parse: /asset/{dbName}/{folder}/{file...}
	const rest = url.pathname.slice(ASSET_PREFIX.length);
	const slashIdx = rest.indexOf('/');
	if (slashIdx === -1) return;

	const dbName = decodeURIComponent(rest.slice(0, slashIdx));
	const assetPath = decodeURIComponent(rest.slice(slashIdx + 1));

	if (!dbName || !assetPath) return;

	event.respondWith(
		readAsset(dbName, assetPath).then((data) => {
			if (!data) {
				return new Response('Not found', { status: 404 });
			}
			return new Response(data, {
				status: 200,
				headers: {
					'Content-Type': mimeFromPath(assetPath),
					'Cache-Control': 'public, max-age=31536000, immutable',
				},
			});
		}).catch(() => new Response('Internal error', { status: 500 })),
	);
});

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

//! Local content-addressed store — the file-based analog of the engine's
//! IndexedDB `IdbBlobStore` + journal. Forward-compatible with the storage
//! architecture the acvus engine is being re-grounded on (content-addressed
//! + append-only + metadata compare-and-swap).
//!
//! Layout under a store root:
//! ```text
//! store/
//! ├── blobs/{sha256-hex}   immutable, content-addressed (a G-Set: a join-
//! │                        semilattice under ∪, so concurrent puts are safe —
//! │                        equal content ⇒ equal hash ⇒ idempotent; distinct
//! │                        content ⇒ distinct path ⇒ commutative)
//! ├── journal              append-only log: one entry line per commit, each
//! │                        {hash, parent?} — forms the turn DAG (undo = parent,
//! │                        branch = sibling children)
//! └── refs/{name}          mutable pointer, CAS-updated (cursor / HEAD)
//! ```
//!
//! Concurrency model — Pomollu is multi-window (unlike layream's single
//! window), so there are concurrent writers even on one device. This is the
//! reason content-addressed + append-only is the right model, not mutable
//! last-write-wins JSON:
//!
//! - **blobs** are a G-Set (commutative ∪) → concurrent `put` is unconditionally
//!   safe, zero loss.
//! - **journal** is append-only → two windows committing off the same parent
//!   produce two child entries = a *branch* in the DAG (the turn tree), not a
//!   lost write. Divergence is recorded, not resolved destructively.
//! - the **only** serialization point is a cursor's CAS. Within one process
//!   (multiple Tauri windows = concurrent async tasks), `cas_lock` makes the
//!   read-compare-write atomic → genuinely linearizable. The loser of a CAS
//!   doesn't lose data (its blob + journal entry were written *before* the CAS,
//!   by monotonicity) — it simply becomes a branch.
//!
//! Each window should hold its own cursor ref (`cursor/{window-id}`): disjoint
//! keys ⇒ no inter-window contention; same-session edits fan out as siblings.
//!
//! Escalation boundary: `cas_lock` is an in-process mutex, correct for one app
//! process with N windows. Multiple OS processes sharing the store (not a mobile
//! scenario) would need OS file locking (flock) instead — documented, not built.
//! Multi-device sync (CLAUDE.md Claim 2) keeps the blob G-Set and adds
//! commutativity to the pointer; also a migration path, not built here.
//!
//! Caveat: SHA-256 collision resistance is a cryptographic assumption, not a
//! structural guarantee.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use crate::error::PomolluError;

/// A content hash (SHA-256, lowercase hex).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlobHash(pub String);

impl BlobHash {
    fn of(bytes: &[u8]) -> Self {
        let digest = Sha256::digest(bytes);
        BlobHash(hex::encode(digest))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A journal entry: a committed blob and its parent in the turn DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalEntry {
    pub hash: BlobHash,
    pub parent: Option<BlobHash>,
}

pub struct FsBlobStore {
    root: PathBuf,
    /// Serializes ref CAS critical sections so concurrent writers in this
    /// process (multiple Tauri windows = concurrent async tasks) get true
    /// linearizable compare-and-swap, not a check-then-act race.
    cas_lock: Mutex<()>,
}

impl FsBlobStore {
    /// Open (creating if absent) a store rooted at `root`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, PomolluError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("blobs"))?;
        fs::create_dir_all(root.join("refs"))?;
        Ok(Self {
            root,
            cas_lock: Mutex::new(()),
        })
    }

    fn blob_path(&self, hash: &BlobHash) -> PathBuf {
        self.root.join("blobs").join(&hash.0)
    }

    fn ref_path(&self, name: &str) -> PathBuf {
        self.root.join("refs").join(name)
    }

    // ── Blob layer (content-addressed, G-Set) ───────────────────────

    /// Store `bytes`, returning their content hash. Idempotent: storing the
    /// same bytes twice writes once and yields the same hash.
    pub fn put(&self, bytes: &[u8]) -> Result<BlobHash, PomolluError> {
        let hash = BlobHash::of(bytes);
        let path = self.blob_path(&hash);
        if !path.exists() {
            // tmp+rename keeps a concurrent reader from seeing a torn blob.
            let tmp = path.with_extension("tmp");
            fs::write(&tmp, bytes)?;
            fs::rename(&tmp, &path)?;
        }
        Ok(hash)
    }

    pub fn get(&self, hash: &BlobHash) -> Result<Option<Vec<u8>>, PomolluError> {
        match fs::read(self.blob_path(hash)) {
            Ok(b) => Ok(Some(b)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn has(&self, hash: &BlobHash) -> bool {
        self.blob_path(hash).exists()
    }

    // ── Ref layer (mutable pointer, CAS) ────────────────────────────

    pub fn ref_get(&self, name: &str) -> Result<Option<BlobHash>, PomolluError> {
        match fs::read_to_string(self.ref_path(name)) {
            Ok(s) => {
                let s = s.trim();
                Ok(if s.is_empty() {
                    None
                } else {
                    Some(BlobHash(s.to_string()))
                })
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Compare-and-swap a named ref. Sets it to `new` only if its current value
    /// equals `expected`; otherwise returns `Err(Mismatch)` carrying the actual
    /// current value. The read-compare-write runs under `cas_lock`, so it is
    /// atomic — and thus linearizable — across concurrent writers in this
    /// process (multiple windows).
    pub fn ref_cas(
        &self,
        name: &str,
        expected: Option<&BlobHash>,
        new: &BlobHash,
    ) -> Result<(), CasError> {
        let _guard = self
            .cas_lock
            .lock()
            .map_err(|_| CasError::Io(PomolluError::Http("cas lock poisoned".into())))?;
        let current = self.ref_get(name).map_err(CasError::Io)?;
        if current.as_ref() != expected {
            return Err(CasError::Mismatch { current });
        }
        let path = self.ref_path(name);
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, &new.0).map_err(|e| CasError::Io(e.into()))?;
        fs::rename(&tmp, &path).map_err(|e| CasError::Io(e.into()))?;
        Ok(())
    }

    // ── Journal layer (append-only DAG) ─────────────────────────────

    /// Append a journal entry linking `hash` to `parent`. Append-only: existing
    /// entries are never rewritten, so history is monotonic.
    pub fn journal_append(&self, entry: &JournalEntry) -> Result<(), PomolluError> {
        use std::io::Write;
        let line = match &entry.parent {
            Some(p) => format!("{} {}\n", entry.hash.0, p.0),
            None => format!("{}\n", entry.hash.0),
        };
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("journal"))?;
        f.write_all(line.as_bytes())?;
        Ok(())
    }

    /// Read all journal entries in append order.
    pub fn journal_entries(&self) -> Result<Vec<JournalEntry>, PomolluError> {
        let text = match fs::read_to_string(self.root.join("journal")) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        let mut entries = Vec::new();
        for line in text.lines() {
            let mut parts = line.split_whitespace();
            let Some(hash) = parts.next() else { continue };
            let parent = parts.next().map(|p| BlobHash(p.to_string()));
            entries.push(JournalEntry {
                hash: BlobHash(hash.to_string()),
                parent,
            });
        }
        Ok(entries)
    }

    /// Commit `payload` as a new turn whose parent is the current value of
    /// `cursor_ref`, then try to advance the cursor.
    ///
    /// The blob and journal entry are written *before* the CAS, so they persist
    /// regardless of the outcome (monotonicity — no loss). If a concurrent
    /// writer advanced the cursor first, the CAS mismatches and this commit
    /// becomes a *branch* (`advanced == false`): its turn is recorded as a
    /// sibling in the DAG, not lost. The caller can then rebase its window
    /// cursor or merge.
    pub fn commit(&self, payload: &[u8], cursor_ref: &str) -> Result<Commit, PomolluError> {
        let hash = self.put(payload)?;
        let parent = self.ref_get(cursor_ref)?;
        self.journal_append(&JournalEntry {
            hash: hash.clone(),
            parent: parent.clone(),
        })?;
        let advanced = match self.ref_cas(cursor_ref, parent.as_ref(), &hash) {
            Ok(()) => true,
            Err(CasError::Mismatch { .. }) => false,
            Err(CasError::Io(e)) => return Err(e),
        };
        Ok(Commit { hash, advanced })
    }

    /// Walk parent links from `hash` back to a root (entry with no parent).
    /// Returns the chain newest-first. Used for undo / history display.
    pub fn ancestry(&self, hash: &BlobHash) -> Result<Vec<BlobHash>, PomolluError> {
        let entries = self.journal_entries()?;
        let parent_of = |h: &BlobHash| -> Option<BlobHash> {
            entries
                .iter()
                .find(|e| &e.hash == h)
                .and_then(|e| e.parent.clone())
        };
        let mut chain = vec![hash.clone()];
        let mut cur = hash.clone();
        // Bound the walk by entry count to avoid cycles (append-only ⇒ acyclic,
        // but a corrupted journal must not loop forever).
        for _ in 0..entries.len() {
            match parent_of(&cur) {
                Some(p) => {
                    chain.push(p.clone());
                    cur = p;
                }
                None => break,
            }
        }
        Ok(chain)
    }
}

/// Outcome of [`FsBlobStore::commit`].
#[derive(Debug, Clone)]
pub struct Commit {
    pub hash: BlobHash,
    /// `true` if this commit advanced the cursor; `false` if a concurrent
    /// writer won the CAS and this commit became a branch (still persisted).
    pub advanced: bool,
}

#[derive(Debug)]
pub enum CasError {
    /// Current value did not match `expected`. Retry by merging onto `current`.
    Mismatch { current: Option<BlobHash> },
    Io(PomolluError),
}

impl std::fmt::Display for CasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CasError::Mismatch { current } => {
                write!(f, "CAS mismatch (current = {current:?})")
            }
            CasError::Io(e) => write!(f, "CAS io error: {e}"),
        }
    }
}

impl std::error::Error for CasError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, FsBlobStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = FsBlobStore::open(dir.path().join("store")).unwrap();
        (dir, store)
    }

    #[test]
    fn put_is_content_addressed_and_idempotent() {
        let (_d, s) = store();
        let h1 = s.put(b"hello").unwrap();
        let h2 = s.put(b"hello").unwrap();
        assert_eq!(h1, h2, "same content ⇒ same hash");
        assert_eq!(s.get(&h1).unwrap().as_deref(), Some(&b"hello"[..]));

        let h3 = s.put(b"world").unwrap();
        assert_ne!(h1, h3, "distinct content ⇒ distinct hash");
    }

    #[test]
    fn get_missing_is_none() {
        let (_d, s) = store();
        assert_eq!(s.get(&BlobHash("deadbeef".into())).unwrap(), None);
    }

    #[test]
    fn ref_cas_succeeds_only_on_expected() {
        let (_d, s) = store();
        let a = s.put(b"a").unwrap();
        let b = s.put(b"b").unwrap();

        // Initial set: expected = None.
        assert_eq!(s.ref_get("cursor").unwrap(), None);
        s.ref_cas("cursor", None, &a).unwrap();
        assert_eq!(s.ref_get("cursor").unwrap(), Some(a.clone()));

        // Wrong expected ⇒ mismatch, value unchanged.
        let err = s.ref_cas("cursor", None, &b).unwrap_err();
        match err {
            CasError::Mismatch { current } => assert_eq!(current, Some(a.clone())),
            other => panic!("expected mismatch, got {other:?}"),
        }
        assert_eq!(s.ref_get("cursor").unwrap(), Some(a.clone()));

        // Correct expected ⇒ swap.
        s.ref_cas("cursor", Some(&a), &b).unwrap();
        assert_eq!(s.ref_get("cursor").unwrap(), Some(b));
    }

    #[test]
    fn journal_forms_traversable_dag() {
        let (_d, s) = store();
        let root = s.put(b"turn0").unwrap();
        let child = s.put(b"turn1").unwrap();
        let grandchild = s.put(b"turn2").unwrap();

        s.journal_append(&JournalEntry { hash: root.clone(), parent: None }).unwrap();
        s.journal_append(&JournalEntry { hash: child.clone(), parent: Some(root.clone()) }).unwrap();
        s.journal_append(&JournalEntry {
            hash: grandchild.clone(),
            parent: Some(child.clone()),
        })
        .unwrap();

        // ancestry walks newest-first to the root.
        let chain = s.ancestry(&grandchild).unwrap();
        assert_eq!(chain, vec![grandchild, child.clone(), root.clone()]);

        // A branch: a sibling child off `root` (undo + new turn).
        let sibling = s.put(b"turn1b").unwrap();
        s.journal_append(&JournalEntry { hash: sibling.clone(), parent: Some(root.clone()) }).unwrap();
        assert_eq!(s.ancestry(&sibling).unwrap(), vec![sibling, root]);
    }

    #[test]
    fn concurrent_commits_branch_without_loss() {
        use std::collections::HashSet;
        use std::sync::Arc;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let s = Arc::new(FsBlobStore::open(dir.path().join("store")).unwrap());

        // A common starting point.
        let root = s.commit(b"root", "main").unwrap();
        assert!(root.advanced);

        // N windows commit concurrently off the live cursor.
        let n = 8;
        let handles: Vec<_> = (0..n)
            .map(|i| {
                let s = Arc::clone(&s);
                thread::spawn(move || s.commit(format!("turn-{i}").as_bytes(), "main").unwrap())
            })
            .collect();
        let commits: Vec<Commit> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // No loss — every concurrent commit's blob persisted, and the journal
        // recorded all of them (root + n).
        for c in &commits {
            assert!(s.has(&c.hash));
        }
        assert_eq!(s.journal_entries().unwrap().len() as u32, n + 1);

        // Linearizable cursor — the final cursor's ancestry is exactly the
        // commits that reported `advanced`; the branched ones sit off-chain
        // (recorded, not lost). Robust to any thread interleaving.
        let cursor = s.ref_get("main").unwrap().unwrap();
        let chain: HashSet<BlobHash> = s.ancestry(&cursor).unwrap().into_iter().collect();
        for c in &commits {
            if c.advanced {
                assert!(chain.contains(&c.hash), "advanced commit must be on the cursor chain");
            } else {
                assert!(!chain.contains(&c.hash), "branched commit must be off-chain");
            }
        }
    }

    #[test]
    fn journal_survives_reopen_append_only() {
        let dir = tempfile::tempdir().unwrap();
        let root_path = dir.path().join("store");
        let h = {
            let s = FsBlobStore::open(&root_path).unwrap();
            let h = s.put(b"x").unwrap();
            s.journal_append(&JournalEntry { hash: h.clone(), parent: None }).unwrap();
            h
        };
        // Reopen: blobs + journal persist.
        let s2 = FsBlobStore::open(&root_path).unwrap();
        assert!(s2.has(&h));
        assert_eq!(s2.journal_entries().unwrap().len(), 1);
    }
}

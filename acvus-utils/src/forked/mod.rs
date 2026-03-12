use core::task;
use std::{
    collections::{HashMap, VecDeque},
    ops::Deref,
    pin::Pin,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
};

use futures::{Stream, StreamExt};
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};

mod slot;
use slot::Slots;

use crate::forked::slot::Lagged;

/// A forked stream that allows for multiple consumers to access the same data
/// without duplicating the underlying data structure. Each consumer gets a unique ID
/// that can be used to identify it. When a consumer is dropped, its ID is released
/// back to the pool of available IDs.
pub struct Forked<S>
where
    S: Stream,
{
    id: usize,
    pos: usize,
    inner: Arc<ForkedInner<S>>,
}

impl<S> Forked<S>
where
    S: Stream,
{
    pub fn new(inner: S, size: usize) -> Self {
        let inner = Arc::new(ForkedInner::new(inner, size));
        let (id, pos) = inner.register_stream();
        Self { id, pos, inner }
    }

    #[allow(dead_code)]
    pub fn fork(&self) -> Self {
        let (id, pos) = self.inner.register_stream();
        Self {
            id,
            pos,
            inner: self.inner.clone(),
        }
    }

    pub fn terminate(&self) {
        self.inner.is_ended.store(true, Ordering::SeqCst);
        self.inner.wake_all();
    }

    pub fn downgrade(&self) -> ForkedWeak<S> {
        ForkedWeak {
            inner: Arc::downgrade(&self.inner),
        }
    }
}

impl<S> Stream for Forked<S>
where
    S: Stream,
    S::Item: Clone,
{
    type Item = Result<S::Item, Lagged>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let result = self.inner.poll_next(cx, self.id, self.pos);
        let Poll::Ready(Some(item)) = &result else {
            return result;
        };

        let pos = &mut self.get_mut().pos;
        *pos = match item {
            Ok(_) => pos.wrapping_add(1),
            Err(Lagged { head, .. }) => *head,
        };

        result
    }
}

impl<S> Drop for Forked<S>
where
    S: Stream,
{
    fn drop(&mut self) {
        self.inner.unregister_stream(self.id);
    }
}

pub struct ForkedWeak<S>
where
    S: Stream,
{
    inner: Weak<ForkedInner<S>>,
}

impl<S> ForkedWeak<S>
where
    S: Stream,
{
    pub fn try_upgrade(&self) -> Option<Forked<S>> {
        if let Some(inner) = self.inner.upgrade() {
            if !inner.is_ended.load(Ordering::SeqCst) {
                let (id, pos) = inner.register_stream();
                return Some(Forked { id, pos, inner });
            }
        }

        None
    }
}

struct ForkedInner<S>
where
    S: Stream,
{
    unique_id_pool: Mutex<UniqueIdPool>,
    is_ended: AtomicBool,

    // It is extremely hard to make type `S` as pin-compatible since mutex
    // does not support structural pinning.
    // Therefore, we use a `Pin<Box<S>>` to ensure that the inner stream is pinned
    inner: Mutex<Pin<Box<S>>>,
    state: RwLock<State>,

    // Buffer to hold items for the stream consumers.
    // This is necessary to ensure that each consumer can access the same data
    // without duplicating the underlying data structure.
    buffer: RwLock<Slots<S::Item>>,

    // Wakers for the stream consumers.
    // This is used to wake up the consumers when new data is available.
    wakers: Mutex<HashMap<usize, task::Waker>>,
}

impl<S> ForkedInner<S>
where
    S: Stream,
{
    fn new(inner: S, size: usize) -> Self {
        Self {
            unique_id_pool: Mutex::new(UniqueIdPool::default()),
            is_ended: AtomicBool::new(false),
            inner: Mutex::new(Box::pin(inner)),
            state: RwLock::new(State::Ready),
            buffer: RwLock::new(Slots::new(size)),
            wakers: Mutex::new(HashMap::new()),
        }
    }

    fn register_stream(&self) -> (usize, usize) {
        // Create a new unique ID for the stream consumer.
        let id = {
            let mut uip = self.unique_id_pool.lock();
            uip.total_members += 1;
            if let Some(id) = uip.free_ids.pop_front() {
                id
            } else {
                let id = uip.next_id;
                uip.next_id += 1;
                id
            }
        };

        let head = self.buffer.read().head_idx();
        (id, head)
    }

    fn unregister_stream(&self, id: usize) {
        // Release the unique ID back to the pool.
        {
            let mut uip = self.unique_id_pool.lock();
            uip.total_members -= 1;
            uip.free_ids.push_back(id);
        }

        let state = self.state.upgradable_read();
        if let State::Waiting { id: waiter_id } = state.deref()
            && *waiter_id != id
        {
            return;
        }

        *RwLockUpgradableReadGuard::upgrade(state) = State::Ready;
        self.wake_all();
    }

    fn wake_all(&self) {
        let wakers = {
            let mut wakers = self.wakers.lock();
            std::mem::take(&mut *wakers)
        };

        for (_, waker) in wakers {
            waker.wake();
        }
    }
}

impl<S> ForkedInner<S>
where
    S: Stream,
    S::Item: Clone,
{
    fn poll_next(
        &self,
        cx: &mut Context<'_>,
        id: usize,
        pos: usize,
    ) -> Poll<Option<Result<S::Item, Lagged>>> {
        // First, try to consume any available items from the buffer.
        if let Some(item) = self.try_consume(pos) {
            return Poll::Ready(Some(item));
        }

        // Check the stream has ended.
        if self.is_ended.load(Ordering::SeqCst) {
            return Poll::Ready(None);
        }

        let state = self.state.upgradable_read();
        let State::Waiting { id: waiter_id } = *state else {
            let mut state = RwLockUpgradableReadGuard::upgrade(state);
            if let State::Ready = *state {
                *state = State::Waiting { id };
            }

            cx.waker().wake_by_ref();
            return Poll::Pending;
        };

        if waiter_id != id {
            self.wakers.lock().insert(id, cx.waker().clone());
            return Poll::Pending;
        }

        let item = {
            let mut inner = self.inner.lock();
            inner.poll_next_unpin(cx)
        };

        if let Poll::Ready(item) = &item {
            if let Some(item) = item {
                self.buffer.write().enqueue(item.clone());
            } else {
                self.is_ended.store(true, Ordering::SeqCst);
            };
        } else {
            return Poll::Pending;
        }

        self.wake_all();

        {
            *RwLockUpgradableReadGuard::upgrade(state) = State::Ready;
        }

        item.map(|item| item.map(Ok))
    }

    fn try_consume(&self, pos: usize) -> Option<Result<S::Item, Lagged>> {
        let buffer = self.buffer.upgradable_read();
        buffer.consume(pos)
    }
}

#[derive(Clone)]
pub enum State {
    Ready,
    Waiting { id: usize },
}

#[derive(Default)]
struct UniqueIdPool {
    free_ids: VecDeque<usize>,
    next_id: usize,
    total_members: usize,
}
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use parking_lot::Mutex;

// ---------------------------------------------------------------------------
// Signal — unified coroutine communication
// ---------------------------------------------------------------------------

enum Signal<V> {
    Empty,
    Yield(V),
    NeedContext {
        name: String,
        bindings: HashMap<String, V>,
    },
    ContextReady(Arc<V>),
}

// ---------------------------------------------------------------------------
// YieldHandle — producer side
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct YieldHandle<V> {
    shared: Arc<Mutex<Signal<V>>>,
}

impl<V> YieldHandle<V> {
    pub fn yield_val(&self, value: V) -> YieldFuture<'_, V> {
        YieldFuture {
            shared: &self.shared,
            value: Some(value),
        }
    }

    pub fn request_context(&self, name: String) -> ContextFuture<'_, V> {
        ContextFuture {
            shared: &self.shared,
            name: Some(name),
            bindings: HashMap::new(),
        }
    }

    pub fn request_context_with(
        &self,
        name: String,
        bindings: HashMap<String, V>,
    ) -> ContextFuture<'_, V> {
        ContextFuture {
            shared: &self.shared,
            name: Some(name),
            bindings,
        }
    }
}

// ---------------------------------------------------------------------------
// YieldFuture
// ---------------------------------------------------------------------------

pub struct YieldFuture<'a, V> {
    shared: &'a Arc<Mutex<Signal<V>>>,
    value: Option<V>,
}

impl<V> Future for YieldFuture<'_, V>
where
    V: Unpin,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        if let Some(value) = this.value.take() {
            *this.shared.lock() = Signal::Yield(value);
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }
}

// ---------------------------------------------------------------------------
// ContextFuture
// ---------------------------------------------------------------------------

pub struct ContextFuture<'a, V> {
    shared: &'a Arc<Mutex<Signal<V>>>,
    name: Option<String>,
    bindings: HashMap<String, V>,
}

impl<V> Future for ContextFuture<'_, V>
where
    V: Unpin,
{
    type Output = Arc<V>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Arc<V>> {
        let this = self.get_mut();
        let mut signal = this.shared.lock();

        if let Some(name) = this.name.take() {
            *signal = Signal::NeedContext {
                name,
                bindings: std::mem::take(&mut this.bindings),
            };
            Poll::Pending
        } else if let Signal::ContextReady(_) = &*signal {
            let Signal::ContextReady(value) = std::mem::replace(&mut *signal, Signal::Empty) else {
                unreachable!()
            };
            Poll::Ready(value)
        } else {
            Poll::Pending
        }
    }
}

// ---------------------------------------------------------------------------
// Resume API — public types
// ---------------------------------------------------------------------------

pub struct ResumeKey<V>(ResumeKeyInner<V>);

enum ResumeKeyInner<V> {
    Start,
    Context(Arc<V>),
}

pub enum Stepped<V> {
    Emit(EmitStepped<V>),
    NeedContext(NeedContextStepped<V>),
    Done,
}

pub struct EmitStepped<V> {
    value: V,
    key: ResumeKey<V>,
}

impl<V> EmitStepped<V> {
    pub fn into_parts(self) -> (V, ResumeKey<V>) {
        (self.value, self.key)
    }
}

pub struct NeedContextStepped<V> {
    name: String,
    bindings: HashMap<String, V>,
}

impl<V> NeedContextStepped<V> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn bindings(&self) -> &HashMap<String, V> {
        &self.bindings
    }

    pub fn into_parts(self) -> (String, HashMap<String, V>) {
        (self.name, self.bindings)
    }

    pub fn into_key(self, value: Arc<V>) -> ResumeKey<V> {
        ResumeKey(ResumeKeyInner::Context(value))
    }
}

// ---------------------------------------------------------------------------
// Coroutine
// ---------------------------------------------------------------------------

pub struct Coroutine<V> {
    shared: Arc<Mutex<Signal<V>>>,
    fut: Option<Pin<Box<dyn Future<Output = ()> + Send>>>,
}

impl<V> Coroutine<V> {
    pub fn resume(&mut self, key: ResumeKey<V>) -> ResumeFuture<'_, V> {
        if let ResumeKeyInner::Context(arc) = key.0 {
            *self.shared.lock() = Signal::ContextReady(arc);
        }
        ResumeFuture { coroutine: self }
    }
}

// ---------------------------------------------------------------------------
// ResumeFuture — async resume
// ---------------------------------------------------------------------------

pub struct ResumeFuture<'a, V> {
    coroutine: &'a mut Coroutine<V>,
}

impl<V> Future for ResumeFuture<'_, V> {
    type Output = Stepped<V>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Stepped<V>> {
        let this = self.get_mut();
        let fut = match &mut this.coroutine.fut {
            Some(f) => f.as_mut(),
            None => return Poll::Ready(Stepped::Done),
        };

        let poll_result = fut.poll(cx);
        let mut signal = this.coroutine.shared.lock();

        match std::mem::replace(&mut *signal, Signal::Empty) {
            Signal::Yield(value) => {
                if poll_result.is_ready() {
                    this.coroutine.fut = None;
                }
                Poll::Ready(Stepped::Emit(EmitStepped {
                    value,
                    key: ResumeKey(ResumeKeyInner::Start),
                }))
            }
            Signal::NeedContext { name, bindings } => {
                Poll::Ready(Stepped::NeedContext(NeedContextStepped { name, bindings }))
            }
            other => {
                // put back unconsumed signal (e.g. ContextReady mid-processing)
                *signal = other;
                if poll_result.is_ready() {
                    this.coroutine.fut = None;
                    Poll::Ready(Stepped::Done)
                } else {
                    Poll::Pending
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

pub fn coroutine<V, F, Fut>(f: F) -> (Coroutine<V>, ResumeKey<V>)
where
    F: FnOnce(YieldHandle<V>) -> Fut,
    Fut: Future<Output = ()> + Send + 'static,
{
    let shared = Arc::new(Mutex::new(Signal::Empty));
    let handle = YieldHandle {
        shared: Arc::clone(&shared),
    };
    let fut = f(handle);
    (
        Coroutine {
            shared,
            fut: Some(Box::pin(fut)),
        },
        ResumeKey(ResumeKeyInner::Start),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    async fn step<V>(
        co: &mut Coroutine<V>,
        key: ResumeKey<V>,
    ) -> Stepped<V> {
        co.resume(key).await
    }

    #[tokio::test]
    async fn empty_coroutine() {
        let (mut co, key) = coroutine::<i32, _, _>(|_handle| async move {});
        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn single_yield() {
        let (mut co, key) = coroutine(|handle| async move {
            handle.yield_val(42).await;
        });

        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, 42);

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn multiple_yields() {
        let (mut co, key) = coroutine(|handle| async move {
            handle.yield_val(1).await;
            handle.yield_val(2).await;
            handle.yield_val(3).await;
        });

        let mut key = key;
        for expected in [1, 2, 3] {
            let Stepped::Emit(emit) = step(&mut co, key).await else {
                panic!("expected Emit({expected})");
            };
            let (value, next_key) = emit.into_parts();
            assert_eq!(value, expected);
            key = next_key;
        }

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn context_request() {
        let (mut co, key) = coroutine(|handle| async move {
            let ctx = handle.request_context("user".into()).await;
            handle.yield_val(format!("got: {ctx}")).await;
        });

        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext");
        };
        assert_eq!(need.name(), "user");
        assert!(need.bindings().is_empty());

        let key = need.into_key(Arc::new("alice".to_string()));
        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, "got: alice");

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn context_request_with_bindings() {
        let (mut co, key) = coroutine(|handle| async move {
            let mut bindings = HashMap::new();
            bindings.insert("role".into(), "admin".into());
            let ctx = handle
                .request_context_with("user".into(), bindings)
                .await;
            handle.yield_val(format!("got: {ctx}")).await;
        });

        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext");
        };
        assert_eq!(need.name(), "user");
        assert_eq!(need.bindings().get("role"), Some(&"admin".to_string()));

        let key = need.into_key(Arc::new("bob".to_string()));
        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, "got: bob");

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn interleaved_yield_and_context() {
        let (mut co, key) = coroutine(|handle| async move {
            handle.yield_val("start".to_string()).await;
            let name = handle.request_context("name".into()).await;
            handle.yield_val(format!("hello {name}")).await;
            let age = handle.request_context("age".into()).await;
            handle.yield_val(format!("{name} is {age}")).await;
        });

        // yield "start"
        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, "start");

        // need context "name"
        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext");
        };
        assert_eq!(need.name(), "name");
        let key = need.into_key(Arc::new("eve".to_string()));

        // yield "hello eve"
        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, "hello eve");

        // need context "age"
        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext");
        };
        assert_eq!(need.name(), "age");
        let key = need.into_key(Arc::new("30".to_string()));

        // yield "eve is 30"
        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, "eve is 30");

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn multiple_context_requests_in_sequence() {
        let (mut co, key) = coroutine(|handle| async move {
            let a = handle.request_context("a".into()).await;
            let b = handle.request_context("b".into()).await;
            handle.yield_val(format!("{a}+{b}")).await;
        });

        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext a");
        };
        assert_eq!(need.name(), "a");
        let key = need.into_key(Arc::new("1".to_string()));

        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext b");
        };
        assert_eq!(need.name(), "b");
        let key = need.into_key(Arc::new("2".to_string()));

        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, "1+2");

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn done_after_done_is_idempotent() {
        let (mut co, key) = coroutine::<i32, _, _>(|_handle| async move {});

        assert!(matches!(step(&mut co, key).await, Stepped::Done));

        let key = ResumeKey(ResumeKeyInner::Start);
        assert!(matches!(step(&mut co, key).await, Stepped::Done));

        let key = ResumeKey(ResumeKeyInner::Start);
        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn yield_handle_clone() {
        let (mut co, key) = coroutine(|handle| async move {
            let h2 = handle.clone();
            handle.yield_val(1).await;
            h2.yield_val(2).await;
        });

        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit(1)");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, 1);

        let Stepped::Emit(emit) = step(&mut co, key).await else {
            panic!("expected Emit(2)");
        };
        let (value, key) = emit.into_parts();
        assert_eq!(value, 2);

        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }

    #[tokio::test]
    async fn context_without_yield() {
        let (mut co, key) = coroutine::<String, _, _>(|handle| async move {
            let _ctx = handle.request_context("ignored".into()).await;
        });

        let Stepped::NeedContext(need) = step(&mut co, key).await else {
            panic!("expected NeedContext");
        };
        let key = need.into_key(Arc::new("value".to_string()));
        assert!(matches!(step(&mut co, key).await, Stepped::Done));
    }
}

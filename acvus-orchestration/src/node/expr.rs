use std::sync::Arc;

use acvus_interpreter::{ExternFnRegistry, Interpreter, RuntimeError, Stepped, Value};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use super::Node;

pub struct ExprNode {
    module: acvus_mir::ir::MirModule,
    extern_fns: ExternFnRegistry,
    interner: Interner,
}

impl ExprNode {
    pub fn new(
        module: acvus_mir::ir::MirModule,
        extern_fns: &ExternFnRegistry,
        interner: &Interner,
    ) -> Self {
        Self {
            module,
            extern_fns: extern_fns.clone(),
            interner: interner.clone(),
        }
    }
}

impl Node for ExprNode {
    fn spawn(
        &self,
        local: FxHashMap<Astr, Arc<Value>>,
    ) -> acvus_utils::Coroutine<Value, RuntimeError> {
        let interp = Interpreter::new(&self.interner, self.module.clone(), &self.extern_fns);
        let mut inner = interp.execute();
        acvus_utils::coroutine(move |handle| async move {
            loop {
                match inner.resume().await {
                    Stepped::Emit(value) => {
                        handle.yield_val(value).await;
                        return Ok(());
                    }
                    Stepped::NeedContext(request) => {
                        let name = request.name();
                        if let Some(arc) = local.get(&name) {
                            request.resolve(Arc::clone(arc));
                        } else {
                            let bindings = request.bindings().clone();
                            let value = if bindings.is_empty() {
                                handle.request_context(name).await
                            } else {
                                handle.request_context_with(name, bindings).await
                            };
                            request.resolve(value);
                        }
                    }
                    Stepped::Done => {
                        handle.yield_val(Value::Unit).await;
                        return Ok(());
                    }
                    Stepped::Error(e) => return Err(e),
                }
            }
        })
    }
}

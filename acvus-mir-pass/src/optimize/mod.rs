pub mod const_dedup;

use acvus_mir::ir::MirModule;

use crate::TransformPass;

pub struct ConstDedupPass;

impl TransformPass for ConstDedupPass {
    type Required<'a> = ();

    fn transform(&self, module: MirModule, _deps: ()) -> MirModule {
        const_dedup::dedup(module)
    }
}

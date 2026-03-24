pub mod const_dedup;
pub mod reg_color;

use crate::ir::MirModule;

use crate::pass::TransformPass;

pub struct ConstDedupPass;

impl TransformPass for ConstDedupPass {
    type Required<'a> = ();

    fn transform(&self, module: MirModule, _deps: ()) -> MirModule {
        const_dedup::dedup(module)
    }
}

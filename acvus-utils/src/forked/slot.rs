use serde::Serialize;

pub struct Slots<T> {
    data: Box<[Option<T>]>,
    head: usize,
}

impl<T> Slots<T> {
    pub fn new(size: usize) -> Self {
        Self {
            data: std::iter::repeat_with(|| None).take(size).collect(),
            head: 0,
        }
    }

    fn to_absolute_idx(&self, idx: usize) -> usize {
        idx % self.data.len()
    }

    pub fn head_idx(&self) -> usize {
        self.head
    }

    pub fn enqueue(&mut self, item: T) {
        self.data[self.to_absolute_idx(self.head)] = Some(item);
        self.head = self.head.wrapping_add(1);
    }
}

impl<T> Slots<T>
where
    T: Clone,
{
    pub fn consume(&self, pos: usize) -> Option<Result<T, Lagged>> {
        // Caller is trying to consume the next item.
        if self.head == pos {
            return None;
        }

        // Caller is trying to consume an old item.
        if self.head < pos || pos + self.data.len() < self.head {
            let behind_by = self.head.wrapping_sub(pos);
            let head = self.head;
            return Some(Err(Lagged { behind_by, head }));
        }

        let item = self.data[self.to_absolute_idx(pos)].clone();
        Some(Ok(item.unwrap()))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Lagged {
    pub behind_by: usize,
    pub head: usize,
}
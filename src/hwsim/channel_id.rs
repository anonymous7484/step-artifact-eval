#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChannelID(pub usize);

impl std::fmt::Display for ChannelID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ChannelID({})", self.0)
    }
}
use std::sync::atomic::{AtomicUsize, Ordering};

static CHANNEL_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl ChannelID {
    pub fn new() -> Self {
        let id = CHANNEL_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        ChannelID(id)
    }
}


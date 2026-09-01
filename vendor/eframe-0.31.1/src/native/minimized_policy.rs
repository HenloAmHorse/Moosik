//! Moosik local patch — when the minimized-viewport sleep may be waived.
//!
//! Upstream sleeps the event loop for 10 ms whenever the viewport it just
//! painted is minimized, so that a minimized window does not spin a core
//! (<https://github.com/emilk/egui/issues/325>). That is right for a lone
//! window and wrong for an *immediate* child viewport: an immediate viewport
//! is painted inside its parent's pass and its repaint requests are redirected
//! to the parent, so pacing the parent paces the child. Sleeping because the
//! parent is minimized therefore throttles a window the user is still looking
//! at.
//!
//! This module is the one place that decides whether that applies. Both the
//! glow and the wgpu backend call [`may_waive_minimized_sleep`] through a
//! small adapter over their own viewport map; the decision itself lives here
//! and is exercised directly by the tests at the bottom of the file.
//!
//! See `PATCH.md` at the root of this vendored crate.

use egui::ViewportId;

/// The per-viewport facts the policy needs, as the backends can supply them.
///
/// Every method is a plain lookup: the policy performs no allocation, takes no
/// lock and does no logging, because it runs once per painted frame.
pub trait Viewports {
    /// How many viewports are live. Bounds the ancestor walk.
    fn count(&self) -> usize;

    /// Call `f` for every live viewport id, stopping at the first `true`.
    fn any(&self, f: impl FnMut(ViewportId) -> bool) -> bool;

    /// The recorded parent of `id`, or `None` if there is no such viewport.
    fn parent_of(&self, id: ViewportId) -> Option<ViewportId>;

    /// Does `id` have a deferred ui callback?
    ///
    /// This is the authoritative immediate-vs-deferred signal in this version;
    /// see [`may_waive_minimized_sleep`] for why the recorded
    /// `ViewportClass` cannot be used.
    fn is_deferred(&self, id: ViewportId) -> bool;

    /// `Window::is_visible` for `id`, verbatim — `None` when unknown, and
    /// `None` when the viewport has no native window yet.
    fn is_visible(&self, id: ViewportId) -> Option<bool>;

    /// `Window::is_minimized` for `id`, verbatim, with the same `None` rules.
    fn is_minimized(&self, id: ViewportId) -> Option<bool>;
}

/// May the upstream minimized sleep be skipped for `current`?
///
/// True only when `current` has a descendant that is linked to it through an
/// unbroken chain of immediate viewports and whose native window is exactly
/// visible and not minimized. Anything unknown keeps the sleep.
///
/// Off Windows this is a constant `false`: the upstream mitigation exists for
/// macOS and is left alone. The whole module is additionally behind the
/// `moosik_windows_linked_viewport` feature, so an unpatched build never
/// reaches here at all.
pub fn may_waive_minimized_sleep<V: Viewports>(viewports: &V, current: ViewportId) -> bool {
    if !cfg!(target_os = "windows") {
        return false;
    }
    has_visible_linked_immediate_descendant(viewports, current)
}

/// The platform-independent half of [`may_waive_minimized_sleep`].
fn has_visible_linked_immediate_descendant<V: Viewports>(
    viewports: &V,
    ancestor: ViewportId,
) -> bool {
    viewports.any(|candidate| qualifies(viewports, candidate, ancestor))
}

/// Does this one viewport keep `ancestor` awake?
fn qualifies<V: Viewports>(viewports: &V, candidate: ViewportId, ancestor: ViewportId) -> bool {
    // The root and the viewport being painted are never their own dependants.
    // The root exclusion is load-bearing rather than defensive: the root has
    // no ui callback either, so `is_deferred` alone would let it through.
    if candidate == ancestor || candidate == ViewportId::ROOT {
        return false;
    }

    // A deferred viewport runs its own pass and is not starved by this sleep.
    if viewports.is_deferred(candidate) {
        return false;
    }

    // Strict, because an unknown native state must not cost us the sleep.
    if viewports.is_visible(candidate) != Some(true)
        || viewports.is_minimized(candidate) != Some(false)
    {
        return false;
    }

    is_all_immediate_descendant_of(viewports, candidate, ancestor)
}

/// Walk parent links from `node` up towards `ancestor`, requiring every
/// intermediate viewport on the way to be immediate as well.
///
/// A deferred viewport anywhere in the middle breaks the dependency: in
/// `root → deferred → immediate`, the immediate grandchild is painted inside
/// the *deferred* child's pass, which the root's sleep does not gate. The
/// root must keep sleeping.
///
/// The walk terminates on every degenerate shape — a missing entry, a
/// self-parent, reaching the root without a match, and (through the iteration
/// bound) a cycle or a chain longer than the map. It allocates nothing.
fn is_all_immediate_descendant_of<V: Viewports>(
    viewports: &V,
    mut node: ViewportId,
    ancestor: ViewportId,
) -> bool {
    for _ in 0..viewports.count() {
        let Some(parent) = viewports.parent_of(node) else {
            return false; // dangling parent link
        };

        // Checked before the root stop, so that `ancestor == ROOT` still
        // matches a legitimate descendant.
        if parent == ancestor {
            return true;
        }
        if parent == node {
            return false; // self-parent; the root is defined this way
        }
        if parent == ViewportId::ROOT {
            return false; // reached the top without meeting `ancestor`
        }

        // An intermediate link. It carries the pass onwards only if it is
        // itself immediate.
        if viewports.is_deferred(parent) {
            return false;
        }

        node = parent;
    }

    false // cycle, or a chain longer than the whole map
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-built viewport graph. Only the data is fabricated — every test
    /// calls the production policy above.
    struct Graph(Vec<Node>);

    struct Node {
        id: ViewportId,
        parent: ViewportId,
        deferred: bool,
        visible: Option<bool>,
        minimized: Option<bool>,
    }

    impl Viewports for Graph {
        fn count(&self) -> usize {
            self.0.len()
        }

        fn any(&self, mut f: impl FnMut(ViewportId) -> bool) -> bool {
            self.0.iter().any(|n| f(n.id))
        }

        fn parent_of(&self, id: ViewportId) -> Option<ViewportId> {
            self.get(id).map(|n| n.parent)
        }

        fn is_deferred(&self, id: ViewportId) -> bool {
            self.get(id).is_some_and(|n| n.deferred)
        }

        fn is_visible(&self, id: ViewportId) -> Option<bool> {
            self.get(id).and_then(|n| n.visible)
        }

        fn is_minimized(&self, id: ViewportId) -> Option<bool> {
            self.get(id).and_then(|n| n.minimized)
        }
    }

    impl Graph {
        fn get(&self, id: ViewportId) -> Option<&Node> {
            self.0.iter().find(|n| n.id == id)
        }
    }

    fn id(name: &str) -> ViewportId {
        ViewportId::from_hash_of(name)
    }

    /// A viewport with a native window that is on screen.
    fn shown(name: &str, parent: ViewportId) -> Node {
        Node {
            id: id(name),
            parent,
            deferred: false,
            visible: Some(true),
            minimized: Some(false),
        }
    }

    fn root() -> Node {
        Node {
            id: ViewportId::ROOT,
            parent: ViewportId::ROOT,
            deferred: false,
            visible: Some(true),
            minimized: Some(true),
        }
    }

    fn waives(graph: &Graph, current: ViewportId) -> bool {
        has_visible_linked_immediate_descendant(graph, current)
    }

    #[test]
    fn direct_visible_immediate_child_qualifies() {
        let g = Graph(vec![root(), shown("spectrum", ViewportId::ROOT)]);
        assert!(waives(&g, ViewportId::ROOT));
    }

    #[test]
    fn an_all_immediate_chain_qualifies() {
        let g = Graph(vec![
            root(),
            shown("mid", ViewportId::ROOT),
            shown("leaf", id("mid")),
        ]);
        assert!(waives(&g, ViewportId::ROOT));
    }

    #[test]
    fn a_deferred_intermediate_breaks_the_chain() {
        let mut mid = shown("mid", ViewportId::ROOT);
        mid.deferred = true;
        let g = Graph(vec![root(), mid, shown("leaf", id("mid"))]);
        assert!(
            !waives(&g, ViewportId::ROOT),
            "an immediate viewport below a deferred one is paced by that \
             deferred viewport's pass, not by the root's"
        );
    }

    #[test]
    fn a_deferred_leaf_does_not_qualify() {
        let mut leaf = shown("spectrum", ViewportId::ROOT);
        leaf.deferred = true;
        let g = Graph(vec![root(), leaf]);
        assert!(!waives(&g, ViewportId::ROOT));
    }

    // `ViewportClass` is deliberately absent from `Viewports`, so a stale
    // class cannot reach the policy at all. That the backends really do read
    // the callback rather than the class is asserted in their own adapter
    // tests, in `glow_integration` and `wgpu_integration`.

    #[test]
    fn the_root_never_qualifies_as_its_own_dependant() {
        let g = Graph(vec![root()]);
        assert!(!waives(&g, ViewportId::ROOT));
    }

    #[test]
    fn the_current_viewport_never_qualifies() {
        let g = Graph(vec![root(), shown("spectrum", ViewportId::ROOT)]);
        assert!(!waives(&g, id("spectrum")));
    }

    #[test]
    fn a_sibling_does_not_qualify() {
        let g = Graph(vec![
            root(),
            shown("a", ViewportId::ROOT),
            shown("b", ViewportId::ROOT),
        ]);
        assert!(!waives(&g, id("a")), "b is a sibling of a, not below it");
    }

    #[test]
    fn a_hidden_child_does_not_qualify() {
        let mut child = shown("spectrum", ViewportId::ROOT);
        child.visible = Some(false);
        let g = Graph(vec![root(), child]);
        assert!(!waives(&g, ViewportId::ROOT));
    }

    #[test]
    fn a_minimized_child_does_not_qualify() {
        let mut child = shown("spectrum", ViewportId::ROOT);
        child.minimized = Some(true);
        let g = Graph(vec![root(), child]);
        assert!(!waives(&g, ViewportId::ROOT));
    }

    #[test]
    fn unknown_visibility_keeps_the_sleep() {
        let mut child = shown("spectrum", ViewportId::ROOT);
        child.visible = None;
        let g = Graph(vec![root(), child]);
        assert!(!waives(&g, ViewportId::ROOT));
    }

    #[test]
    fn unknown_minimized_state_keeps_the_sleep() {
        let mut child = shown("spectrum", ViewportId::ROOT);
        child.minimized = None;
        let g = Graph(vec![root(), child]);
        assert!(!waives(&g, ViewportId::ROOT));
    }

    #[test]
    fn a_missing_parent_terminates() {
        let g = Graph(vec![root(), shown("orphan", id("gone"))]);
        assert!(!waives(&g, ViewportId::ROOT));
    }

    #[test]
    fn a_self_parent_terminates() {
        let g = Graph(vec![root(), shown("loop", id("loop"))]);
        assert!(!waives(&g, ViewportId::ROOT));
    }

    #[test]
    fn a_cycle_without_the_ancestor_terminates() {
        let g = Graph(vec![
            root(),
            shown("a", id("b")),
            shown("b", id("c")),
            shown("c", id("a")),
        ]);
        assert!(!waives(&g, ViewportId::ROOT));
    }

    #[test]
    fn a_long_chain_is_bounded_by_the_map() {
        // 512 links, none of which reach `absent`. The walk is capped at
        // `count()` steps per candidate, so this returns rather than hanging.
        let mut nodes = vec![root()];
        let mut parent = ViewportId::ROOT;
        for i in 0..512 {
            let name = format!("link{i}");
            nodes.push(shown(&name, parent));
            parent = id(&name);
        }
        let g = Graph(nodes);
        assert!(!waives(&g, id("absent")));
    }

    #[test]
    fn the_platform_gate_is_part_of_the_decision() {
        let g = Graph(vec![root(), shown("spectrum", ViewportId::ROOT)]);
        assert_eq!(
            may_waive_minimized_sleep(&g, ViewportId::ROOT),
            cfg!(target_os = "windows"),
            "the waiver is Windows-only; elsewhere the upstream sleep stands"
        );
    }
}

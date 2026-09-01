mod app_icon;
mod epi_integration;
mod event_loop_context;
pub mod run;

/// File storage which can be used by native backends.
#[cfg(feature = "persistence")]
pub mod file_storage;

pub(crate) mod winit_integration;

/// Moosik local patch: shared minimized-sleep policy. See `PATCH.md`.
#[cfg(all(
    feature = "moosik_windows_linked_viewport",
    any(feature = "glow", feature = "wgpu")
))]
mod minimized_policy;

#[cfg(feature = "glow")]
mod glow_integration;

#[cfg(feature = "wgpu")]
mod wgpu_integration;

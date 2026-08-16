//! GPU compute for the superlet transform.
//!
//! The transform is dominated by FFT convolution: 1024 bars × several wavelets ×
//! tens of thousands of frames, with no dependency between bars. That is close
//! to the ideal shape for a GPU, and it is *offline* work — a pre-process, not
//! playback — so it can never contend with the audio path.
//!
//! This module is deliberately built bottom-up and validated against the CPU
//! before it is wired to anything. The hazard with GPU numerics is not a crash,
//! it is plausible-looking wrong output; the existing CPU routes are the oracle
//! that catches it.
//!
//! wgpu rather than CUDA: it is already in the dependency tree (eframe pulls it
//! in) and it runs on Vulkan, DX12 and Metal, so this works on AMD, Intel and
//! Apple rather than only NVIDIA.

/// A batched power-of-two complex FFT on the GPU.
///
/// Stockham autosort, radix 2: it reads the two halves of its input and writes
/// scattered, which leaves the output in natural order with no separate
/// bit-reversal pass. One dispatch per stage, ping-ponging between two buffers.
pub struct GpuFft {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    conv: wgpu::ComputePipeline,
    conv_layout: wgpu::BindGroupLayout,
    extract: wgpu::ComputePipeline,
    extract_layout: wgpu::BindGroupLayout,
    /// Largest storage buffer this adapter will bind, in bytes.
    max_buffer: u64,
}

/// Complex sample, matching WGSL's `vec2<f32>`.
pub type C32 = [f32; 2];

const WORKGROUP: u32 = 64;

/// Split a workgroup count across two dispatch dimensions.
///
/// A dimension caps at 65535 workgroups; the largest transforms here need
/// hundreds of thousands. Returns `(x, y, span)`, where `span` is how many
/// threads one full x-row covers so the shader can rebuild its linear index.
fn dispatch_2d(groups: u32) -> (u32, u32, u32) {
    const MAX: u32 = 65_535;
    if groups <= MAX { return (groups, 1, groups * WORKGROUP); }
    let x = MAX;
    (x, groups.div_ceil(x), x * WORKGROUP)
}

/// Uniform buffer offsets must be aligned; 256 is the universal requirement.
const UNIFORM_STRIDE: u64 = 256;

const SHADER: &str = r#"
struct Params {
    n: u32,
    stage: u32,
    batch: u32,
    inverse: u32,
    // Threads per dispatch row. A dimension caps at 65535 workgroups, which the
    // largest sizes here blow through, so work is spread over x and y.
    span: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read>       src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             p: Params;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_n = p.n >> 1u;
    let total  = half_n * p.batch;
    let t = gid.x + gid.y * p.span;
    if (t >= total) { return; }

    // Each thread owns one butterfly of one transform in the batch.
    let b = t / half_n;
    let i = t % half_n;
    let base = b * p.n;

    // Stockham indexing: `j` is the position inside the current sub-transform,
    // `k` the start of the doubled output block.
    let m = 1u << p.stage;
    let j = i & (m - 1u);
    let k = (i >> p.stage) << (p.stage + 1u);

    let a  = src[base + i];
    let bv = src[base + i + half_n];

    let ang = -6.28318530717958647692 * f32(j) / f32(2u * m);
    var w = vec2<f32>(cos(ang), sin(ang));
    if (p.inverse == 1u) { w.y = -w.y; }

    let tw = cmul(w, bv);
    dst[base + k + j]     = a + tw;
    dst[base + k + j + m] = a - tw;
}
"#;

/// Pointwise product of one signal block with one kernel spectrum, plus the
/// magnitude extraction that turns inverse-transform output into frames.
///
/// Both stay on the device on purpose. The isolated-FFT benchmark lost most of
/// its advantage to upload and readback; keeping the signal spectrum resident
/// and returning only the finished `kernels × frames` grid removes almost all
/// of that traffic.
const CONV_SHADER: &str = r#"
struct Conv {
    n: u32,
    blocks: u32,
    kernels: u32,
    span: u32,
};

@group(0) @binding(0) var<storage, read>       sig: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       ker: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> prod: array<vec2<f32>>;
@group(0) @binding(3) var<uniform>             c: Conv;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = c.n * c.blocks * c.kernels;
    let t = gid.x + gid.y * c.span;
    if (t >= total) { return; }
    let i = t % c.n;                 // position within the transform
    let rest = t / c.n;
    let b = rest % c.blocks;         // which signal block
    let w = rest / c.blocks;         // which kernel
    let a = sig[b * c.n + i];
    let k = ker[w * c.n + i];
    prod[t] = vec2<f32>(a.x * k.x - a.y * k.y, a.x * k.y + a.y * k.x);
}
"#;

/// Reads finished convolution output and writes one magnitude per frame.
///
/// Consecutive blocks overlap, so more than one could supply a given frame.
/// Each frame picks the *last* block that covers it — the same one the CPU
/// route ends up with, since there it is simply overwritten in block order.
/// Choosing explicitly makes it deterministic and removes any write race.
const EXTRACT_SHADER: &str = r#"
struct Ex {
    n: u32,
    stride: u32,
    blocks: u32,
    kernels: u32,
    frames: u32,
    hop: u32,
    n_sig: u32,
    span: u32,
};

// Per kernel: x = taps, y = half-width.
@group(0) @binding(0) var<storage, read>       prod: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       kmeta: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> mags: array<f32>;
@group(0) @binding(3) var<uniform>             e: Ex;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = e.kernels * e.frames;
    let t = gid.x + gid.y * e.span;
    if (t >= total) { return; }
    let w  = t / e.frames;
    let fi = t % e.frames;

    let k    = kmeta[w].x;
    let half = kmeta[w].y;

    // Position of this frame's output in the linear convolution.
    let m = fi * e.hop + half;
    if (m < k - 1u || m >= e.n_sig) { mags[t] = -1.0; return; }

    // Last block whose valid region [base + k - 1, base + n) covers m.
    let b = min((m - (k - 1u)) / e.stride, e.blocks - 1u);
    let base = b * e.stride;
    if (m < base + k - 1u || m >= base + e.n) { mags[t] = -1.0; return; }

    let c = prod[(w * e.blocks + b) * e.n + (m - base)];
    // 2/n: the inverse transform is unscaled, and the wavelet pair carries a
    // factor of two, exactly as on the CPU route.
    mags[t] = 2.0 * sqrt(c.x * c.x + c.y * c.y) / f32(e.n);
}
"#;

/// A signal already uploaded and forward-transformed on the device.
///
/// Split out from the convolution because it must happen *once per block size*,
/// not once per batch of kernels. Folding it into `convolve` meant the caller's
/// chunking silently re-uploaded and re-transformed the whole signal for every
/// handful of bars — precisely the redundancy the CPU route had just been
/// restructured to remove, reintroduced on the GPU.
pub struct GpuSignal {
    buf: wgpu::Buffer,
    n: usize,
    stride: usize,
    blocks: usize,
    n_sig: usize,
}

/// One kernel's taps, as the pipeline needs them.
pub struct GpuKernel<'a> {
    pub re: &'a [f32],
    pub im: &'a [f32],
    /// Half-width; the kernel spans `centre ± half`.
    pub half: usize,
}

impl GpuFft {
    /// Acquire a compute device, or `None` if this machine cannot provide one.
    ///
    /// A dedicated instance rather than borrowing eframe's: analysis then works
    /// regardless of which render backend eframe chose, and stays structurally
    /// separate from anything touching the window. `None` is a normal outcome —
    /// every caller must keep the CPU path.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        ))?;

        // Ask for as much storage-buffer headroom as the adapter allows: the
        // largest transform this transform uses is 2^21 complex samples, which
        // is 16 MB, and batching several at once is the entire point.
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter_limits.max_buffer_size;
        limits.max_compute_workgroups_per_dimension =
            adapter_limits.max_compute_workgroups_per_dimension;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("moosik-compute"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )).ok()?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aslt-fft"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let storage = |read_only: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aslt-fft-bind"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: storage(true), count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: storage(false), count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    // Dynamic, so every stage's parameters can be written once
                    // up front and selected by offset instead of rewriting the
                    // buffer between dispatches.
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aslt-fft-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aslt-fft-pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Convolution and extraction share the same shape: N storage buffers
        // plus one uniform, all compute-visible.
        let make = |src: &str, label: &str, n_storage: usize, rw: &[usize]|
            -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
            let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..n_storage).map(|i| {
                wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: !rw.contains(&i) },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            }).collect();
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: n_storage as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label), entries: &entries,
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label), bind_group_layouts: &[&bgl], push_constant_ranges: &[],
            });
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: Some(&pl), module: &m,
                entry_point: Some("main"), compilation_options: Default::default(), cache: None,
            });
            (p, bgl)
        };
        let (conv, conv_layout) = make(CONV_SHADER, "aslt-conv", 3, &[2]);
        let (extract, extract_layout) = make(EXTRACT_SHADER, "aslt-extract", 3, &[2]);

        Some(Self {
            device, queue, pipeline, layout, conv, conv_layout, extract, extract_layout,
            max_buffer: adapter_limits.max_storage_buffer_binding_size as u64,
        })
    }

    /// Record one batched FFT over an existing buffer pair, returning `true`
    /// when the result landed in `b` rather than `a`.
    ///
    /// Split out so the pipeline can transform buffers that never leave the
    /// device — the whole point of the resident design.
    fn encode_fft(
        &self, enc: &mut wgpu::CommandEncoder, a: &wgpu::Buffer, b: &wgpu::Buffer,
        n: usize, batch: usize, inverse: bool,
    ) -> bool {
        let stages = n.trailing_zeros();
        if stages == 0 || batch == 0 { return false; }

        let params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft-params"),
            size: UNIFORM_STRIDE * stages as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (gx, gy, span) = dispatch_2d(((n / 2 * batch) as u32).div_ceil(WORKGROUP));
        for s in 0..stages {
            let p: [u32; 8] = [n as u32, s, batch as u32, u32::from(inverse), span, 0, 0, 0];
            self.queue.write_buffer(&params, UNIFORM_STRIDE * s as u64, bytemuck::cast_slice(&p));
        }
        let bind = |src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fft-bind"), layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &params, offset: 0,
                            size: std::num::NonZeroU64::new(32),
                        }),
                    },
                ],
            })
        };
        let a_to_b = bind(a, b);
        let b_to_a = bind(b, a);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        for s in 0..stages {
            pass.set_bind_group(0, if s % 2 == 0 { &a_to_b } else { &b_to_a },
                                &[UNIFORM_STRIDE as u32 * s]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        stages % 2 == 1
    }

    fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size: bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn uniform(&self, label: &str, words: &[u32]) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (words.len() * 4).max(16) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(words));
        buf
    }

    /// Every kernel of one block size convolved against the signal, returning
    /// `kernels × frames` magnitudes.
    ///
    /// The signal is transformed once and stays resident; each kernel then costs
    /// a multiply, an inverse and a magnitude pass, all on the device. Only the
    /// finished grid is read back, which is what the isolated-FFT benchmark was
    /// paying dearly for.
    ///
    /// A frame whose window runs past the signal comes back as `-1.0`, meaning
    /// "the caller must compute this one directly" — the edge frames renormalise
    /// over the taps that survive, which a convolution cannot express.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_signal(
        &self, signal: &[f32], n: usize, stride: usize,
    ) -> Result<GpuSignal, String> {
        if !n.is_power_of_two() || n == 0 { return Err("n must be a power of two".into()); }
        if stride == 0 || stride > n { return Err("stride out of range".into()); }

        let blocks = signal.len().div_ceil(stride).max(1);
        let cplx = std::mem::size_of::<C32>() as u64;
        let sig_bytes = blocks as u64 * n as u64 * cplx;
        if sig_bytes > self.max_buffer {
            return Err(format!("signal blocks need {sig_bytes} bytes"));
        }
        let mut host = vec![[0.0f32; 2]; blocks * n];
        for b in 0..blocks {
            let base = b * stride;
            for i in 0..n {
                host[b * n + i] = [signal.get(base + i).copied().unwrap_or(0.0), 0.0];
            }
        }
        let sig_a = self.storage("sig-a", sig_bytes);
        let sig_b = self.storage("sig-b", sig_bytes);
        self.queue.write_buffer(&sig_a, 0, bytemuck::cast_slice(&host));
        let mut enc = self.device.create_command_encoder(&Default::default());
        let in_b = self.encode_fft(&mut enc, &sig_a, &sig_b, n, blocks, false);
        self.queue.submit(Some(enc.finish()));
        Ok(GpuSignal {
            buf: if in_b { sig_b } else { sig_a },
            n, stride, blocks, n_sig: signal.len(),
        })
    }

    /// Convolve a prepared signal against a batch of kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn convolve_with(
        &self, prepared: &GpuSignal, signal_len: usize, hop: usize, frames: usize,
        kernels: &[GpuKernel],
    ) -> Result<Vec<Vec<f32>>, String> {
        if kernels.is_empty() || frames == 0 { return Ok(Vec::new()); }
        let (n, stride, blocks) = (prepared.n, prepared.stride, prepared.blocks);
        let cplx = std::mem::size_of::<C32>() as u64;
        let sig = &prepared.buf;
        let _ = signal_len;

        // How many kernels fit at once: the product buffer is the big one.
        let per_kernel = blocks as u64 * n as u64 * cplx;
        let budget = self.max_buffer.min(256 << 20);
        let chunk = ((budget / per_kernel.max(1)) as usize).clamp(1, kernels.len());

        let mut out = vec![Vec::new(); kernels.len()];
        for start in (0..kernels.len()).step_by(chunk) {
            let end = (start + chunk).min(kernels.len());
            let group = &kernels[start..end];
            let w = group.len();

            // Kernels, time-reversed and zero-padded, transformed on device.
            let mut khost = vec![[0.0f32; 2]; w * n];
            for (gi, k) in group.iter().enumerate() {
                let taps = k.re.len().min(n);
                for i in 0..taps {
                    khost[gi * n + i] = [k.re[taps - 1 - i], k.im[taps - 1 - i]];
                }
            }
            let ker_a = self.storage("ker-a", (w * n) as u64 * cplx);
            let ker_b = self.storage("ker-b", (w * n) as u64 * cplx);
            self.queue.write_buffer(&ker_a, 0, bytemuck::cast_slice(&khost));

            let prod_a = self.storage("prod-a", w as u64 * per_kernel);
            let prod_b = self.storage("prod-b", w as u64 * per_kernel);
            let outbuf = self.storage("mag", (w * frames * 4) as u64);
            let meta: Vec<u32> = group.iter()
                .flat_map(|k| [k.re.len() as u32, k.half as u32])
                .collect();
            let metabuf = self.storage("meta", (meta.len() * 4).max(8) as u64);
            self.queue.write_buffer(&metabuf, 0, bytemuck::cast_slice(&meta));

            let mut enc = self.device.create_command_encoder(&Default::default());
            let k_in_b = self.encode_fft(&mut enc, &ker_a, &ker_b, n, w, false);
            let ker = if k_in_b { &ker_b } else { &ker_a };

            let (cgx, cgy, cspan) =
                dispatch_2d(((n * blocks * w) as u32).div_ceil(WORKGROUP));
            let cu = self.uniform("conv-u", &[n as u32, blocks as u32, w as u32, cspan]);
            let cbind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("conv-bind"), layout: &self.conv_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: sig.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ker.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: prod_a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: cu.as_entire_binding() },
                ],
            });
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.conv);
                pass.set_bind_group(0, &cbind, &[]);
                pass.dispatch_workgroups(cgx, cgy, 1);
            }

            let p_in_b = self.encode_fft(&mut enc, &prod_a, &prod_b, n, blocks * w, true);
            let prod = if p_in_b { &prod_b } else { &prod_a };

            let (egx, egy, espan) = dispatch_2d(((w * frames) as u32).div_ceil(WORKGROUP));
            let eu = self.uniform("ex-u", &[
                n as u32, stride as u32, blocks as u32, w as u32,
                frames as u32, hop as u32, prepared.n_sig as u32, espan,
            ]);
            let ebind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ex-bind"), layout: &self.extract_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: prod.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: metabuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: outbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: eu.as_entire_binding() },
                ],
            });
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.extract);
                pass.set_bind_group(0, &ebind, &[]);
                pass.dispatch_workgroups(egx, egy, 1);
            }

            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mag-read"), size: (w * frames * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            enc.copy_buffer_to_buffer(&outbuf, 0, &staging, 0, (w * frames * 4) as u64);
            self.queue.submit(Some(enc.finish()));

            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
            {
                let view = slice.get_mapped_range();
                let mags: &[f32] = bytemuck::cast_slice(&view);
                for (gi, col) in out[start..end].iter_mut().enumerate() {
                    *col = mags[gi * frames..(gi + 1) * frames].to_vec();
                }
            }
            staging.unmap();
        }
        Ok(out)
    }

    /// The process-wide compute device, created once on first use.
    ///
    /// Acquiring a device costs driver initialisation and a shader compile, and
    /// nothing about it is per-analysis, so a run that opened its own would pay
    /// that every track. Sharing it also keeps concurrent callers — several
    /// tests, or an analysis alongside a probe — from opening several devices
    /// against one adapter at the same time.
    ///
    /// `None` means this machine has no usable adapter, which is a normal
    /// outcome that every caller must handle by staying on the CPU.
    pub fn shared() -> Option<&'static GpuFft> {
        static SHARED: std::sync::OnceLock<Option<GpuFft>> = std::sync::OnceLock::new();
        SHARED.get_or_init(GpuFft::new).as_ref()
    }

    /// Adapter name, for reporting which device the work ran on.
    /// Adapter name, for display and for keying per-machine calibration.
    ///
    /// Cached, because this builds an `Instance` and enumerates adapters — tens
    /// of milliseconds. The debug overlay asks once a frame, and an uncached
    /// version of this took the spectrum from 60 fps to about 32 whenever the
    /// panel was open.
    pub fn describe() -> Option<String> {
        static NAME: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        NAME.get_or_init(Self::describe_uncached).clone()
    }

    fn describe_uncached() -> Option<String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        ))?;
        let info = adapter.get_info();
        Some(format!("{} ({:?}, {:?})", info.name, info.device_type, info.backend))
    }

    /// How many transforms of length `n` fit in one batch on this device.
    pub fn max_batch(&self, n: usize) -> usize {
        let per = (n * std::mem::size_of::<C32>()) as u64;
        if per == 0 { return 0; }
        (self.max_buffer / per).max(1) as usize
    }

    /// Transform `data` in place: `batch` contiguous blocks of `n` complex
    /// samples each. `n` must be a power of two.
    ///
    /// The inverse is *unscaled* here — like `rustfft`, which leaves the 1/n to
    /// the caller. Matching that convention keeps the CPU route the oracle
    /// rather than something that has to be adjusted for.
    pub fn fft_batch(&self, data: &mut [C32], n: usize, inverse: bool) -> Result<(), String> {
        if n == 0 || !n.is_power_of_two() { return Err(format!("n must be a power of two, got {n}")); }
        if !data.len().is_multiple_of(n) {
            return Err("data length is not a multiple of n".into());
        }
        let batch = data.len() / n;
        if batch == 0 { return Ok(()); }
        let bytes = std::mem::size_of_val(data) as u64;
        if bytes > self.max_buffer {
            return Err(format!("{bytes} bytes exceeds this device's {} limit", self.max_buffer));
        }

        let stages = n.trailing_zeros();
        // A single-point transform is the identity, and there is no butterfly
        // to run — bail before dispatching zero stages against live buffers.
        if stages == 0 { return Ok(()); }

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let buf_a = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft-a"), size: bytes, usage, mapped_at_creation: false,
        });
        let buf_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft-b"), size: bytes, usage, mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(data));

        let mut enc = self.device.create_command_encoder(&Default::default());
        let in_b = self.encode_fft(&mut enc, &buf_a, &buf_b, n, batch, inverse);

        // After an odd number of stages the result sits in B, after an even
        // number in A.
        let result = if in_b { &buf_b } else { &buf_a };
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft-read"), size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(result, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        data.copy_from_slice(bytemuck::cast_slice(&slice.get_mapped_range()));
        staging.unmap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::num_complex::Complex;

    /// Skips rather than fails when there is no usable adapter — CI and
    /// headless boxes are a normal case, and the CPU path covers them.
    fn gpu() -> Option<&'static GpuFft> { GpuFft::shared() }

    fn cpu_fft(data: &[C32], n: usize, inverse: bool) -> Vec<C32> {
        let mut planner = rustfft::FftPlanner::<f32>::new();
        let fft = if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
        let mut buf: Vec<Complex<f32>> =
            data.iter().map(|c| Complex { re: c[0], im: c[1] }).collect();
        for chunk in buf.chunks_mut(n) { fft.process(chunk); }
        buf.iter().map(|c| [c.re, c.im]).collect()
    }

    fn noise(len: usize, seed: u32) -> Vec<C32> {
        let mut s = seed | 1;
        (0..len).map(|_| {
            let mut r = || {
                s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            };
            [r(), r()]
        }).collect()
    }

    /// Worst relative error against rustfft, normalised by the transform's own
    /// peak so it means the same thing at every size.
    fn worst_err(got: &[C32], want: &[C32]) -> f32 {
        let peak = want.iter()
            .map(|c| c[0].abs().max(c[1].abs()))
            .fold(0.0f32, f32::max)
            .max(1e-20);
        got.iter().zip(want).map(|(g, w)| {
            ((g[0] - w[0]).abs()).max((g[1] - w[1]).abs()) / peak
        }).fold(0.0f32, f32::max)
    }

    /// The whole point of building this bottom-up: the GPU transform has to
    /// agree with the one already in use before anything is wired to it.
    ///
    /// `cargo test --bin moosik -- --nocapture gpu_fft_matches_cpu`
    #[test]
    fn gpu_fft_matches_cpu() {
        let Some(g) = gpu() else {
            println!("no GPU adapter — skipping (CPU path covers this machine)");
            return;
        };
        println!("adapter: {}", GpuFft::describe().unwrap_or_default());

        // Every block size the transform actually uses is a power of two from
        // 4096 (the smallest kernel that routes through an FFT) upward.
        for log2n in [1u32, 2, 3, 8, 12, 14, 16] {
            let n = 1usize << log2n;
            for &inverse in &[false, true] {
                let src = noise(n, 0x5EED ^ log2n);
                let want = cpu_fft(&src, n, inverse);
                let mut got = src.clone();
                g.fft_batch(&mut got, n, inverse).expect("gpu fft failed");
                let e = worst_err(&got, &want);
                println!("n={n:<6} inverse={inverse:<5} worst rel err {e:.2e}");
                // f32 FFT accumulates error with depth; 1e-4 is loose enough
                // for 2^16 and far tighter than any wrong indexing would give.
                assert!(e < 1e-4, "n={n} inverse={inverse}: {e:.3e}");
            }
        }
    }

    /// Batching is where the speed comes from, so a batched result must equal
    /// the same transforms done one at a time — not merely look plausible.
    #[test]
    fn batching_does_not_change_the_answer() {
        let Some(g) = gpu() else { return };
        let (n, batch) = (1usize << 10, 17); // deliberately not a round batch
        let src = noise(n * batch, 0xC0FFEE);
        let want = cpu_fft(&src, n, false);
        let mut got = src.clone();
        g.fft_batch(&mut got, n, false).expect("gpu fft failed");
        assert!(worst_err(&got, &want) < 1e-4);
    }

    /// Forward then inverse must return the input, scaled by n.
    #[test]
    fn round_trip_returns_the_input() {
        let Some(g) = gpu() else { return };
        let n = 1usize << 12;
        let src = noise(n, 0xBEEF);
        let mut buf = src.clone();
        g.fft_batch(&mut buf, n, false).unwrap();
        g.fft_batch(&mut buf, n, true).unwrap();
        let inv = 1.0 / n as f32;
        let worst = src.iter().zip(&buf)
            .map(|(a, b)| (a[0] - b[0] * inv).abs().max((a[1] - b[1] * inv).abs()))
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "round trip drifted by {worst:.3e}");
    }

    /// The whole pipeline against the CPU route it would replace.
    ///
    /// This is the one that matters. The FFT agreeing proves nothing about the
    /// multiply, the block ownership rule or the frame indexing — and those are
    /// exactly where a mistake produces a plausible spectrum rather than an
    /// error.
    #[test]
    fn convolve_matches_the_cpu_route() {
        use crate::spectrum::aslt::{fft_block_len, frame_count, hop_for_fps, Morlet, SignalBlocks};
        let Some(g) = gpu() else {
            println!("no GPU adapter — skipping");
            return;
        };
        let sr = 16_000u32;
        let secs = 3.0f32;
        let n_len = (secs * sr as f32) as usize;
        let tau = std::f32::consts::TAU;
        let sig: Vec<f32> = (0..n_len)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.6 * (tau * 60.0 * t).sin() + 0.4 * (tau * 205.0 * t).sin()
            })
            .collect();
        let hop = hop_for_fps(sr, 60.0);
        let frames = frame_count(sig.len(), hop);

        // Several kernels of one block size, which is exactly how the pipeline
        // is used: one signal transform, many kernels sharing it.
        // A spread of frequencies and cycle counts, then whichever block size
        // the most of them land on — the pipeline exists to serve many kernels
        // from one signal transform, so testing it with one proves little.
        let mut mors: Vec<Morlet> = Vec::new();
        for f in [55.0f32, 60.0, 75.0, 110.0, 150.0, 205.0, 300.0] {
            for c in [30.0f32, 40.0, 55.0, 70.0] {
                mors.push(Morlet::new(f, c, sr as f32));
            }
        }
        let mut by_size: std::collections::HashMap<usize, Vec<&Morlet>> = Default::default();
        for m in &mors {
            by_size.entry(fft_block_len(m.taps())).or_default().push(m);
        }
        let (&n_block, keep) = by_size.iter()
            .max_by_key(|(_, v)| v.len())
            .expect("no kernels built");
        assert!(keep.len() >= 3, "test needs several kernels sharing a block size");

        let stride = n_block / 2;
        let blocks = SignalBlocks::build(&sig, n_block);
        let kernels: Vec<GpuKernel> = keep.iter()
            .map(|m| GpuKernel { re: m.re_taps(), im: m.im_taps(), half: m.half_width() })
            .collect();
        let prep = g.prepare_signal(&sig, n_block, stride).expect("prepare failed");
        let got = g.convolve_with(&prep, sig.len(), hop, frames, &kernels)
            .expect("gpu convolve failed");

        let mut worst = 0.0f32;
        let mut compared = 0usize;
        for (gi, m) in keep.iter().enumerate() {
            let want = m.magnitudes_via_shared_for_test(&sig, &blocks, hop, frames)
                .expect("cpu shared route declined");
            let peak = want.iter().cloned().fold(0.0f32, f32::max).max(1e-12);
            for (fi, (&a, &b)) in got[gi].iter().zip(want.iter()).enumerate() {
                // -1 marks an edge frame the GPU deliberately leaves to the
                // caller; those are not part of this comparison.
                if a < 0.0 { continue; }
                let _ = fi;
                worst = worst.max((a - b).abs() / peak);
                compared += 1;
            }
        }
        println!("n={n_block} kernels={} compared {compared} frames, worst {worst:.2e}",
                 keep.len());
        assert!(compared > frames / 2, "too few interior frames compared");
        assert!(worst < 1e-4, "gpu pipeline differs from CPU by {worst:.3e}");
    }

    /// The decisive comparison: the resident pipeline against the CPU route it
    /// would replace, on a realistic workload, with the CPU using every core.
    ///
    /// `cargo test --release --bin moosik -- --ignored --nocapture pipeline_vs_cpu`
    #[test]
    #[ignore = "benchmark — run explicitly"]
    fn pipeline_vs_cpu() {
        use crate::spectrum::aslt::{fft_block_len, frame_count, hop_for_fps, Morlet, SignalBlocks};
        use rayon::prelude::*;
        use std::time::Instant;
        let Some(g) = gpu() else { println!("no GPU adapter"); return };
        println!("\nadapter: {}", GpuFft::describe().unwrap_or_default());

        let sr = 48_000u32;
        let secs = 20.0f32;
        let n_len = (secs * sr as f32) as usize;
        let sig = noise_real(n_len);
        let hop = hop_for_fps(sr, 180.0);
        let frames = frame_count(sig.len(), hop);

        println!("{:>8} {:>8} {:>11} {:>11} {:>9}",
                 "n", "kernels", "cpu ms", "gpu ms", "speedup");
        // One row per block size, with enough kernels on each to look like a
        // real run: a whole preset puts hundreds of kernels on some sizes.
        for (target_n, base_f) in [(1usize << 15, 300.0f32), (1 << 17, 80.0), (1 << 19, 22.0)] {
            let mut mors: Vec<Morlet> = Vec::new();
            let mut c = 20.0f32;
            while mors.len() < 32 && c < 4000.0 {
                let m = Morlet::new(base_f, c, sr as f32);
                if fft_block_len(m.taps()) == target_n { mors.push(m); }
                c *= 1.05;
            }
            if mors.len() < 4 { continue; }
            let stride = target_n / 2;

            let blocks = SignalBlocks::build(&sig, target_n);
            let t0 = Instant::now();
            let _: Vec<Vec<f32>> = mors.par_iter()
                .map(|m| m.magnitudes_via_shared_for_test(&sig, &blocks, hop, frames)
                          .unwrap_or_default())
                .collect();
            let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let kernels: Vec<GpuKernel> = mors.iter()
                .map(|m| GpuKernel { re: m.re_taps(), im: m.im_taps(), half: m.half_width() })
                .collect();
            // Warm, so driver and pipeline setup is not charged to the run.
            let prep = g.prepare_signal(&sig, target_n, stride).expect("prepare failed");
            let _ = g.convolve_with(&prep, sig.len(), hop, frames, &kernels[..1]);
            let t0 = Instant::now();
            let got = g.convolve_with(&prep, sig.len(), hop, frames, &kernels);
            let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if got.is_err() { println!("{target_n:>8} {:>8} gpu declined: {:?}", mors.len(), got.err()); continue; }

            println!("{target_n:>8} {:>8} {cpu_ms:>11.1} {gpu_ms:>11.1} {:>8.2}x",
                     mors.len(), cpu_ms / gpu_ms.max(1e-9));
        }
        println!("cpu uses every core via rayon; gpu includes upload and readback.");
    }

    fn noise_real(len: usize) -> Vec<f32> {
        let mut s = 0x1234_5678u32;
        (0..len).map(|_| {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        }).collect()
    }

    /// Is the GPU actually faster at the sizes this transform uses — including
    /// upload and readback, which is where naive GPU ports lose?
    ///
    /// `cargo test --release --bin moosik -- --ignored --nocapture gpu_vs_cpu_bench`
    #[test]
    #[ignore = "benchmark — run explicitly"]
    fn gpu_vs_cpu_bench() {
        use std::time::Instant;
        let Some(g) = gpu() else { println!("no GPU adapter"); return };
        println!("\nadapter: {}", GpuFft::describe().unwrap_or_default());
        println!("{:>8} {:>7} {:>10} {:>10} {:>9}", "n", "batch", "cpu ms", "gpu ms", "speedup");

        // Block sizes the transform actually uses: `fft_block_len` is
        // (2·taps).next_power_of_two(), so 2^12 upward.
        for (log2n, batch) in [(12u32, 64usize), (14, 32), (16, 16), (18, 8), (20, 4)] {
            let n = 1usize << log2n;
            let src = noise(n * batch, 0xA11CE ^ log2n);

            let t0 = Instant::now();
            let _ = cpu_fft(&src, n, false);
            let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // Warm the pipeline so the first dispatch's driver work is not
            // charged to the measurement.
            let mut warm = src.clone();
            let _ = g.fft_batch(&mut warm, n, false);

            let mut got = src.clone();
            let t0 = Instant::now();
            g.fft_batch(&mut got, n, false).expect("gpu fft failed");
            let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

            println!("{n:>8} {batch:>7} {cpu_ms:>10.2} {gpu_ms:>10.2} {:>8.1}x",
                     cpu_ms / gpu_ms.max(1e-9));
        }
        println!("gpu ms includes upload, dispatch and readback.");
    }

    #[test]
    fn bad_sizes_are_refused_rather_than_silently_wrong() {
        let Some(g) = gpu() else { return };
        let mut d = vec![[0.0f32; 2]; 12];
        assert!(g.fft_batch(&mut d, 6, false).is_err(), "6 is not a power of two");
        assert!(g.fft_batch(&mut d, 8, false).is_err(), "12 is not a multiple of 8");
        // A length-1 transform is the identity, not an error.
        let mut one = vec![[1.0f32, 2.0]];
        assert!(g.fft_batch(&mut one, 1, false).is_ok());
        assert_eq!(one[0], [1.0, 2.0]);
    }

    /// The batch limit has to leave room for at least one transform, or the
    /// caller has no legal size to ask for.
    #[test]
    fn max_batch_is_at_least_one() {
        let Some(g) = gpu() else { return };
        for log2n in [12u32, 16, 21] {
            assert!(g.max_batch(1 << log2n) >= 1);
        }
    }
}

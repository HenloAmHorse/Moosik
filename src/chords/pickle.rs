//! Reads the float tensors out of a legacy (pre-1.6, non-zip) `torch.save`
//! file without running anything: a small pickle machine that knows only the
//! opcodes and the three globals such a state dict uses, and refuses the rest.
//!
//! Layout: four pickles (magic number, protocol, system info, the object),
//! a pickle listing the storage keys, then each storage in that order as a
//! little-endian i64 element count and the raw little-endian f32 data.

use std::collections::HashMap;

#[derive(Clone, Debug)]
enum V {
    /// None, and the booleans and floats nothing here reads.
    None,
    Int(i64),
    Str(String),
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict(Vec<(V, V)>),
    Global(&'static str),
    Storage(String),
    Tensor { storage: String, offset: usize, size: Vec<usize>, stride: Vec<usize> },
    Mark,
}

/// A tensor: its shape and its values, row-major.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

const GLOBALS: [&str; 3] = ["collections OrderedDict", "torch._utils _rebuild_tensor_v2", "torch FloatStorage"];

struct Machine<'a> {
    b: &'a [u8],
    at: usize,
    stack: Vec<V>,
    memo: HashMap<u32, V>,
}

impl<'a> Machine<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.at..self.at + n).ok_or("the file ends inside a pickle")?;
        self.at += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn pop(&mut self) -> Result<V, String> {
        self.stack.pop().ok_or_else(|| "pickle stack underflow".to_string())
    }
    fn pop_mark(&mut self) -> Result<Vec<V>, String> {
        let at = self.stack.iter().rposition(|v| matches!(v, V::Mark)).ok_or("no mark")?;
        let items = self.stack.split_off(at + 1);
        self.stack.pop();
        Ok(items)
    }
    fn line(&mut self) -> Result<String, String> {
        let end = self.b[self.at..].iter().position(|&c| c == b'\n').ok_or("unterminated line")?;
        let s = std::str::from_utf8(&self.b[self.at..self.at + end]).map_err(|_| "bad text")?.to_string();
        self.at += end + 1;
        Ok(s)
    }

    /// Runs one pickle to its STOP and returns the object.
    fn run(&mut self) -> Result<V, String> {
        self.stack.clear();
        self.memo.clear();
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    self.u8()?; // PROTO
                }
                b'.' => return self.pop(), // STOP
                b'(' => self.stack.push(V::Mark),
                b'}' => self.stack.push(V::Dict(Vec::new())),
                b']' => self.stack.push(V::List(Vec::new())),
                b')' => self.stack.push(V::Tuple(Vec::new())),
                b'N' => self.stack.push(V::None),
                0x88 | 0x89 => self.stack.push(V::None),
                b'J' => {
                    let v = self.u32()? as i32;
                    self.stack.push(V::Int(v as i64))
                }
                b'K' => {
                    let v = self.u8()?;
                    self.stack.push(V::Int(v as i64))
                }
                b'M' => {
                    let v = self.u16()?;
                    self.stack.push(V::Int(v as i64))
                }
                0x8a => {
                    // LONG1: little-endian two's complement, up to 8 bytes here
                    let n = self.u8()? as usize;
                    let bytes = self.take(n)?;
                    if n > 8 {
                        // Only the optimiser's counters get this big; never read.
                        self.stack.push(V::None);
                        continue;
                    }
                    let mut v: i64 = 0;
                    for (i, &x) in bytes.iter().enumerate() {
                        v |= (x as i64) << (8 * i);
                    }
                    if n > 0 && n < 8 && bytes[n - 1] & 0x80 != 0 {
                        v -= 1i64 << (8 * n);
                    }
                    self.stack.push(V::Int(v))
                }
                b'G' => {
                    self.take(8)?;
                    self.stack.push(V::None)
                }
                b'X' => {
                    let n = self.u32()? as usize;
                    let s = std::str::from_utf8(self.take(n)?).map_err(|_| "bad text")?.to_string();
                    self.stack.push(V::Str(s))
                }
                0x8c => {
                    let n = self.u8()? as usize;
                    let s = std::str::from_utf8(self.take(n)?).map_err(|_| "bad text")?.to_string();
                    self.stack.push(V::Str(s))
                }
                b'q' => {
                    let k = self.u8()? as u32;
                    let v = self.stack.last().ok_or("empty stack")?.clone();
                    self.memo.insert(k, v);
                }
                b'r' => {
                    let k = self.u32()?;
                    let v = self.stack.last().ok_or("empty stack")?.clone();
                    self.memo.insert(k, v);
                }
                b'h' => {
                    let k = self.u8()? as u32;
                    self.stack.push(self.memo.get(&k).ok_or("unknown memo")?.clone())
                }
                b'j' => {
                    let k = self.u32()?;
                    self.stack.push(self.memo.get(&k).ok_or("unknown memo")?.clone())
                }
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(V::Tuple(items))
                }
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(V::Tuple(vec![a]))
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(V::Tuple(vec![a, b]))
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(V::Tuple(vec![a, b, c]))
                }
                b'c' => {
                    let name = format!("{} {}", self.line()?, self.line()?);
                    let g = GLOBALS.iter().find(|g| **g == name).ok_or_else(|| format!("refused global {name}"))?;
                    self.stack.push(V::Global(g))
                }
                b'Q' => {
                    // BINPERSID: ('storage', type, key, location, numel)
                    let pid = self.pop()?;
                    let V::Tuple(t) = pid else { return Err("bad persistent id".into()) };
                    match (t.first(), t.get(1), t.get(2)) {
                        (Some(V::Str(s)), Some(V::Global("torch FloatStorage")), Some(V::Str(key))) if s == "storage" => {
                            self.stack.push(V::Storage(key.clone()))
                        }
                        _ => return Err("only float storages are read".into()),
                    }
                }
                b'R' => {
                    let args = self.pop()?;
                    let f = self.pop()?;
                    let V::Tuple(a) = args else { return Err("bad call".into()) };
                    match f {
                        V::Global("collections OrderedDict") => self.stack.push(V::Dict(Vec::new())),
                        V::Global("torch._utils _rebuild_tensor_v2") => {
                            let ints = |v: &V| -> Result<Vec<usize>, String> {
                                match v {
                                    V::Tuple(xs) => xs
                                        .iter()
                                        .map(|x| match x {
                                            V::Int(i) if *i >= 0 => Ok(*i as usize),
                                            _ => Err("bad tensor shape".to_string()),
                                        })
                                        .collect(),
                                    _ => Err("bad tensor shape".into()),
                                }
                            };
                            match (a.first(), a.get(1)) {
                                (Some(V::Storage(s)), Some(V::Int(off))) if *off >= 0 => self.stack.push(V::Tensor {
                                    storage: s.clone(),
                                    offset: *off as usize,
                                    size: ints(a.get(2).ok_or("no size")?)?,
                                    stride: ints(a.get(3).ok_or("no stride")?)?,
                                }),
                                _ => return Err("bad tensor".into()),
                            }
                        }
                        _ => return Err("refused call".into()),
                    }
                }
                b'b' => {
                    self.pop()?; // BUILD: state for an OrderedDict (metadata); not needed
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    match self.stack.last_mut() {
                        Some(V::Dict(d)) => d.push((k, v)),
                        _ => return Err("SETITEM without a dict".into()),
                    }
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    let Some(V::Dict(d)) = self.stack.last_mut() else { return Err("SETITEMS without a dict".into()) };
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        d.push((k, v));
                    }
                }
                b'a' => {
                    let v = self.pop()?;
                    match self.stack.last_mut() {
                        Some(V::List(l)) => l.push(v),
                        _ => return Err("APPEND without a list".into()),
                    }
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    match self.stack.last_mut() {
                        Some(V::List(l)) => l.extend(items),
                        _ => return Err("APPENDS without a list".into()),
                    }
                }
                _ => return Err(format!("refused pickle opcode 0x{op:02x}")),
            }
        }
    }
}

/// The tensors of `obj[field]` (a state dict), by name.
pub fn read_state_dict(bytes: &[u8], field: &str) -> Result<HashMap<String, Tensor>, String> {
    let mut m = Machine { b: bytes, at: 0, stack: Vec::new(), memo: HashMap::new() };
    for _ in 0..3 {
        m.run()?; // magic number, protocol, system info
    }
    let obj = m.run()?;
    let V::List(keys) = m.run()? else { return Err("no storage list".into()) };
    // The storages follow, in the listed order.
    let mut storages: HashMap<String, (usize, usize)> = HashMap::new();
    for k in keys {
        let V::Str(k) = k else { return Err("bad storage key".into()) };
        let n = i64::from_le_bytes(m.take(8)?.try_into().unwrap());
        if n < 0 {
            return Err("bad storage size".into());
        }
        let start = m.at;
        m.take(n as usize * 4)?;
        storages.insert(k, (start, n as usize));
    }
    let V::Dict(top) = obj else { return Err("not a dict".into()) };
    let net = top
        .into_iter()
        .find_map(|(k, v)| matches!(&k, V::Str(s) if s == field).then_some(v))
        .ok_or_else(|| format!("no `{field}` in the file"))?;
    let V::Dict(net) = net else { return Err(format!("`{field}` is not a dict")) };
    let mut out = HashMap::new();
    for (k, v) in net {
        let (V::Str(name), V::Tensor { storage, offset, size, stride }) = (k, v) else { continue };
        let &(start, len) = storages.get(&storage).ok_or("missing storage")?;
        // Only contiguous row-major tensors are expected in a state dict.
        let mut want = 1;
        for (d, s) in size.iter().zip(&stride).rev() {
            if *d > 1 && *s != want {
                return Err(format!("{name} is not contiguous"));
            }
            want *= d;
        }
        let count: usize = size.iter().product();
        if offset + count > len {
            return Err(format!("{name} runs past its storage"));
        }
        let raw = &bytes[start + offset * 4..start + (offset + count) * 4];
        let data = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        out.insert(name, Tensor { shape: size, data });
    }
    Ok(out)
}

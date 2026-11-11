# PyTorch code

Read this when the code touches `torch`. The rules in `scientific.md` about shapes, units and
seeded randomness apply unchanged. This file covers what torch adds: state that lives on the
wrong device or in the global RNG, graphs that are longer or shorter than intended, and syncs
that make a step loop crawl.

## Randomness

- Every stochastic function takes a `generator: torch.Generator`. Never call
  `torch.manual_seed`, `torch.seed` or an unseeded `torch.rand*` in library code. The global RNG
  is process state that any import or test can disturb.
- Data streams use CPU generators and move the finished block to the training device. A
  generator is bound to one device, and CPU generation is what makes a data stream reproducible
  across CPU, MPS and CUDA.
- Module initialization is the one place torch offers no generator argument. Seed it inside
  `torch.random.fork_rng()` with the seed passed in, so that the global RNG is restored
  afterwards.
- Distinct streams, training data, reference data, evaluation data, weight initialization,
  noise, get distinct generators derived from one run seed. Reusing one generator across streams
  makes any change to one stream reorder the others.
- Bitwise reproducibility holds only within a device and a torch version. Record both wherever a
  seed is recorded.

## Devices and dtypes

- No `.cuda()` or `.cpu()` as a way to pick a device, and no hard-coded device strings. The
  device arrives from configuration once. Tensors are created with `device=` or moved with
  `.to(device)` at the boundary.
- Create tensors on the device they are used on. A tensor built on CPU inside a step loop and
  moved every iteration is a hidden transfer.
- `float32` is the default and is stated in docstrings only when a function deviates. Do not let
  `float64` in by way of NumPy. `torch.as_tensor(np_array)` inherits the NumPy dtype, so convert
  explicitly at the boundary.
- Functions accept and return `torch.Tensor`. Convert to NumPy only in analysis and plotting
  code, once, with `.detach().cpu().numpy()`.
- Not every op has an MPS kernel. When a shim exists for that reason, the comment says which op
  and which backend, and the fast path stays the default on CPU and CUDA.

## Autograd and state

- State carried across optimizer steps, a recurrent state, a replay buffer, a running estimate,
  is detached explicitly at the window boundary. An undetached carry makes the graph grow across
  windows until memory runs out, or worse, backpropagates through the entire history.
- A buffer that must contribute gradients holds live tensors. A buffer that stores detached
  tensors trains nothing through them. Decide which one it is and say so in the docstring.
- No in-place ops, `x += ...`, `x[mask] = ...`, `relu_`, on tensors that require grad or that
  another tensor's graph still references. Prefer functional forms. The error, when it comes,
  points at the backward pass and not at the culprit.
- Evaluation and analysis run under `torch.no_grad()` or `torch.inference_mode()`, and
  `model.eval()` is set when the model has mode-dependent layers. Restore training mode
  afterwards.
- Parameters and buffers that must survive a checkpoint are registered on the module with
  `nn.Parameter` or `register_buffer`, not held as plain attributes.

## Step loops

- Avoid `.item()`, `float(tensor)`, `bool(tensor)` and `if tensor:` inside a step loop. Each is a
  host sync. Accumulate on the device and sync once per logging interval.
- Prefer one batched op over a Python loop over the batch. A Python loop over time steps is
  normal for a recurrent model. A Python loop over agents or units is a bug.
- Preallocate outputs with `torch.empty` and fill, or collect in a list and `torch.stack` once.
  Do not grow a tensor with repeated `torch.cat`.
- `torch.compile` and mixed precision are opt-in from configuration, never on by default in a
  library function.

## Shapes

- State the axis convention once per module and keep it, batch-major or time-major, and say
  which `batch_first` an `nn.RNN`-style module uses.
- Use `einsum` or explicit `unsqueeze` over broadcasting that works because two sizes happen to
  match. A batch of 64 against 64 units broadcasts without complaint and without meaning.
- Reduce over named axes, `x.sum(dim=-1)`, never a bare `x.sum()` unless a scalar is the point.
- Test shapes at the boundary: a batch of one, a single time step, an empty batch where the code
  claims to support it.

## Tests

- Gradient checks for any custom loss or operator: `torch.autograd.gradcheck` in `float64` on a
  tiny input.
- Determinism: two calls with equal generators give equal outputs, and a changed seed changes
  them.
- Device-agnostic tests run on CPU only. A CUDA or MPS test is marked and skipped when the
  backend is absent.
- Numerical comparison with `torch.testing.assert_close` and an explicit tolerance, never `==`.

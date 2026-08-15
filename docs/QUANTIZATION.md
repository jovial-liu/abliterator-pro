# Quantization contract

“FP4” is not one checkpoint format.

| CLI value | Backend/representation | Editable in place? | Intended workflow |
|---|---|---:|---|
| `bnb-fp4` | bitsandbytes FP4 codebook and block scales | No | discover → dense edit → bnb quantize |
| `bnb-nf4` | bitsandbytes normal-float codebook | No | discover → dense edit → bnb quantize |
| `nvfp4` | NVIDIA E2M1 with NVFP4 scaling | Only with a verified master/repack path | dense edit → FP-Quant/ModelOpt export |
| `mxfp4` | E2M1 with MX block scaling | Only with a verified master/repack path | dense edit → FP-Quant export |

Packed tensors are byte/code indices plus scale state, not ordinary matrices. Arithmetic on their
`.data` would corrupt both logical values and quantizer invariants. The bench therefore allows
activation capture and residual hooks on quantized models but blocks permanent packed-weight
projection.

The preferred persistent workflow is:

```text
quantized or dense runtime
  → discover and validate direction
canonical BF16/FP16 checkpoint
  → chunked FP32 projection
edited dense safetensors checkpoint
  → backend-specific quantization/export
reload quantized checkpoint
  → repeat behavioral tests and direction-leakage measurement
```

Install the backend explicitly:

```bash
pip install -e '.[bnb]'       # bitsandbytes FP4/NF4
pip install -e '.[fpquant]'   # fp_quant + a tested Transformers lower bound
```

Transformers documents whole-model bitsandbytes `dequantize()`, but it requires enough memory for
the expanded model and cannot undo quantization loss. `abpro apply --dequantize` exposes that
explicit fallback.

## Hardware notes

- bitsandbytes FP4/NF4 supports substantially broader hardware than native NVFP4.
- Native accelerated NVFP4 inference requires NVIDIA Blackwell-class hardware.
- Native FP-Quant kernels require QuTLASS and Blackwell-generation NVIDIA hardware.
- FP-Quant pseudoquantization still requires the `fp_quant`/Triton runtime and a supported Linux
  GPU/XPU environment; it avoids QuTLASS and provides no speed or storage benefit. It is not a
  CPU/macOS fallback.
- DeepSeek and Kimi checkpoints may ship in FP8 or INT4. Those formats are not reinterpreted as
  FP4; packed/scaled modules and nested quantization metadata fail closed. Persistent conversion
  starts from a compatible canonical dense source.

Export verifies that the runtime actually contains the requested backend’s packed modules, that
the saved config names the requested format, and that non-empty weight payloads were written. The
quantization manifest records the dense checkpoint content hash, ablation-manifest hash, backend
mode (including pseudoquantization), source dtype, package versions, and verification evidence.

Primary references:

- <https://huggingface.co/docs/transformers/quantization/bitsandbytes>
- <https://huggingface.co/docs/transformers/en/quantization/fp_quant>
- <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq>

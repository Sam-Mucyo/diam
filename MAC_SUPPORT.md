# Dia Text-to-Speech for Mac

This document provides information about running Dia on macOS, particularly on Apple Silicon Macs (M1/M2/M3).

## Mac-Optimized Version

The standard version of Dia may encounter compatibility issues with Metal Performance Shaders (MPS) on Apple Silicon Macs. We've created a Mac-optimized version that addresses these issues and provides better performance.

### Features

- Custom attention implementation compatible with MPS
- Selective CPU fallbacks for problematic operations
- Support for Grouped Query Attention (GQA) with MPS
- Improved error handling and diagnostics

## Running Dia on Mac

To run the Mac-optimized version:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run the Mac-optimized version
python run_dia_mac.py
```

This will start a Gradio web interface accessible at http://127.0.0.1:7861 in your browser.

## Performance Notes

- Performance on Mac will be slower than on dedicated NVIDIA GPUs
- Generation speed is approximately 14-15 tokens per second on Apple Silicon
- The realtime factor is about 0.17x (takes ~6x longer than real-time)
- Shorter inputs will generate faster
- You can adjust the `max_tokens` parameter to balance quality and speed

## Technical Details

The Mac optimization addresses several issues:

1. **MPS Compatibility**: PyTorch's `scaled_dot_product_attention` function has compatibility issues with MPS. We've implemented a custom version that works with MPS.

2. **Dimension Mismatch**: The original implementation had issues with tensor dimension mismatches when running on MPS. Our version properly handles different numbers of heads between queries and keys/values.

3. **Precision**: We use float32 precision for better compatibility with MPS.

## Troubleshooting

If you encounter issues:

- Try reducing the `max_tokens` parameter
- Use shorter text inputs
- If all else fails, you can force CPU mode with `python run_dia_mac.py --device cpu`

## Files

- `run_dia_mac.py`: Mac-optimized Gradio interface
- `dia/mps_compat.py`: MPS compatibility patches and custom attention implementation

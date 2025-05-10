#!/usr/bin/env python3
"""
Dia Text-to-Speech for Mac
A Mac-optimized version of the Dia TTS application with MPS compatibility fixes
"""

import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from dia.model import Dia
from dia.mps_compat import apply_mps_patches


# --- Parse Arguments ---
parser = argparse.ArgumentParser(description="Mac-optimized Dia TTS Interface")
parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
parser.add_argument("--float32", action="store_true", default=True, help="Use float32 precision for better MPS compatibility")

args = parser.parse_args()


# --- Setup Device ---
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    # Enable MPS fallbacks
    torch.backends.mps.allow_ops_in_cpu_mode = True
    print("Using MPS (Metal Performance Shaders) with compatibility patches")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# --- Load Model ---
print("Loading Dia model...")
try:
    # Use float32 for MPS compatibility
    compute_dtype = "float32" if (device.type == "mps" or args.float32) else "float16"
    
    # Apply MPS patches if using MPS
    if device.type == "mps":
        apply_mps_patches()
    
    # Load the model
    model = Dia.from_pretrained(
        "nari-labs/Dia-1.6B", 
        compute_dtype=compute_dtype, 
        device=device
    )
except Exception as e:
    print(f"Error loading Dia model: {e}")
    raise


def run_inference(
    text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
):
    """
    Runs Dia inference with error handling and MPS compatibility
    """
    global model, device

    if not text_input or text_input.isspace():
        raise gr.Error("Text input cannot be empty.")

    temp_audio_prompt_path = None
    output_audio = (44100, np.zeros(1, dtype=np.float32))

    try:
        # Process audio prompt if provided
        prompt_path_for_generate = None
        if audio_prompt_input is not None:
            sr, audio_data = audio_prompt_input
            if audio_data is None or audio_data.size == 0 or audio_data.max() == 0:
                gr.Warning("Audio prompt seems empty or silent, ignoring prompt.")
            else:
                # Save prompt audio to a temporary WAV file
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
                    temp_audio_prompt_path = f_audio.name
                    
                    # Basic audio preprocessing
                    if np.issubdtype(audio_data.dtype, np.integer):
                        max_val = np.iinfo(audio_data.dtype).max
                        audio_data = audio_data.astype(np.float32) / max_val
                    elif not np.issubdtype(audio_data.dtype, np.floating):
                        gr.Warning(f"Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion.")
                        try:
                            audio_data = audio_data.astype(np.float32)
                        except Exception as conv_e:
                            raise gr.Error(f"Failed to convert audio prompt to float32: {conv_e}")

                    # Ensure mono
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 2:  # Assume (2, N)
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] == 2:  # Assume (N, 2)
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            gr.Warning(f"Audio prompt has unexpected shape {audio_data.shape}, taking first channel.")
                            if audio_data.shape[0] < 10:  # Likely (C, N) format
                                audio_data = audio_data[0]
                            else:  # Likely (N, C) format
                                audio_data = audio_data[:, 0]

                    # Write to temporary file
                    sf.write(temp_audio_prompt_path, audio_data, sr)
                    prompt_path_for_generate = temp_audio_prompt_path

        # --- Run Inference ---
        start_time = time.time()
        
        try:
            print(f"Generating audio from text: {text_input[:50]}...")
            output_audio_np = model.generate(
                text_input,
                max_tokens=max_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                use_torch_compile=False,
                audio_prompt=prompt_path_for_generate,
                verbose=True,
            )
        except RuntimeError as e:
            error_msg = str(e)
            print(f"Error during generation: {error_msg}")
            
            if "incompatible dimensions" in error_msg and device.type == "mps":
                print("MPS compatibility error detected. Falling back to CPU for this operation.")
                # Try again with CPU
                cpu_device = torch.device("cpu")
                original_device = model.device
                model.device = cpu_device
                
                output_audio_np = model.generate(
                    text_input,
                    max_tokens=max_tokens,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=cfg_filter_top_k,
                    use_torch_compile=False,
                    audio_prompt=prompt_path_for_generate,
                    verbose=True,
                )
                
                # Restore device
                model.device = original_device
            else:
                # Re-raise if it's not the specific MPS error we're handling
                raise

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")

        # Process generated audio
        if output_audio_np is not None:
            output_sr = 44100
            original_len = len(output_audio_np)
            
            # Apply speed adjustment
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(original_len / speed_factor)
            
            if target_len != original_len and target_len > 0:
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                output_audio = (output_sr, resampled_audio_np.astype(np.float32))
                print(f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed.")
            else:
                output_audio = (output_sr, output_audio_np.astype(np.float32))
        else:
            print("Warning: No audio was generated.")
            output_audio = (44100, np.zeros(1, dtype=np.float32))

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error during inference: {e}")

    finally:
        # Clean up temporary files
        if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
            try:
                Path(temp_audio_prompt_path).unlink()
            except Exception as e:
                print(f"Warning: Failed to delete temporary audio prompt file: {e}")

    return output_audio


# --- Gradio Interface ---
with gr.Blocks(title="Dia Text-to-Speech for Mac") as demo:
    gr.Markdown(
        """
        # Dia Text-to-Speech for Mac
        
        Generate realistic dialogue from text with Dia, a 1.6B parameter text-to-speech model by Nari Labs.
        This version includes Mac-specific optimizations for Apple Silicon.
        
        ## Tips:
        - Always use speaker tags: `[S1]` and `[S2]`
        - For non-verbal sounds, use tags like `(laughs)`, `(coughs)`, etc.
        - For voice cloning, upload audio and provide its transcript before your generation text
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text Input",
                placeholder="[S1] Hello there! [S2] Hi, how are you? [S1] I'm doing great! (laughs)",
                lines=10,
            )
            audio_prompt_input = gr.Audio(
                label="Audio Prompt (Optional)",
                type="numpy",
                sources=["upload"],
            )

            with gr.Accordion("Advanced Settings", open=False):
                max_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=512,
                    maximum=4096,
                    value=3072,
                    step=128,
                    info="Maximum number of tokens to generate (longer = more audio).",
                )
                cfg_scale = gr.Slider(
                    label="CFG Scale",
                    minimum=1.0,
                    maximum=5.0,
                    value=3.0,
                    step=0.1,
                    info="How closely to follow the input text (higher = more faithful).",
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.5,
                    maximum=2.0,
                    value=1.3,
                    step=0.05,
                    info="Lower values make the output more deterministic, higher values increase randomness.",
                )
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    minimum=0.80,
                    maximum=1.0,
                    value=0.95,
                    step=0.01,
                    info="Filters vocabulary to the most likely tokens cumulatively reaching probability P.",
                )
                cfg_filter_top_k = gr.Slider(
                    label="CFG Filter Top K",
                    minimum=15,
                    maximum=50,
                    value=30,
                    step=1,
                    info="Top k filter for CFG guidance.",
                )
                speed_factor_slider = gr.Slider(
                    label="Speed Factor",
                    minimum=0.8,
                    maximum=1.0,
                    value=0.94,
                    step=0.02,
                    info="Adjusts the speed of the generated audio (1.0 = original speed).",
                )

            run_button = gr.Button("Generate Audio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="numpy",
                autoplay=False,
            )
            
            status_output = gr.Markdown(
                "Ready to generate audio. Enter text and click 'Generate Audio'."
            )

    # Link button click to function
    run_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_input,
            max_tokens,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor_slider,
        ],
        outputs=[audio_output],
        api_name="generate_audio",
    )

    # Add examples
    example_prompt_path = "./example_prompt.mp3"
    examples_list = [
        [
            "[S1] Hello! Welcome to Dia text-to-speech running on Mac. [S2] This version includes special optimizations for Apple Silicon. [S1] Try it out and let us know what you think!",
            None,
            2048,
            3.0,
            1.3,
            0.95,
            35,
            0.94,
        ],
        [
            "[S1] Open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] I'm biased, but I think we clearly won. \n[S2] Hard to disagree. (laughs)",
            example_prompt_path if Path(example_prompt_path).exists() else None,
            2048,
            3.0,
            1.3,
            0.95,
            35,
            0.94,
        ],
    ]

    if examples_list:
        gr.Examples(
            examples=examples_list,
            inputs=[
                text_input,
                audio_prompt_input,
                max_tokens,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                speed_factor_slider,
            ],
            outputs=[audio_output],
            fn=run_inference,
            cache_examples=False,
            label="Examples (Click to Run)",
        )

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Dia for Mac interface...")
    demo.launch(share=args.share)

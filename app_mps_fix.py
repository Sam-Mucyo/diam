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


# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Dia TTS")
parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
parser.add_argument("--mps_fallback", action="store_true", default=True, help="Enable MPS fallbacks for problematic operations")

args = parser.parse_args()


# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
# Simplified MPS check for broader compatibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) with fallbacks for problematic operations")
    # Enable fallbacks for MPS
    torch.backends.mps.allow_ops_in_cpu_mode = True
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load Dia model and config
print("Loading Dia model...")
try:
    # Use the function from inference.py with float32 for better MPS compatibility
    model = Dia.from_pretrained(
        "nari-labs/Dia-1.6B", 
        compute_dtype="float32" if device.type == "mps" else "float16", 
        device=device
    )
except Exception as e:
    print(f"Error loading Dia model: {e}")
    raise


def run_inference(
    text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
):
    """
    Runs Dia inference using the globally loaded model and provided inputs.
    Uses temporary files for text and audio prompt compatibility with inference.generate.
    """
    global model, device  # Access global model, config, device

    if not text_input or text_input.isspace():
        raise gr.Error("Text input cannot be empty.")

    temp_txt_file_path = None
    temp_audio_prompt_path = None
    output_audio = (44100, np.zeros(1, dtype=np.float32))

    try:
        prompt_path_for_generate = None
        if audio_prompt_input is not None:
            sr, audio_data = audio_prompt_input
            # Check if audio_data is valid
            if audio_data is None or audio_data.size == 0 or audio_data.max() == 0:  # Check for silence/empty
                gr.Warning("Audio prompt seems empty or silent, ignoring prompt.")
            else:
                # Save prompt audio to a temporary WAV file
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
                    temp_audio_prompt_path = f_audio.name  # Store path for cleanup

                    # Basic audio preprocessing for consistency
                    # Convert to float32 in [-1, 1] range if integer type
                    if np.issubdtype(audio_data.dtype, np.integer):
                        max_val = np.iinfo(audio_data.dtype).max
                        audio_data = audio_data.astype(np.float32) / max_val
                    elif not np.issubdtype(audio_data.dtype, np.floating):
                        gr.Warning(f"Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion.")
                        # Attempt conversion, might fail for complex types
                        try:
                            audio_data = audio_data.astype(np.float32)
                        except Exception as conv_e:
                            raise gr.Error(f"Failed to convert audio prompt to float32: {conv_e}")

                    # Ensure mono (average channels if stereo)
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 2:  # Assume (2, N)
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] == 2:  # Assume (N, 2)
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            gr.Warning(
                                f"Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis."
                            )
                            # Take first dimension/channel
                            if audio_data.shape[0] < 10:  # Likely (C, N) format
                                audio_data = audio_data[0]
                            else:  # Likely (N, C) format
                                audio_data = audio_data[:, 0]

                    # Write to temporary file
                    sf.write(temp_audio_prompt_path, audio_data, sr)
                    prompt_path_for_generate = temp_audio_prompt_path

        # --- Run Inference ---
        start_time = time.time()
        
        # Add error handling for MPS
        try:
            output_audio_np = model.generate(
                text_input,
                max_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,  # Pass the value here
                use_torch_compile=False,  # Keep False for Gradio stability
                audio_prompt=prompt_path_for_generate,
            )
        except RuntimeError as e:
            if "incompatible dimensions" in str(e) and device.type == "mps":
                print("MPS compatibility error detected. Falling back to CPU for this operation.")
                # Try again with CPU
                cpu_device = torch.device("cpu")
                model.device = cpu_device
                output_audio_np = model.generate(
                    text_input,
                    max_tokens=max_new_tokens,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=cfg_filter_top_k,
                    use_torch_compile=False,
                    audio_prompt=prompt_path_for_generate,
                )
                # Restore device
                model.device = device
            else:
                # Re-raise if it's not the specific MPS error we're handling
                raise

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")

        # 4. Convert Codes to Audio
        if output_audio_np is not None:
            # Get sample rate from the loaded DAC model
            output_sr = 44100

            # --- Slow down audio ---
            original_len = len(output_audio_np)
            # Ensure speed_factor is positive and not excessively small/large to avoid issues
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(original_len / speed_factor)  # Target length based on speed_factor
            if target_len != original_len and target_len > 0:  # Only interpolate if length changes and is valid
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                output_audio = (
                    output_sr,
                    resampled_audio_np.astype(np.float32),
                )  # Use resampled audio
                print(f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed.")
            else:
                output_audio = (
                    output_sr,
                    output_audio_np.astype(np.float32),
                )  # Use original audio
        else:
            print("Warning: No audio was generated.")
            output_audio = (44100, np.zeros(1, dtype=np.float32))  # Empty audio

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
with gr.Blocks(title="Dia Text-to-Speech") as demo:
    gr.Markdown(
        """
        # Dia Text-to-Speech
        
        Generate realistic dialogue from text with Dia, a 1.6B parameter text-to-speech model by Nari Labs.
        
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
                max_new_tokens = gr.Slider(
                    label="Max New Tokens",
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
                    value=0.95,  # Default from inference.py
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

    # Link button click to function
    run_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_input,
            max_new_tokens,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor_slider,
        ],
        outputs=[audio_output],  # Add status_output here if using it
        api_name="generate_audio",
    )

    # Add examples (ensure the prompt path is correct or remove it if example file doesn't exist)
    example_prompt_path = "./example_prompt.mp3"  # Adjust if needed
    examples_list = [
        [
            "[S1] Oh fire! Oh my goodness! What's the procedure? What to we do people? The smoke could be coming through an air duct! \n[S2] Oh my god! Okay.. it's happening. Everybody stay calm! \n[S1] What's the procedure... \n[S2] Everybody stay fucking calm!!!... Everybody fucking calm down!!!!! \n[S1] No! No! If you touch the handle, if its hot there might be a fire down the hallway! ",
            None,
            3072,
            3.0,
            1.3,
            0.95,
            35,
            0.94,
        ],
        [
            "[S1] Open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] I'm biased, but I think we clearly won. \n[S2] Hard to disagree. (laughs) \n[S1] Thanks for listening to this demo. \n[S2] Try it now on Git hub and Hugging Face. \n[S1] If you liked our model, please give us a star and share to your friends. \n[S2] This was Nari Labs.",
            example_prompt_path if Path(example_prompt_path).exists() else None,
            3072,
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
                max_new_tokens,
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
    else:
        gr.Markdown("_(No examples configured or example prompt file missing)_")

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")

    # set `GRADIO_SERVER_NAME`, `GRADIO_SERVER_PORT` env vars to override default values
    # use `GRADIO_SERVER_NAME=0.0.0.0` for Docker
    demo.launch(share=args.share)

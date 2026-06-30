import os
import re
import glob
import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# sRGB ↔ linear-light helpers
# ---------------------------------------------------------------------------

def srgb_to_linear(c):
    """Convert sRGB [0,1] to linear light [0,1]."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    """Convert linear light [0,1] to sRGB [0,1]."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


# ---------------------------------------------------------------------------
# Smooth interpolation (avoids hard boundary artifacts)
# ---------------------------------------------------------------------------

def _smoothstep(x, edge0, edge1):
    """Hermite smoothstep: 0 below edge0, 1 above edge1, smooth in between."""
    # Handle edge case where edge0 == edge1 (binary mask)
    if edge0 >= edge1:
        return np.where(x >= edge0, 1.0, 0.0)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Bloom / glow around the sun (lens flare simulation)
# ---------------------------------------------------------------------------

def _bloom(src, levels=6, strength=1.0, decay=0.82, base_sigma=2.0):
    """
    Multi-scale bloom: downsample → blur → upsample → accumulate.
    Simulates the lens glow / flare that a real camera produces around the sun.
    """
    h, w = src.shape[:2]
    acc = np.zeros((h, w), np.float32)
    cur = src.astype(np.float32)
    weight = 1.0
    for _ in range(levels):
        cur = cv2.pyrDown(cur)
        blurred = cv2.GaussianBlur(cur, (0, 0), base_sigma)
        acc += weight * cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
        weight *= decay
    return strength * acc


# ---------------------------------------------------------------------------
# Core: synthesize a realistic EV-2 image from an EV-0 image
# ---------------------------------------------------------------------------

def synth_ev_minus2(image_path,
                    ev_stops=-2.0,
                    # --- sun isolation: smooth gradient for sun pixels ---
                    sun_knee_8bit=240.0,       # Start ramp at 240 (captures sun gradient)
                    sun_peak_8bit=255.0,       # Full sun mask at 255
                    highlight_boost=3.5,       # Boost sun to keep it bright after -2 EV
                    # --- bloom (lens glow around the sun) ---
                    bloom_knee_8bit=240.0,     # glow sourced from bright sun region
                    bloom_strength=0.15,       # Subtle bloom for natural lens flare
                    bloom_levels=4,            # Multi-scale bloom
                    bloom_tint=(0.92, 0.97, 1.0),  # slight warm-cool tint on glow
                    # --- surroundings darkening ---
                    surround_darken=0.25,      # Darken surroundings (2 stops = 1/4 brightness)
                    # --- noise simulation ---
                    add_noise=True,
                    full_well=12000.0,         # simulated full-well capacity
                    read_noise_e=3.0,          # read noise in electrons
                    seed=None):
    """
    Synthesize a realistic EV-2 image from an EV-0 image.

    Key principles (matching real camera behaviour):
    1. Exposure change is done in **linear light** (÷4 for 2 stops).
       This naturally makes dark pixels much darker (ratio ~8 in sRGB)
       while bright pixels stay relatively bright (ratio ~1.7 in sRGB).
    2. Sun pixels that are clipped in EV-0 are **highlight-reconstructed**
       (their true radiance is much higher than 255), so they remain a
       visible bright orb in EV-2 instead of turning grey.
    3. A **bloom / glow** is added around the sun to simulate lens flare.
    4. Smooth transitions (smoothstep) avoid hard boundary artifacts.
    5. Optional **shot + read noise** makes dark regions look realistic.

    Args:
        image_path: Path to the EV-0 input image.

    Returns:
        Synthesized EV-2 image as a uint8 BGR array, or None on failure.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        return None

    rng = np.random.default_rng(seed)

    # --- 1. Convert to linear light ---
    srgb = bgr.astype(np.float32) / 255.0
    lin = srgb_to_linear(srgb)

    # --- 2. Luminance map (used to identify sun pixels) ---
    # Using ITU-R BT.601 luma weights (BGR order)
    lum8 = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]

    # --- 3. Sun mask: smooth 0→1 transition for near-white pixels ---
    #   ~0 everywhere, ramps to 1 only for the sun core
    sun_mask = _smoothstep(lum8, sun_knee_8bit, sun_peak_8bit)  # HxW in [0,1]

    # --- 4. Highlight reconstruction on sun pixels ---
    # Clipped pixels at 255 actually represent much higher radiance.
    # We boost them so they remain bright after the exposure reduction.
    boost = 1.0 + sun_mask * (highlight_boost - 1.0)
    lin_reconstructed = lin * boost[:, :, None]

    # --- 5. Apply surroundings darkening (BEFORE bloom to avoid darkening the glow) ---
    # Create a gain map: sun regions get 1.0, non-sun regions get surround_darken
    # This makes the surroundings much darker while protecting the sun
    surround_gain = surround_darken + (1.0 - surround_darken) * sun_mask
    lin_reconstructed = lin_reconstructed * surround_gain[:, :, None]

    # --- 6. Bloom / glow sourced strictly from the sun ---
    lum_lin = 0.114 * lin_reconstructed[:, :, 0] + \
              0.587 * lin_reconstructed[:, :, 1] + \
              0.299 * lin_reconstructed[:, :, 2]
    bloom_knee = srgb_to_linear(np.float32(bloom_knee_8bit / 255.0))
    bloom_src = np.clip(lum_lin - bloom_knee, 0.0, None) * sun_mask
    glow = _bloom(bloom_src, levels=bloom_levels, strength=bloom_strength)
    lin_reconstructed = lin_reconstructed + glow[:, :, None] * np.array(bloom_tint, np.float32)

    # --- 7. Apply exposure change in linear light ---
    # EV-2 = EV-0 × 2^(-2) = EV-0 ÷ 4
    out = lin_reconstructed * (2.0 ** ev_stops)

    # --- 7. Optional noise simulation ---
    # Real EV-2 images have visible noise in dark areas (shot noise + read noise)
    if add_noise:
        # Shot noise: Poisson-like, proportional to signal
        expected_electrons = np.clip(out, 0.0, None) * full_well
        sigma = np.sqrt(expected_electrons + read_noise_e ** 2)
        noise = rng.standard_normal(expected_electrons.shape).astype(np.float32) * sigma
        noisy_electrons = expected_electrons + noise
        out = np.clip(noisy_electrons, 0.0, None) / full_well

    # --- 8. Convert back to sRGB ---
    out = np.clip(out, 0.0, 1.0)
    result = np.clip(linear_to_srgb(out) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# Single image conversion
# ---------------------------------------------------------------------------

def convert_single_image_to_ev2(input_path, output_path=None, add_noise=True, seed=0):
    """
    Convert a single image to EV-2 exposure.

    Args:
        input_path: Path to the input EV-0 image.
        output_path: Path to save the output EV-2 image. If None, returns the image array.
        add_noise: Whether to add realistic sensor noise.
        seed: Random seed for reproducible noise.

    Returns:
        The converted EV-2 image as a uint8 BGR array, or None on failure.
        If output_path is provided, the image is also saved to that path.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}")
        return None

    # Generate the EV-2 image
    ev_minus2_image = synth_ev_minus2(input_path, ev_stops=-2.0, add_noise=add_noise, seed=seed)

    if ev_minus2_image is None:
        print(f"Failed to process: {input_path}")
        return None

    # Save if output path is provided
    if output_path is not None:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, ev_minus2_image)
        print(f"EV-2 image saved to: {output_path}")

    return ev_minus2_image


# ---------------------------------------------------------------------------
# Batch processing (DEFAULT - processes all images in a directory)
# ---------------------------------------------------------------------------

def generate_ev_minus2_images(input_dir, output_dir, add_noise=True, seed=0):
    """
    Generate EV-2 images from EV-0 images in a directory.

    Args:
        input_dir:  Directory containing EV-0 images.
        output_dir: Directory to save generated EV-2 images.
        add_noise:  Whether to add realistic sensor noise.
        seed:       Random seed for reproducible noise (None for random).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect image files (both lowercase and uppercase extensions)
    exts = ['.jpg', '.jpeg', '.png', '.bmp']
    files = set()
    for ext in exts:
        files.update(glob.glob(os.path.join(input_dir, '*' + ext)))
        files.update(glob.glob(os.path.join(input_dir, '*' + ext.upper())))
    files = sorted(files)

    if not files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(files)} ev0 images in {input_dir}")
    print(f"Generating ev-2 images (realistic EV-2 synthesis with sRGB gamma, "
          f"highlight reconstruction, bloom, noise)")
    print(f"Output directory: {output_dir}")

    success_count = 0
    failed_count = 0

    for i, image_path in enumerate(tqdm(files, desc="Generating ev-2 images")):
        try:
            ev_minus2_image = synth_ev_minus2(
                image_path,
                ev_stops=-2.0,
                add_noise=add_noise,
                seed=(seed + i) if seed is not None else None
            )

            if ev_minus2_image is None:
                print(f"\nFailed to read: {image_path}")
                failed_count += 1
                continue

            # Create output filename
            filename = os.path.basename(image_path)
            if 'ev0' in filename.lower():
                # Replace ev0/ev_0 with ev-2/ev_-2 using regex (case-insensitive)
                out_fn = re.sub(r'ev_?0', 'ev_-2', filename, flags=re.IGNORECASE)
            else:
                name, ext = os.path.splitext(filename)
                out_fn = f"{name}_ev-2{ext}"

            output_path = os.path.join(output_dir, out_fn)
            cv2.imwrite(output_path, ev_minus2_image)
            success_count += 1

        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")
            failed_count += 1

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Successfully generated: {success_count} images")
    print(f"Failed: {failed_count} images")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    input_directory = "/home/test/agnik/Set1_IQ_LAB_DATA/IQLabData/ev0/Non_Neon_Light_White_(Negative)"
    output_directory = "/home/test/agnik/Set1_IQ_LAB_DATA/IQLabData/ev-2/Non_Neon_Light_White_(Negative)"
    generate_ev_minus2_images(input_directory, output_directory, add_noise=True, seed=0)

"""Numerical Problem Solving Engine for Nalanda AI.

Provides step-by-step parametric solving for academic numericals (e.g., 1D Convolutional Autoencoders),
computing all intermediate convolution products, latent bottleneck projections, decoder expansions,
transpose convolution outputs, and MSE loss calculations with full mathematical typography.
"""
import re
from typing import Dict, Any, List, Optional, Tuple


class NumericalSolverService:
    """Parametric solver for multi-stage neural network and signal numericals."""

    @classmethod
    def is_numerical_query(cls, query: str) -> bool:
        """Detect if user query is asking to solve, recalculate, or compute a numerical problem."""
        q_lower = query.lower().strip()
        triggers = [
            r"\b(solve|calculate|compute|recalculate|evaluate|find)\b.*\b(numerical|loss|mse|latent|encoder|decoder|convolution|conv|weights?|kernel)\b",
            r"\b(what\s+if|when|with)\b.*\b(input|x|kernel|w|stride)\s*(?:=|is|are|becomes)\b",
            r"\bautoencoder\s+numerical\b",
            r"\bcompute\s+(?:the\s+)?(?:reconstruction\s+)?loss\b",
            r"\bcalculate\s+(?:the\s+)?latent\s+vector\b",
            r"\[[\d\s,\.\-]+\]",
        ]
        return any(re.search(pat, q_lower) for pat in triggers)

    @classmethod
    def extract_custom_values(cls, query: str) -> Dict[str, Any]:
        """Extract custom user-provided vectors or parameters from the chat query."""
        params: Dict[str, Any] = {}

        # 1. Extract input vector x:
        # Supports "input is [2, 3, 4, 5]", "what if input is [2, 3, 4, 5]", "x = [2.0, 3.0, 4.0, 5.0]", "input: 2, 3, 4, 5"
        x_match = re.search(r"(?:input\s*vector|input|x)\s*(?:=|is|are|becomes|:)?\s*\[([0-9\s,\.\-]+)\]", query, re.I)
        if not x_match:
            x_match = re.search(r"(?:input\s*vector|input|x)\s*(?:=|is|are|becomes|:)\s*([0-9]+(?:\.[0-9]+)?(?:\s*,\s*-?[0-9]+(?:\.[0-9]+)?)+)", query, re.I)
        if not x_match:
            bracket_m = re.search(r"\[([0-9\s,\.\-]+)\]", query)
            if bracket_m and not re.search(r"(?:kernel|w_conv|weight|filter)", query, re.I):
                x_match = bracket_m

        if x_match:
            try:
                nums = [float(v.strip()) for v in x_match.group(1).split(",") if v.strip()]
                if len(nums) >= 2:
                    params["x"] = nums
            except Exception:
                pass

        # 2. Extract kernel weights w_conv:
        # Supports "kernel is [0.8, -0.2]", "w = [1, -1]", "w_conv = [0.5, -0.5]"
        w_match = re.search(r"(?:kernel\s*weights?|kernel|w_conv|weights?|filter|w)\s*(?:=|is|are|becomes|:)?\s*\[([0-9\s,\.\-]+)\]", query, re.I)
        if not w_match:
            w_match = re.search(r"(?:kernel\s*weights?|kernel|w_conv|weights?|filter|w)\s*(?:=|is|are|becomes|:)\s*([0-9\-]+(?:\.[0-9]+)?(?:\s*,\s*-?[0-9]+(?:\.[0-9]+)?)+)", query, re.I)

        if w_match:
            try:
                nums = [float(v.strip()) for v in w_match.group(1).split(",") if v.strip()]
                if len(nums) == 2:
                    params["w_conv"] = nums
            except Exception:
                pass

        # 3. Extract stride: e.g. "stride = 1" or "stride is 2"
        stride_match = re.search(r"\bstride\s*(?:=|is|are|becomes|:)?\s*(\d+)\b", query, re.I)
        if stride_match:
            try:
                params["stride"] = int(stride_match.group(1))
            except Exception:
                pass

        return params

    @classmethod
    def solve_autoencoder_numerical(
        cls,
        x: Optional[List[float]] = None,
        w_conv: Optional[List[float]] = None,
        stride: int = 1,
        b_conv: float = 0.0,
    ) -> Dict[str, Any]:
        """Execute full mathematical derivation of the 1D Convolutional Autoencoder numerical."""
        # Standard default parameters from lecture slide
        input_x = [float(v) for v in (x or [1.0, 2.0, 3.0, 4.0])]
        kernel_w = [float(v) for v in (w_conv or [0.5, -0.5])]
        str_val = stride if stride in (1, 2) else 1

        # Stage 1: Convolution (Encoder)
        k_len = len(kernel_w)
        x_len = len(input_x)
        out_len = max(1, ((x_len - k_len) // str_val) + 1)

        conv_steps: List[Dict[str, Any]] = []
        conv_output: List[float] = []

        for idx in range(out_len):
            start = idx * str_val
            window = input_x[start : start + k_len]
            t1 = kernel_w[0] * window[0] if len(window) > 0 else 0.0
            t2 = kernel_w[1] * window[1] if len(window) > 1 else 0.0
            step_sum = round(t1 + t2 + b_conv, 4)
            conv_output.append(step_sum)
            conv_steps.append({
                "step_num": idx + 1,
                "window": window,
                "calc_str": f"{kernel_w[0]} · {window[0]} + ({kernel_w[1]}) · {window[1]} = {round(t1, 3)} + ({round(t2, 3)}) = {step_sum}",
                "result": step_sum,
            })

        # Stage 2: Latent Bottleneck (Linear FC Layer)
        # Fixed lecture weight matrix: W_fc = [[1, 0, -1], [0.5, 0.5, 0.5]], b = [0, 0]
        c_feats = list(conv_output[:3])
        while len(c_feats) < 3:
            c_feats.append(0.0)

        c1, c2, c3 = c_feats[0], c_feats[1], c_feats[2]

        z1 = round(1.0 * c1 + 0.0 * c2 + (-1.0) * c3, 4)
        z2 = round(0.5 * c1 + 0.5 * c2 + 0.5 * c3, 4)
        latent_z = [z1, z2]

        # Stage 3: Decoder Linear FC Layer
        # W_fc_decoder = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]], b = [0, 0, 0]
        f1 = round(1.0 * z1 + 0.0 * z2, 4)
        f2 = round(0.0 * z1 + 1.0 * z2, 4)
        f3 = round(-1.0 * z1 + 0.5 * z2, 4)
        decoder_fc_output = [f1, f2, f3]

        # Stage 4: Transpose Convolution (3 -> 4)
        w0 = kernel_w[0]
        w1 = kernel_w[1]
        y0 = round(f1 * w0, 4)
        y1 = round(f1 * w1 + f2 * w0, 4)
        y2 = round(f2 * w1 + f3 * w0, 4)
        y3 = round(f3 * w1, 4)
        reconstructed = [y0, y1, y2, y3]

        # Stage 5: MSE Reconstruction Loss
        # MSE = 1/N * sum((x_hat_i - x_i)^2)
        diff_sqs: List[Dict[str, Any]] = []
        sum_sq = 0.0
        n_elements = min(len(reconstructed), len(input_x))

        for i in range(n_elements):
            diff = round(reconstructed[i] - input_x[i], 4)
            sq = round(diff ** 2, 4)
            sum_sq += sq
            diff_sqs.append({
                "i": i + 1,
                "x_hat": reconstructed[i],
                "x": input_x[i],
                "diff": diff,
                "sq": sq,
            })

        mse_loss = round(sum_sq / n_elements, 4)

        return {
            "input_x": input_x,
            "kernel_w": kernel_w,
            "stride": str_val,
            "bias_conv": b_conv,
            "conv_steps": conv_steps,
            "conv_output": conv_output,
            "latent_z": latent_z,
            "z1_calc": f"1 · ({c1}) + 0 · ({c2}) + (-1) · ({c3}) = {z1}",
            "z2_calc": f"0.5 · ({c1}) + 0.5 · ({c2}) + 0.5 · ({c3}) = {z2}",
            "decoder_fc_output": decoder_fc_output,
            "reconstructed": reconstructed,
            "diff_sqs": diff_sqs,
            "sum_sq": round(sum_sq, 4),
            "mse_loss": mse_loss,
        }

    @classmethod
    def format_solution_markdown(cls, solution: Dict[str, Any]) -> str:
        """Format the solved numerical problem into clean, human-understandable academic Markdown with proper mathematical italic typography without raw slashes, dollar signs, or stray asterisks."""
        x = solution["input_x"]
        w = solution["kernel_w"]
        s = solution["stride"]
        conv_out = solution["conv_output"]
        z = solution["latent_z"]
        dec_fc = solution["decoder_fc_output"]
        recon = solution["reconstructed"]
        mse = solution["mse_loss"]

        lines: List[str] = []
        lines.append("### Comprehensive Solved Numerical: 1D Convolutional Autoencoder")
        lines.append(f"Here is the complete step-by-step mathematical derivation for input vector 𝑥 = {x} and convolution kernel 𝑊_conv = {w}:\n")

        # Given Data Block
        lines.append("#### 1. Given Parameters & Architectural Dimensions")
        lines.append(f"- **Input Sequence Vector** (𝑥): `{x}` (Dimensions: 1 × {len(x)}, 1 channel)")
        lines.append(f"- **Convolution Kernel** (𝑊_conv): `{w}` (Filter Size 𝐾 = {len(w)}, Stride 𝑆 = {s}, No Padding)")
        lines.append(f"- **Expected Feature Map Length** (𝐿_out): (N - K) / S + 1 = ({len(x)} - {len(w)}) / {s} + 1 = **{len(conv_out)}**\n")

        # Step 1: Convolution Encoder
        lines.append("#### 2. Encoder Convolution Layer Calculation")
        lines.append("The 1D kernel slides across the input sequence in steps of 1:")
        for cs in solution["conv_steps"]:
            lines.append(f"- **Window {cs['step_num']}:** Input elements `{cs['window']}` → `{cs['calc_str']}`")
        lines.append(f"\n> **Convolution Feature Map Output:** `{conv_out}`\n")

        # Step 2: Latent Bottleneck
        lines.append("#### 3. Latent Bottleneck Projection (Fully Connected Layer)")
        lines.append("The convolution feature map is flattened and multiplied by the latent weight matrix 𝑊_fc (2 × 3):")
        lines.append(f"- 𝑧₁ = {solution['z1_calc']}")
        lines.append(f"- 𝑧₂ = {solution['z2_calc']}")
        lines.append(f"\n> **Compressed Latent Vector** (𝑧): `{z}`\n")

        # Step 3: Decoder Feature Map Expansion
        lines.append("#### 4. Decoder Feature Map Reconstruction")
        lines.append("The compressed 2D latent representation is expanded back into a 3D feature map using 𝑊_fc_decoder (3 × 2):")
        lines.append(f"- 𝑓₁ = 1.0 · ({z[0]}) + 0.0 · ({z[1]}) = {dec_fc[0]}")
        lines.append(f"- 𝑓₂ = 0.0 · ({z[0]}) + 1.0 · ({z[1]}) = {dec_fc[1]}")
        lines.append(f"- 𝑓₃ = -1.0 · ({z[0]}) + 0.5 · ({z[1]}) = {dec_fc[2]}")
        lines.append(f"\n> **Decoder Feature Map Output** (𝑓): `{dec_fc}`\n")

        # Step 4: Transpose Convolution
        lines.append("#### 5. Transpose Convolution Upsampling (3 → 4)")
        lines.append(f"Upsampling the 3-element feature map back to the original 4-element sequence dimension using symmetric filter weights (𝑤₀ = {w[0]}, 𝑤₁ = {w[1]}):")
        lines.append(f"- Output[0] = 𝑤₀ · ({dec_fc[0]}) = {recon[0]}")
        lines.append(f"- Output[1] = 𝑤₁ · ({dec_fc[0]}) + 𝑤₀ · ({dec_fc[1]}) = {recon[1]}")
        lines.append(f"- Output[2] = 𝑤₁ · ({dec_fc[1]}) + 𝑤₀ · ({dec_fc[2]}) = {recon[2]}")
        lines.append(f"- Output[3] = 𝑤₁ · ({dec_fc[2]}) = {recon[3]}")
        lines.append(f"\n> **Reconstructed Vector** (𝑥̂): `{recon}`\n")

        # Step 5: Mean Squared Error Loss
        lines.append("#### 6. Reconstruction Loss Evaluation (Mean Squared Error)")
        lines.append("Comparing the reconstructed signal 𝑥̂ with the original input sequence 𝑥:")
        lines.append("- Formula: **MSE** = (1/N) · ∑ (𝑥̂ᵢ - 𝑥ᵢ)²")

        sq_terms = []
        for d in solution["diff_sqs"]:
            lines.append(f"- Element {d['i']}: ({d['x_hat']} - {d['x']})² = ({d['diff']})² = {d['sq']}")
            sq_terms.append(str(d["sq"]))

        sum_str = " + ".join(sq_terms)
        lines.append(f"\n> **MSE Loss Calculation:** (1/4) · [{sum_str}] = (1/4) · [{solution['sum_sq']}] = **{mse}**\n")

        # Step 6: Summary Table
        lines.append("#### 7. Verification Summary Matrix")
        lines.append("| Pipeline Stage | Mathematical Operation | Output Vector | Dimensions |")
        lines.append("| :--- | :--- | :--- | :--- |")
        lines.append(f"| **1. Input** | Raw Image Sequence | `{x}` | 1 × {len(x)} |")
        lines.append(f"| **2. Encoder Conv** | Valid 1D Convolution (𝑊 = {w}) | `{conv_out}` | 1 × {len(conv_out)} |")
        lines.append(f"| **3. Latent Bottleneck** | Linear Projection (𝑊_fc) | `{z}` | 1 × 2 |")
        lines.append(f"| **4. Decoder FC** | Feature Expansion (𝑊_fc_decoder) | `{dec_fc}` | 1 × 3 |")
        lines.append(f"| **5. Transpose Conv** | Symmetry Upsampling | `{recon}` | 1 × 4 |")
        lines.append(f"| **6. Final Evaluation** | Mean Squared Error (MSE) | **{mse}** | Scalar Loss |")

        lines.append(f"\n*Executing these stages sequentially demonstrates how an autoencoder compresses input data into a latent bottleneck and reconstructs it with minimal error.*")

        return "\n".join(lines)

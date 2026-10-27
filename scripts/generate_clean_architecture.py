import os
import subprocess

# -------------------------------------------------------------------------
# 1. GENERATE CLEAN PUBLICATION SVG (300 DPI)
# -------------------------------------------------------------------------
# Clean, spacious, high-contrast block diagram designed for single-column reports
width = 1750
height = 860

svg = f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="'DejaVu Sans', 'Arial', sans-serif">
  <!-- Clean White Background -->
  <rect width="{width}" height="{height}" fill="#ffffff"/>

  <defs>
    <!-- Card Shadow -->
    <filter id="cardShadow" x="-5%" y="-5%" width="110%" height="110%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>

    <!-- Arrow Markers -->
    <marker id="arrBlue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1d4ed8"/>
    </marker>
    <marker id="arrTeal" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#0f766e"/>
    </marker>
    <marker id="arrPurple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#7c3aed"/>
    </marker>
    <marker id="arrRed" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#b91c1c"/>
    </marker>
    <marker id="arrAmber" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#d97706"/>
    </marker>
    <marker id="arrDark" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#334155"/>
    </marker>
  </defs>

  <!-- ========================================================================= -->
  <!-- 1. INPUT PORTRAIT STAGE (Left) -->
  <!-- ========================================================================= -->
  <g transform="translate(40, 320)" filter="url(#cardShadow)">
    <rect width="180" height="220" rx="10" fill="#f8fafc" stroke="#64748b" stroke-width="2"/>
    <rect width="180" height="38" rx="10" fill="#1e293b"/>
    <rect width="180" height="10" y="28" fill="#1e293b"/>
    <text x="90" y="24" font-size="14" font-weight="700" fill="#ffffff" text-anchor="middle">INPUT PORTRAIT</text>
    
    <!-- Image Icon Box -->
    <rect x="35" y="55" width="110" height="95" rx="6" fill="#e2e8f0" stroke="#94a3b8" stroke-width="1.5"/>
    <circle cx="70" cy="85" r="14" fill="#64748b"/>
    <path d="M 50 135 C 50 110, 90 110, 90 135 Z" fill="#64748b"/>
    <circle cx="110" cy="85" r="14" fill="#94a3b8"/>
    <path d="M 90 135 C 90 110, 130 110, 130 135 Z" fill="#94a3b8"/>

    <text x="90" y="172" font-size="13" font-weight="700" fill="#0f172a" text-anchor="middle">Input Image I</text>
    <text x="90" y="193" font-size="12" fill="#475569" text-anchor="middle">224 × 224 × 3 RGB</text>
  </g>

  <!-- Split Connectors from Input -->
  <!-- To Stream A (Up) -->
  <path d="M 220 400 L 265 400 L 265 170 L 305 170" fill="none" stroke="#1d4ed8" stroke-width="3" marker-end="url(#arrBlue)"/>
  <text x="250" y="275" font-size="11.5" font-weight="700" fill="#1d4ed8" text-anchor="middle" transform="rotate(-90, 250, 275)">SPATIAL PATH</text>

  <!-- To Stream B (Down) -->
  <path d="M 220 460 L 265 460 L 265 690 L 305 690" fill="none" stroke="#0f766e" stroke-width="3" marker-end="url(#arrTeal)"/>
  <text x="250" y="585" font-size="11.5" font-weight="700" fill="#0f766e" text-anchor="middle" transform="rotate(-90, 250, 585)">SPECTRAL PATH</text>

  <!-- ========================================================================= -->
  <!-- 2. STREAM A: SPATIAL TRANSFORMER (Top Stream, Blue) -->
  <!-- ========================================================================= -->
  <g transform="translate(310, 40)" filter="url(#cardShadow)">
    <rect width="660" height="260" rx="12" fill="#f0f7ff" stroke="#3b82f6" stroke-width="2.5"/>
    <!-- Header Badge -->
    <rect width="660" height="42" rx="12" fill="#1d4ed8"/>
    <rect width="660" height="12" y="30" fill="#1d4ed8"/>
    <text x="330" y="27" font-size="15" font-weight="700" fill="#ffffff" text-anchor="middle">STREAM A: SPATIAL VISION TRANSFORMER (Landmark Warping &amp; Blending Seams)</text>

    <!-- Sub-block 1: DINOv2 Backbone -->
    <g transform="translate(20, 60)">
      <rect width="190" height="175" rx="8" fill="#ffffff" stroke="#93c5fd" stroke-width="1.8"/>
      <rect width="190" height="28" rx="8" fill="#dbeafe"/>
      <rect width="190" height="8" y="20" fill="#dbeafe"/>
      <text x="95" y="19" font-size="12" font-weight="700" fill="#1e40af" text-anchor="middle">DINOv2 ViT-S/14</text>
      
      <text x="95" y="52" font-size="11.5" font-weight="600" fill="#1e293b" text-anchor="middle">Frozen Base (21.6M)</text>
      <text x="95" y="72" font-size="11" fill="#475569" text-anchor="middle">Embedding D = 384</text>
      
      <!-- LoRA Badge -->
      <rect x="15" y="88" width="160" height="42" rx="6" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.2"/>
      <text x="95" y="105" font-size="11" font-weight="700" fill="#1d4ed8" text-anchor="middle">+ LoRA (Rank-16, α=32)</text>
      <text x="95" y="122" font-size="10" fill="#2563eb" text-anchor="middle">Blocks 6–11 (0.59M Params)</text>

      <text x="95" y="152" font-size="11" font-weight="600" fill="#0f172a" text-anchor="middle">Tokens: z_cls + 256 Patches</text>
    </g>

    <!-- Arrow 1 -> 2 -->
    <line x1="210" y1="147" x2="235" y2="147" stroke="#1d4ed8" stroke-width="2.5" marker-end="url(#arrBlue)"/>

    <!-- Sub-block 2: Multi-Scale Aggregator -->
    <g transform="translate(240, 60)">
      <rect width="210" height="175" rx="8" fill="#ffffff" stroke="#93c5fd" stroke-width="1.8"/>
      <rect width="210" height="28" rx="8" fill="#dbeafe"/>
      <rect width="210" height="8" y="20" fill="#dbeafe"/>
      <text x="105" y="19" font-size="12" font-weight="700" fill="#1e40af" text-anchor="middle">Multi-Scale Aggregation</text>
      
      <text x="105" y="50" font-size="11" font-weight="700" fill="#1e293b" text-anchor="middle">1536-D Spatial Vector f:</text>
      
      <rect x="15" y="62" width="180" height="26" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
      <text x="105" y="79" font-size="11" fill="#0f172a" text-anchor="middle">z_cls (Class Token: 384)</text>

      <rect x="15" y="93" width="180" height="26" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
      <text x="105" y="110" font-size="11" fill="#0f172a" text-anchor="middle">μ_patch (Global Mean: 384)</text>

      <rect x="15" y="124" width="180" height="26" rx="4" fill="#fef2f2" stroke="#fca5a5"/>
      <text x="105" y="141" font-size="11" font-weight="700" fill="#991b1b" text-anchor="middle">σ_patch (Seam Detector: 384)</text>

      <rect x="15" y="155" width="180" height="26" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
      <text x="105" y="172" font-size="11" fill="#0f172a" text-anchor="middle">z_max (Peak Anomaly: 384)</text>
    </g>

    <!-- Arrow 2 -> 3 -->
    <line x1="450" y1="147" x2="475" y2="147" stroke="#1d4ed8" stroke-width="2.5" marker-end="url(#arrBlue)"/>

    <!-- Sub-block 3: Spatial Projector MLP -->
    <g transform="translate(480, 60)">
      <rect width="160" height="175" rx="8" fill="#ffffff" stroke="#93c5fd" stroke-width="1.8"/>
      <rect width="160" height="28" rx="8" fill="#dbeafe"/>
      <rect width="160" height="8" y="20" fill="#dbeafe"/>
      <text x="80" y="19" font-size="12" font-weight="700" fill="#1e40af" text-anchor="middle">Spatial Projector</text>
      
      <text x="80" y="60" font-size="11.5" font-weight="600" fill="#1e293b" text-anchor="middle">Bottleneck MLP</text>
      <text x="80" y="80" font-size="11" fill="#475569" text-anchor="middle">1536 → 512 → 256</text>
      <text x="80" y="100" font-size="11" fill="#475569" text-anchor="middle">GELU + BatchNorm</text>

      <rect x="12" y="125" width="136" height="38" rx="6" fill="#1d4ed8"/>
      <text x="80" y="142" font-size="11" font-weight="700" fill="#ffffff" text-anchor="middle">Spatial Embedding</text>
      <text x="80" y="157" font-size="11" font-weight="700" fill="#93c5fd" text-anchor="middle">e_spatial ∈ ℝ^256</text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- 3. STREAM B: FREQUENCY RESIDUALS (Bottom Stream, Teal) -->
  <!-- ========================================================================= -->
  <g transform="translate(310, 560)" filter="url(#cardShadow)">
    <rect width="660" height="260" rx="12" fill="#f0fdfa" stroke="#0d9488" stroke-width="2.5"/>
    <!-- Header Badge -->
    <rect width="660" height="42" rx="12" fill="#0f766e"/>
    <rect width="660" height="12" y="30" fill="#0f766e"/>
    <text x="330" y="27" font-size="15" font-weight="700" fill="#ffffff" text-anchor="middle">STREAM B: FREQUENCY RESIDUAL EXTRACTOR (Generative Latent Artifacts &amp; Halftone Noise)</text>

    <!-- Sub-block 1: Non-trainable filter bank -->
    <g transform="translate(20, 60)">
      <rect width="190" height="175" rx="8" fill="#ffffff" stroke="#5eead4" stroke-width="1.8"/>
      <rect width="190" height="28" rx="8" fill="#ccfbf1"/>
      <rect width="190" height="8" y="20" fill="#ccfbf1"/>
      <text x="95" y="19" font-size="12" font-weight="700" fill="#0f766e" text-anchor="middle">Fixed Filter Bank</text>
      
      <text x="95" y="55" font-size="11.5" font-weight="700" fill="#1e293b" text-anchor="middle">Depthwise Laplacian:</text>
      <text x="95" y="73" font-size="11" fill="#475569" text-anchor="middle">K = ⅓(K_Lap + K_h + K_v)</text>
      <text x="95" y="91" font-size="11" fill="#0d9488" text-anchor="middle">R = |I ∗ K|</text>

      <line x1="20" y1="105" x2="170" y2="105" stroke="#e2e8f0" stroke-width="1"/>

      <text x="95" y="125" font-size="11.5" font-weight="700" fill="#1e293b" text-anchor="middle">2D Haar Wavelet:</text>
      <text x="95" y="143" font-size="11" fill="#475569" text-anchor="middle">High-High Subband HH(x,y)</text>
      <text x="95" y="161" font-size="10.5" fill="#0d9488" text-anchor="middle">Bilinear Upsample to 224²</text>
    </g>

    <!-- Arrow 1 -> 2 -->
    <line x1="210" y1="147" x2="235" y2="147" stroke="#0f766e" stroke-width="2.5" marker-end="url(#arrTeal)"/>

    <!-- Sub-block 2: Composite Map -->
    <g transform="translate(240, 60)">
      <rect width="200" height="175" rx="8" fill="#ffffff" stroke="#5eead4" stroke-width="1.8"/>
      <rect width="200" height="28" rx="8" fill="#ccfbf1"/>
      <rect width="200" height="8" y="20" fill="#ccfbf1"/>
      <text x="100" y="19" font-size="12" font-weight="700" fill="#0f766e" text-anchor="middle">Residual Composite</text>
      
      <text x="100" y="55" font-size="12" font-weight="700" fill="#0f766e" text-anchor="middle">Composite Map C:</text>
      <text x="100" y="78" font-size="12" fill="#1e293b" text-anchor="middle">C = 0.5 R + 0.5 |HH|↑</text>
      <text x="100" y="98" font-size="11" fill="#475569" text-anchor="middle">Shape: 224 × 224 × 3</text>

      <rect x="15" y="115" width="170" height="48" rx="6" fill="#f0fdfa" stroke="#0d9488" stroke-width="1.2"/>
      <text x="100" y="134" font-size="10.5" font-weight="700" fill="#0f766e" text-anchor="middle">Isolates Periodic Nyquist</text>
      <text x="100" y="150" font-size="10" fill="#0d9488" text-anchor="middle">Upsampling Checkerboards</text>
    </g>

    <!-- Arrow 2 -> 3 -->
    <line x1="440" y1="147" x2="465" y2="147" stroke="#0f766e" stroke-width="2.5" marker-end="url(#arrTeal)"/>

    <!-- Sub-block 3: 4-Stage ConvNet -->
    <g transform="translate(470, 60)">
      <rect width="170" height="175" rx="8" fill="#ffffff" stroke="#5eead4" stroke-width="1.8"/>
      <rect width="170" height="28" rx="8" fill="#ccfbf1"/>
      <rect width="170" height="8" y="20" fill="#ccfbf1"/>
      <text x="85" y="19" font-size="12" font-weight="700" fill="#0f766e" text-anchor="middle">4-Stage ConvNet</text>
      
      <text x="85" y="55" font-size="11" font-weight="600" fill="#1e293b" text-anchor="middle">Lightweight Encoder (0.22M)</text>
      <text x="85" y="74" font-size="11" fill="#475569" text-anchor="middle">3 → 32 → 64 → 128 → 128</text>
      <text x="85" y="93" font-size="11" fill="#475569" text-anchor="middle">Stride-2 Conv + BatchNorm</text>
      <text x="85" y="112" font-size="10.5" fill="#475569" text-anchor="middle">AvgPool ∥ MaxPool</text>

      <rect x="15" y="125" width="140" height="38" rx="6" fill="#0f766e"/>
      <text x="85" y="142" font-size="11" font-weight="700" fill="#ffffff" text-anchor="middle">Spectral Embedding</text>
      <text x="85" y="157" font-size="11" font-weight="700" fill="#a7f3d0" text-anchor="middle">e_freq ∈ ℝ^128</text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- 4. CROSS-STREAM GATING & FUSION (Middle, Purple) -->
  <!-- ========================================================================= -->
  <g transform="translate(1015, 290)" filter="url(#cardShadow)">
    <rect width="260" height="280" rx="12" fill="#faf5ff" stroke="#9333ea" stroke-width="2.5"/>
    <rect width="260" height="38" rx="12" fill="#7c3aed"/>
    <rect width="260" height="12" y="26" fill="#7c3aed"/>
    <text x="130" y="25" font-size="14" font-weight="700" fill="#ffffff" text-anchor="middle">CROSS-STREAM FUSION</text>

    <!-- Freq Dimension Alignment -->
    <rect x="15" y="52" width="230" height="42" rx="6" fill="#ffffff" stroke="#d8b4fe"/>
    <text x="125" y="69" font-size="11.5" font-weight="700" fill="#6b21a8" text-anchor="middle">Dimension Alignment</text>
    <text x="125" y="85" font-size="11" fill="#581c87" text-anchor="middle">f_aligned = W_align(e_freq) ∈ ℝ^256</text>

    <!-- Channel Gating Vector -->
    <rect x="15" y="104" width="230" height="52" rx="6" fill="#ffffff" stroke="#d8b4fe"/>
    <text x="125" y="122" font-size="11.5" font-weight="700" fill="#6b21a8" text-anchor="middle">Channel Gating Vector g</text>
    <text x="125" y="138" font-size="10.5" fill="#581c87" text-anchor="middle">g = σ(W_g [e_spat ∥ f_align] + b)</text>
    <text x="125" y="151" font-size="10" fill="#7e22ce" text-anchor="middle">Dynamic Attention per Channel</text>

    <!-- Gated combination & MLP -->
    <rect x="15" y="166" width="230" height="45" rx="6" fill="#ffffff" stroke="#d8b4fe"/>
    <text x="125" y="183" font-size="11.5" font-weight="700" fill="#6b21a8" text-anchor="middle">Gated Combination</text>
    <text x="125" y="199" font-size="11" fill="#581c87" text-anchor="middle">z_fuse = e_spat + g ⊙ f_align</text>

    <!-- Output Vector -->
    <rect x="15" y="222" width="230" height="44" rx="6" fill="#7c3aed"/>
    <text x="125" y="240" font-size="12" font-weight="700" fill="#ffffff" text-anchor="middle">Fusion MLP Projection</text>
    <text x="125" y="256" font-size="12" font-weight="700" fill="#e9d5ff" text-anchor="middle">e_fuse ∈ ℝ^256</text>
  </g>

  <!-- Connectors into Fusion -->
  <!-- Stream A into Fusion -->
  <path d="M 970 207 L 1050 207 L 1050 290" fill="none" stroke="#1d4ed8" stroke-width="2.5" marker-end="url(#arrBlue)"/>
  <!-- Stream B into Fusion -->
  <path d="M 970 727 L 1050 727 L 1050 570" fill="none" stroke="#0f766e" stroke-width="2.5" marker-end="url(#arrTeal)"/>

  <!-- ========================================================================= -->
  <!-- 5. TRAINING VS INFERENCE DUAL-HEADS (Right Stage) -->
  <!-- ========================================================================= -->
  <!-- Upper Right: Decoupled Multi-Task ArcFace (Training) -->
  <g transform="translate(1310, 40)" filter="url(#cardShadow)">
    <rect width="400" height="360" rx="12" fill="#fff1f2" stroke="#e11d48" stroke-width="2.5"/>
    <rect width="400" height="38" rx="12" fill="#be123c"/>
    <rect width="400" height="12" y="26" fill="#be123c"/>
    <text x="200" y="25" font-size="13.5" font-weight="700" fill="#ffffff" text-anchor="middle">TRAINING: DECOUPLED MULTI-TASK ARCFACE</text>

    <!-- Head 1: Spatial -->
    <rect x="20" y="55" width="360" height="60" rx="8" fill="#ffffff" stroke="#fca5a5" stroke-width="1.5"/>
    <rect x="20" y="55" width="8" height="60" rx="4" fill="#1d4ed8"/>
    <text x="40" y="78" font-size="12" font-weight="700" fill="#1d4ed8">Head 1: Spatial ArcFace</text>
    <text x="40" y="96" font-size="11" fill="#475569">e_spatial (256) → 2 Classes | Angular Margin m = 0.25, s = 32.0</text>

    <!-- Head 2: Frequency -->
    <rect x="20" y="125" width="360" height="60" rx="8" fill="#ffffff" stroke="#fca5a5" stroke-width="1.5"/>
    <rect x="20" y="125" width="8" height="60" rx="4" fill="#0f766e"/>
    <text x="40" y="148" font-size="12" font-weight="700" fill="#0f766e">Head 2: Frequency ArcFace</text>
    <text x="40" y="166" font-size="11" fill="#475569">e_freq (128) → 2 Classes | Angular Margin m = 0.25, s = 32.0</text>

    <!-- Head 3: Fused -->
    <rect x="20" y="195" width="360" height="60" rx="8" fill="#ffffff" stroke="#fca5a5" stroke-width="1.5"/>
    <rect x="20" y="195" width="8" height="60" rx="4" fill="#7c3aed"/>
    <text x="40" y="218" font-size="12" font-weight="700" fill="#7c3aed">Head 3: Fused ArcFace</text>
    <text x="40" y="236" font-size="11" fill="#475569">e_fuse (256) → 2 Classes | Angular Margin m = 0.25, s = 32.0</text>

    <!-- Total Loss Banner -->
    <rect x="20" y="270" width="360" height="72" rx="8" fill="#ffe4e6" stroke="#f43f5e" stroke-width="1.5"/>
    <text x="200" y="293" font-size="12" font-weight="700" fill="#9f1239" text-anchor="middle">Total Multi-Task Joint Loss:</text>
    <text x="200" y="313" font-size="12" font-weight="700" fill="#881337" text-anchor="middle">L_total = L_fuse + 0.5 L_spatial + 0.5 L_freq</text>
    <text x="200" y="331" font-size="10.5" fill="#be123c" text-anchor="middle">✓ Eliminates Modality Competition &amp; Gradient Dominance</text>
  </g>

  <!-- Lower Right: Bona Fide Z-Score Calibration (Inference) -->
  <g transform="translate(1310, 460)" filter="url(#cardShadow)">
    <rect width="400" height="360" rx="12" fill="#fffbeb" stroke="#f59e0b" stroke-width="2.5"/>
    <rect width="400" height="38" rx="12" fill="#d97706"/>
    <rect width="400" height="12" y="26" fill="#d97706"/>
    <text x="200" y="25" font-size="13.5" font-weight="700" fill="#ffffff" text-anchor="middle">INFERENCE: BONA FIDE Z-SCORE CALIBRATION</text>

    <!-- Step 1: Linear Margins -->
    <rect x="20" y="55" width="360" height="65" rx="8" fill="#ffffff" stroke="#fde68a" stroke-width="1.5"/>
    <text x="200" y="76" font-size="11.5" font-weight="700" fill="#92400e" text-anchor="middle">1. Unconstrained Linear Cosine Margins</text>
    <text x="200" y="95" font-size="11" fill="#78350f" text-anchor="middle">S_s = cos θ_morph,s − cos θ_bf,s</text>
    <text x="200" y="111" font-size="11" fill="#78350f" text-anchor="middle">S_f = cos θ_morph,f − cos θ_bf,f</text>

    <!-- Step 2: Z-Score Normalization -->
    <rect x="20" y="130" width="360" height="75" rx="8" fill="#ffffff" stroke="#fde68a" stroke-width="1.5"/>
    <text x="200" y="151" font-size="11.5" font-weight="700" fill="#92400e" text-anchor="middle">2. Bona Fide Distribution Standardization</text>
    <text x="200" y="172" font-size="11" fill="#78350f" text-anchor="middle">Z_s = (S_s − μ_bf,s) / (σ_bf,s + ε)</text>
    <text x="200" y="190" font-size="11" fill="#78350f" text-anchor="middle">Z_f = (S_f − μ_bf,f) / (σ_bf,f + ε)</text>

    <!-- Step 3: Decision Fusion -->
    <rect x="20" y="215" width="360" height="60" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="200" y="237" font-size="12" font-weight="700" fill="#b45309" text-anchor="middle">3. Equal-Weighted Calibrated Decision</text>
    <text x="200" y="259" font-size="14" font-weight="800" fill="#78350f" text-anchor="middle">Z_final = 0.5 Z_s + 0.5 Z_f</text>

    <!-- Operational Decision Rule -->
    <rect x="20" y="285" width="360" height="58" rx="8" fill="#15803d"/>
    <text x="200" y="306" font-size="12" font-weight="700" fill="#ffffff" text-anchor="middle">Operational ISO/IEC 30107-3 Verdict</text>
    <text x="200" y="326" font-size="12" font-weight="700" fill="#bbf7d0" text-anchor="middle">Z_final ≥ τ*  ⟹  MORPH ATTACK (Reject)</text>
  </g>

  <!-- Connectors into ArcFace & Decision -->
  <!-- Direct Stream lines to ArcFace -->
  <path d="M 970 120 L 1310 120" fill="none" stroke="#1d4ed8" stroke-width="2.5" marker-end="url(#arrBlue)"/>
  <path d="M 1275 410 L 1310 410 L 1310 250" fill="none" stroke="#7c3aed" stroke-width="2.5"/>
  <path d="M 970 780 L 1310 780 L 1310 270" fill="none" stroke="#0f766e" stroke-width="2.5"/>

  <!-- Fusion Output to Inference -->
  <path d="M 1275 450 L 1310 450 L 1310 490" fill="none" stroke="#d97706" stroke-width="3" marker-end="url(#arrAmber)"/>

</svg>
"""

svg_path = "/tmp/system_architecture_clean.svg"
png_path = "/home/crimsond/project/ieee_report/figures/system_architecture.png"

with open(svg_path, "w") as f:
    f.write(svg)

subprocess.run(["magick", "-density", "300", svg_path, "-strip", png_path], check=True)
print(f"Rendered crystal-clear 300 DPI architecture diagram to {png_path} ({os.path.getsize(png_path)/1024:.1f} KB)")


# -------------------------------------------------------------------------
# 2. GENERATE DRAW.IO EDITABLE XML (.drawio)
# -------------------------------------------------------------------------
drawio_xml = """<mxfile host="app.diagrams.net" modified="2026-10-27T00:00:00.000Z" agent="Mozilla/5.0" version="22.0.0" type="device">
  <diagram id="UID_MAD_Architecture" name="UID-MAD System Architecture">
    <mxGraphModel dx="1422" dy="794" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="900" background="#ffffff" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- Title Block -->
        <mxCell id="title" value="UID-MAD: Unified Interpretable Dual-Stream Face Morph Attack Detection" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=20;fontStyle=1;fontColor=#0F172A;" vertex="1" parent="1">
          <mxGeometry x="350" y="20" width="900" height="40" as="geometry" />
        </mxCell>

        <!-- 1. Input Node -->
        <mxCell id="input_node" value="&lt;b&gt;Input Portrait Image I&lt;/b&gt;&lt;br&gt;224 × 224 × 3 RGB&lt;br&gt;&lt;i&gt;ICAO Doc 9303 Credential&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#64748B;strokeWidth=2;fontSize=13;fontColor=#0F172A;" vertex="1" parent="1">
          <mxGeometry x="40" y="380" width="180" height="100" as="geometry" />
        </mxCell>

        <!-- Stream A Container -->
        <mxCell id="streamA_group" value="STREAM A: SPATIAL TRANSFORMER (Landmark Warping &amp; Blending Seams)" style="swimlane;whiteSpace=wrap;html=1;fillColor=#EFF6FF;strokeColor=#2563EB;strokeWidth=2;fontSize=14;fontStyle=1;fontColor=#1E40AF;startSize=30;" vertex="1" parent="1">
          <mxGeometry x="280" y="90" width="670" height="290" as="geometry" />
        </mxCell>

        <mxCell id="vit_node" value="&lt;b&gt;DINOv2 ViT-S/14&lt;/b&gt;&lt;br&gt;Frozen Base (21.6M)&lt;br&gt;&lt;b style=&quot;color:#1D4ED8;&quot;&gt;+ Rank-16 LoRA&lt;/b&gt;&lt;br&gt;(Blocks 6–11, α=32)&lt;br&gt;&lt;br&gt;&lt;i&gt;z_cls + 256 Patches&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#93C5FD;strokeWidth=1.5;fontSize=12;fontColor=#0F172A;" vertex="1" parent="streamA_group">
          <mxGeometry x="20" y="50" width="170" height="210" as="geometry" />
        </mxCell>

        <mxCell id="pool_node" value="&lt;b&gt;Multi-Scale Aggregation&lt;/b&gt;&lt;br&gt;&lt;b&gt;f ∈ ℝ^1536&lt;/b&gt;&lt;br&gt;&lt;hr&gt;• &lt;b&gt;z_cls&lt;/b&gt; (Class: 384)&lt;br&gt;• &lt;b&gt;μ_patch&lt;/b&gt; (Mean: 384)&lt;br&gt;• &lt;b style=&quot;color:#DC2626;&quot;&gt;σ_patch&lt;/b&gt; (Seam: 384)&lt;br&gt;• &lt;b&gt;z_max&lt;/b&gt; (Peak: 384)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#93C5FD;strokeWidth=1.5;fontSize=12;fontColor=#0F172A;" vertex="1" parent="streamA_group">
          <mxGeometry x="230" y="50" width="200" height="210" as="geometry" />
        </mxCell>

        <mxCell id="spat_proj_node" value="&lt;b&gt;Spatial Projector&lt;/b&gt;&lt;br&gt;Bottleneck MLP&lt;br&gt;1536 → 512 → 256&lt;br&gt;GELU + BatchNorm&lt;br&gt;&lt;br&gt;&lt;b style=&quot;color:#1D4ED8;&quot;&gt;e_spatial ∈ ℝ^256&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=2;fontSize=12;fontColor=#1E40AF;" vertex="1" parent="streamA_group">
          <mxGeometry x="470" y="50" width="170" height="210" as="geometry" />
        </mxCell>

        <mxCell id="arr_a1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2563EB;strokeWidth=2;" edge="1" source="vit_node" target="pool_node" parent="streamA_group">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="arr_a2" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2563EB;strokeWidth=2;" edge="1" source="pool_node" target="spat_proj_node" parent="streamA_group">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- Stream B Container -->
        <mxCell id="streamB_group" value="STREAM B: FREQUENCY RESIDUAL EXTRACTOR (Generative Latent Artifacts &amp; Print-Scan Noise)" style="swimlane;whiteSpace=wrap;html=1;fillColor=#F0FDFA;strokeColor=#0D9488;strokeWidth=2;fontSize=14;fontStyle=1;fontColor=#0F766E;startSize=30;" vertex="1" parent="1">
          <mxGeometry x="280" y="470" width="670" height="290" as="geometry" />
        </mxCell>

        <mxCell id="filter_node" value="&lt;b&gt;Fixed Filter Bank&lt;/b&gt;&lt;br&gt;&lt;hr&gt;&lt;b&gt;Laplacian Residual:&lt;/b&gt;&lt;br&gt;K = ⅓(K_Lap + K_h + K_v)&lt;br&gt;R = |I ∗ K|&lt;br&gt;&lt;br&gt;&lt;b&gt;2D Haar Wavelet:&lt;/b&gt;&lt;br&gt;High-High Subband HH(x,y)&lt;br&gt;Bilinear Upsample" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#5EEAD4;strokeWidth=1.5;fontSize=12;fontColor=#0F172A;" vertex="1" parent="streamB_group">
          <mxGeometry x="20" y="50" width="180" height="210" as="geometry" />
        </mxCell>

        <mxCell id="comp_node" value="&lt;b&gt;Residual Composite Map&lt;/b&gt;&lt;br&gt;&lt;b&gt;C = 0.5 R + 0.5 |HH|↑&lt;/b&gt;&lt;br&gt;Shape: 224 × 224 × 3&lt;br&gt;&lt;hr&gt;&lt;i&gt;Isolates periodic neural Nyquist checkerboard patterns&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#5EEAD4;strokeWidth=1.5;fontSize=12;fontColor=#0F172A;" vertex="1" parent="streamB_group">
          <mxGeometry x="240" y="50" width="180" height="210" as="geometry" />
        </mxCell>

        <mxCell id="conv_node" value="&lt;b&gt;4-Stage ConvNet&lt;/b&gt;&lt;br&gt;3 → 32 → 64 → 128 → 128&lt;br&gt;Stride-2 Conv + BatchNorm&lt;br&gt;Dual Adaptive Pooling&lt;br&gt;&lt;br&gt;&lt;b style=&quot;color:#0F766E;&quot;&gt;e_freq ∈ ℝ^128&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#CCFBF1;strokeColor=#0D9488;strokeWidth=2;fontSize=12;fontColor=#0F766E;" vertex="1" parent="streamB_group">
          <mxGeometry x="460" y="50" width="180" height="210" as="geometry" />
        </mxCell>

        <mxCell id="arr_b1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#0D9488;strokeWidth=2;" edge="1" source="filter_node" target="comp_node" parent="streamB_group">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="arr_b2" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#0D9488;strokeWidth=2;" edge="1" source="comp_node" target="conv_node" parent="streamB_group">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- Input Connectors -->
        <mxCell id="arr_in_a" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2563EB;strokeWidth=3;" edge="1" source="input_node" target="streamA_group" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array points="240" y="410" />
            <Array points="240" y="235" />
          </mxGeometry>
        </mxCell>
        <mxCell id="arr_in_b" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#0D9488;strokeWidth=3;" edge="1" source="input_node" target="streamB_group" parent="1">
          <mxGeometry relative="1" as="geometry">
            <Array points="240" y="450" />
            <Array points="240" y="615" />
          </mxGeometry>
        </mxCell>

        <!-- Fusion Node -->
        <mxCell id="fusion_node" value="&lt;b&gt;CROSS-STREAM GATING &amp; FUSION&lt;/b&gt;&lt;br&gt;&lt;hr&gt;• Alignment: f_aligned = W_align(e_freq) ∈ ℝ^256&lt;br&gt;• Gating: g = σ(W_g [e_spat ∥ f_align] + b)&lt;br&gt;• Fusion: z_fused = e_spat + g ⊙ f_align&lt;br&gt;&lt;br&gt;&lt;b style=&quot;color:#7C3AED;&quot;&gt;e_fuse = FusionMLP(z_fused) ∈ ℝ^256&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FAF5FF;strokeColor=#9333EA;strokeWidth=2;fontSize=12;fontColor=#581C87;" vertex="1" parent="1">
          <mxGeometry x="1000" y="365" width="230" height="150" as="geometry" />
        </mxCell>

        <!-- Fusion Connectors -->
        <mxCell id="arr_fuse_a" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2563EB;strokeWidth=2.5;" edge="1" source="spat_proj_node" target="fusion_node" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="arr_fuse_b" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#0D9488;strokeWidth=2.5;" edge="1" source="conv_node" target="fusion_node" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- Training Heads Node -->
        <mxCell id="arcface_node" value="&lt;b&gt;DECOUPLED MULTI-TASK ARCFACE (Training)&lt;/b&gt;&lt;br&gt;&lt;hr&gt;• &lt;b&gt;Spatial Head:&lt;/b&gt; e_spatial → 2 Classes (m=0.25, s=32)&lt;br&gt;• &lt;b&gt;Frequency Head:&lt;/b&gt; e_freq → 2 Classes (m=0.25, s=32)&lt;br&gt;• &lt;b&gt;Fused Head:&lt;/b&gt; e_fuse → 2 Classes (m=0.25, s=32)&lt;br&gt;&lt;br&gt;&lt;b style=&quot;color:#BE123C;&quot;&gt;L_total = L_fuse + 0.5 L_spatial + 0.5 L_freq&lt;/b&gt;&lt;br&gt;&lt;i&gt;Prevents CNN gradient dominance over ViT&lt;/i&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF1F2;strokeColor=#E11D48;strokeWidth=2;fontSize=12;fontColor=#881337;" vertex="1" parent="1">
          <mxGeometry x="1270" y="110" width="290" height="190" as="geometry" />
        </mxCell>

        <!-- Inference Decision Node -->
        <mxCell id="inference_node" value="&lt;b&gt;BONA FIDE Z-SCORE CALIBRATION (Inference)&lt;/b&gt;&lt;br&gt;&lt;hr&gt;• Linear Margins: S_s = cos θ_m,s − cos θ_bf,s&lt;br&gt;• Standardize against genuine bona fide set:&lt;br&gt;Z_s = (S_s − μ_bf,s) / σ_bf,s&lt;br&gt;Z_f = (S_f − μ_bf,f) / σ_bf,f&lt;br&gt;&lt;br&gt;&lt;b style=&quot;color:#B45309;font-size:13px;&quot;&gt;Z_final = 0.5 Z_s + 0.5 Z_f&lt;/b&gt;&lt;br&gt;&lt;hr&gt;&lt;b style=&quot;color:#15803D;&quot;&gt;Verdict: Z_final ≥ τ* ⟹ MORPH ATTACK&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FEF3C7;strokeColor=#F59E0B;strokeWidth=2;fontSize=12;fontColor=#78350F;" vertex="1" parent="1">
          <mxGeometry x="1270" y="470" width="290" height="210" as="geometry" />
        </mxCell>

        <!-- Final Flow Connectors -->
        <mxCell id="arr_to_arc" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#BE123C;strokeWidth=2;" edge="1" source="fusion_node" target="arcface_node" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="arr_to_inf" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#D97706;strokeWidth=2.5;" edge="1" source="fusion_node" target="inference_node" parent="1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

drawio_dir = "/home/crimsond/project/ieee_report/figures"
drawio_file = os.path.join(drawio_dir, "uid_mad_architecture.drawio")

with open(drawio_file, "w") as f:
    f.write(drawio_xml)

print(f"Created editable Draw.io XML file at {drawio_file} ({os.path.getsize(drawio_file)/1024:.1f} KB)")

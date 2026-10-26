import os

svg_content = """<svg width="1400" height="880" viewBox="0 0 1400 880" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif">
  <defs>
    <!-- Gradients -->
    <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f8fafc"/>
      <stop offset="100%" stop-color="#f1f5f9"/>
    </linearGradient>
    
    <linearGradient id="streamAGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#eff6ff"/>
      <stop offset="100%" stop-color="#dbeafe"/>
    </linearGradient>
    
    <linearGradient id="streamBGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f0fdfa"/>
      <stop offset="100%" stop-color="#ccfbf1"/>
    </linearGradient>
    
    <linearGradient id="fuseGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#faf5ff"/>
      <stop offset="100%" stop-color="#f3e8ff"/>
    </linearGradient>
    
    <linearGradient id="trainGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#fff1f2"/>
      <stop offset="100%" stop-color="#ffe4e6"/>
    </linearGradient>
    
    <linearGradient id="inferGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#fffbeb"/>
      <stop offset="100%" stop-color="#fef3c7"/>
    </linearGradient>

    <!-- Filters for drop shadows -->
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>
    
    <filter id="cardShadow" x="-4%" y="-4%" width="108%" height="112%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.06"/>
    </filter>

    <!-- Marker Arrowheads -->
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569"/>
    </marker>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563eb"/>
    </marker>
    <marker id="arrowTeal" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#0d9488"/>
    </marker>
    <marker id="arrowPurple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#9333ea"/>
    </marker>
  </defs>

  <!-- Background -->
  <rect width="1400" height="880" fill="url(#bgGrad)" rx="16"/>
  <rect width="1396" height="876" x="2" y="2" fill="none" stroke="#e2e8f0" stroke-width="2" rx="15"/>

  <!-- TOP INPUT IMAGE -->
  <g transform="translate(480, 25)" filter="url(#shadow)">
    <rect width="440" height="60" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>
    <rect width="6" height="60" rx="3" fill="#3b82f6"/>
    <text x="220" y="27" font-size="16" font-weight="700" fill="#0f172a" text-anchor="middle">Input Facial Portrait Image</text>
    <text x="220" y="47" font-size="13" font-weight="500" fill="#64748b" text-anchor="middle">I ∈ ℝ^(3 × 224 × 224) (Normalized ImageNet Stats)</text>
  </g>

  <!-- SPLIT ARROWS FROM INPUT -->
  <path d="M 540 85 L 540 125 L 350 125 L 350 145" fill="none" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arrowBlue)"/>
  <path d="M 860 85 L 860 125 L 1050 125 L 1050 145" fill="none" stroke="#0d9488" stroke-width="2.5" marker-end="url(#arrowTeal)"/>

  <!-- ==================== STREAM A CONTAINER (LEFT) ==================== -->
  <rect x="50" y="145" width="600" height="340" rx="14" fill="url(#streamAGrad)" stroke="#93c5fd" stroke-width="2" filter="url(#shadow)"/>
  <rect x="65" y="155" width="230" height="26" rx="6" fill="#2563eb"/>
  <text x="180" y="173" font-size="12" font-weight="700" fill="#ffffff" text-anchor="middle">STREAM A: SPATIAL TRANSFORMER</text>
  <text x="310" y="173" font-size="12" font-weight="600" fill="#1e40af">Targeting Landmark Blending Seams</text>

  <!-- Block A1: ViT Backbone -->
  <g transform="translate(70, 195)" filter="url(#cardShadow)">
    <rect width="560" height="68" rx="8" fill="#ffffff" stroke="#bfdbfe" stroke-width="1.5"/>
    <text x="20" y="26" font-size="14" font-weight="700" fill="#1e3a8a">DINOv2 ViT-S/14 Backbone (Frozen) + LoRA Adapters</text>
    <text x="20" y="46" font-size="12" fill="#475569">• 21.6M base parameters, patch size 14×14, D = 384, N = 256 patches</text>
    <text x="20" y="61" font-size="12" fill="#475569">• Rank-16 LoRA (r=16, α=32) injected on attention &amp; MLP layers of blocks 6–11</text>
  </g>

  <!-- Arrow A1 -> A2 -->
  <path d="M 350 263 L 350 285" fill="none" stroke="#2563eb" stroke-width="2" marker-end="url(#arrowBlue)"/>

  <!-- Block A2: Multi-Scale Aggregator -->
  <g transform="translate(70, 285)" filter="url(#cardShadow)">
    <rect width="560" height="85" rx="8" fill="#ffffff" stroke="#bfdbfe" stroke-width="1.5"/>
    <text x="20" y="24" font-size="14" font-weight="700" fill="#1e3a8a">Multi-Scale Spatial Token Aggregation</text>
    <text x="20" y="44" font-size="13" font-weight="600" fill="#0284c7">f = [ z_cls ∥ μ_patch ∥ σ_patch ∥ z_max ] ∈ ℝ^1536</text>
    <text x="20" y="62" font-size="12" fill="#64748b">• σ_patch: Mathematical Seam Detector (elevated patch variance across blending edges)</text>
    <text x="20" y="77" font-size="12" fill="#64748b">• z_max: Preserves peak local anomaly cues that average pooling suppresses</text>
  </g>

  <!-- Arrow A2 -> A3 -->
  <path d="M 350 370 L 350 392" fill="none" stroke="#2563eb" stroke-width="2" marker-end="url(#arrowBlue)"/>

  <!-- Block A3: Spatial Projector -->
  <g transform="translate(70, 392)" filter="url(#cardShadow)">
    <rect width="560" height="75" rx="8" fill="#ffffff" stroke="#bfdbfe" stroke-width="1.5"/>
    <text x="20" y="24" font-size="14" font-weight="700" fill="#1e3a8a">Spatial Projector (Bottleneck MLP)</text>
    <text x="20" y="44" font-size="12" fill="#475569">• Linear(1536 → 512) → GELU → BatchNorm → Dropout(0.3) → Linear(512 → 256)</text>
    <text x="20" y="63" font-size="13" font-weight="700" fill="#2563eb">Output: Spatial Embedding e_spatial ∈ ℝ^256</text>
  </g>

  <!-- ==================== STREAM B CONTAINER (RIGHT) ==================== -->
  <rect x="750" y="145" width="600" height="340" rx="14" fill="url(#streamBGrad)" stroke="#99f6e4" stroke-width="2" filter="url(#shadow)"/>
  <rect x="765" y="155" width="225" height="26" rx="6" fill="#0d9488"/>
  <text x="877" y="173" font-size="12" font-weight="700" fill="#ffffff" text-anchor="middle">STREAM B: FREQUENCY RESIDUAL</text>
  <text x="1005" y="173" font-size="12" font-weight="600" fill="#0f766e">Targeting Generative Latent Artifacts</text>

  <!-- Block B1: Residual Filters -->
  <g transform="translate(770, 195)" filter="url(#cardShadow)">
    <rect width="560" height="68" rx="8" fill="#ffffff" stroke="#99f6e4" stroke-width="1.5"/>
    <text x="20" y="24" font-size="14" font-weight="700" fill="#0f766e">Non-Trainable Filter Bank &amp; Haar Wavelet</text>
    <text x="20" y="44" font-size="12" fill="#475569">• Laplacian Bank: K = 1/3(K_Lap + K_h + K_v) ⟹ Residual R = |I * K|</text>
    <text x="20" y="60" font-size="12" fill="#475569">• 2D Haar High-High Subband HH(x,y) (Rectified &amp; Upsampled) ⟹ C = 0.5R + 0.5|HH|↑</text>
  </g>

  <!-- Arrow B1 -> B2 -->
  <path d="M 1050 263 L 1050 285" fill="none" stroke="#0d9488" stroke-width="2" marker-end="url(#arrowTeal)"/>

  <!-- Block B2: ConvNet -->
  <g transform="translate(770, 285)" filter="url(#cardShadow)">
    <rect width="560" height="85" rx="8" fill="#ffffff" stroke="#99f6e4" stroke-width="1.5"/>
    <text x="20" y="24" font-size="14" font-weight="700" fill="#0f766e">4-Stage Lightweight Convolutional Encoder</text>
    <text x="20" y="44" font-size="12" fill="#475569">• Conv Stages: 3 → 32 → 64 → 128 → 128 (3×3 Kernels, Stride 2)</text>
    <text x="20" y="62" font-size="12" fill="#475569">• Batch Normalization + GELU non-linearities isolate periodic upsampling energy</text>
    <text x="20" y="77" font-size="12" fill="#475569">• Operates without downsampling spatial features prematurely</text>
  </g>

  <!-- Arrow B2 -> B3 -->
  <path d="M 1050 370 L 1050 392" fill="none" stroke="#0d9488" stroke-width="2" marker-end="url(#arrowTeal)"/>

  <!-- Block B3: Pooling & Head -->
  <g transform="translate(770, 392)" filter="url(#cardShadow)">
    <rect width="560" height="75" rx="8" fill="#ffffff" stroke="#99f6e4" stroke-width="1.5"/>
    <text x="20" y="24" font-size="14" font-weight="700" fill="#0f766e">Dual Adaptive Pooling &amp; Projection</text>
    <text x="20" y="44" font-size="12" fill="#475569">• Dual Adaptive Pool [AvgPool ∥ MaxPool] (256-d) → Linear(256 → 128)</text>
    <text x="20" y="63" font-size="13" font-weight="700" fill="#0d9488">Output: Frequency Embedding e_freq ∈ ℝ^128</text>
  </g>

  <!-- ==================== CROSS-STREAM ALIGNMENT & GATING ==================== -->
  <!-- Arrows feeding into fusion -->
  <path d="M 350 467 L 350 515 L 430 515" fill="none" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arrowBlue)"/>
  <path d="M 1050 467 L 1050 515 L 970 515" fill="none" stroke="#0d9488" stroke-width="2.5" marker-end="url(#arrowTeal)"/>

  <g transform="translate(440, 485)" filter="url(#shadow)">
    <rect width="520" height="66" rx="10" fill="url(#fuseGrad)" stroke="#d8b4fe" stroke-width="2"/>
    <text x="260" y="24" font-size="14" font-weight="700" fill="#6b21a8" text-anchor="middle">Feature Alignment &amp; Channel Gating Fusion</text>
    <text x="260" y="42" font-size="12" fill="#475569" text-anchor="middle">f_aligned = Linear(128 → 256),   Gate g = σ(W_g [e_spatial ∥ f_aligned] + b_g) ∈ [0, 1]^256</text>
    <text x="260" y="58" font-size="12" font-weight="600" fill="#7e22ce" text-anchor="middle">z_fused = e_spatial + g ⊙ f_aligned ⟹ e_fuse = FusionMLP(z_fused) ∈ ℝ^256</text>
  </g>

  <!-- ==================== THREE ARCFACE CLASSIFICATION HEADS ==================== -->
  <!-- Downward arrows from streams to heads -->
  <path d="M 230 467 L 230 575" fill="none" stroke="#2563eb" stroke-width="2" marker-end="url(#arrowBlue)"/>
  <path d="M 700 551 L 700 575" fill="none" stroke="#9333ea" stroke-width="2" marker-end="url(#arrowPurple)"/>
  <path d="M 1170 467 L 1170 575" fill="none" stroke="#0d9488" stroke-width="2" marker-end="url(#arrowTeal)"/>

  <!-- Head 1: Spatial ArcFace -->
  <g transform="translate(100, 575)" filter="url(#cardShadow)">
    <rect width="260" height="68" rx="8" fill="#ffffff" stroke="#93c5fd" stroke-width="2"/>
    <rect width="260" height="6" rx="3" fill="#2563eb"/>
    <text x="130" y="28" font-size="13" font-weight="700" fill="#1e3a8a" text-anchor="middle">Spatial ArcFace Head</text>
    <text x="130" y="47" font-size="12" fill="#475569" text-anchor="middle">Input: e_spatial ∈ ℝ^256 → 2</text>
    <text x="130" y="62" font-size="11" font-weight="600" fill="#2563eb" text-anchor="middle">Scale s = 32.0, Margin m = 0.25</text>
  </g>

  <!-- Head 2: Fused ArcFace -->
  <g transform="translate(570, 575)" filter="url(#cardShadow)">
    <rect width="260" height="68" rx="8" fill="#ffffff" stroke="#d8b4fe" stroke-width="2"/>
    <rect width="260" height="6" rx="3" fill="#9333ea"/>
    <text x="130" y="28" font-size="13" font-weight="700" fill="#6b21a8" text-anchor="middle">Fused ArcFace Head</text>
    <text x="130" y="47" font-size="12" fill="#475569" text-anchor="middle">Input: e_fuse ∈ ℝ^256 → 2</text>
    <text x="130" y="62" font-size="11" font-weight="600" fill="#9333ea" text-anchor="middle">Scale s = 32.0, Margin m = 0.25</text>
  </g>

  <!-- Head 3: Freq ArcFace -->
  <g transform="translate(1040, 575)" filter="url(#cardShadow)">
    <rect width="260" height="68" rx="8" fill="#ffffff" stroke="#99f6e4" stroke-width="2"/>
    <rect width="260" height="6" rx="3" fill="#0d9488"/>
    <text x="130" y="28" font-size="13" font-weight="700" fill="#0f766e" text-anchor="middle">Frequency ArcFace Head</text>
    <text x="130" y="47" font-size="12" fill="#475569" text-anchor="middle">Input: e_freq ∈ ℝ^128 → 2</text>
    <text x="130" y="62" font-size="11" font-weight="600" fill="#0d9488" text-anchor="middle">Scale s = 32.0, Margin m = 0.25</text>
  </g>

  <!-- ==================== TRAINING VS INFERENCE BOTTOM BLOCKS ==================== -->
  <!-- Split Paths to Training and Inference -->
  <path d="M 230 643 L 230 675 L 340 675 L 340 695" fill="none" stroke="#e11d48" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M 640 643 L 640 675 L 420 675 L 420 695" fill="none" stroke="#e11d48" stroke-width="2" marker-end="url(#arrow)"/>
  
  <path d="M 760 643 L 760 675 L 980 675 L 980 695" fill="none" stroke="#d97706" stroke-width="2" marker-end="url(#arrow)"/>
  <path d="M 1170 643 L 1170 675 L 1060 675 L 1060 695" fill="none" stroke="#d97706" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Left: Training Loss Block -->
  <g transform="translate(50, 695)" filter="url(#shadow)">
    <rect width="600" height="155" rx="12" fill="url(#trainGrad)" stroke="#fda4af" stroke-width="2"/>
    <rect x="20" y="15" width="220" height="24" rx="5" fill="#e11d48"/>
    <text x="130" y="31" font-size="12" font-weight="700" fill="#ffffff" text-anchor="middle">TRAINING: MULTI-TASK LOSS</text>
    
    <text x="20" y="62" font-size="13" font-weight="700" fill="#9f1239">Decoupled Multi-Task Angular Margin Objective:</text>
    <text x="20" y="84" font-size="14" font-weight="700" fill="#be123c">L_total = L_ArcFace(e_fuse, y) + 0.5 L_ArcFace(e_spatial, y) + 0.5 L_ArcFace(e_freq, y)</text>
    
    <text x="20" y="112" font-size="12" fill="#475569">• Independent heads prevent convolutional gradients from disrupting slow ViT updates.</text>
    <text x="20" y="130" font-size="12" fill="#475569">• Resolves early concatenation feature inversion (Trial 1 AUC collapsed to 0.2422).</text>
    <text x="20" y="148" font-size="12" fill="#475569">• Stage 1 warmup (Epochs 1-4) ➔ Stage 2 rank-16 LoRA fine-tuning (Epochs 5-8).</text>
  </g>

  <!-- Right: Inference Block -->
  <g transform="translate(750, 695)" filter="url(#shadow)">
    <rect width="600" height="155" rx="12" fill="url(#inferGrad)" stroke="#fde68a" stroke-width="2"/>
    <rect x="20" y="15" width="230" height="24" rx="5" fill="#d97706"/>
    <text x="135" y="31" font-size="12" font-weight="700" fill="#ffffff" text-anchor="middle">INFERENCE: DECISION FUSION</text>
    
    <text x="20" y="62" font-size="13" font-weight="700" fill="#92400e">Bona Fide Reference Z-Score Standardization:</text>
    <text x="20" y="84" font-size="13" font-weight="700" fill="#b45309">Z_s = (S_s - μ_bf,s) / (σ_bf,s + ε),   Z_f = (S_f - μ_bf,f) / (σ_bf,f + ε)</text>
    <text x="20" y="104" font-size="14" font-weight="800" fill="#78350f">Final Calibrated Score: Z_final = 0.5 Z_s + 0.5 Z_f</text>
    
    <text x="20" y="128" font-size="12" fill="#475569">• Discards saturated softmax probabilities; operates directly in linear cosine space.</text>
    <text x="20" y="146" font-size="12" fill="#475569">• Achieves 0.992 mean ROC-AUC across all 15 ISO/IEC 30107-3 FRLL-Morphs slices.</text>
  </g>
</svg>
"""

with open('/tmp/system_architecture.svg', 'w') as f:
    f.write(svg_content)

print("SVG written successfully.")

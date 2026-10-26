import os
import subprocess

svg = """<svg width="1700" height="700" viewBox="0 0 1700 700" xmlns="http://www.w3.org/2000/svg" font-family="'DejaVu Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif">
  <defs>
    <!-- Gradients -->
    <linearGradient id="streamABg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f0f7ff"/>
      <stop offset="100%" stop-color="#e0effe"/>
    </linearGradient>
    <linearGradient id="streamBBg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f0fdf9"/>
      <stop offset="100%" stop-color="#ccfbf1"/>
    </linearGradient>
    <linearGradient id="fuseBg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#faf5ff"/>
      <stop offset="100%" stop-color="#f3e8ff"/>
    </linearGradient>
    <linearGradient id="arcBg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#fff1f2"/>
      <stop offset="100%" stop-color="#ffe4e6"/>
    </linearGradient>
    <linearGradient id="zscoreBg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#fffbeb"/>
      <stop offset="100%" stop-color="#fef3c7"/>
    </linearGradient>
    
    <!-- Colored Header Bars -->
    <linearGradient id="blueBar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#1d4ed8"/>
      <stop offset="100%" stop-color="#3b82f6"/>
    </linearGradient>
    <linearGradient id="tealBar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#0f766e"/>
      <stop offset="100%" stop-color="#14b8a6"/>
    </linearGradient>
    <linearGradient id="purpleBar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#6d28d9"/>
      <stop offset="100%" stop-color="#9333ea"/>
    </linearGradient>
    <linearGradient id="roseBar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#be123c"/>
      <stop offset="100%" stop-color="#f43f5e"/>
    </linearGradient>
    <linearGradient id="amberBar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#b45309"/>
      <stop offset="100%" stop-color="#f59e0b"/>
    </linearGradient>

    <!-- Isometric Conv Block Gradients -->
    <linearGradient id="cubeTop" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#5eead4"/>
      <stop offset="100%" stop-color="#2dd4bf"/>
    </linearGradient>
    <linearGradient id="cubeFront" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0f766e"/>
      <stop offset="100%" stop-color="#115e59"/>
    </linearGradient>
    <linearGradient id="cubeSide" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#14b8a6"/>
      <stop offset="100%" stop-color="#0d9488"/>
    </linearGradient>

    <!-- Drop Shadows -->
    <filter id="cardGlow" x="-6%" y="-6%" width="112%" height="112%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.06"/>
    </filter>
    <filter id="boxShadow" x="-8%" y="-8%" width="116%" height="116%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="#0f172a" flood-opacity="0.05"/>
    </filter>

    <!-- Markers -->
    <marker id="arrBlue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563eb"/>
    </marker>
    <marker id="arrTeal" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#0d9488"/>
    </marker>
    <marker id="arrPurple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#7c3aed"/>
    </marker>
    <marker id="arrRose" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#e11d48"/>
    </marker>
    <marker id="arrAmber" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#d97706"/>
    </marker>
  </defs>

  <!-- Clean Pure White Background (Blends natively with LaTeX) -->
  <rect width="1700" height="700" fill="#ffffff"/>

  <!-- ========================================================================= -->
  <!-- 1. INPUT STAGE (Left x: 25 to 175, Center y: 350) -->
  <!-- ========================================================================= -->
  <g transform="translate(30, 245)" filter="url(#cardGlow)">
    <rect width="145" height="210" rx="12" fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>
    <rect width="145" height="34" rx="12" fill="#0f172a"/>
    <rect width="145" height="12" y="22" fill="#0f172a"/>
    <text x="72" y="22" font-size="11.5" font-weight="700" fill="#ffffff" text-anchor="middle">INPUT PORTRAIT</text>
    
    <!-- Stylized Face Outline -->
    <circle cx="72" cy="95" r="32" fill="#f8fafc" stroke="#475569" stroke-width="1.8"/>
    <ellipse cx="62" cy="90" rx="3.5" ry="2.5" fill="#2563eb"/>
    <ellipse cx="82" cy="90" rx="3.5" ry="2.5" fill="#2563eb"/>
    <path d="M 70 96 L 74 96 L 72 102 Z" fill="#64748b"/>
    <path d="M 64 110 Q 72 116 80 110" fill="none" stroke="#475569" stroke-width="1.8" stroke-linecap="round"/>
    
    <!-- Coordinate Mesh / Patch Grid Overlay -->
    <rect x="44" y="67" width="56" height="56" fill="none" stroke="#3b82f6" stroke-width="1" stroke-dasharray="2,2" opacity="0.6"/>
    <line x1="58" y1="67" x2="58" y2="123" stroke="#3b82f6" stroke-width="0.8" opacity="0.5"/>
    <line x1="72" y1="67" x2="72" y2="123" stroke="#3b82f6" stroke-width="0.8" opacity="0.5"/>
    <line x1="86" y1="67" x2="86" y2="123" stroke="#3b82f6" stroke-width="0.8" opacity="0.5"/>
    <line x1="44" y1="81" x2="100" y2="81" stroke="#3b82f6" stroke-width="0.8" opacity="0.5"/>
    <line x1="44" y1="95" x2="100" y2="95" stroke="#3b82f6" stroke-width="0.8" opacity="0.5"/>
    <line x1="44" y1="109" x2="100" y2="109" stroke="#3b82f6" stroke-width="0.8" opacity="0.5"/>

    <text x="72" y="152" font-size="11" font-weight="700" fill="#0f172a" text-anchor="middle">Facial Image I</text>
    <text x="72" y="168" font-size="10" font-weight="500" fill="#64748b" text-anchor="middle">224 × 224 × 3 (RGB)</text>
    <text x="72" y="186" font-size="9" font-weight="600" fill="#2563eb" text-anchor="middle">ImageNet Standardized</text>
    <text x="72" y="199" font-size="8.5" fill="#94a3b8" text-anchor="middle">ISO/IEC 19794-5 Compliant</text>
  </g>

  <!-- Split Dataflow Arrows from Input -->
  <path d="M 175 320 C 205 320, 205 170, 235 170" fill="none" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arrBlue)"/>
  <path d="M 175 380 C 205 380, 205 530, 235 530" fill="none" stroke="#0d9488" stroke-width="2.5" marker-end="url(#arrTeal)"/>

  <!-- ========================================================================= -->
  <!-- 2. STREAM A: SPATIAL TRANSFORMER (Top track: x: 240 to 690, y: 20 to 330) -->
  <!-- ========================================================================= -->
  <rect x="240" y="20" width="460" height="315" rx="14" fill="url(#streamABg)" stroke="#93c5fd" stroke-width="2" filter="url(#cardGlow)"/>
  
  <!-- Stream A Header Badge -->
  <rect x="255" y="32" width="240" height="25" rx="6" fill="url(#blueBar)"/>
  <text x="375" y="49" font-size="11" font-weight="700" fill="#ffffff" text-anchor="middle">STREAM A: SPATIAL TRANSFORMER</text>
  <text x="508" y="49" font-size="11" font-weight="700" fill="#1e40af">Landmark Blending Seams</text>

  <!-- Block A1: ViT Backbone -->
  <g transform="translate(255, 68)" filter="url(#boxShadow)">
    <rect width="430" height="68" rx="8" fill="#ffffff" stroke="#bfdbfe" stroke-width="1.5"/>
    <text x="14" y="22" font-size="12" font-weight="700" fill="#1e3a8a">DINOv2 ViT-S/14 Backbone (Frozen, 21.6M)</text>
    <text x="14" y="39" font-size="10.5" fill="#475569">Patch size 14×14 ⇒ N = 16×16 = 256 tokens, Embedding D = 384</text>
    
    <!-- LoRA Badge Inside A1 -->
    <rect x="14" y="46" width="310" height="16" rx="4" fill="#eff6ff" stroke="#3b82f6" stroke-width="1"/>
    <text x="169" y="58" font-size="9" font-weight="700" fill="#1d4ed8" text-anchor="middle">LoRA Adapters: r = 16, α = 32 (Blocks 6–11, 0.59M Trainable)</text>
  </g>

  <!-- Arrow A1 -> A2 -->
  <path d="M 470 136 L 470 152" fill="none" stroke="#2563eb" stroke-width="2" marker-end="url(#arrBlue)"/>

  <!-- Block A2: Multi-Scale Patch Token Aggregation -->
  <g transform="translate(255, 152)" filter="url(#boxShadow)">
    <rect width="430" height="82" rx="8" fill="#ffffff" stroke="#bfdbfe" stroke-width="1.5"/>
    <text x="14" y="19" font-size="12" font-weight="700" fill="#1e3a8a">Multi-Scale Spatial Token Aggregation</text>
    <text x="14" y="34" font-size="10.5" font-weight="600" fill="#0284c7">f = [ z_cls ∥ μ_patch ∥ σ_patch ∥ z_max ] ∈ ℝ^1536</text>
    
    <!-- 4 Colored Segment Bars (Properly sized and spaced) -->
    <g transform="translate(14, 41)">
      <rect x="0" y="0" width="94" height="18" rx="3" fill="#2563eb"/>
      <text x="47" y="13" font-size="8.5" font-weight="700" fill="#ffffff" text-anchor="middle">z_cls (384-d)</text>
      
      <rect x="100" y="0" width="94" height="18" rx="3" fill="#0284c7"/>
      <text x="147" y="13" font-size="8.5" font-weight="700" fill="#ffffff" text-anchor="middle">μ_patch (384-d)</text>
      
      <rect x="200" y="0" width="102" height="18" rx="3" fill="#0891b2"/>
      <text x="251" y="13" font-size="8.5" font-weight="700" fill="#ffffff" text-anchor="middle">σ_patch (Seams)</text>
      
      <rect x="308" y="0" width="94" height="18" rx="3" fill="#0f766e"/>
      <text x="355" y="13" font-size="8.5" font-weight="700" fill="#ffffff" text-anchor="middle">z_max (Peak)</text>
    </g>
    <text x="14" y="74" font-size="9" fill="#64748b">σ_patch acts as seam detector; z_max captures localized blending cutouts</text>
  </g>

  <!-- Arrow A2 -> A3 -->
  <path d="M 470 234 L 470 248" fill="none" stroke="#2563eb" stroke-width="2" marker-end="url(#arrBlue)"/>

  <!-- Block A3: Spatial Projector -->
  <g transform="translate(255, 248)" filter="url(#boxShadow)">
    <rect width="430" height="72" rx="8" fill="#ffffff" stroke="#bfdbfe" stroke-width="1.5"/>
    <text x="14" y="20" font-size="12" font-weight="700" fill="#1e3a8a">Spatial Projector (Bottleneck MLP)</text>
    <text x="14" y="37" font-size="10" fill="#475569">Linear(1536 → 512) → GELU → BatchNorm → Dropout(0.3) → Linear(512 → 256)</text>
    <rect x="14" y="46" width="220" height="18" rx="4" fill="#dbeafe"/>
    <text x="124" y="59" font-size="10" font-weight="700" fill="#1e40af" text-anchor="middle">Output: e_spatial ∈ ℝ^256</text>
  </g>

  <!-- ========================================================================= -->
  <!-- 3. STREAM B: FREQUENCY RESIDUAL (Bottom track: x: 240 to 690, y: 365 to 680) -->
  <!-- ========================================================================= -->
  <rect x="240" y="365" width="460" height="315" rx="14" fill="url(#streamBBg)" stroke="#99f6e4" stroke-width="2" filter="url(#cardGlow)"/>
  
  <!-- Stream B Header Badge -->
  <rect x="255" y="377" width="240" height="25" rx="6" fill="url(#tealBar)"/>
  <text x="375" y="394" font-size="11" font-weight="700" fill="#ffffff" text-anchor="middle">STREAM B: SPECTRAL RESIDUALS</text>
  <text x="508" y="394" font-size="11" font-weight="700" fill="#0f766e">Generative Latent Artifacts</text>

  <!-- Block B1: Filter Bank & Wavelet -->
  <g transform="translate(255, 413)" filter="url(#boxShadow)">
    <rect width="430" height="66" rx="8" fill="#ffffff" stroke="#99f6e4" stroke-width="1.5"/>
    <text x="14" y="20" font-size="12" font-weight="700" fill="#0f766e">Non-Trainable Filter Bank &amp; 2D Haar Wavelet</text>
    <text x="14" y="37" font-size="10.5" fill="#334155">Laplacian Bank: K = ⅓(K_Lap + K_h + K_v) ⇒ Spatial Residual R = |I * K|</text>
    <text x="14" y="54" font-size="10.5" fill="#334155">2D Haar High-High Subband HH(x,y) ⇒ Composite C = 0.5R + 0.5|HH|↑</text>
  </g>

  <!-- Arrow B1 -> B2 -->
  <path d="M 470 479 L 470 493" fill="none" stroke="#0d9488" stroke-width="2" marker-end="url(#arrTeal)"/>

  <!-- Block B2: 4-Stage ConvNet with 3D Isometric Feature Maps -->
  <g transform="translate(255, 493)" filter="url(#boxShadow)">
    <rect width="430" height="88" rx="8" fill="#ffffff" stroke="#99f6e4" stroke-width="1.5"/>
    <text x="14" y="20" font-size="12" font-weight="700" fill="#0f766e">4-Stage Lightweight Convolutional Encoder</text>
    
    <!-- 3D Feature Map Volumes -->
    <!-- Volume 1 (32x112x112) -->
    <g transform="translate(18, 28)">
      <polygon points="0,8 10,0 22,0 12,8" fill="url(#cubeTop)"/>
      <polygon points="12,8 22,0 22,36 12,44" fill="url(#cubeSide)"/>
      <polygon points="0,8 12,8 12,44 0,44" fill="url(#cubeFront)"/>
      <text x="11" y="53" font-size="7.5" font-weight="700" fill="#0f766e" text-anchor="middle">32×112</text>
    </g>
    <text x="44" y="52" font-size="11" font-weight="700" fill="#0d9488">→</text>

    <!-- Volume 2 (64x56x56) -->
    <g transform="translate(56, 33)">
      <polygon points="0,6 12,0 26,0 14,6" fill="url(#cubeTop)"/>
      <polygon points="14,6 26,0 26,28 14,34" fill="url(#cubeSide)"/>
      <polygon points="0,6 14,6 14,34 0,34" fill="url(#cubeFront)"/>
      <text x="13" y="44" font-size="7.5" font-weight="700" fill="#0f766e" text-anchor="middle">64×56</text>
    </g>
    <text x="89" y="52" font-size="11" font-weight="700" fill="#0d9488">→</text>

    <!-- Volume 3 (128x28x28) -->
    <g transform="translate(101, 38)">
      <polygon points="0,5 14,0 32,0 18,5" fill="url(#cubeTop)"/>
      <polygon points="18,5 32,0 32,22 18,27" fill="url(#cubeSide)"/>
      <polygon points="0,5 18,5 18,27 0,27" fill="url(#cubeFront)"/>
      <text x="16" y="36" font-size="7.5" font-weight="700" fill="#0f766e" text-anchor="middle">128×28</text>
    </g>
    <text x="140" y="52" font-size="11" font-weight="700" fill="#0d9488">→</text>

    <!-- Volume 4 (128x14x14) -->
    <g transform="translate(152, 42)">
      <polygon points="0,4 16,0 36,0 20,4" fill="url(#cubeTop)"/>
      <polygon points="20,4 36,0 36,16 20,20" fill="url(#cubeSide)"/>
      <polygon points="0,4 20,4 20,20 0,20" fill="url(#cubeFront)"/>
      <text x="18" y="29" font-size="7.5" font-weight="700" fill="#0f766e" text-anchor="middle">128×14</text>
    </g>

    <!-- Conv Specs Text -->
    <text x="200" y="44" font-size="10.5" fill="#334155">3×3 Stride-2 Convolutions + BatchNorm + GELU</text>
    <text x="200" y="60" font-size="9" fill="#64748b">Suppresses face semantics; isolates periodic Nyquist checkerboards</text>
  </g>

  <!-- Arrow B2 -> B3 -->
  <path d="M 470 581 L 470 595" fill="none" stroke="#0d9488" stroke-width="2" marker-end="url(#arrTeal)"/>

  <!-- Block B3: Dual Adaptive Pool & Output -->
  <g transform="translate(255, 595)" filter="url(#boxShadow)">
    <rect width="430" height="72" rx="8" fill="#ffffff" stroke="#99f6e4" stroke-width="1.5"/>
    <text x="14" y="20" font-size="12" font-weight="700" fill="#0f766e">Dual Adaptive Pooling &amp; Projection</text>
    <text x="14" y="37" font-size="10" fill="#475569">[AvgPool(128) ∥ MaxPool(128)] ⇒ Linear(256 → 128)</text>
    <rect x="14" y="46" width="220" height="18" rx="4" fill="#ccfbf1"/>
    <text x="124" y="59" font-size="10" font-weight="700" fill="#0f766e" text-anchor="middle">Output: e_freq ∈ ℝ^128</text>
  </g>

  <!-- ========================================================================= -->
  <!-- 4. GATING & FUSION NETWORK (Center-right x: 740 to 980, y: 220 to 480) -->
  <!-- ========================================================================= -->
  <!-- Connecting Curves to Fusion -->
  <path d="M 700 284 C 720 284, 720 310, 740 310" fill="none" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arrBlue)"/>
  <path d="M 700 631 C 720 631, 720 400, 740 400" fill="none" stroke="#0d9488" stroke-width="2.5" marker-end="url(#arrTeal)"/>

  <rect x="740" y="225" width="250" height="260" rx="14" fill="url(#fuseBg)" stroke="#d8b4fe" stroke-width="2" filter="url(#cardGlow)"/>
  
  <rect x="753" y="237" width="224" height="24" rx="6" fill="url(#purpleBar)"/>
  <text x="865" y="253" font-size="11" font-weight="700" fill="#ffffff" text-anchor="middle">CHANNEL GATING FUSION</text>

  <!-- Alignment -->
  <g transform="translate(753, 270)" filter="url(#boxShadow)">
    <rect width="224" height="46" rx="6" fill="#ffffff" stroke="#e9d5ff" stroke-width="1.2"/>
    <text x="10" y="18" font-size="11" font-weight="700" fill="#6b21a8">Dimension Alignment</text>
    <text x="10" y="34" font-size="10" fill="#475569">f_aligned = W_align · e_freq ∈ ℝ^256</text>
  </g>

  <!-- Gating Mechanism -->
  <g transform="translate(753, 324)" filter="url(#boxShadow)">
    <rect width="224" height="74" rx="6" fill="#ffffff" stroke="#e9d5ff" stroke-width="1.2"/>
    <text x="10" y="18" font-size="11" font-weight="700" fill="#6b21a8">Channel Gating Vector</text>
    <text x="10" y="34" font-size="9.5" font-weight="600" fill="#7c3aed">g = σ(W_g [e_s ∥ f_align] + b_g) ∈ [0,1]^256</text>
    <text x="10" y="52" font-size="9.5" font-weight="600" fill="#7c3aed">z_fused = e_spatial + g ⊙ f_aligned</text>
    <text x="10" y="66" font-size="9" fill="#64748b">Dynamic channel reweighting</text>
  </g>

  <!-- Fusion MLP -->
  <g transform="translate(753, 406)" filter="url(#boxShadow)">
    <rect width="224" height="66" rx="6" fill="#ffffff" stroke="#e9d5ff" stroke-width="1.2"/>
    <text x="10" y="18" font-size="11" font-weight="700" fill="#6b21a8">Fusion MLP Projector</text>
    <text x="10" y="35" font-size="10" fill="#475569">Linear(256 → 256) → GELU → BN</text>
    <rect x="10" y="42" width="180" height="16" rx="3" fill="#f3e8ff"/>
    <text x="100" y="54" font-size="9" font-weight="700" fill="#7c3aed" text-anchor="middle">Output: e_fuse ∈ ℝ^256</text>
  </g>

  <!-- ========================================================================= -->
  <!-- 5. MULTI-TASK DECOUPLED ARCFACE HEADS (x: 1030 to 1330) -->
  <!-- ========================================================================= -->
  <!-- Connections to ArcFace Heads -->
  <!-- Spatial Head Arrow (Direct from e_spatial) routed above Gating block -->
  <path d="M 700 260 C 740 120, 970 95, 1030 95" fill="none" stroke="#2563eb" stroke-width="2.2" marker-end="url(#arrBlue)"/>
  <!-- Fused Head Arrow -->
  <path d="M 990 355 L 1030 355" fill="none" stroke="#7c3aed" stroke-width="2.5" marker-end="url(#arrPurple)"/>
  <!-- Frequency Head Arrow (Direct from e_freq) routed below Gating block -->
  <path d="M 700 645 C 740 680, 970 605, 1030 605" fill="none" stroke="#0d9488" stroke-width="2.2" marker-end="url(#arrTeal)"/>

  <!-- ArcFace Container -->
  <rect x="1030" y="20" width="310" height="660" rx="14" fill="url(#arcBg)" stroke="#fecdd3" stroke-width="2" filter="url(#cardGlow)"/>
  
  <rect x="1045" y="32" width="280" height="25" rx="6" fill="url(#roseBar)"/>
  <text x="1185" y="49" font-size="11" font-weight="700" fill="#ffffff" text-anchor="middle">DECOUPLED MULTI-TASK ARCFACE</text>

  <!-- Head 1: Spatial Head -->
  <g transform="translate(1045, 66)" filter="url(#boxShadow)">
    <rect width="280" height="96" rx="8" fill="#ffffff" stroke="#fda4af" stroke-width="1.2"/>
    <rect width="6" height="96" rx="3" fill="#2563eb"/>
    <text x="18" y="20" font-size="12" font-weight="700" fill="#1e3a8a">Head 1: Spatial ArcFace</text>
    <text x="18" y="37" font-size="10.5" fill="#475569">Input: e_spatial ∈ ℝ^256 → 2 Classes</text>
    <text x="18" y="53" font-size="10" font-weight="600" fill="#be123c">Scale s = 32.0, Angular Margin m = 0.25</text>
    <text x="18" y="69" font-size="9.5" fill="#64748b">Loss: L_ArcFace(e_spatial, y)</text>
    <rect x="18" y="74" width="240" height="15" rx="3" fill="#eff6ff"/>
    <text x="138" y="85" font-size="9" font-weight="700" fill="#1d4ed8" text-anchor="middle">Cosine Margin: S_s = cos θ_morph - cos θ_bf</text>
  </g>

  <!-- Training Loss Summary Box inside ArcFace Block -->
  <g transform="translate(1045, 172)" filter="url(#boxShadow)">
    <rect width="280" height="118" rx="8" fill="#ffffff" stroke="#fb7185" stroke-width="1.5"/>
    <text x="14" y="20" font-size="11.5" font-weight="700" fill="#9f1239">Training Multi-Task Joint Loss:</text>
    <text x="14" y="40" font-size="10.5" font-weight="700" fill="#e11d48">L_total = L_fuse + 0.5 L_s + 0.5 L_f</text>
    <text x="14" y="60" font-size="9" fill="#475569">• Independent heads eliminate feature inversion</text>
    <text x="14" y="75" font-size="9" fill="#475569">• Conv gradients cannot overpower slow ViT updates</text>
    <text x="14" y="90" font-size="9" fill="#475569">• Stage 1: Warmup (Ep 1–4) ⇒ Stage 2: LoRA (Ep 5–8)</text>
    <text x="14" y="106" font-size="9.5" font-weight="700" fill="#047857">Achieves 0.0000% Val EER on SMDD</text>
  </g>

  <!-- Head 2: Fused Head -->
  <g transform="translate(1045, 305)" filter="url(#boxShadow)">
    <rect width="280" height="96" rx="8" fill="#ffffff" stroke="#fda4af" stroke-width="1.2"/>
    <rect width="6" height="96" rx="3" fill="#7c3aed"/>
    <text x="18" y="20" font-size="12" font-weight="700" fill="#581c87">Head 2: Fused ArcFace</text>
    <text x="18" y="37" font-size="10.5" fill="#475569">Input: e_fuse ∈ ℝ^256 → 2 Classes</text>
    <text x="18" y="53" font-size="10" font-weight="600" fill="#be123c">Scale s = 32.0, Angular Margin m = 0.25</text>
    <text x="18" y="69" font-size="9.5" fill="#64748b">Loss: L_ArcFace(e_fuse, y)</text>
    <rect x="18" y="74" width="240" height="15" rx="3" fill="#faf5ff"/>
    <text x="138" y="85" font-size="9" font-weight="700" fill="#7c3aed" text-anchor="middle">Cosine Margin: S_fuse = cos θ_morph - cos θ_bf</text>
  </g>

  <!-- Head 3: Frequency Head -->
  <g transform="translate(1045, 570)" filter="url(#boxShadow)">
    <rect width="280" height="96" rx="8" fill="#ffffff" stroke="#fda4af" stroke-width="1.2"/>
    <rect width="6" height="96" rx="3" fill="#0d9488"/>
    <text x="18" y="20" font-size="12" font-weight="700" fill="#0f766e">Head 3: Frequency ArcFace</text>
    <text x="18" y="37" font-size="10.5" fill="#475569">Input: e_freq ∈ ℝ^128 → 2 Classes</text>
    <text x="18" y="53" font-size="10" font-weight="600" fill="#be123c">Scale s = 32.0, Angular Margin m = 0.25</text>
    <text x="18" y="69" font-size="9.5" fill="#64748b">Loss: L_ArcFace(e_freq, y)</text>
    <rect x="18" y="74" width="240" height="15" rx="3" fill="#f0fdf4"/>
    <text x="138" y="85" font-size="9" font-weight="700" fill="#0f766e" text-anchor="middle">Cosine Margin: S_f = cos θ_morph - cos θ_bf</text>
  </g>

  <!-- ========================================================================= -->
  <!-- 6. INFERENCE: BONA FIDE Z-SCORE CALIBRATION (x: 1375 to 1675) -->
  <!-- ========================================================================= -->
  <!-- Connection from Margins S_s and S_f to Calibration -->
  <path d="M 1325 114 C 1350 114, 1350 200, 1375 200" fill="none" stroke="#2563eb" stroke-width="2.2" marker-end="url(#arrBlue)"/>
  <path d="M 1325 618 C 1350 618, 1350 500, 1375 500" fill="none" stroke="#0d9488" stroke-width="2.2" marker-end="url(#arrTeal)"/>

  <rect x="1375" y="100" width="295" height="490" rx="14" fill="url(#zscoreBg)" stroke="#fde68a" stroke-width="2" filter="url(#cardGlow)"/>
  
  <rect x="1390" y="112" width="265" height="25" rx="6" fill="url(#amberBar)"/>
  <text x="1522" y="129" font-size="11" font-weight="700" fill="#ffffff" text-anchor="middle">INFERENCE: Z-SCORE FUSION</text>

  <!-- Standardisation Formula Box -->
  <g transform="translate(1390, 148)" filter="url(#boxShadow)">
    <rect width="265" height="118" rx="8" fill="#ffffff" stroke="#fcd34d" stroke-width="1.2"/>
    <text x="12" y="20" font-size="11" font-weight="700" fill="#92400e">Bona Fide Reference Standardization</text>
    <text x="12" y="42" font-size="11" font-weight="600" fill="#b45309">Z_s = (S_s - μ_bf,s) / (σ_bf,s + ε)</text>
    <text x="12" y="64" font-size="11" font-weight="600" fill="#b45309">Z_f = (S_f - μ_bf,f) / (σ_bf,f + ε)</text>
    <text x="12" y="86" font-size="9.5" fill="#475569">• Discards saturated softmax probabilities</text>
    <text x="12" y="102" font-size="9.5" fill="#475569">• Operates directly in linear cosine margin space</text>
  </g>

  <!-- Final Ensemble Score -->
  <g transform="translate(1390, 280)" filter="url(#boxShadow)">
    <rect width="265" height="96" rx="8" fill="#ffffff" stroke="#fcd34d" stroke-width="1.5"/>
    <text x="12" y="20" font-size="11.5" font-weight="700" fill="#92400e">Calibrated Decision Output</text>
    <rect x="12" y="28" width="241" height="26" rx="4" fill="#fef3c7"/>
    <text x="132" y="46" font-size="12" font-weight="700" fill="#b45309" text-anchor="middle">Z_final = 0.5 Z_s + 0.5 Z_f</text>
    
    <text x="12" y="72" font-size="9.5" font-weight="700" fill="#047857">✓ Mean 0.992 ROC-AUC across 15 Slices</text>
    <text x="12" y="88" font-size="9.5" font-weight="700" fill="#047857">✓ StyleGAN2 Clean: 0.9399 AUC (vs 0.4988)</text>
  </g>

  <!-- ISO/IEC Evaluation Metrics Badge -->
  <g transform="translate(1390, 390)" filter="url(#boxShadow)">
    <rect width="265" height="185" rx="8" fill="#ffffff" stroke="#fcd34d" stroke-width="1.2"/>
    <text x="12" y="20" font-size="11" font-weight="700" fill="#92400e">ISO/IEC 30107-3 Standard Metrics</text>
    <text x="12" y="38" font-size="9.5" fill="#334155">• Sweep threshold τ over dynamic Z range</text>
    <text x="12" y="55" font-size="9.5" fill="#334155">• APCER(τ) = (1/N_PA) ∑ I(Z_i &lt; τ)</text>
    <text x="12" y="72" font-size="9.5" fill="#334155">• BPCER(τ) = (1/N_BF) ∑ I(Z_i ≥ τ)</text>
    <text x="12" y="92" font-size="10" font-weight="700" fill="#b45309">EER @ APCER(τ*) = BPCER(τ*)</text>
    <text x="12" y="112" font-size="9.5" font-weight="600" fill="#1e293b">BPCER @ 1.0% &amp; 5.0% APCER</text>
    <text x="12" y="132" font-size="9.5" font-weight="700" fill="#047857">✓ Latency: 17.3 ms (e-Gate Budget: &lt; 50 ms)</text>
    <text x="12" y="152" font-size="9" fill="#64748b">• Tested zero-shot across 5 generators</text>
    <text x="12" y="168" font-size="9" fill="#64748b">• Print-Scan: &gt; 0.997 frequency AUC surge</text>
  </g>

</svg>
"""

with open("/tmp/system_architecture_pro.svg", "w") as f:
    f.write(svg)

# Render with ImageMagick at 300 DPI
subprocess.run(["magick", "-density", "300", "/tmp/system_architecture_pro.svg", "/home/crimsond/project/ieee_report/figures/system_architecture.png"], check=True)
print("Rendered successfully to /home/crimsond/project/ieee_report/figures/system_architecture.png")

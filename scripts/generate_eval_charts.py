import subprocess

# Generate high-resolution quantitative evaluation charts (Figure 2) as 300 DPI PNG
# Replaces slow pgfplots runtime rendering to make Overleaf compile in 1-2 seconds

svg = """<svg width="1800" height="520" viewBox="0 0 1800 520" xmlns="http://www.w3.org/2000/svg" font-family="'DejaVu Sans', Arial, sans-serif">
  <rect width="1800" height="520" fill="#ffffff"/>

  <!-- ========================================================================= -->
  <!-- SUBPLOT 1: Phase 1 vs Phase 2 (Clean Images) -->
  <!-- ========================================================================= -->
  <g transform="translate(60, 40)">
    <!-- Subplot Title -->
    <text x="230" y="20" font-size="14.5" font-weight="700" fill="#111827" text-anchor="middle">(a) Phase 1 vs. Phase 2 (Clean Images)</text>
    
    <!-- Y-Axis Label -->
    <text x="-35" y="200" font-size="12" font-weight="600" fill="#374151" text-anchor="middle" transform="rotate(-90, -35, 200)">ROC-AUC Score</text>

    <!-- Grid lines & Y-ticks (Y-range: 0.4 to 1.0) -->
    <!-- Y: 1.0 -> y=60, 0.8 -> y=140, 0.6 -> y=220, 0.4 -> y=300 -->
    <g stroke="#e5e7eb" stroke-dasharray="4,4" stroke-width="1">
      <line x1="20" y1="60" x2="480" y2="60"/>
      <line x1="20" y1="120" x2="480" y2="120"/>
      <line x1="20" y1="180" x2="480" y2="180"/>
      <line x1="20" y1="240" x2="480" y2="240"/>
      <line x1="20" y1="300" x2="480" y2="300"/>
    </g>
    <!-- Y Labels -->
    <text x="12" y="64" font-size="10.5" fill="#4b5563" text-anchor="end">1.0</text>
    <text x="12" y="124" font-size="10.5" fill="#4b5563" text-anchor="end">0.9</text>
    <text x="12" y="184" font-size="10.5" fill="#4b5563" text-anchor="end">0.8</text>
    <text x="12" y="244" font-size="10.5" fill="#4b5563" text-anchor="end">0.6</text>
    <text x="12" y="304" font-size="10.5" fill="#4b5563" text-anchor="end">0.4</text>

    <!-- X-Axis Line -->
    <line x1="20" y1="300" x2="480" y2="300" stroke="#374151" stroke-width="1.2"/>
    <line x1="20" y1="60" x2="20" y2="300" stroke="#374151" stroke-width="1.2"/>

    <!-- Bars: 5 categories. Each category width=80. Centers: 70, 160, 250, 340, 430 -->
    <!-- Group 1: OpenCV (0.9993, 0.9997) -->
    <!-- height = (val - 0.4) / 0.6 * 240 -->
    <!-- val 0.9993 -> h = 239.7, y = 60.3 -->
    <rect x="52" y="60.3" width="16" height="239.7" fill="#3b82f6" rx="1.5"/>
    <rect x="72" y="60.1" width="16" height="239.9" fill="#8b5cf6" rx="1.5"/>
    <text x="70" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">OpenCV</text>

    <!-- Group 2: FaceMorpher (0.9958, 0.9999) -->
    <rect x="142" y="61.7" width="16" height="238.3" fill="#3b82f6" rx="1.5"/>
    <rect x="162" y="60.0" width="16" height="240.0" fill="#8b5cf6" rx="1.5"/>
    <text x="160" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">FaceMorpher</text>

    <!-- Group 3: WebMorph (0.9886, 0.9968) -->
    <rect x="232" y="64.6" width="16" height="235.4" fill="#3b82f6" rx="1.5"/>
    <rect x="252" y="61.3" width="16" height="238.7" fill="#8b5cf6" rx="1.5"/>
    <text x="250" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">WebMorph</text>

    <!-- Group 4: AMSL (0.9461, 0.9938) -->
    <rect x="322" y="81.6" width="16" height="218.4" fill="#3b82f6" rx="1.5"/>
    <rect x="342" y="62.5" width="16" height="237.5" fill="#8b5cf6" rx="1.5"/>
    <text x="340" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">AMSL</text>

    <!-- Group 5: StyleGAN2 (0.4988, 0.9399) - THE SURGE -->
    <rect x="412" y="260.5" width="16" height="39.5" fill="#3b82f6" rx="1.5"/>
    <rect x="432" y="84.0" width="16" height="216.0" fill="#8b5cf6" rx="1.5"/>
    <text x="430" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">StyleGAN2</text>
    <!-- Surge Arrow Callout -->
    <text x="440" y="74" font-size="9" font-weight="700" fill="#7c3aed" text-anchor="middle">+0.4411</text>

    <!-- Legend -->
    <g transform="translate(130, 355)">
      <rect x="0" y="0" width="14" height="10" fill="#3b82f6" rx="1.5"/>
      <text x="20" y="9" font-size="10" fill="#374151">Phase 1 (Spatial Baseline)</text>
      <rect x="160" y="0" width="14" height="10" fill="#8b5cf6" rx="1.5"/>
      <text x="180" y="9" font-size="10" fill="#374151">Phase 2 (UID-MAD Calibrated)</text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- SUBPLOT 2: Stream Orthogonality (Print-Scan) -->
  <!-- ========================================================================= -->
  <g transform="translate(660, 40)">
    <!-- Subplot Title -->
    <text x="230" y="20" font-size="14.5" font-weight="700" fill="#111827" text-anchor="middle">(b) Stream Orthogonality (Print-Scan)</text>
    
    <!-- Y-Axis Label -->
    <text x="-35" y="200" font-size="12" font-weight="600" fill="#374151" text-anchor="middle" transform="rotate(-90, -35, 200)">ROC-AUC under Print-Scan</text>

    <!-- Grid lines & Y-ticks (Y-range: 0.2 to 1.0) -->
    <!-- 1.0 -> y=60, 0.8 -> y=120, 0.6 -> y=180, 0.4 -> y=240, 0.2 -> y=300 -->
    <g stroke="#e5e7eb" stroke-dasharray="4,4" stroke-width="1">
      <line x1="20" y1="60" x2="480" y2="60"/>
      <line x1="20" y1="120" x2="480" y2="120"/>
      <line x1="20" y1="180" x2="480" y2="180"/>
      <line x1="20" y1="240" x2="480" y2="240"/>
      <line x1="20" y1="300" x2="480" y2="300"/>
    </g>
    <!-- Y Labels -->
    <text x="12" y="64" font-size="10.5" fill="#4b5563" text-anchor="end">1.0</text>
    <text x="12" y="124" font-size="10.5" fill="#4b5563" text-anchor="end">0.8</text>
    <text x="12" y="184" font-size="10.5" fill="#4b5563" text-anchor="end">0.6</text>
    <text x="12" y="244" font-size="10.5" fill="#4b5563" text-anchor="end">0.4</text>
    <text x="12" y="304" font-size="10.5" fill="#4b5563" text-anchor="end">0.2</text>

    <!-- X-Axis Line -->
    <line x1="20" y1="300" x2="480" y2="300" stroke="#374151" stroke-width="1.2"/>
    <line x1="20" y1="60" x2="20" y2="300" stroke="#374151" stroke-width="1.2"/>

    <!-- Bars: 5 categories. Each category width=80. Centers: 70, 160, 250, 340, 430 -->
    <!-- height = (val - 0.2) / 0.8 * 240 -->
    <!-- Group 1: OpenCV (0.9143, 0.9992) -->
    <rect x="52" y="85.7" width="16" height="214.3" fill="#f97316" rx="1.5"/>
    <rect x="72" y="60.2" width="16" height="239.8" fill="#0d9488" rx="1.5"/>
    <text x="70" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">OpenCV</text>

    <!-- Group 2: FaceMorpher (0.9342, 0.9996) -->
    <rect x="142" y="79.7" width="16" height="220.3" fill="#f97316" rx="1.5"/>
    <rect x="162" y="60.1" width="16" height="239.9" fill="#0d9488" rx="1.5"/>
    <text x="160" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">FaceMorpher</text>

    <!-- Group 3: WebMorph (0.6355, 0.9976) -->
    <rect x="232" y="169.3" width="16" height="130.7" fill="#f97316" rx="1.5"/>
    <rect x="252" y="60.7" width="16" height="239.3" fill="#0d9488" rx="1.5"/>
    <text x="250" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">WebMorph</text>

    <!-- Group 4: AMSL (0.3743, 0.9981) -->
    <rect x="322" y="247.7" width="16" height="52.3" fill="#f97316" rx="1.5"/>
    <rect x="342" y="60.6" width="16" height="239.4" fill="#0d9488" rx="1.5"/>
    <text x="340" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">AMSL</text>

    <!-- Group 5: StyleGAN2 (0.3155, 0.9977) -->
    <rect x="412" y="265.3" width="16" height="34.7" fill="#f97316" rx="1.5"/>
    <rect x="432" y="60.7" width="16" height="239.3" fill="#0d9488" rx="1.5"/>
    <text x="430" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">StyleGAN2</text>

    <!-- Legend -->
    <g transform="translate(145, 355)">
      <rect x="0" y="0" width="14" height="10" fill="#f97316" rx="1.5"/>
      <text x="20" y="9" font-size="10" fill="#374151">Spatial Stream AUC</text>
      <rect x="145" y="0" width="14" height="10" fill="#0d9488" rx="1.5"/>
      <text x="165" y="9" font-size="10" fill="#374151">Frequency Stream AUC</text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- SUBPLOT 3: Calibrated EER across Conditions -->
  <!-- ========================================================================= -->
  <g transform="translate(1260, 40)">
    <!-- Subplot Title -->
    <text x="230" y="20" font-size="14.5" font-weight="700" fill="#111827" text-anchor="middle">(c) Calibrated EER across Conditions</text>
    
    <!-- Y-Axis Label -->
    <text x="-35" y="200" font-size="12" font-weight="600" fill="#374151" text-anchor="middle" transform="rotate(-90, -35, 200)">Equal Error Rate (EER %)</text>

    <!-- Grid lines & Y-ticks (Y-range: 0 to 16%) -->
    <!-- 16% -> y=60, 12% -> y=120, 8% -> y=180, 4% -> y=240, 0% -> y=300 -->
    <g stroke="#e5e7eb" stroke-dasharray="4,4" stroke-width="1">
      <line x1="20" y1="60" x2="480" y2="60"/>
      <line x1="20" y1="120" x2="480" y2="120"/>
      <line x1="20" y1="180" x2="480" y2="180"/>
      <line x1="20" y1="240" x2="480" y2="240"/>
      <line x1="20" y1="300" x2="480" y2="300"/>
    </g>
    <!-- Y Labels -->
    <text x="12" y="64" font-size="10.5" fill="#4b5563" text-anchor="end">16%</text>
    <text x="12" y="124" font-size="10.5" fill="#4b5563" text-anchor="end">12%</text>
    <text x="12" y="184" font-size="10.5" fill="#4b5563" text-anchor="end">8%</text>
    <text x="12" y="244" font-size="10.5" fill="#4b5563" text-anchor="end">4%</text>
    <text x="12" y="304" font-size="10.5" fill="#4b5563" text-anchor="end">0%</text>

    <!-- X-Axis Line -->
    <line x1="20" y1="300" x2="480" y2="300" stroke="#374151" stroke-width="1.2"/>
    <line x1="20" y1="60" x2="20" y2="300" stroke="#374151" stroke-width="1.2"/>

    <!-- Bars: 5 categories. Each category width=80. Centers: 70, 160, 250, 340, 430 -->
    <!-- 3 bars per group: width=12 each -->
    <!-- height = val / 16 * 240 -->
    <!-- Group 1: OpenCV (0.74, 0.25, 2.33) -->
    <rect x="46" y="288.9" width="13" height="11.1" fill="#2563eb" rx="1"/>
    <rect x="62" y="296.2" width="13" height="3.8" fill="#dc2626" rx="1"/>
    <rect x="78" y="265.0" width="13" height="35.0" fill="#64748b" rx="1"/>
    <text x="69" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">OpenCV</text>

    <!-- Group 2: FaceMorpher (0.41, 0.49, 1.02) -->
    <rect x="136" y="293.8" width="13" height="6.2" fill="#2563eb" rx="1"/>
    <rect x="152" y="292.6" width="13" height="7.4" fill="#dc2626" rx="1"/>
    <rect x="168" y="284.7" width="13" height="15.3" fill="#64748b" rx="1"/>
    <text x="159" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">FaceMorpher</text>

    <!-- Group 3: WebMorph (1.96, 2.21, 3.39) -->
    <rect x="226" y="270.6" width="13" height="29.4" fill="#2563eb" rx="1"/>
    <rect x="242" y="266.8" width="13" height="33.2" fill="#dc2626" rx="1"/>
    <rect x="258" y="249.1" width="13" height="50.9" fill="#64748b" rx="1"/>
    <text x="249" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">WebMorph</text>

    <!-- Group 4: AMSL (2.92, 3.23, 6.73) -->
    <rect x="316" y="256.2" width="13" height="43.8" fill="#2563eb" rx="1"/>
    <rect x="332" y="251.5" width="13" height="48.5" fill="#dc2626" rx="1"/>
    <rect x="348" y="199.0" width="13" height="101.0" fill="#64748b" rx="1"/>
    <text x="339" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">AMSL</text>

    <!-- Group 5: StyleGAN2 (14.66, 14.33, 3.92) -->
    <rect x="406" y="80.1" width="13" height="219.9" fill="#2563eb" rx="1"/>
    <rect x="422" y="85.0" width="13" height="215.0" fill="#dc2626" rx="1"/>
    <rect x="438" y="241.2" width="13" height="58.8" fill="#64748b" rx="1"/>
    <text x="429" y="322" font-size="10.5" font-weight="600" fill="#374151" text-anchor="middle">StyleGAN2</text>

    <!-- Legend -->
    <g transform="translate(110, 355)">
      <rect x="0" y="0" width="14" height="10" fill="#2563eb" rx="1"/>
      <text x="18" y="9" font-size="10" fill="#374151">Clean Digital</text>
      <rect x="105" y="0" width="14" height="10" fill="#dc2626" rx="1"/>
      <text x="123" y="9" font-size="10" fill="#374151">JPEG (q=30)</text>
      <rect x="210" y="0" width="14" height="10" fill="#64748b" rx="1"/>
      <text x="228" y="9" font-size="10" fill="#374151">Print-Scan</text>
    </g>
  </g>
</svg>
"""

with open("/tmp/eval_charts.svg", "w") as f:
    f.write(svg)

subprocess.run(["magick", "-density", "300", "/tmp/eval_charts.svg", "/home/crimsond/project/ieee_report/figures/quantitative_eval_charts.png"], check=True)
print("Rendered quantitative evaluation charts to /home/crimsond/project/ieee_report/figures/quantitative_eval_charts.png")

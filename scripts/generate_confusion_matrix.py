import subprocess

svg = """<svg width="1500" height="500" viewBox="0 0 1500 500" xmlns="http://www.w3.org/2000/svg" font-family="'DejaVu Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif">
  <defs>
    <!-- Card and Box Shadows -->
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.06"/>
    </filter>
  </defs>

  <!-- Clean Background -->
  <rect width="1500" height="500" fill="#ffffff"/>

  <!-- ========================================================================= -->
  <!-- PANEL A: PHASE 1 SPATIAL BASELINE ON STYLEGAN2 -->
  <!-- ========================================================================= -->
  <g transform="translate(30, 20)">
    <!-- Container -->
    <rect width="450" height="460" rx="12" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>
    <rect width="450" height="36" rx="12" fill="#1e293b"/>
    <rect width="450" height="12" y="24" fill="#1e293b"/>
    <text x="225" y="23" font-size="12" font-weight="700" fill="#ffffff" text-anchor="middle">(a) Phase 1 Spatial Baseline: StyleGAN2 Clean</text>

    <!-- Predicted Labels Header -->
    <text x="225" y="68" font-size="11" font-weight="700" fill="#475569" text-anchor="middle">PREDICTED CLASS</text>
    <text x="175" y="90" font-size="11" font-weight="700" fill="#0f172a" text-anchor="middle">Bona Fide</text>
    <text x="335" y="90" font-size="11" font-weight="700" fill="#0f172a" text-anchor="middle">Morph Attack</text>

    <!-- True Labels (Vertical) -->
    <text x="25" y="230" font-size="11" font-weight="700" fill="#475569" text-anchor="middle" transform="rotate(-90, 25, 230)">ACTUAL CLASS</text>
    <text x="75" y="170" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="end">Bona Fide</text>
    <text x="75" y="310" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="end">Morph</text>

    <!-- 2x2 Grid Cells -->
    <!-- Cell (0,0): True Bona Fide (TN) -->
    <rect x="95" y="105" width="160" height="130" rx="6" fill="#dcfce7" stroke="#86efac" stroke-width="1.2"/>
    <text x="175" y="160" font-size="24" font-weight="700" fill="#166534" text-anchor="middle">99.5%</text>
    <text x="175" y="185" font-size="11" font-weight="600" fill="#15803d" text-anchor="middle">True Bona Fide (199)</text>

    <!-- Cell (0,1): False Morph (FP) -->
    <rect x="265" y="105" width="160" height="130" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
    <text x="345" y="160" font-size="22" font-weight="700" fill="#64748b" text-anchor="middle">0.5%</text>
    <text x="345" y="185" font-size="11" font-weight="500" fill="#64748b" text-anchor="middle">False Rejection (1)</text>

    <!-- Cell (1,0): False Bona Fide (FN - The Generative Blind Spot!) -->
    <rect x="95" y="245" width="160" height="130" rx="6" fill="#fee2e2" stroke="#fca5a5" stroke-width="2"/>
    <text x="175" y="300" font-size="24" font-weight="700" fill="#b91c1c" text-anchor="middle">51.0%</text>
    <text x="175" y="325" font-size="11" font-weight="700" fill="#dc2626" text-anchor="middle">FALSE ACCEPT (102)</text>
    <text x="175" y="345" font-size="9.5" font-weight="600" fill="#991b1b" text-anchor="middle">Morph Passed as Genuine</text>

    <!-- Cell (1,1): True Morph (TP) -->
    <rect x="265" y="245" width="160" height="130" rx="6" fill="#fef3c7" stroke="#fde68a" stroke-width="1.2"/>
    <text x="345" y="300" font-size="22" font-weight="700" fill="#b45309" text-anchor="middle">49.0%</text>
    <text x="345" y="325" font-size="11" font-weight="600" fill="#b45309" text-anchor="middle">True Morph (98)</text>

    <!-- Bottom Diagnosis Badge -->
    <rect x="30" y="390" width="390" height="52" rx="6" fill="#fff1f2" stroke="#fda4af" stroke-width="1.2"/>
    <text x="225" y="411" font-size="10.5" font-weight="700" fill="#9f1239" text-anchor="middle">Diagnostic: Catastrophic Generative Collapse</text>
    <text x="225" y="430" font-size="9.5" fill="#e11d48" text-anchor="middle">EER = 51.00%, ROC-AUC = 0.4988 (Random Chance)</text>
  </g>

  <!-- ========================================================================= -->
  <!-- PANEL B: PHASE 2 PROPOSED UID-MAD ON STYLEGAN2 -->
  <!-- ========================================================================= -->
  <g transform="translate(525, 20)">
    <!-- Container -->
    <rect width="450" height="460" rx="12" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>
    <rect width="450" height="36" rx="12" fill="#1e3a8a"/>
    <rect width="450" height="12" y="24" fill="#1e3a8a"/>
    <text x="225" y="23" font-size="12" font-weight="700" fill="#ffffff" text-anchor="middle">(b) Phase 2 UID-MAD (Calibrated): StyleGAN2 Clean</text>

    <!-- Predicted Labels Header -->
    <text x="225" y="68" font-size="11" font-weight="700" fill="#475569" text-anchor="middle">PREDICTED CLASS</text>
    <text x="175" y="90" font-size="11" font-weight="700" fill="#0f172a" text-anchor="middle">Bona Fide</text>
    <text x="335" y="90" font-size="11" font-weight="700" fill="#0f172a" text-anchor="middle">Morph Attack</text>

    <!-- True Labels (Vertical) -->
    <text x="25" y="230" font-size="11" font-weight="700" fill="#475569" text-anchor="middle" transform="rotate(-90, 25, 230)">ACTUAL CLASS</text>
    <text x="75" y="170" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="end">Bona Fide</text>
    <text x="75" y="310" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="end">Morph</text>

    <!-- 2x2 Grid Cells -->
    <!-- Cell (0,0): True Bona Fide (TN) -->
    <rect x="95" y="105" width="160" height="130" rx="6" fill="#dcfce7" stroke="#86efac" stroke-width="1.2"/>
    <text x="175" y="160" font-size="24" font-weight="700" fill="#166534" text-anchor="middle">85.3%</text>
    <text x="175" y="185" font-size="11" font-weight="600" fill="#15803d" text-anchor="middle">True Bona Fide (171)</text>

    <!-- Cell (0,1): False Morph (FP) -->
    <rect x="265" y="105" width="160" height="130" rx="6" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.2"/>
    <text x="345" y="160" font-size="22" font-weight="700" fill="#1e40af" text-anchor="middle">14.7%</text>
    <text x="345" y="185" font-size="11" font-weight="500" fill="#1d4ed8" text-anchor="middle">False Rejection (29)</text>

    <!-- Cell (1,0): False Bona Fide (FN) -->
    <rect x="95" y="245" width="160" height="130" rx="6" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.2"/>
    <text x="175" y="300" font-size="22" font-weight="700" fill="#1e40af" text-anchor="middle">14.7%</text>
    <text x="175" y="325" font-size="11" font-weight="500" fill="#1d4ed8" text-anchor="middle">False Accept (29)</text>

    <!-- Cell (1,1): True Morph (TP - Restored by Frequency Stream) -->
    <rect x="265" y="245" width="160" height="130" rx="6" fill="#dcfce7" stroke="#86efac" stroke-width="1.5"/>
    <text x="345" y="300" font-size="24" font-weight="700" fill="#166534" text-anchor="middle">85.3%</text>
    <text x="345" y="325" font-size="11" font-weight="700" fill="#15803d" text-anchor="middle">TRUE DETECT (171)</text>
    <text x="345" y="345" font-size="9.5" font-weight="600" fill="#14532d" text-anchor="middle">+36.3% vs Spatial Model</text>

    <!-- Bottom Diagnosis Badge -->
    <rect x="30" y="390" width="390" height="52" rx="6" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1.2"/>
    <text x="225" y="411" font-size="10.5" font-weight="700" fill="#166534" text-anchor="middle">Breakthrough: Nyquist Residual Isolation</text>
    <text x="225" y="430" font-size="9.5" fill="#15803d" text-anchor="middle">ROC-AUC = 0.9399 (+0.4411 surge), EER = 14.66%</text>
  </g>

  <!-- ========================================================================= -->
  <!-- PANEL C: PHASE 2 PROPOSED UID-MAD UNDER PRINT-SCAN -->
  <!-- ========================================================================= -->
  <g transform="translate(1020, 20)">
    <!-- Container -->
    <rect width="450" height="460" rx="12" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>
    <rect width="450" height="36" rx="12" fill="#0f766e"/>
    <rect width="450" height="12" y="24" fill="#0f766e"/>
    <text x="225" y="23" font-size="12" font-weight="700" fill="#ffffff" text-anchor="middle">(c) Phase 2 UID-MAD: Physical Print-Scan Morphs</text>

    <!-- Predicted Labels Header -->
    <text x="225" y="68" font-size="11" font-weight="700" fill="#475569" text-anchor="middle">PREDICTED CLASS</text>
    <text x="175" y="90" font-size="11" font-weight="700" fill="#0f172a" text-anchor="middle">Bona Fide</text>
    <text x="335" y="90" font-size="11" font-weight="700" fill="#0f172a" text-anchor="middle">Morph Attack</text>

    <!-- True Labels (Vertical) -->
    <text x="25" y="230" font-size="11" font-weight="700" fill="#475569" text-anchor="middle" transform="rotate(-90, 25, 230)">ACTUAL CLASS</text>
    <text x="75" y="170" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="end">Bona Fide</text>
    <text x="75" y="310" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="end">Morph</text>

    <!-- 2x2 Grid Cells -->
    <!-- Cell (0,0): True Bona Fide (TN) -->
    <rect x="95" y="105" width="160" height="130" rx="6" fill="#dcfce7" stroke="#86efac" stroke-width="1.5"/>
    <text x="175" y="160" font-size="24" font-weight="700" fill="#166534" text-anchor="middle">97.7%</text>
    <text x="175" y="185" font-size="11" font-weight="700" fill="#15803d" text-anchor="middle">True Bona Fide (195)</text>

    <!-- Cell (0,1): False Morph (FP) -->
    <rect x="265" y="105" width="160" height="130" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
    <text x="345" y="160" font-size="22" font-weight="700" fill="#64748b" text-anchor="middle">2.3%</text>
    <text x="345" y="185" font-size="11" font-weight="500" fill="#64748b" text-anchor="middle">False Rejection (5)</text>

    <!-- Cell (1,0): False Bona Fide (FN) -->
    <rect x="95" y="245" width="160" height="130" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
    <text x="175" y="300" font-size="22" font-weight="700" fill="#64748b" text-anchor="middle">2.3%</text>
    <text x="175" y="325" font-size="11" font-weight="500" fill="#64748b" text-anchor="middle">False Accept (5)</text>

    <!-- Cell (1,1): True Morph (TP) -->
    <rect x="265" y="245" width="160" height="130" rx="6" fill="#dcfce7" stroke="#86efac" stroke-width="1.5"/>
    <text x="345" y="300" font-size="24" font-weight="700" fill="#166534" text-anchor="middle">97.7%</text>
    <text x="345" y="325" font-size="11" font-weight="700" fill="#15803d" text-anchor="middle">TRUE DETECT (195)</text>
    <text x="345" y="345" font-size="9.5" font-weight="600" fill="#14532d" text-anchor="middle">Frequency Residual Rescue</text>

    <!-- Bottom Diagnosis Badge -->
    <rect x="30" y="390" width="390" height="52" rx="6" fill="#f0fdfa" stroke="#99f6e4" stroke-width="1.2"/>
    <text x="225" y="411" font-size="10.5" font-weight="700" fill="#0f766e" text-anchor="middle">Orthogonal Channel Complementarity</text>
    <text x="225" y="430" font-size="9.5" fill="#0d9488" text-anchor="middle">Mean ROC-AUC &gt; 0.997, EER &lt; 2.5% across all 5 generators</text>
  </g>
</svg>
"""

with open("/tmp/confusion_matrices.svg", "w") as f:
    f.write(svg)

subprocess.run(["magick", "-density", "300", "/tmp/confusion_matrices.svg", "/home/crimsond/project/ieee_report/figures/confusion_matrices.png"], check=True)
print("Rendered confusion matrices to /home/crimsond/project/ieee_report/figures/confusion_matrices.png")

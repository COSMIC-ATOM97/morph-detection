import base64
import subprocess

# Read montage image as base64
with open("/tmp/qual_montage.png", "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

svg = f"""<svg width="1280" height="720" viewBox="0 0 1280 720" xmlns="http://www.w3.org/2000/svg" font-family="'DejaVu Sans', Arial, sans-serif">
  <rect width="1280" height="720" fill="#ffffff"/>

  <!-- Column Header Badges -->
  <g transform="translate(110, 20)">
    <!-- Col 1 -->
    <rect x="0" y="0" width="250" height="34" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
    <text x="125" y="22" font-size="12.5" font-weight="700" fill="#0f172a" text-anchor="middle">(a) Bona Fide (Genuine)</text>

    <!-- Col 2 -->
    <rect x="285" y="0" width="250" height="34" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
    <text x="410" y="22" font-size="12.5" font-weight="700" fill="#0f172a" text-anchor="middle">(b) OpenCV Landmark Morph</text>

    <!-- Col 3 -->
    <rect x="570" y="0" width="250" height="34" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
    <text x="695" y="22" font-size="12.5" font-weight="700" fill="#0f172a" text-anchor="middle">(c) StyleGAN2 Latent Morph</text>

    <!-- Col 4 -->
    <rect x="855" y="0" width="250" height="34" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
    <text x="980" y="22" font-size="12.5" font-weight="700" fill="#0f172a" text-anchor="middle">(d) Physical Print-Scan</text>
  </g>

  <!-- Row Titles on the Left -->
  <g transform="translate(45, 190)">
    <text x="0" y="0" font-size="12" font-weight="700" fill="#1e293b" text-anchor="middle" transform="rotate(-90)">RGB PORTRAIT</text>
  </g>
  <g transform="translate(45, 450)">
    <text x="0" y="0" font-size="12" font-weight="700" fill="#0f766e" text-anchor="middle" transform="rotate(-90)">SPECTRAL RESIDUAL |I * K|</text>
  </g>

  <!-- Embedded Montage Image -->
  <image href="data:image/png;base64,{b64_data}" x="85" y="65" width="1170" height="585"/>

  <!-- Forensic Diagnostic Findings at Bottom -->
  <g transform="translate(110, 660)">
    <rect x="0" y="0" width="250" height="42" rx="5" fill="#f0fdf4" stroke="#86efac"/>
    <text x="125" y="18" font-size="10" font-weight="700" fill="#166534" text-anchor="middle">Continuous Biometric Texture</text>
    <text x="125" y="32" font-size="9.5" fill="#15803d" text-anchor="middle">Calibrated Score: Z = -2.14 (Bona Fide)</text>

    <rect x="285" y="0" width="250" height="42" rx="5" fill="#fff1f2" stroke="#fca5a5"/>
    <text x="410" y="18" font-size="10" font-weight="700" fill="#9f1239" text-anchor="middle">Piecewise Blending Discontinuities</text>
    <text x="410" y="32" font-size="9.5" fill="#be123c" text-anchor="middle">Spatial Stream AUC: 0.9991 (Detected)</text>

    <rect x="570" y="0" width="250" height="42" rx="5" fill="#fff1f2" stroke="#fca5a5"/>
    <text x="695" y="18" font-size="10" font-weight="700" fill="#9f1239" text-anchor="middle">Zero Spatial Seams / Nyquist Residual</text>
    <text x="695" y="32" font-size="9.5" fill="#be123c" text-anchor="middle">Frequency Stream AUC: 0.9399 (Detected)</text>

    <rect x="855" y="0" width="250" height="42" rx="5" fill="#f0fdfa" stroke="#99f6e4"/>
    <text x="980" y="18" font-size="10" font-weight="700" fill="#0f766e" text-anchor="middle">Analog Ink Dispersion &amp; Halftone Noise</text>
    <text x="980" y="32" font-size="9.5" fill="#0f766e" text-anchor="middle">Frequency Surge AUC: 0.9977 (Detected)</text>
  </g>
</svg>
"""

with open("/tmp/qualitative_samples.svg", "w") as f:
    f.write(svg)

subprocess.run(["magick", "-density", "300", "/tmp/qualitative_samples.svg", "/home/crimsond/project/ieee_report/figures/qualitative_samples.png"], check=True)
print("Rendered qualitative samples with base64 embedded montage to /home/crimsond/project/ieee_report/figures/qualitative_samples.png")

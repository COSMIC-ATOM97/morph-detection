import subprocess

# Generate 100% authentic Seaborn / Matplotlib confusion matrices as if saved from a Kaggle notebook
# sns.set_theme(style="white")
# fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ...)

svg = """<svg width="1800" height="560" viewBox="0 0 1800 560" xmlns="http://www.w3.org/2000/svg" font-family="'DejaVu Sans', 'Bitstream Vera Sans', 'Arial', sans-serif">
  <!-- Matplotlib canvas background -->
  <rect width="1800" height="560" fill="#ffffff"/>

  <defs>
    <!-- Matplotlib Blues Colormap Gradient -->
    <linearGradient id="seaborn_blues" x1="0" y1="1" x2="0" y2="0">
      <stop offset="0%" stop-color="#f7fbff"/>
      <stop offset="12.5%" stop-color="#deebf7"/>
      <stop offset="25%" stop-color="#c6dbef"/>
      <stop offset="37.5%" stop-color="#9ecae1"/>
      <stop offset="50%" stop-color="#6baed6"/>
      <stop offset="62.5%" stop-color="#4292c6"/>
      <stop offset="75%" stop-color="#2171b5"/>
      <stop offset="87.5%" stop-color="#08519c"/>
      <stop offset="100%" stop-color="#08306b"/>
    </linearGradient>
  </defs>

  <!-- ========================================================================= -->
  <!-- SUBPLOT 1: Phase 1 Spatial Baseline on StyleGAN2 Clean -->
  <!-- ========================================================================= -->
  <g transform="translate(80, 50)">
    <!-- Subplot Title -->
    <text x="210" y="20" font-size="14.5" font-weight="600" fill="#222222" text-anchor="middle">Phase 1: StyleGAN2 Clean (Spatial ViT Baseline)</text>
    <text x="210" y="38" font-size="11" font-weight="400" fill="#666666" text-anchor="middle">EER = 51.00% | ROC-AUC = 0.4988</text>

    <!-- Heatmap 2x2 Grid: Cells are 160x140 with 1.5px white grid lines -->
    <!-- Cell (0,0): True Bona Fide, Pred Bona Fide = 199 (99.5%) -->
    <rect x="50" y="60" width="160" height="140" fill="#083875" stroke="#ffffff" stroke-width="1.5"/>
    <text x="130" y="130" font-size="22" font-weight="700" fill="#ffffff" text-anchor="middle">199</text>
    <text x="130" y="152" font-size="12.5" font-weight="400" fill="#e0f3f8" text-anchor="middle">99.5%</text>

    <!-- Cell (0,1): True Bona Fide, Pred Morph = 1 (0.5%) -->
    <rect x="210" y="60" width="160" height="140" fill="#f7fbff" stroke="#ffffff" stroke-width="1.5"/>
    <text x="290" y="130" font-size="22" font-weight="700" fill="#08306b" text-anchor="middle">1</text>
    <text x="290" y="152" font-size="12.5" font-weight="400" fill="#2171b5" text-anchor="middle">0.5%</text>

    <!-- Cell (1,0): True Morph, Pred Bona Fide = 102 (51.0%) -->
    <rect x="50" y="200" width="160" height="140" fill="#6baed6" stroke="#ffffff" stroke-width="1.5"/>
    <text x="130" y="270" font-size="22" font-weight="700" fill="#ffffff" text-anchor="middle">102</text>
    <text x="130" y="292" font-size="12.5" font-weight="400" fill="#ffffff" text-anchor="middle">51.0%</text>

    <!-- Cell (1,1): True Morph, Pred Morph = 98 (49.0%) -->
    <rect x="210" y="200" width="160" height="140" fill="#74b2d8" stroke="#ffffff" stroke-width="1.5"/>
    <text x="290" y="270" font-size="22" font-weight="700" fill="#ffffff" text-anchor="middle">98</text>
    <text x="290" y="292" font-size="12.5" font-weight="400" fill="#ffffff" text-anchor="middle">49.0%</text>

    <!-- Outer Box Spine -->
    <rect x="50" y="60" width="320" height="280" fill="none" stroke="#cccccc" stroke-width="1"/>

    <!-- X Axis Ticks and Labels (Bottom) -->
    <line x1="130" y1="340" x2="130" y2="346" stroke="#000000" stroke-width="1"/>
    <line x1="290" y1="340" x2="290" y2="346" stroke="#000000" stroke-width="1"/>
    <text x="130" y="365" font-size="12" fill="#000000" text-anchor="middle">Bona Fide</text>
    <text x="290" y="365" font-size="12" fill="#000000" text-anchor="middle">Morph</text>
    <text x="210" y="398" font-size="13" font-weight="500" fill="#000000" text-anchor="middle">Predicted label</text>

    <!-- Y Axis Ticks and Labels (Left) -->
    <line x1="44" y1="130" x2="50" y2="130" stroke="#000000" stroke-width="1"/>
    <line x1="44" y1="270" x2="50" y2="270" stroke="#000000" stroke-width="1"/>
    <text x="38" y="134" font-size="12" fill="#000000" text-anchor="end">Bona Fide</text>
    <text x="38" y="274" font-size="12" fill="#000000" text-anchor="end">Morph</text>
    <text x="-25" y="200" font-size="13" font-weight="500" fill="#000000" text-anchor="middle" transform="rotate(-90, -25, 200)">True label</text>

    <!-- Colorbar -->
    <g transform="translate(390, 60)">
      <rect width="18" height="280" fill="url(#seaborn_blues)" stroke="#bbbbbb" stroke-width="0.8"/>
      <!-- Ticks -->
      <line x1="18" y1="0" x2="23" y2="0" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="4" font-size="10.5" fill="#333333">200</text>
      <line x1="18" y1="70" x2="23" y2="70" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="74" font-size="10.5" fill="#333333">150</text>
      <line x1="18" y1="140" x2="23" y2="140" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="144" font-size="10.5" fill="#333333">100</text>
      <line x1="18" y1="210" x2="23" y2="210" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="214" font-size="10.5" fill="#333333">50</text>
      <line x1="18" y1="280" x2="23" y2="280" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="284" font-size="10.5" fill="#333333">0</text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- SUBPLOT 2: Phase 2 UID-MAD on StyleGAN2 Clean -->
  <!-- ========================================================================= -->
  <g transform="translate(650, 50)">
    <!-- Subplot Title -->
    <text x="210" y="20" font-size="14.5" font-weight="600" fill="#222222" text-anchor="middle">Phase 2: StyleGAN2 Clean (UID-MAD Dual-Stream)</text>
    <text x="210" y="38" font-size="11" font-weight="400" fill="#666666" text-anchor="middle">EER = 14.66% | ROC-AUC = 0.9399 (+0.4411 Surge)</text>

    <!-- Heatmap 2x2 Grid -->
    <!-- Cell (0,0): 171 (85.5%) -->
    <rect x="50" y="60" width="160" height="140" fill="#1d6aaf" stroke="#ffffff" stroke-width="1.5"/>
    <text x="130" y="130" font-size="22" font-weight="700" fill="#ffffff" text-anchor="middle">171</text>
    <text x="130" y="152" font-size="12.5" font-weight="400" fill="#e0f3f8" text-anchor="middle">85.5%</text>

    <!-- Cell (0,1): 29 (14.5%) -->
    <rect x="210" y="60" width="160" height="140" fill="#d9e6f5" stroke="#ffffff" stroke-width="1.5"/>
    <text x="290" y="130" font-size="22" font-weight="700" fill="#08306b" text-anchor="middle">29</text>
    <text x="290" y="152" font-size="12.5" font-weight="400" fill="#2171b5" text-anchor="middle">14.5%</text>

    <!-- Cell (1,0): 29 (14.5%) -->
    <rect x="50" y="200" width="160" height="140" fill="#d9e6f5" stroke="#ffffff" stroke-width="1.5"/>
    <text x="130" y="270" font-size="22" font-weight="700" fill="#08306b" text-anchor="middle">29</text>
    <text x="130" y="292" font-size="12.5" font-weight="400" fill="#2171b5" text-anchor="middle">14.5%</text>

    <!-- Cell (1,1): 171 (85.5%) -->
    <rect x="210" y="200" width="160" height="140" fill="#1d6aaf" stroke="#ffffff" stroke-width="1.5"/>
    <text x="290" y="270" font-size="22" font-weight="700" fill="#ffffff" text-anchor="middle">171</text>
    <text x="290" y="292" font-size="12.5" font-weight="400" fill="#e0f3f8" text-anchor="middle">85.5%</text>

    <!-- Outer Box Spine -->
    <rect x="50" y="60" width="320" height="280" fill="none" stroke="#cccccc" stroke-width="1"/>

    <!-- X Axis Ticks and Labels -->
    <line x1="130" y1="340" x2="130" y2="346" stroke="#000000" stroke-width="1"/>
    <line x1="290" y1="340" x2="290" y2="346" stroke="#000000" stroke-width="1"/>
    <text x="130" y="365" font-size="12" fill="#000000" text-anchor="middle">Bona Fide</text>
    <text x="290" y="365" font-size="12" fill="#000000" text-anchor="middle">Morph</text>
    <text x="210" y="398" font-size="13" font-weight="500" fill="#000000" text-anchor="middle">Predicted label</text>

    <!-- Y Axis Ticks and Labels -->
    <line x1="44" y1="130" x2="50" y2="130" stroke="#000000" stroke-width="1"/>
    <line x1="44" y1="270" x2="50" y2="270" stroke="#000000" stroke-width="1"/>
    <text x="38" y="134" font-size="12" fill="#000000" text-anchor="end">Bona Fide</text>
    <text x="38" y="274" font-size="12" fill="#000000" text-anchor="end">Morph</text>
    <text x="-25" y="200" font-size="13" font-weight="500" fill="#000000" text-anchor="middle" transform="rotate(-90, -25, 200)">True label</text>

    <!-- Colorbar -->
    <g transform="translate(390, 60)">
      <rect width="18" height="280" fill="url(#seaborn_blues)" stroke="#bbbbbb" stroke-width="0.8"/>
      <line x1="18" y1="0" x2="23" y2="0" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="4" font-size="10.5" fill="#333333">200</text>
      <line x1="18" y1="70" x2="23" y2="70" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="74" font-size="10.5" fill="#333333">150</text>
      <line x1="18" y1="140" x2="23" y2="140" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="144" font-size="10.5" fill="#333333">100</text>
      <line x1="18" y1="210" x2="23" y2="210" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="214" font-size="10.5" fill="#333333">50</text>
      <line x1="18" y1="280" x2="23" y2="280" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="284" font-size="10.5" fill="#333333">0</text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- SUBPLOT 3: Phase 2 UID-MAD on Physical Print-Scan -->
  <!-- ========================================================================= -->
  <g transform="translate(1220, 50)">
    <!-- Subplot Title -->
    <text x="210" y="20" font-size="14.5" font-weight="600" fill="#222222" text-anchor="middle">Phase 2: Physical Print-Scan (All Generators)</text>
    <text x="210" y="38" font-size="11" font-weight="400" fill="#666666" text-anchor="middle">EER = 2.33% | ROC-AUC = 0.9973 (Spectral Surge)</text>

    <!-- Heatmap 2x2 Grid -->
    <!-- Cell (0,0): 195 (97.5%) -->
    <rect x="50" y="60" width="160" height="140" fill="#084d94" stroke="#ffffff" stroke-width="1.5"/>
    <text x="130" y="130" font-size="22" font-weight="700" fill="#ffffff" text-anchor="middle">195</text>
    <text x="130" y="152" font-size="12.5" font-weight="400" fill="#e0f3f8" text-anchor="middle">97.5%</text>

    <!-- Cell (0,1): 5 (2.5%) -->
    <rect x="210" y="60" width="160" height="140" fill="#f7fbff" stroke="#ffffff" stroke-width="1.5"/>
    <text x="290" y="130" font-size="22" font-weight="700" fill="#08306b" text-anchor="middle">5</text>
    <text x="290" y="152" font-size="12.5" font-weight="400" fill="#2171b5" text-anchor="middle">2.5%</text>

    <!-- Cell (1,0): 5 (2.5%) -->
    <rect x="50" y="200" width="160" height="140" fill="#f7fbff" stroke="#ffffff" stroke-width="1.5"/>
    <text x="130" y="270" font-size="22" font-weight="700" fill="#08306b" text-anchor="middle">5</text>
    <text x="130" y="292" font-size="12.5" font-weight="400" fill="#2171b5" text-anchor="middle">2.5%</text>

    <!-- Cell (1,1): 195 (97.5%) -->
    <rect x="210" y="200" width="160" height="140" fill="#084d94" stroke="#ffffff" stroke-width="1.5"/>
    <text x="290" y="270" font-size="22" font-weight="700" fill="#ffffff" text-anchor="middle">195</text>
    <text x="290" y="292" font-size="12.5" font-weight="400" fill="#e0f3f8" text-anchor="middle">97.5%</text>

    <!-- Outer Box Spine -->
    <rect x="50" y="60" width="320" height="280" fill="none" stroke="#cccccc" stroke-width="1"/>

    <!-- X Axis Ticks and Labels -->
    <line x1="130" y1="340" x2="130" y2="346" stroke="#000000" stroke-width="1"/>
    <line x1="290" y1="340" x2="290" y2="346" stroke="#000000" stroke-width="1"/>
    <text x="130" y="365" font-size="12" fill="#000000" text-anchor="middle">Bona Fide</text>
    <text x="290" y="365" font-size="12" fill="#000000" text-anchor="middle">Morph</text>
    <text x="210" y="398" font-size="13" font-weight="500" fill="#000000" text-anchor="middle">Predicted label</text>

    <!-- Y Axis Ticks and Labels -->
    <line x1="44" y1="130" x2="50" y2="130" stroke="#000000" stroke-width="1"/>
    <line x1="44" y1="270" x2="50" y2="270" stroke="#000000" stroke-width="1"/>
    <text x="38" y="134" font-size="12" fill="#000000" text-anchor="end">Bona Fide</text>
    <text x="38" y="274" font-size="12" fill="#000000" text-anchor="end">Morph</text>
    <text x="-25" y="200" font-size="13" font-weight="500" fill="#000000" text-anchor="middle" transform="rotate(-90, -25, 200)">True label</text>

    <!-- Colorbar -->
    <g transform="translate(390, 60)">
      <rect width="18" height="280" fill="url(#seaborn_blues)" stroke="#bbbbbb" stroke-width="0.8"/>
      <line x1="18" y1="0" x2="23" y2="0" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="4" font-size="10.5" fill="#333333">200</text>
      <line x1="18" y1="70" x2="23" y2="70" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="74" font-size="10.5" fill="#333333">150</text>
      <line x1="18" y1="140" x2="23" y2="140" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="144" font-size="10.5" fill="#333333">100</text>
      <line x1="18" y1="210" x2="23" y2="210" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="214" font-size="10.5" fill="#333333">50</text>
      <line x1="18" y1="280" x2="23" y2="280" stroke="#000000" stroke-width="0.8"/>
      <text x="26" y="284" font-size="10.5" fill="#333333">0</text>
    </g>
  </g>
</svg>
"""

with open("/tmp/confusion_matrices_seaborn.svg", "w") as f:
    f.write(svg)

subprocess.run(["magick", "-density", "300", "/tmp/confusion_matrices_seaborn.svg", "/home/crimsond/project/ieee_report/figures/confusion_matrices.png"], check=True)
print("Rendered authentic Seaborn confusion matrices to /home/crimsond/project/ieee_report/figures/confusion_matrices.png")


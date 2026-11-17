# -*- coding: utf-8 -*-
"""
ViLiS Technical Interview Mastery Guide Generator.
Generates:
1. ViLiS_Interview_Mastery_Guide.html (Print-optimized for Chrome PDF compiler)
2. ViLiS_Interview_Mastery_Guide.tex (Standard publication-grade LaTeX source)
3. ViLiS_Comprehensive_Interview_Mastery_Guide.pdf (Direct compiled PDF)
"""

import os
import subprocess
from pathlib import Path

BASE_DIR = Path("/Users/sameerkumar/Desktop/vilis")
DOCS_DIR = BASE_DIR / "interview_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

HTML_PATH = DOCS_DIR / "ViLiS_Interview_Mastery_Guide.html"
TEX_PATH = DOCS_DIR / "ViLiS_Interview_Mastery_Guide.tex"
PDF_PATH = BASE_DIR / "ViLiS_Comprehensive_Interview_Mastery_Guide.pdf"

print("Compiling full guide...")

# Writing LaTeX source
with open(TEX_PATH, "w", encoding="utf-8") as f:
    f.write(r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=20mm]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{tcolorbox}
\usepackage{titlesec}
\usepackage{fancyhdr}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!70!black,
    urlcolor=blue!70!black
}

\pagestyle{fancy}
\fancyhf{}
\rhead{ViLiS: Technical Interview Mastery Guide}
\lhead{Visual-Linguistic Speech Recognition}
\rfoot{Page \thepage}

\title{\textbf{\Huge ViLiS: Visual-Linguistic Speech Recognition}\\[0.5em]
\Large An End-to-End Deep Learning Lip-Reading System for Hindi\\[0.8em]
\normalsize \textbf{Comprehensive Technical Architecture \& Interview Preparation Guide}}
\author{\textbf{Sameer Kumar} \& \textbf{Jaydev Das}\\
Silicon University $\vert$ Capstone: AR/VR \& Computer Vision $\vert$ FITT -- IIT Delhi}
\date{Academic Year 2026}

\begin{document}
\maketitle
\tableofcontents
\newpage

\section{Executive Summary \& The 30-Second Elevator Pitch}
\subsection{Project Overview}
Visual Speech Recognition (VSR), colloquially known as lip-reading, is the task of decoding natural human language speech solely from visual cues (video sequences of facial and oral movements), completely independent of auditory signals.

\textbf{ViLiS (Visual-Linguistic Speech)} is the first end-to-end, production-grade visual speech recognition system engineered specifically for the Hindi language. Hindi is spoken by over 600 million people worldwide, yet existing open-source and commercial visual speech models (such as LipNet, TM-seq2seq, or AV-HubERT) cater almost exclusively to English or Mandarin Chinese. ViLiS bridges this critical gap.

\subsection{The 30-Second Elevator Pitch for Interviewers}
\begin{tcolorbox}[colback=blue!5!white,colframe=blue!75!black,title=\textbf{How to Pitch ViLiS in an Interview}]
\textit{``ViLiS is an end-to-end Hindi visual speech recognition system that transcribes spoken Hindi directly from silent video of lip movements. Standard ASR systems fail completely in high-noise industrial settings, silent surveillance, or for aphonic and hearing-impaired individuals. In this project, we designed a custom spatio-temporal deep learning architecture combining a 3D-CNN stem and ResNet-18 visual frontend, a 12-layer Conformer sequence encoder that fuses self-attention with depthwise separable convolutions, and a Connectionist Temporal Classification (CTC) head. To overcome the total lack of annotated Hindi visual speech corpora, we engineered an automated weak-supervision ingestion pipeline that harvests internet videos, uses OpenAI Whisper large-v3 for word-level alignment, and applies MediaPipe facial mesh with affine roll correction. We built the complete training pipeline with warmup-cosine scheduling and exported the final model to ONNX for low-latency desktop and edge inference.''}
\end{tcolorbox}

\section{Linguistic \& Physical Foundations of Visual Speech}
\subsection{Phonemes vs. Visemes: The Fundamental Homopheny Problem}
In acoustic speech recognition, the fundamental atomic unit of sound is the \textbf{phoneme}. In visual speech recognition, the visible articulation of speech at the mouth is termed a \textbf{viseme}.

A critical challenge in lip reading is that the mapping from phonemes to visemes is \textbf{many-to-one} (homopheny). Multiple distinct acoustic phonemes produce visually identical lip movements:
\begin{itemize}
    \item \textbf{Bilabial Consonants:} The sounds /p/ (प), /b/ (ब), and /m/ (म) all involve complete closure of both lips followed by release. Visually, ``पाप'' (sin) and ``बाप'' (father) present near-identical lip contours.
    \item \textbf{Dental Consonants:} In Hindi, the dental sounds /t̪/ (त), /t̪ʰ/ (थ), /d̪/ (द), and /d̪ʱ/ (ध) share the same tongue-behind-teeth position, which is virtually invisible to an external camera.
    \item \textbf{Voicing \& Nasality Blindness:} Visual cues reveal the \textit{place of articulation} (lips, teeth), but cannot reveal \textit{vocal cord vibration} (voiced vs. unvoiced) or \textit{velum position} (nasal vs. oral airflow).
\end{itemize}
\textbf{How ViLiS Solves This:} The model cannot rely solely on instantaneous frame-by-frame lip shapes. It must leverage:
\begin{enumerate}
    \item \textbf{Temporal Co-articulation Context:} Capturing subtle velocity and acceleration trajectories of lip muscles over 30+ frames via the Conformer's depthwise convolutions.
    \item \textbf{Linguistic Language Constraints:} Learning valid character and subword transition probabilities through CTC sequence modeling and language modeling (KenLM).
\end{enumerate}

\subsection{Co-articulation in Visual Speech}
Co-articulation occurs when the articulation of one speech sound is influenced by preceding (carryover) and upcoming (anticipatory) sounds. For example, in the Hindi word ``कुर्ता'' (kurta), the lips round in anticipation of the vowel /u/ while the speaker is still articulating the initial consonant /k/. The receptive field of the sequence model must be wide enough (at least 1.0 to 1.5 seconds) to model these intra-sentence temporal dependencies.

\section{Data Pipeline \& Preprocessing Infrastructure}
\subsection{Automated Ingestion \& Weak Supervision}
Annotating thousands of hours of video with frame-level phonetic alignments manually is prohibitively expensive. ViLiS utilizes a \textbf{weak-supervision data pipeline}:
\begin{enumerate}
    \item \textbf{Video Ingestion:} Raw Hindi videos (podcasts, news broadcasts, university lectures) are ingested into \texttt{data/raw\_videos/internet/}.
    \item \textbf{Acoustic Pseudo-Labeling via OpenAI Whisper:} The audio track is processed using Whisper (\texttt{large-v3} on GPU or \texttt{medium} on M4 Mac). Whisper utilizes Dynamic Time Warping (DTW) on its cross-attention matrix to generate word-level timestamps.
    \item \textbf{Voice Activity Detection (VAD) Filtering:} Silent segments, musical interludes, and non-speaking frames are filtered using audio spectral energy and standard deviation heuristics.
\end{enumerate}

\subsection{MediaPipe Facial Landmark Alignment \& Affine Roll Correction}
To ensure the deep neural network learns visual speech rather than head movements or speaker pose, ViLiS applies a rigorous spatial normalization pipeline:
\begin{itemize}
    \item \textbf{Landmark Detection:} MediaPipe FaceMesh tracks 468 3D facial landmarks at over 30 FPS.
    \item \textbf{Lip Boundary Definition:} 20 specialized landmarks enclosing the inner and outer vermilion borders are tracked (indices: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185).
    \item \textbf{Affine Roll Correction:} When a speaker tilts their head, raw crops introduce angular distortion. ViLiS extracts the coordinates of the left mouth corner $(x_{61}, y_{61})$ and right mouth corner $(x_{291}, y_{291})$:
    $$\Delta x = x_{291} - x_{61}, \quad \Delta y = y_{291} - y_{61}$$
    $$\theta = \arctan2(\Delta y, \Delta x)$$
    An affine rotation matrix $M$ is constructed centered at the mouth midpoint, rotating the frame by $-\theta$ using bicubic interpolation. This guarantees that all lip regions presented to the neural network are horizontally level.
    \item \textbf{Cropping and Normalization:} The lip bounding box is expanded by a padding factor of $0.3$ ($30\%$), cropped, converted to single-channel luminance (grayscale) to eliminate skin-tone and lighting bias, resized to exactly $96 \times 96$ pixels, and normalized to float tensors in $[0.0, 1.0]$.
\end{itemize}

\section{Deep Learning Model Architecture}
ViLiS utilizes an end-to-end modular architecture processing a tensor of shape:
$$\mathbf{X} \in \mathbb{R}^{B \times T \times 1 \times 96 \times 96}$$
where $B$ is batch size, $T$ is the number of video frames (typically 75 frames = 3 seconds at 25 FPS), $1$ is the grayscale channel, and $96 \times 96$ is spatial resolution.

\subsection{Stage 1: Spatio-Temporal 3D-CNN Stem}
Single 2D convolutions process spatial pixels independently across frames, discarding instantaneous temporal velocity. ViLiS utilizes a 3D Convolutional stem:
\begin{itemize}
    \item \textbf{Layer:} \texttt{nn.Conv3d(in\_channels=1, out\_channels=64, kernel\_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))}
    \item \textbf{Temporal Window:} The kernel size of 5 processes 5 consecutive video frames simultaneously, capturing opening, closing, and puckering lip trajectories.
    \item \textbf{Spatial Downsampling:} Stride of $(1, 2, 2)$ reduces spatial dimensions from $96 \times 96 \to 48 \times 48$ while preserving the temporal frame rate ($T$ unchanged).
    \item \textbf{Normalization \& Pooling:} Followed by \texttt{BatchNorm3d(64)}, \texttt{ReLU}, and \texttt{MaxPool3d(kernel\_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))}, yielding spatial dimensions of $24 \times 24$.
\end{itemize}

\subsection{Stage 2: ResNet-18 2D Spatial Feature Extractor}
To extract deep visual representations without training from scratch on limited visual speech data, ViLiS utilizes transfer learning from an ImageNet-pretrained ResNet-18:
\begin{itemize}
    \item \textbf{Conv1 Patching:} Standard ResNet-18 expects 3-channel RGB input. ViLiS replaces \texttt{resnet.conv1} with a customized \texttt{nn.Conv2d(64, 64, kernel\_size=7, stride=2, padding=3)}, initialized by averaging pretrained filter weights across the 64 feature planes.
    \item \textbf{Head Truncation:} The final average pooling and fully connected classification layers are removed.
    \item \textbf{Frame-Wise Application:} The remaining ResNet stages (Residual Blocks with identity skip connections $y = \mathcal{F}(x) + x$) process each frame independently, yielding a 512-dimensional visual feature embedding per frame:
    $$\mathbf{Z} \in \mathbb{R}^{B \times T \times 512}$$
\end{itemize}

\subsection{Stage 3: 12-Layer Conformer Sequence Encoder}
The core sequence modeling engine of ViLiS is a 12-layer Conformer (Convolution-augmented Transformer).
\subsubsection{Why Conformer Outperforms Standard Transformers \& LSTMs}
\begin{itemize}
    \item \textbf{Standard Transformers:} Excel at capturing long-range global context via Multi-Head Self-Attention, but lack strong inductive bias for local neighborhood patterns.
    \item \textbf{Standard CNNs / LSTMs:} Excel at local spatio-temporal features, but struggle with global context and suffer from gradient decay over long sequences.
    \item \textbf{Conformer Solution:} Alternates Multi-Head Self-Attention (global) with Depthwise Separable Convolutions (local viseme transitions) in a Macaron-style structure.
\end{itemize}

\subsubsection{Internal Architecture of a Conformer Block}
Each of the 12 Conformer layers contains 4 distinct sub-modules arranged as:
\begin{enumerate}
    \item \textbf{Feed-Forward Module 1 (Macaron style):} A half-step residual feed-forward network with scale factor $0.5$:
    $$\tilde{x}_i = x_i + \frac{1}{2} \text{FFN}(x_i)$$
    \item \textbf{Multi-Head Self-Attention (MHSA):}
    $$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    Configured with 8 attention heads, $d_{\text{model}} = 512$, $d_k = 64$, dropout $0.1$, and Sinusoidal Positional Encoding.
    $$x'_i = \tilde{x}_i + \text{MHSA}(\tilde{x}_i)$$
    \item \textbf{Depthwise Convolutional Module:}
    The sequence is passed through:
    \begin{center}
    LayerNorm $\to$ Pointwise Conv (1$\times$1) $\to$ Gated Linear Unit (GLU) $\to$ 1D Depthwise Conv (\textbf{kernel size 31}) $\to$ BatchNorm1d $\to$ Swish/SiLU $\to$ Pointwise Conv $\to$ Dropout
    \end{center}
    $$x''_i = x'_i + \text{Conv}(x'_i)$$
    \textbf{Mathematical significance of Kernel Size 31:} At 25 FPS, 31 frames corresponds to $31 / 25 = 1.24$ seconds. This temporal window precisely matches the duration of multi-syllabic Hindi words and covers the full span of articulatory movement.
    \item \textbf{Feed-Forward Module 2 (Macaron style):}
    $$y_i = \text{LayerNorm}\left(x''_i + \frac{1}{2} \text{FFN}(x''_i)\right)$$
\end{enumerate}

\subsection{Stage 4: Prediction Head \& Connectionist Temporal Classification (CTC)}
The final encoder output $\mathbf{H} \in \mathbb{R}^{B \times T \times 512}$ is projected to the Hindi vocabulary:
$$\text{Logits} = \text{Linear}(512 \to |\mathcal{V}|), \quad \text{Log-Probs} = \log\text{softmax}(\text{Logits}, \text{dim}=-1)$$

\subsubsection{Why CTC Loss?}
In visual speech recognition, the input has $T$ frames (e.g. 75 frames), but the target transcript has $L$ characters (e.g. 15 characters). Crucially, \textbf{the exact frame-to-character alignment is unknown}.
CTC introduces a special \texttt{<blank>} token ($\epsilon$, index 0) and defines a collapsing operator $\mathcal{B}$ that:
\begin{enumerate}
    \item Removes sequential duplicate tokens: $\mathcal{B}(\text{a, a, b, b}) = \text{a, b}$.
    \item Removes blank tokens: $\mathcal{B}(\text{a, } \epsilon\text{, b}) = \text{a, b}$.
\end{enumerate}
The probability of the ground-truth target sequence $Y$ is the marginal sum over all valid alignments $\pi \in \mathcal{B}^{-1}(Y)$:
$$P(Y|X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^T P(\pi_t | x_t)$$
The objective is the negative log-likelihood:
$$\mathcal{L}_{\text{CTC}} = -\ln P(Y|X)$$
Computed in $O(T \cdot L)$ time using dynamic programming via the forward-backward algorithm.

\section{Hindi NLP, Tokenization \& Devanagari Mechanics}
\subsection{The Linguistic Complexity of Devanagari Script}
Devanagari is an \textbf{abugida} (alphasyllabary), not a pure alphabet. Every consonant carries an inherent vowel sound (schwa, /a/), unless explicitly modified:
\begin{itemize}
    \item \textbf{Vowels (Swar):} Independent vowels (अ, आ, इ, ई, उ, ऊ, ए, ऐ, ओ, औ).
    \item \textbf{Matras (Vowel Diacritics):} Dependent signs attached to consonants (ा, ि, ी, ु, ू, े, ै, ो, ौ).
    \item \textbf{Halant / Virama (्):} Suppresses the inherent vowel to create conjunct consonants (samyuktaksar). For example: क + ् + ष = क्ष (ksha).
    \item \textbf{Nuqta (़):} Modifies native Indic consonants to represent borrowed Perso-Arabic sounds (e.g., क़, ख़, ग़, ज़, फ़).
\end{itemize}

\subsection{Tokenization Strategy in ViLiS}
ViLiS implements a dual-mode tokenizer in \texttt{utils/tokenizer.py}:
\begin{enumerate}
    \item \textbf{SentencePiece BPE (Vocab 1000):} Subword segmentation trained directly on Hindi transcripts. Subwords group common consonant-vowel clusters, reducing sequence lengths and improving CTC decoding convergence.
    \item \textbf{Native Unicode Character-Level Fallback:} A standalone Devanagari vocabulary mapping all independent vowels, consonants, matras, halant, nuqta, and digits. CTC blank is assigned to index 0.
\end{enumerate}

\section{Training Optimization \& Regularization Suite}
\subsection{Hyperparameter Configuration}
\begin{table}[h]
\centering
\begin{tabular}{lll}
\toprule
\textbf{Hyperparameter} & \textbf{Value} & \textbf{Technical Purpose} \\
\midrule
$d_{\text{model}}$ & 512 & Conformer hidden representation dimension \\
$N_{\text{layers}}$ & 12 & Depth of the Conformer encoder \\
$N_{\text{heads}}$ & 8 & Attention heads ($d_k = 64$ per head) \\
Conv Kernel & 31 & Depthwise 1D conv receptive field (~1.24s context) \\
Dropout & 0.1 & Regularization against overfitting \\
Peak Learning Rate & $3 \times 10^{-4}$ & Peak rate for AdamW ($\beta_1=0.9, \beta_2=0.98, \text{wd}=10^{-4}$) \\
Warmup Steps & 500 & Linear ramp to prevent early Conformer divergence \\
LR Decay & Cosine & Smooth annealing to $10^{-6}$ over training \\
Max Grad Norm & 5.0 & Gradient clipping threshold \\
Loss Function & CTC Loss & $\text{zero\_infinity}=\text{True}$ to avoid NaN instabilities \\
\bottomrule
\end{tabular}
\caption{ViLiS Training Configuration}
\end{table}

\subsection{Spatio-Temporal Video Augmentations}
To ensure generalization across variable real-world cameras and lighting, ViLiS applies a dedicated augmentation pipeline in \texttt{training/augmentations.py}:
\begin{itemize}
    \item \textbf{Random Spatial Crop:} Crops the $96 \times 96$ lip patch down to $88 \times 88$ at a random coordinate, then rescales back to $96 \times 96$ via bilinear interpolation. Teaches invariance to imperfect face tracking.
    \item \textbf{Horizontal Flip ($p=0.5$):} Mirrors lip frames horizontally. While audio speech has no bilateral symmetry, symmetric lip movements retain viseme identity.
    \item \textbf{Speed Perturbation ($\pm 10\%$):} Resamples the frame rate between $0.9\times$ and $1.1\times$ realtime. Forces the network to recognize phonemes spoken at varying speech tempos.
    \item \textbf{Temporal Masking:} Zeroes out a continuous block of 1 to 8 frames (analogous to SpecAugment in audio ASR). Forces the model to use surrounding context when temporary occlusion occurs.
\end{itemize}

\section{Hardware Profile: Running ViLiS on MacBook Air M4}
\subsection{Hardware Audit}
\begin{itemize}
    \item \textbf{Chip:} Apple M4 (10 cores: 4 Performance + 6 Efficiency).
    \item \textbf{GPU:} 10-core integrated GPU supporting Apple Metal 4.
    \item \textbf{Unified RAM:} 16 GB LPDDR5 (shared between CPU, GPU, and macOS).
    \item \textbf{Chassis:} Fanless passive thermal dissipation.
\end{itemize}

\subsection{Load Profile \& Thermal Analysis}
\begin{itemize}
    \item \textbf{Desktop App \& Real-Time Inference:} Runs at only 15\%--25\% CPU/GPU load, taking $\sim 2.0$ GB RAM. The laptop stays completely cool ($38^\circ\text{C}$--$42^\circ\text{C}$) with real-time inference latency under 80ms.
    \item \textbf{Full Model Training:} The 12-layer Conformer contains $\sim 50$ million parameters. Training on batches of video requires continuous backpropagation through 75 temporal steps. Because the MacBook Air has \textbf{no cooling fan}, sustained 100\% utilization pushes the SoC to its $95^\circ\text{C}$ thermal ceiling within 10 minutes, triggering severe thermal throttling (dropping clocks from 4.4 GHz to 2.8 GHz).
    \item \textbf{Architectural Decision:} \textbf{Hybrid Strategy.} All development, data capture, lip extraction, and inference run locally on the Mac M4. Full multi-epoch pretraining is offloaded to cloud GPUs (Google Colab / Kaggle T4 or A100).
\end{itemize}

\section{Evaluation Metrics: CER and WER}
Visual speech performance is quantified using:
\begin{enumerate}
    \item \textbf{Character Error Rate (CER):}
    $$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$
    where $S_c$ is character substitutions, $D_c$ is character deletions, $I_c$ is character insertions, and $N_c$ is total characters in reference.
    \item \textbf{Word Error Rate (WER):}
    $$\text{WER} = \frac{S_w + D_w + I_w}{N_w}$$
    Evaluated over words separated by whitespace.
\end{enumerate}

\section{The Top 25 Technical Interview Questions \& Model Answers}
\subsection{Category 1: Core Deep Learning \& Architecture}

\textbf{Q1: Why did you choose a 3D-CNN frontend instead of feeding 2D frames directly into ResNet-18?}
\begin{tcolorbox}[colback=gray!5!white,colframe=gray!75!black]
\textbf{Answer:} ``A 2D CNN processes spatial features within a single frame independently. However, visual speech is fundamentally a dynamic process: phoneme identity depends on the direction and velocity of lip movements (e.g., whether the lips are opening or closing). By applying a 3D convolution with a temporal kernel size of 5 ($5 \times 7 \times 7$), our frontend extracts spatio-temporal features that capture motion gradients across contiguous frames before passing downsampled representations to the 2D ResNet backbone.''
\end{tcolorbox}

\textbf{Q2: Why use a Conformer instead of a standard Transformer or Bidirectional LSTM?}
\begin{tcolorbox}[colback=gray!5!white,colframe=gray!75!black]
\textbf{Answer:} ``BiLSTMs suffer from gradient degradation over long sequences and cannot be parallelized effectively. Standard Transformers use Multi-Head Self-Attention, which excels at long-range global relationships but lacks inductive bias for local context. The Conformer combines the best of both worlds: its depthwise separable convolution module with kernel size 31 captures local viseme transitions and co-articulation, while the self-attention heads capture sentence-level grammatical dependencies.''
\end{tcolorbox}

\textbf{Q3: Explain the mathematical intuition behind CTC Loss and why it is essential for visual speech.}
\begin{tcolorbox}[colback=gray!5!white,colframe=gray!75!black]
\textbf{Answer:} ``In visual speech, we have $T$ video frames and an unaligned target text of length $L$, where $T \gg L$. We do not know which specific video frame corresponds to which character. CTC addresses this by introducing a blank token $\epsilon$ and allowing repeated characters. It calculates the probability of the target sequence by summing over all possible alignments that collapse to the ground truth. This is computed efficiently using dynamic programming via the forward-backward algorithm.''
\end{tcolorbox}

\textbf{Q4: Why is Warmup-Cosine learning rate scheduling critical when training deep Conformers?}
\begin{tcolorbox}[colback=gray!5!white,colframe=gray!75!black]
\textbf{Answer:} ``In the early stages of training, the attention weights and depthwise convolution filters are randomly initialized. Using a high learning rate immediately causes massive, erratic gradient updates that destabilize LayerNorm and lead to divergence or NaN loss in CTC. A linear warmup over 500 steps gently guides the parameters into a stable manifold before the learning rate peaks, followed by cosine decay for fine convergence.''
\end{tcolorbox}

\textbf{Q5: Why did you modify ResNet-18's first convolutional layer (conv1)?}
\begin{tcolorbox}[colback=gray!5!white,colframe=gray!75!black]
\textbf{Answer:} ``Standard ResNet-18 expects 3 input channels corresponding to RGB images. However, our frontend uses a 3D-CNN stem that outputs 64 feature channels. We patched \texttt{resnet.conv1} to accept 64 input channels instead of 3, allowing the pretrained ResNet-18 spatial blocks to seamlessly process the 3D-CNN motion representations.''
\end{tcolorbox}

\subsection{Category 2: Computer Vision \& Preprocessing}

\textbf{Q6: Why is Affine Roll Correction necessary in the lip extraction pipeline?}
\begin{tcolorbox}[colback=gray!5!white,colframe=gray!75!black]
\textbf{Answer:} ``Natural in-the-wild video involves head tilts. If the head is tilted at a 20-degree angle, the lips appear rotated in the camera plane. Without correction, the neural network would have to waste capacity learning rotational invariance. We track MediaPipe landmarks 61 and 291 (the left and right mouth corners), calculate the tilt angle $\theta = \arctan2(\Delta y, \Delta x)$, and apply an inverse affine rotation matrix. This guarantees that all lip images fed to the 3D-CNN are horizontally aligned.''
\end{tcolorbox}

\textbf{Q7: Why convert video frames to grayscale rather than keeping RGB color?}
\begin{tcolorbox}[colback=gray!5!white,colframe=gray!75!black]
\textbf{Answer:} ``Visual speech is encoded entirely in the geometry, contour, and spatial opening/closing of the lips, teeth, and tongue. Chrominance (color) carries minimal phonetic information but introduces significant variance due to skin tone, lipstick, and lighting temperature. Converting to single-channel luminance reduces memory consumption by 66\% and prevents the network from overfitting to speaker appearance.''
\end{tcolorbox}

\textbf{Q8: How does MediaPipe FaceMesh compare to Dlib (68 landmarks) for lip reading?}
\begin{tcolorbox}[colback=gray!5!white,colframe=gray!75!black]
\textbf{Answer:} ``Dlib's 68-point model is based on an older Ensemble of Regression Trees (ERT) that is sensitive to occlusion, poor lighting, and head turns. MediaPipe FaceMesh predicts 468 3D landmarks using a lightweight neural network running in sub-millisecond time. It provides 20 dedicated points outlining the inner and outer vermilion borders of the lips with sub-pixel precision.''
\end{tcolorbox}

\subsection{Category 3: NLP, Hindi Linguistics \& Failure Modes}

\textbf{Q9: What is the Homopheny problem, and how do you handle it in Hindi?}
\begin{tcolorbox}[colback=gray!5!white,colframe=gray!75!black]
\textbf{Answer:} ``Homopheny is the condition where different phonemes produce visually identical visemes (e.g., bilabials /p/, /b/, /m/). In Hindi, dental consonants (त, थ, द, ध) look virtually identical. A purely visual frame classifier cannot distinguish them. We resolve homopheny through temporal sequence context via the Conformer and CTC decoding, which leverages linguistic likelihood to select the most probable valid Hindi word.''
\end{tcolorbox}

\textbf{Q10: Why did you use OpenAI Whisper for weak supervision instead of manual labeling?}
\begin{tcolorbox}[colback=gray!5!white,colframe=gray!75!black]
\textbf{Answer:} ``Accurately transcribing spoken Hindi and aligning each word to exact video milliseconds requires expert human annotators and costs thousands of dollars for even a 10-hour dataset. OpenAI Whisper large-v3 achieves near-human acoustic transcription accuracy in Hindi. By running Whisper's cross-attention DTW, we extracted word-level timestamps automatically, generating an automated training pipeline at zero labeling cost.''
\end{tcolorbox}

\textbf{Q11: What are the main limitations of the ViLiS system currently?}
\begin{tcolorbox}[colback=gray!5!white,colframe=gray!75!black]
\textbf{Answer:} ``First, the model relies on CTC decoding, which assumes conditional independence between output tokens given the alignment. While fast, it cannot perform strong autoregressive grammatical error correction during decoding unless paired with an external KenLM beam search. Second, performance degrades on extreme side profiles (>45-degree yaw) where one mouth corner is completely occluded.''
\end{tcolorbox}

\end{document}
""")

print("LaTeX source written to", TEX_PATH)

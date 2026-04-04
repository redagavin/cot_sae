
\section{Introduction}
The emergence of Chain-of-Thought (CoT) reasoning has significantly enhanced the problem-solving capabilities of Large Language Models (LLMs). However, a critical shadow looms over this progress: unfaithfulness. Recent evidence suggests that a model’s textual reasoning often serves as a "post-hoc" rationalization rather than a true trace of its underlying logic. For instance, models frequently exploit subtle hints or biases in a prompt while producing a CoT that claims to follow a neutral, step-by-step derivation~\cite{chen2025reasoning}. This discrepancy creates a fundamental trust gap; if the CoT does not reflect the actual internal computation, our primary window into monitoring and regulating AI behavior is effectively broken.

Traditionally, efforts to detect and measure this unfaithfulness have relied almost exclusively on behavioral interventions. These methods typically involve perturbing the input prompt, manually editing reasoning steps, or paraphrasing the CoT to observe changes in the final output~\cite{turpin2023language,lanham2023measuring}. While these approaches have successfully demonstrated that models can be deceptive, they suffer from a fundamental irony: they attempt to verify a model’s internal integrity without ever examining its internal states. Consequently, purely textual methods remain "black-box" evaluations that can identify the presence of unfaithfulness but are incapable of pinpointing exactly where or how the internal computation diverges from the stated reasoning.

To bridge this gap, we propose a transition from behavioral observation to mechanistic transparency using Sparse Autoencoders (SAEs). By decomposing complex model activations into millions of sparse, interpretable features~\cite{bricken2023towards,templeton2024scaling}, SAEs allow us to audit what a model is "thinking" at each token position and layer. We hypothesize that SAE-based feature analysis can detect unfaithfulness with far greater sensitivity than textual perturbations. By mapping the activation of specific semantic features against the generated CoT, we aim to uncover the precise mechanistic divergence where the model’s internal logic departs from its public-facing explanation, providing a more rigorous foundation for AI safety and interpretability.

Concretely, our study aims to demonstrate that internal feature analysis provides a stronger signal of reasoning faithfulness than surface-level inspection. In particular, we seek to identify SAE features whose activations differ systematically between faithful and unfaithful reasoning conditions, revealing information that is invisible from the generated text alone. By tracking these features along the generation trajectory, we aim to localize a divergence point where internal representations begin to reflect the misleading hint before the CoT text does. As a further validation step, we explore targeted interventions that suppress hint-correlated features to test whether reducing these activations can mitigate hint-following behavior while preserving coherent reasoning.

\section{Problem Setup}

We investigate the problem of unfaithful Chain-of-Thought (CoT) reasoning in large language models (LLMs), which occurs when a model's generated rationale fails to accurately reflect its true internal decision-making process. Given an LLM $f$ and a prompt $P$ containing a misleading hint, the model autoregressively generates a sequence of reasoning tokens $C$ followed by a final prediction $y$, denoted as $$(C, y) = f(P)$$ where $C = (c_1, c_2, \dots, c_T)$. Unfaithfulness specifically manifests when the model's internal computation heavily relies on the misleading hint to produce the final answer $y$, yet it generates a plausible, post-hoc reasoning sequence $C$ that completely conceals this reliance, maintaining a facade of objective deduction. 

To uncover where this deception begins, we examine the sequence of the model's internal hidden states $H = (\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_T)$ produced during generation. We hypothesize that there exists a specific "divergence point": an early token position where the internal representation $\mathbf{h}_t$ silently commits to the biased hint, even though the emitted text context $c_{1:t}$ remains seemingly logical and unbiased. Formally, we aim to localize the earliest generation step
\[
\tau = \min \left\{ t \in [1, T] \mid \mathcal{D}(\mathbf{h}_t, c_{1:t}) > \epsilon \right\},
\]
where $\mathcal{D}$ measures the discrepancy between the biased internal computation and the externally stated reasoning, and $\epsilon$ is a divergence threshold. Given this critical gap between hidden mechanisms and stated rationales, our core question is: How can we systematically analyze an LLM's internal trajectory to pinpoint the exact token position $\tau$ where the model first begins to fake its reasoning?


\section{Related Work}
\textbf{Textual methods for faithfulness evaluation.} \citet{turpin2023language} showed biasing prompts cause models to rationalize biased answers without mentioning the bias. \citet{lanham2023measuring} found an inverse scaling pattern—larger models produce less faithful reasoning. \citet{chen2025reasoning} found that reasoning models exploit incorrect hints >99\% of the time while admitting it <2\% in CoT. These methods are valuable but cannot localize where in a reasoning trace unfaithfulness begins, nor characterize how it happens.

\smallskip\noindent\textbf{SAEs for mechanistic interpretability.} Anthropic's "Towards Monosemanticity"~\cite{bricken2023towards} and "Scaling Monosemanticity"~\cite{templeton2024scaling} established SAEs as a tool for extracting interpretable features from LLMs. Their circuit tracing work~\cite{lindsey2025on} used cross-layer transcoders to trace computational graphs in Claude 3.5 Haiku, directly revealing cases where internal computation diverges from output text. Pre-trained SAEs are now available for Gemma 2 \cite{lieberum2024gemma}, Llama-3.1-8B , and GPT-4 \cite{gao2025scaling}, with SAELens~\cite{bloom2024saetrainingcodebase} providing a unified interface.

\smallskip\noindent\textbf{The gap between CoT and SAE.} A few recent papers connect SAEs and CoT. \citet{chen2025does} used SAEs with activation patching on Pythia to study CoT effectiveness; \citet{galichin2025have} identified reasoning-specific SAE features in DeepSeek-R1-Distill, but none use SAEs to assess faithfulness, none compare SAE-based analysis to textual baselines, and none attempt to localize the point where reasoning becomes unfaithful under adversarial conditions.

\section{Proposed Approach}

\smallskip\noindent\textbf{Inducing hint-following.} We adapt the hint-insertion protocol from \citet{chen2025reasoning}. Using Gemma 2 9B-it, we run ~200–500 MMLU questions (correctly answered at baseline) under three conditions: no hint, true hint, and false hint. Since Gemma 2 9B-it is not a native reasoning model, we prompt it to produce step-by-step reasoning before its final answer. For each trial, we record the final answer, the full CoT text, and whether the CoT mentions the hint.

\smallskip\noindent\textbf{Defining hint-following and unfaithfulness.} A case is hint-following when the model's answer under the false-hint condition differs from its (correct) no-hint answer in the direction of the hint. Among hint-following cases, a case is textually unfaithful when the CoT does not mention the hint—the model was influenced but hides it. A case is textually faithful when the CoT does mention the hint. Our core question applies to all hint-following cases: can SAE features reveal hint influence on internal computation, regardless of whether the CoT text is honest about it?

\smallskip\noindent\textbf{SAE feature extraction and divergence localization.} To inspect the internal mechanisms, we extract feature activations using Gemma Scope Sparse Autoencoders (SAEs) at middle and late model depths (e.g., layers 12, 18, 24, and 30). We compute the cosine distance and Jensen-Shannon divergence between the SAE feature trajectories of the no-hint and false-hint conditions. Furthermore, we isolate cases where the model generates a final answer influenced by the false hint, yet the CoT text shows no semantic trace of it. By examining SAE features in these specific instances, we directly measure whether internal representations provide visibility into unfaithfulness that remains entirely hidden from standard behavioral inspection. To contextualize these findings, we analyze differentially active features through Neuronpedia to assign interpretable semantic labels to the latent concepts driving the unfaithfulness. We subsequently quantify the visibility gap, defined as the fraction of demonstrably unfaithful generations that exhibit evidence of hint influence under SAE analysis compared to surface-level textual inspection.

\smallskip\noindent\textbf{Causal validation.} Because correlation between SAE features and unfaithful outputs does not strictly entail causation, we propose targeted feature clamping as a stretch objective to rule out epiphenomenal effects. During the false-hint generation process, we dynamically clamp the identified candidate unfaithfulness features to their baseline no-hint activation levels. If this targeted intervention enables the model to recover the correct final answer while maintaining a coherent CoT, it provides strong causal evidence that these specific latent features actively drive the hint-following behavior rather than merely correlating with it.

\smallskip\noindent\textbf{Ablations.} To ensure the robustness of our findings, we conduct ablations across several axes. First, we vary the layer at which SAE features are extracted, comparing early, middle, and late model layers to understand where hint-related signals emerge during generation. Second, we evaluate the effect of SAE dictionary width by comparing feature sets with 16K, 65K, and 131K features. Third, we test different hint formulations, including authority-based hints, metadata hints, and peer-influence hints, to examine whether the divergence signal depends on the specific style of misleading information. Finally, we evaluate the approach across multiple task domains, including MMLU and GSM8K, to determine whether the observed SAE-based signals generalize beyond a single benchmark.

\section{Evaluation Plan}

Ground truth for hint-following is established by construction: a case is hint-following when the model's answer under the false-hint condition differs from its (correct) no-hint answer in the direction of the hint. Among hint-following cases, textual faithfulness is determined by whether the CoT mentions the hint. Visibility rates then measure what fraction of hint-following cases each method reveals evidence of hint influence in.

\begin{table}[h]
\centering
\renewcommand{\arraystretch}{1.3}
\caption{Summary of metrics for evaluating CoT faithfulness using SAE features.}
\begin{tabular}{p{0.32\textwidth} p{0.63\textwidth}}
\hline
\textbf{Metric} & \textbf{What it measures} \\ \hline
Hint-following rate & \% of false-hint trials where the model changes its final answer in the direction of the hint \\

SAE visibility rate & \% of hint-following cases where SAE features show significant divergence from no-hint baseline \\

Text mention rate & \% of hint-following cases where CoT text mentions the hint \\

Visibility gap & SAE visibility rate $-$ text mention rate (core claim) \\

Feature divergence onset & Earliest token position where SAE features diverge significantly from the no-hint baseline \\

Time gap & Feature divergence onset $-$ text divergence onset \\

Ablation recovery rate & \% of false-hint trials where feature clamping restores the correct answer (stretch goal) \\ \hline
\end{tabular}
\end{table}
\subsection{Experimental Rigor}

\paragraph{Baselines and Sanity Checks}
We implement random feature ablation (which should not reduce hint-following), a true-hint control (where divergence should be qualitatively different), and a linear probe baseline to verify if SAE decomposition provides value beyond a standard black-box classifier.

\paragraph{Failure Modes and Pivots}
Potential failure is defined by:
\begin{itemize}
    \item \textbf{No divergence signal:} Hint processing may occur below SAE resolution (a meaningful negative result).
    \item \textbf{Global shift:} Divergence is observed everywhere, making it non-localizable.
    \item \textbf{No visibility gap:} SAEs offer no measurable advantage over analyzing CoT text.
\end{itemize}
If signals are insufficient, we will pivot to a simpler task or a different architecture, such as \texttt{DeepSeek-R1-Distill-Llama-8B} using Llama Scope SAEs.

\section{Feasibility}

Gemma 2 9B fits on a single A100 (80GB). SAE feature extraction is a forward-pass operation with minimal overhead. For 500 questions $\times$ 3 conditions $\times$ $\sim$200 tokens, compute is modest. We use the Northeastern Discovery cluster (or NDIF if needed). All data (MMLU, GSM8K), SAEs (Gemma Scope on HuggingFace), and libraries (SAELens, TransformerLens) are publicly available.

\begin{table}[h]
\centering
\caption{Project risks, deadlines, and mitigation strategies.}
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{p{0.35\linewidth}|l|p{0.45\linewidth}}
\toprule
\textbf{Risk} & \textbf{Deadline} & \textbf{Mitigation} \\
\midrule
Gemma 2 9B doesn't follow false hints reliably & Week 1 & Switch to DeepSeek-R1-Distill-Llama-8B + Llama Scope \\
SAE features too noisy for divergence detection & Week 2 & Restrict to top features; try larger SAE widths; compare to linear probe \\
Ablation infeasible in time & Week 3 & Stretch goal; core analysis is a complete study without it \\
\bottomrule
\end{tabular}
\vspace{2mm} % Optional spacing
\label{tab:feasibility_risks}
\end{table}

\section{Timeline}

\begin{table}[h]
\centering
\caption{Proposed project timeline and weekly milestones.}
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{l|p{0.75\linewidth}}
\toprule
\textbf{Week} & \textbf{Tasks} \\
\midrule
1 (Mar 11--17) & Infrastructure setup, MMLU subset preparation, hint-insertion templates, pilot run (50 questions $\times$ 3 conditions) \\
2 (Mar 18--24) & Full data generation, SAE feature extraction at 4 layers, begin divergence analysis and visualization \\
3 (Mar 25--31) & Complete analysis, compute visibility gap, characterize features, run sanity checks, begin ablation if time permits \\
4 (Apr 1--7)   & Finalize results, write final paper \\
\bottomrule
\end{tabular}
\vspace{2mm}
\label{tab:timeline}
\end{table}

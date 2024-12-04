# PIELM_Numpy

Physcial Informed Extreme Learning Machine(PIELM) method to solve PDEs, such as Possion problem and Biharmonic problem

# Corresponding Papers

## ugmented physics informed extreme learning machine to solve thebiharmonic equations via Fourier expansions
created by Xi’an Li, Jinran Wu, Yujia Huang, Jiaxin Deng, Zhe Ding, You-Gan Wang, Xin Tai, Liang Liu

[[Paper]](https://arxiv.org/pdf/2310.13947.pdf)

### Ideas
By carefully calculating the differential and boundary operators of the biharmonicequation on discretized collections, the solution for this high-order equation is reformulated as a linearleast squares minimization problem.

### Abstract: 
To address the sensitivity of parameters and limited precision for physics-informed extreme learning machines(PIELM) with common activation functions, such as sigmoid, tangent, and Gaussian, in solving highorder partial differential equations (PDEs) relevant to scientific computation and engineering applications, this work develops a Fourier-induced PIELM (FPIELM) method. This approach aims to approximatesolutions for a class of fourth-order biharmonic equations with two boundary conditions on both unitized and non-unitized domains. By carefully calculating the differential and boundary operators of the biharmonicequation on discretized collections, the solution for this high-order equation is reformulated as a linearleast squares minimization problem. We further evaluate the FPIELM with varying hidden nodes andscaling factors for uniform distribution initialization, and then determine the optimal range for these twohyperparameters. Numerical experiments and comparative analyses demonstrate that the proposed FPIELMmethod is more stable, robust, precise, and efficient than other PIELM approaches in solving biharmonicequations across both regular and irregular domains.

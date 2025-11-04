# ZVIR
ZVIR: Zero-Shot Implicit Deep Image Prior with Prior Activation for Infrared and Visible Image Fusion  
https://www.sciencedirect.com/science/article/abs/pii/S0031320325013299  
Pattern Recognition,  
2025,  
112666,  
ISSN 0031-3203,  
https://doi.org/10.1016/j.patcog.2025.112666.  
# Abstract
  Infrared and visible image fusion aims to generate an image that simultaneously contains features from both modalities, thereby enhancing its information richness and expressive capability. Although the deep image prior has demonstrated excellent performance in traditional inverse problems such as image restoration and denoising, conventional network structures struggle to effectively capture the joint features required for multimodal image fusion. To address this issue, we propose a fusion framework that takes the visible image as the primary degraded input and models the fusion task as the reconstruction of a fused feature image based on the visible modality. This framework leverages the internal recurrence of the fused image and introduces connection modules during the downsampling stage to enhance infrared feature representation, thereby utilizing the network structure to form a deep image prior that guides the fusion process. Experimental results further demonstrate that although the network is affected by the degraded input, its structure can still implicitly extract effective priors required for image fusion. By replacing the input image in the network path, the deep image prior can be activated to generate the target fused image. We perform both qualitative and quantitative comparisons with mainstream methods on several public infrared-visible datasets, demonstrating the effectiveness of the proposed method in terms of fusion quality and downstream task performance.



python train.py

# Language-Guided Fine-grained Image Editing
## Intro
Language-guided image editing has grown increasingly popular, largely due to the accessibility of naturallanguage interfaces, as demonstrated by modelslike GPT-4. However, existing approaches often struggle with fine-grained control. 

For example:\
*"Help me remove the guy at the rightmost corner of the image who's wearing a black top and pants"*
|Source|Target|
|--|--|
|<img width="200" src="https://github.com/user-attachments/assets/be9fba52-ca9e-4e91-b673-e3b26e4e21af" />|<img src="https://github.com/user-attachments/assets/d752d782-a4d1-4f68-b4ce-781eee67e829" width="200"/>|


The model removes the specified figure, but also changes the central subject's face and alters background semantics.

To address this limitation, we present (1) a zero-shot pipeline and (2) an two-stage pipeline for fine-grained, language-guided image editing. 
## Zero-shot Pipeline
### Method
Given an input image and a natural language prompt (e.g. "replace the dog on the man's lap with a cap"), zero-shot pipeline performs: (1) generate object mask using grounding dino and sam2, (2) dilate the mask for more reasonable output, and (3) inpaint the new object.

<img width="1984" height="456" alt="image" src="https://github.com/user-attachments/assets/e5a7ca3c-3d87-48b6-8847-c994f94147e0" />

### Result
<img width="300" src="https://github.com/user-attachments/assets/e000b977-bb8a-46b8-a9be-ec0fc3cec657"/>

## Two-Stage Pipeline
### Method
Given an input image and a natural language prompt, the two-stage pipeline performs: (1) object mask generation, (2) spatial description via LLaVA, (3) inpainting the background using Stable Diffusion XL, (4) predicting the replacement object’s mask, and (5) final inpainting to insert the new object.

<img width="1984" src="https://github.com/user-attachments/assets/5b0511da-0356-41f9-9b87-240bb86ed294"/>

### Result
*Due to unexpected loss of access to critical GPU resources, we were unable to finish fine-tuning a custom mask generator, so we manually draw a mask as a proof-of-concept.*
![final_result_grid_page-0001_low](https://github.com/user-attachments/assets/0c78aa59-9a2f-49b0-aa77-7285fb323127)





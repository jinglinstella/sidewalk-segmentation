import torch
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SamModel, SamConfig, SamProcessor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
# model = SamModel(config=model_config)
model = SamModel.from_pretrained('kitooo/sidewalk-seg-base')
model.to(device)

def segment_sidewalk(image):
    width, height = image.size
    prompt = [0, 0, width, height]
    inputs = processor(image, input_boxes=[[prompt]], return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    prob_map = torch.sigmoid(outputs.pred_masks.squeeze()).cpu().detach()
    prediction = (prob_map > 0.5).float()
    prob_map, prediction = prob_map.numpy(), prediction.numpy()
    save_image(image, 'image.png')
    save_image(prediction, 'mask.png', cmap='gray')
    save_image(prob_map, 'prob.png', cmap='jet')
    return Image.open('image.png'), Image.open('mask.png'), Image.open('prob.png')

def save_image(image, path, **kwargs):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, interpolation='nearest', **kwargs)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type='pil', label='Original TIFF Image')
            button = gr.Button('Get Sidewalk Mask')
        with gr.Column():
            mask = gr.Image(type='pil', label='Predicted Mask')
            prob_map = gr.Image(type='pil', label='Predicted Probability Map')
    button.click(
        segment_sidewalk,  
        inputs=[image_input], 
        outputs=[image_input, mask, prob_map]
    )
demo.launch(debug=True, show_error=True)
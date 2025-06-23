import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
print(image.height, image.width)  # [480, 640]

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
patch_size = model.config.patch_size

inputs = processor(images=image, return_tensors="pt")
print(inputs.pixel_values.shape)  # [1, 3, 224, 224]
batch_size, rgb, img_height, img_width = inputs.pixel_values.shape
num_patches_height, num_patches_width = (
    img_height // patch_size,
    img_width // patch_size,
)
num_patches_flat = num_patches_height * num_patches_width

outputs = model(**inputs)
last_hidden_states = outputs[0]
print(last_hidden_states.shape)  # [1, 1 + 256, 768]
assert last_hidden_states.shape == (
    batch_size,
    1 + num_patches_flat,
    model.config.hidden_size,
)

cls_token = last_hidden_states[:, 0, :]
patch_features = last_hidden_states[:, 1:, :].unflatten(
    1, (num_patches_height, num_patches_width)
)

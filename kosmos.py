import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from draw_bounding_box import draw_entity_boxes_on_image 
from utility_functions import save_json_locally, get_base_filename_without_extension

base_path = os.getcwd()
input_images_folder = f'{base_path}/input_images'
jsons_folder = f'{base_path}/jsons'
output_images_folder = f'{base_path}/output_images'

def process_image(image_path, json_path, output_image_path, prompt = "<grounding> Describe this image in detail:"):
    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    image = Image.open(image_path) # open image
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Specify `cleanup_and_extract=False` in order to see the raw model generation.
    processed_text_unclean = processor.post_process_generation(generated_text, cleanup_and_extract=False)
    # By default, the generated  text is cleanup and the entities are extracted.
    processed_text, entities = processor.post_process_generation(generated_text)
    # to save result as json file
    save_json_locally(prompt=prompt, filename_path=json_path, entities=entities, description=processed_text, unclean_description=processed_text_unclean)
    # save bounding box images 
    draw_entity_boxes_on_image(image, entities, show=False, save_path=output_image_path)

    

if __name__ == "__main__":
    if not (os.path.exists(input_images_folder)): os.mkdir(input_images_folder)
    if not (os.path.exists(output_images_folder)): os.mkdir(output_images_folder)
    if not (os.path.exists(jsons_folder)): os.mkdir(jsons_folder)

    valid_extensions=('.jpg', '.jpeg', '.png', '.gif')

    for filename in os.listdir(input_images_folder):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(input_images_folder, filename)
            base_filename = get_base_filename_without_extension(image_path)

            json_path = os.path.join(jsons_folder, f'{base_filename}.json')
            output_image_path = os.path.join(output_images_folder, f'{base_filename}.jpg')
            process_image(image_path ,json_path, output_image_path)
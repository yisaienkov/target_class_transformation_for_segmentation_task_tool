import gradio as gr
import cv2

from models import Model


gen = Model()

def draw_rectangle(image, *arr):
    output_image = image.copy()
    gen_output_image = image.copy()
    for i in range(0, len(arr), 4):
        x1, y1, x2, y2 = int(arr[i + 0]), int(arr[i + 1]), int(arr[i + 2]), int(arr[i + 3])
        if x1 == 0 and x2 == 0:
            continue

        output_image = cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        tooth = image[y1:y2, x1:x2]
        res = gen(tooth)
        gen_output_image[y1:y2, x1:x2] = (cv2.cvtColor(cv2.resize(res, (x2 - x1, y2 - y1)), cv2.COLOR_GRAY2RGB) * 255).astype(int)

    return output_image, gen_output_image


def variable_outputs(k):
    k = int(k) * 4
    return [gr.Textbox(visible=True)] * k + [gr.Textbox(visible=False)] * (20 - k)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image()
            coords = []
            n_boxes = gr.Number(1, label="Boxes", min_width=80, maximum=5, minimum=0)
            for i in range(5):
                with gr.Row():
                    visible = False
                    if i == 0:
                        visible = True
                    coords.append(gr.Number(0, label="xs", min_width=80, visible=visible))
                    coords.append(gr.Number(0, label="ys", min_width=80, visible=visible))
                    coords.append(gr.Number(0, label="xe", min_width=80, visible=visible))
                    coords.append(gr.Number(0, label="ye", min_width=80, visible=visible))
        
        with gr.Column():
            image_rect_output = gr.Image()
            image_gen_output = gr.Image()
    
    n_boxes.change(variable_outputs, n_boxes, coords)
    image_button = gr.Button("Generate")

    image_button.click(draw_rectangle, inputs=[image_input, *coords], outputs=[image_rect_output, image_gen_output])

if __name__ == "__main__":
    demo.launch()
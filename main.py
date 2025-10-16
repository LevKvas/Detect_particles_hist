import numpy as np
import gradio as gr
import spaces
import cv2
from cellpose import models
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import os, io, base64
from PIL import Image
from cellpose.io import imread, imsave
import glob

from huggingface_hub import hf_hub_download

img = np.zeros((96, 128), dtype=np.uint8)
fp0 = Image.fromarray(img)


# fp0 = "0.png"
# imsave(fp0, img)

# data  retrieval
def download_weights():
    return hf_hub_download(repo_id="mouseland/cellpose-sam", filename="cpsam")

    # os.system("wget -q https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam")


def download_weights_old():
    import os, requests

    fname = ['cpsam']

    url = ["https://osf.io/d7c8e/download"]

    for j in range(len(url)):
        if not os.path.isfile(fname[j]):
            ntries = 0
            while ntries < 10:
                try:
                    r = requests.get(url[j])
                except:
                    print("!!! Failed to download data !!!")
                    ntries += 1
                    print(ntries)

        if r.status_code != requests.codes.ok:
            print("!!! Failed to download data !!!")
        else:
            with open(fname[j], "wb") as fid:
                fid.write(r.content)


try:
    fpath = download_weights()
    model = models.CellposeModel(gpu=True, pretrained_model=fpath)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


def plot_flows(y):
    Y = (np.clip(normalize99(y[0][0]), 0, 1) - 0.5) * 2
    X = (np.clip(normalize99(y[1][0]), 0, 1) - 0.5) * 2
    H = (np.arctan2(Y, X) + np.pi) / (2 * np.pi)
    S = normalize99(y[0][0] ** 2 + y[1][0] ** 2)
    HSV = np.concatenate((H[:, :, np.newaxis], S[:, :, np.newaxis], S[:, :, np.newaxis]), axis=-1)
    HSV = np.clip(HSV, 0.0, 1.0)
    flow = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return flow


def plot_outlines(img, masks): # in this func I can return contours
    img = normalize99(img)
    img = np.clip(img, 0, 1)
    outpix = []
    contours, hierarchy = cv2.findContours(masks.astype(np.int32), mode=cv2.RETR_FLOODFILL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        pix = contours[c].astype(int).squeeze()
        if len(pix) > 4:
            peri = cv2.arcLength(contours[c], True)
            approx = cv2.approxPolyDP(contours[c], 0.001, True)[:, 0, :]
            outpix.append(approx)

    figsize = (6, 6)
    if img.shape[0] > img.shape[1]:
        figsize = (6 * img.shape[1] / img.shape[0], 6)
    else:
        figsize = (6, 6 * img.shape[0] / img.shape[1])
    fig = plt.figure(figsize=figsize, facecolor='k')
    ax = fig.add_axes([0.0, 0.0, 1, 1])
    ax.set_xlim([0, img.shape[1]])
    ax.set_ylim([0, img.shape[0]])
    ax.imshow(img[::-1], origin='upper', aspect='auto')
    if outpix is not None:
        for o in outpix:
            ax.plot(o[:, 0], img.shape[0] - o[:, 1], color=[1, 0, 0], lw=1)
    ax.axis('off')

    # bytes_image = io.BytesIO()
    # plt.savefig(bytes_image, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    # bytes_image.seek(0)
    # img_arr = np.frombuffer(bytes_image.getvalue(), dtype=np.uint8)
    # bytes_image.close()
    # img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # del bytes_image
    # fig.clf()
    # plt.close(fig)

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    pil_img = Image.open(buf)

    plt.close(fig)

    return pil_img


def plot_overlay(img, masks):
    if img.ndim > 2:
        img_gray = img.astype(np.float32).mean(axis=-1)
    else:
        img_gray = img.astype(np.float32)

    img = normalize99(img_gray)
    # img = np.clip(img, 0, 1)
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:, :, 2] = np.clip(img * 1.5, 0, 1.0)
    for n in range(int(masks.max())):
        ipix = (masks == n + 1).nonzero()
        HSV[ipix[0], ipix[1], 0] = np.random.rand()
        HSV[ipix[0], ipix[1], 1] = 1.0
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB


def normalize99(img):
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (1e-10 + np.percentile(X, 99) - np.percentile(X, 1))
    return X


def image_resize(img, resize=400):
    ny, nx = img.shape[:2]
    if np.array(img.shape).max() > resize:
        if ny > nx:
            nx = int(nx / ny * resize)
            ny = resize
        else:
            ny = int(ny / nx * resize)
            nx = resize
        shape = (nx, ny)
        img = cv2.resize(img, shape)
    img = img.astype(np.uint8)
    return img


@spaces.GPU(duration=10)
def run_model_gpu(img, max_iter, flow_threshold, cellprob_threshold):
    masks, flows, _ = model.eval(img, niter=max_iter, flow_threshold=flow_threshold,
                                 cellprob_threshold=cellprob_threshold)
    return masks, flows


@spaces.GPU(duration=60)
def run_model_gpu60(img, max_iter, flow_threshold, cellprob_threshold):
    masks, flows, _ = model.eval(img, niter=max_iter, flow_threshold=flow_threshold,
                                 cellprob_threshold=cellprob_threshold)
    return masks, flows


@spaces.GPU(duration=240)
def run_model_gpu240(img, max_iter, flow_threshold, cellprob_threshold):
    masks, flows, _ = model.eval(img, niter=max_iter, flow_threshold=flow_threshold,
                                 cellprob_threshold=cellprob_threshold)
    return masks, flows


import datetime
from zipfile import ZipFile


def get_contour_coordinates(masks):
    """Получить координаты всех контуров"""
    contours, hierarchy = cv2.findContours(masks.astype(np.int32),
                                           mode=cv2.RETR_FLOODFILL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

    all_contours = []
    for i, contour in enumerate(contours):
        if len(contour) > 4:  # Игнорируем слишком маленькие
            # Упрощаем контур
            approx = cv2.approxPolyDP(contour, 0.001, True)
            all_contours.append({
                'particle_id': i + 1,
                'coordinates': approx.squeeze().tolist(),  # [[x,y], [x,y], ...]
                'area': cv2.contourArea(contour),
                'perimeter': cv2.arcLength(contour, True)
            })

    return all_contours


def calculate_eccentricity(major_axis, minor_axis):
    """Calculate the eccentricity of an ellipse"""
    if major_axis >= minor_axis:
        return np.sqrt(1 - (minor_axis**2) / (major_axis**2))
    else:
        return np.sqrt(1 - (major_axis**2) / (minor_axis**2))


def build_eccentricity_dist(some_array, bins=10):
    """Calculate the distribution of the number of particles from eccentricity values"""
    eccentricity_bins = np.linspace(0, 1.0, bins + 1)
    hist, bin_edges = np.histogram(some_array, bins=eccentricity_bins)

    plt.figure(figsize=(10, 6))

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # create hist
    bars = plt.bar(bin_centers, hist, width=bin_width * 0.8,
                   alpha=0.7, color='skyblue', edgecolor='black')

    # add values on columns
    for bar, count in zip(bars, hist):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{count}', ha='center', va='bottom')

    plt.xlabel('Eccentricity')
    plt.ylabel('Number of Particles')
    plt.title(f'Eccentricity Distribution (n={len(some_array)})')  # ← ДОБАВИЛ ЗАГОЛОВОК
    plt.grid(True, alpha=0.3)
    plt.xticks(bin_edges)

    # statistics
    total_particles = np.sum(hist)
    plt.text(0.02, 0.98, f'Total particles: {total_particles}',
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    # convert in image for Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def work_with_contours(contours_data_all):
    array_with_eccentricity = []

    for contour_data in contours_data_all:
        coordinates = contour_data['coordinates']

        contour_points = np.array(coordinates, dtype=np.float32)

        if len(contour_points) < 5:
            continue

        try:
            ellipse = cv2.fitEllipse(contour_points)
            center, axes, angle = ellipse

            major_axis = max(axes)
            minor_axis = min(axes)
            eccentricity = calculate_eccentricity(major_axis, minor_axis)

            array_with_eccentricity.append(eccentricity)

        except Exception as e:
            print(f"Error fitting ellipse: {e}")
            continue

    if not array_with_eccentricity:
        return create_empty_histogram()

    return build_eccentricity_dist(array_with_eccentricity)


def create_empty_histogram():
    """Создать пустую гистограмму с сообщением"""
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, 'No eccentricity data available',
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def cellpose_segment(filepath, resize=1000, max_iter=250, flow_threshold=0.4, cellprob_threshold=0):
    zip_path = os.path.splitext(filepath[-1])[0] + "_masks.zip"
    json_path = os.path.splitext(filepath[-1])[0] + "_contours.json"

    contour_data_all = []
    # zip_path = 'masks.zip'
    with ZipFile(zip_path, 'w') as myzip:
        for j in range((len(filepath))):
            now = datetime.datetime.now()
            formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

            img_input = imread(filepath[j])
            # img_input = np.array(img_pil)
            img = image_resize(img_input, resize=resize)

            maxsize = np.max(img.shape)
            if maxsize <= 1000:
                masks, flows = run_model_gpu(img, max_iter, flow_threshold, cellprob_threshold)
            elif maxsize < 5000:
                masks, flows = run_model_gpu60(img, max_iter, flow_threshold, cellprob_threshold)
            elif maxsize < 20000:
                masks, flows = run_model_gpu240(img, max_iter, flow_threshold, cellprob_threshold)
            else:
                raise ValueError("Image size must be less than 20,000")

            contours_data = get_contour_coordinates(masks)
            contour_data_all.extend(contours_data)

            print(formatted_now, j, masks.max(), os.path.split(filepath[j])[-1])

            target_size = (img_input.shape[1], img_input.shape[0])
            if (target_size[0] != img.shape[1] or target_size[1] != img.shape[0]):
                # scale it back to keep the orignal size
                masks_rsz = cv2.resize(masks.astype('uint16'), target_size, interpolation=cv2.INTER_NEAREST).astype(
                    'uint16')
            else:
                masks_rsz = masks.copy()

            fname_masks = os.path.splitext(filepath[j])[0] + "_masks.tif"
            imsave(fname_masks, masks_rsz)

            myzip.write(fname_masks, arcname=os.path.split(fname_masks)[-1])

    # masks, flows, _ = model.eval(img, channels=[0,0])
    flows = flows[0]
    # masks = np.zeros(img.shape[:2])
    # flows = np.zeros_like(img)

    outpix = plot_outlines(img, masks)
    histogram_image = work_with_contours(contour_data_all)  # build based on all data

    import json
    with open(json_path, 'w') as f:
        json.dump(contour_data_all, f, indent=2)

    # overlay = plot_overlay(img, masks)

    # crand = .2 + .8 * np.random.rand(np.max(masks.flatten()).astype('int')+1,).astype('float32')
    # crand[0] = 0

    # overlay = Image.fromarray(overlay)
    flows = Image.fromarray(flows)

    Ly, Lx = img.shape[:2]
    outpix = outpix.resize((Lx, Ly), resample=Image.BICUBIC)
    # overlay = overlay.resize((Lx, Ly), resample  = Image.BICUBIC)
    flows = flows.resize((Lx, Ly), resample=Image.BICUBIC)

    fname_out = os.path.splitext(filepath[-1])[0] + "_outlines.png"
    outpix.save(fname_out)  # "outlines.png")

    # fname_flows  = os.path.splitext(filepath[-1])[0]+"_flows.png"
    # flows.save(fname_flows) #"outlines.png")

    if len(filepath) > 1:
        b1 = gr.DownloadButton(visible=True, value=zip_path)
    else:
        b1 = gr.DownloadButton(visible=True, value=fname_masks)
    b2 = gr.DownloadButton(visible=True, value=fname_out)  # "outlines.png")
    b3 = gr.DownloadButton(visible=True, value=json_path)

    return outpix, flows, b1, b2, b3, histogram_image


def download_function():
    b1 = gr.DownloadButton("Download masks as TIFF", visible=False)
    b2 = gr.DownloadButton("Download outline image as PNG", visible=False)
    return b1, b2


def tif_view(filepath):
    fpath, fext = os.path.splitext(filepath)
    if fext in ['tiff', 'tif']:
        img = imread(filepath[-1])
        if img.ndim == 2:
            img = np.tile(img[:, :, np.newxis], [1, 1, 3])
        elif img.ndim == 3:
            imin = np.argmin(img.shape)
            if imin < 2:
                img = np.tranpose(img, [2, imin])
        else:
            raise ValueError("TIF cannot have more than three dimensions")

        Ly, Lx, nchan = img.shape
        imgi = np.zeros((Ly, Lx, 3))
        nn = np.minimum(3, img.shape[-1])
        imgi[:, :, :nn] = img[:, :, :nn]

        # filepath = fpath+'.png'
        imsave(filepath, imgi)
    return filepath


def norm_path(filepath):
    img = imread(filepath)
    img = normalize99(img)
    img = np.clip(img, 0, 1)
    fpath, fext = os.path.splitext(filepath)
    filepath = fpath + '.png'
    pil_image = Image.fromarray((255. * img).astype(np.uint8))
    pil_image.save(filepath)
    # imsave(filepath, pil_image)
    return filepath


def update_image(filepath):
    for f in filepath:
        f = tif_view(f)
    filepath_show = norm_path(filepath[-1])
    return filepath_show, filepath, fp0, fp0


def update_button(filepath):
    filepath = tif_view(filepath)
    filepath_show = norm_path(filepath)
    return filepath_show, [filepath], fp0, fp0


with gr.Blocks(title="Hello",
               css=".gradio-container {background:purple;}") as demo:
    # filepath = ""
    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML("""<div style="font-family:'Times New Roman', 'Serif'; font-size:20pt; font-weight:bold; text-align:center; color:white;">Cellpose-SAM for cellular 
            segmentation <a style="color:#cfe7fe; font-size:14pt;" href="https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1" target="_blank">[paper]</a> 
            <a style="color:white; font-size:14pt;" href="https://github.com/MouseLand/cellpose" target="_blank">[github]</a>
            <a style="color:white; font-size:14pt;" href="https://www.youtube.com/watch?v=KIdYXgQemcI" target="_blank">[talk]</a>                        
            </div>""")
            gr.HTML(
                """<h4 style="color:white;">You may need to login/refresh for 5 minutes of free GPU compute per day (enough to process hundreds of images). </h4>""")

            input_image = gr.Image(label="Input", type="filepath")

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        resize = gr.Number(label='max resize', value=1000)
                        max_iter = gr.Number(label='max iterations', value=250)
                        flow_threshold = gr.Number(label='flow threshold', value=0.4)
                        cellprob_threshold = gr.Number(label='cellprob threshold', value=0)

                    up_btn = gr.UploadButton("Multi-file upload (png, jpg, tif etc)", visible=True,
                                             file_count="multiple")

                    # gr.HTML("""<h4 style="color:white;"> Note2: Only the first image of a tif will display the segmentations, but you can download segmentations for all planes. </h4>""")

                with gr.Column(scale=1):
                    send_btn = gr.Button("Run Cellpose-SAM")
                    down_btn = gr.DownloadButton("Download masks (TIF)", visible=False)
                    down_btn2 = gr.DownloadButton("Download outlines (PNG)", visible=False)
                    down_btn3 = gr.DownloadButton("Download contours (JSON)", visible=False)

        with gr.Column(scale=2):
            outlines = gr.Image(label="Outlines", type="pil", format='png',
                                value=fp0)  # , width = "50vw", height = "20vw")
            # img_overlay = gr.Image(label = "Overlay", type = "pil", format = 'png') #, width = "50vw", height = "20vw")
            flows = gr.Image(label="Cellpose flows", type="pil", format='png',
                             value=fp0)  # , width = "50vw", height = "20vw")

            histogram_output = gr.Image(label="Eccentricity Distribution", type="pil",
                                        format='png', value=fp0)

    sample_list = glob.glob("samples/*.png")
    # sample_list = []
    # for j in range(23):
    #    sample_list.append("samples/img%0.2d.png"%j)

    gr.Examples(sample_list, fn=update_button, inputs=input_image, outputs=[input_image, up_btn, outlines, flows],
                examples_per_page=50, label="Click on an example to try it")
    input_image.upload(update_button, input_image, [input_image, up_btn, outlines, flows])
    up_btn.upload(update_image, up_btn, [input_image, up_btn, outlines, flows])

    send_btn.click(cellpose_segment, [up_btn, resize, max_iter, flow_threshold, cellprob_threshold],
                   [outlines, flows, down_btn, down_btn2, down_btn3, histogram_output])

    # down_btn.click(download_function, None, [down_btn, down_btn2])

    gr.HTML("""<h4 style="color:white;"> Notes:<br> 
                    <li>you can load and process 2D, multi-channel tifs.
                    <li>the smallest dimension of a tif --> channels
                    <li>you can upload multiple files and download a zip of the segmentations
                    <li>install Cellpose-SAM locally for full functionality.
                    </h4>""")

demo.launch()

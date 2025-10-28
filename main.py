import numpy as np
import gradio as gr
import spaces
import math
import cv2
from cellpose import models
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import os, io
from PIL import Image
from cellpose.io import imread, imsave
import glob
import tempfile
import json
import datetime
from zipfile import ZipFile

img = np.zeros((96, 128), dtype=np.uint8)
fp0 = Image.fromarray(img)

# 117 пкс - 20 нм
# 1 пкс - 0.171 нм
K_PX_TO_NM = 0.171  # to convert pixels to nm


# data retrieval
def download_weights():
    return hf_hub_download(repo_id="mouseland/cellpose-sam", filename="cpsam")


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


def plot_outlines(img, masks):
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

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    pil_img = Image.open(buf)
    plt.close(fig)
    return pil_img


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


def calculate_eccentricity(a, b):
    """a and b are semi-axes"""
    if a >= b:
        return np.sqrt(1 - (b ** 2) / (a ** 2))
    else:
        return np.sqrt(1 - (a ** 2) / (b ** 2))


def build_eccentricity_dist(some_array, bins=10):
    """Calculate the distribution with smart text positioning"""
    if not some_array:
        return create_empty_histogram()

    eccentricity_bins = np.linspace(0, 1.0, bins + 1)
    hist, bin_edges = np.histogram(some_array, bins=eccentricity_bins)

    plt.figure(figsize=(12, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    bars = plt.bar(bin_centers, hist, width=bin_width * 0.8,
                   alpha=0.7, color='skyblue', edgecolor='black')

    max_height = max(hist) if len(hist) > 0 else 1

    for bar, count in zip(bars, hist):
        if count > 0:
            bar_height = bar.get_height()
            bar_center = bar.get_x() + bar.get_width() / 2

            if bins <= 15:
                y_pos = bar_height + max_height * 0.02
                va = 'bottom'
                color = 'black'
            else:
                if bar_height > max_height * 0.3:
                    y_pos = bar_height * 0.7
                    color = 'black'
                else:
                    y_pos = bar_height * 0.5
                    color = 'black' if bar_height > max_height * 0.1 else 'white'
                va = 'center'

            fontsize = 8 if bins > 20 else (9 if bins > 15 else 10)
            plt.text(bar_center, y_pos, f'{count}',
                     ha='center', va=va, color=color,
                     fontsize=fontsize, fontweight='bold')

    plt.xlabel('Eccentricity')
    plt.ylabel('Number of Particles')
    plt.title(f'Eccentricity Distribution (n={len(some_array)}, bins={bins})')
    plt.grid(True, alpha=0.3)

    if bins > 20:
        plt.xticks(bin_edges[::2])
    elif bins > 15:
        plt.xticks(bin_edges[::2])
    else:
        plt.xticks(bin_edges)

    total_particles = np.sum(hist)
    plt.text(0.02, 0.98, f'Total particles: {total_particles}\nBins: {bins}',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    plt.ylim(0, max_height * 1.15)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return Image.open(buf)


def build_square_dist(some_array, bins=10):
    """Calculate the distribution with smart text positioning"""
    if not some_array:
        return create_empty_histogram()

    min_square = min(some_array)
    max_square = max(some_array)

    if max_square < 1.0:
        max_square = 1.0
    else:
        max_square = max_square * 1.1

    square_bins = np.linspace(min_square, max_square, bins + 1)
    hist, bin_edges = np.histogram(some_array, bins=square_bins)

    plt.figure(figsize=(12, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    bars = plt.bar(bin_centers, hist, width=bin_width * 0.8,
                   alpha=0.7, color='lightgreen', edgecolor='black')

    max_height = max(hist) if len(hist) > 0 else 1

    for bar, count in zip(bars, hist):
        if count > 0:
            bar_height = bar.get_height()
            bar_center = bar.get_x() + bar.get_width() / 2

            if bins <= 15:
                y_pos = bar_height + max_height * 0.02
                va = 'bottom'
                color = 'black'
            else:
                if bar_height > max_height * 0.4:
                    y_pos = bar_height * 0.8
                    color = 'black'
                elif bar_height > max_height * 0.15:
                    y_pos = bar_height * 0.5
                    color = 'black'
                else:
                    y_pos = bar_height * 0.3
                    color = 'white'
                va = 'center'

            fontsize = 8 if bins > 20 else (9 if bins > 15 else 10)
            plt.text(bar_center, y_pos, f'{count}',
                     ha='center', va=va, color=color,
                     fontsize=fontsize, fontweight='bold')

    plt.xlabel('Square, nm^2')
    plt.ylabel('Number of Particles')
    plt.title(f'Square Distribution (n={len(some_array)}, bins={bins})')
    plt.grid(True, alpha=0.3)

    if bins > 20:
        plt.xticks(bin_edges[::3])
    elif bins > 15:
        plt.xticks(bin_edges[::2])
    else:
        plt.xticks(bin_edges)

    total_particles = np.sum(hist)
    mean_square = np.mean(some_array) if len(some_array) > 0 else 0
    plt.text(0.02, 0.98, f'Total particles: {total_particles}\nBins: {bins}\nMean: {mean_square:.2f} nm²',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    if max_height > 0:
        plt.ylim(0, max_height * 1.15)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return Image.open(buf)


def create_empty_histogram():
    """Создать пустую гистограмму с сообщением"""
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, 'No data available',
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return Image.open(buf)


def create_histogram_data(contours_data_all, bins=10):
    """Создает данные для гистограммы эксцентриситета"""
    array_with_eccentricity = []

    for contour_data in contours_data_all:
        coordinates = contour_data['coordinates']
        contour_points = np.array(coordinates, dtype=np.float32)

        if len(contour_points) < 5:
            continue

        try:
            ellipse = cv2.fitEllipse(contour_points)
            center, axes, angle = ellipse

            major_axis = max(axes) * K_PX_TO_NM
            minor_axis = min(axes) * K_PX_TO_NM
            eccentricity = calculate_eccentricity(major_axis / 2, minor_axis / 2)
            array_with_eccentricity.append(eccentricity)
        except Exception as e:
            continue

    if not array_with_eccentricity:
        return None, None

    eccentricity_bins = np.linspace(0, 1.0, bins + 1)
    hist, bin_edges = np.histogram(array_with_eccentricity, bins=eccentricity_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers.tolist(), hist.tolist()


def create_square_data(contours_data_all, bins=10):
    """Создает данные для гистограммы площадей"""
    array_with_squares = []

    for contour_data in contours_data_all:
        coordinates = contour_data['coordinates']
        contour_points = np.array(coordinates, dtype=np.float32)

        if len(contour_points) < 5:
            continue

        try:
            ellipse = cv2.fitEllipse(contour_points)
            center, axes, angle = ellipse

            major_axis = max(axes) * K_PX_TO_NM
            minor_axis = min(axes) * K_PX_TO_NM
            square = math.pi * (major_axis / 2) * (minor_axis / 2)
            array_with_squares.append(square)
        except Exception as e:
            continue

    if not array_with_squares:
        return None, None

    min_square = min(array_with_squares)
    max_square = max(array_with_squares) * 1.1
    square_bins = np.linspace(min_square, max_square, bins + 1)
    hist, bin_edges = np.histogram(array_with_squares, bins=square_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers.tolist(), hist.tolist()


def create_histogram_txt_file(contours_data_all, bins=10):
    """Создает TXT файл с данными гистограмм"""
    ecc_centers, ecc_counts = create_histogram_data(contours_data_all, bins)
    sq_centers, sq_counts = create_square_data(contours_data_all, bins)

    content = "HISTOGRAM DATA\n"
    content += "=" * 50 + "\n\n"

    # Данные эксцентриситета
    content += "ECCENTRICITY HISTOGRAM\n"
    content += "-" * 30 + "\n"
    content += "Bin_Center\tParticle_Count\n"

    if ecc_centers and ecc_counts:
        for center, count in zip(ecc_centers, ecc_counts):
            content += f"{center:.4f}\t{count}\n"
    else:
        content += "No eccentricity data available\n"

    content += "\n\n"

    # Данные площадей
    content += "SQUARE HISTOGRAM (nm²)\n"
    content += "-" * 30 + "\n"
    content += "Bin_Center\tParticle_Count\n"

    if sq_centers and sq_counts:
        for center, count in zip(sq_centers, sq_counts):
            content += f"{center:.2f}\t{count}\n"
    else:
        content += "No square data available\n"

    # Создаем временный файл
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.write(content)
    temp_file.close()
    return temp_file.name


def work_with_contours(contours_data_all, bins=10):
    array_with_eccentricity = []
    array_with_squares = []

    for contour_data in contours_data_all:
        coordinates = contour_data['coordinates']
        contour_points = np.array(coordinates, dtype=np.float32)

        if len(contour_points) < 5:
            continue

        try:
            ellipse = cv2.fitEllipse(contour_points)
            center, axes, angle = ellipse

            major_axis = max(axes) * K_PX_TO_NM
            minor_axis = min(axes) * K_PX_TO_NM
            square = math.pi * (major_axis / 2) * (minor_axis / 2)
            eccentricity = calculate_eccentricity(major_axis / 2, minor_axis / 2)

            array_with_eccentricity.append(eccentricity)
            array_with_squares.append(square)
        except Exception as e:
            continue

    if not array_with_eccentricity:
        return create_empty_histogram(), create_empty_histogram()
    if not array_with_squares:
        return create_empty_histogram(), create_empty_histogram()

    return build_eccentricity_dist(array_with_eccentricity, bins), build_square_dist(array_with_squares, bins)


def cellpose_segment(filepath, max_iter=250, flow_threshold=0.4, cellprob_threshold=0, bins=10):
    zip_path = os.path.splitext(filepath[-1])[0] + "_masks.zip"
    json_path = os.path.splitext(filepath[-1])[0] + "_contours.json"

    contour_data_all = []

    with ZipFile(zip_path, 'w') as myzip:
        for j in range((len(filepath))):
            now = datetime.datetime.now()
            formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

            img_input = imread(filepath[j])
            processing_resize = 100
            img_processed = image_resize(img_input, resize=processing_resize)

            masks_small, flows = run_model_gpu(img_processed, max_iter, flow_threshold, cellprob_threshold)

            target_size = (img_input.shape[1], img_input.shape[0])
            masks = cv2.resize(masks_small.astype('uint16'), target_size,
                               interpolation=cv2.INTER_NEAREST).astype('uint16')

            contours_data = get_contour_coordinates(masks)
            contour_data_all.extend(contours_data)

            print(
                f"{formatted_now} {j} {masks.max()} {os.path.split(filepath[j])[-1]} (processed at {processing_resize}px)")

            fname_masks = os.path.splitext(filepath[j])[0] + "_masks.tif"
            imsave(fname_masks, masks)
            myzip.write(fname_masks, arcname=os.path.split(fname_masks)[-1])

    flows = flows[0]

    # Создаем файл с данными гистограмм
    hist_txt_path = create_histogram_txt_file(contour_data_all, bins)

    # Остальной код
    outpix = plot_outlines(img_input, masks)
    hist_eccentricity, hist_squares = work_with_contours(contour_data_all, bins)

    with open(json_path, 'w') as f:
        json.dump(contour_data_all, f, indent=2)

    flows = Image.fromarray(flows)

    Ly, Lx = img_input.shape[:2]
    max_display_size = 1200
    if Ly > max_display_size or Lx > max_display_size:
        scale = max_display_size / max(Ly, Lx)
        new_size = (int(Lx * scale), int(Ly * scale))
        outpix = outpix.resize(new_size, resample=Image.BICUBIC)
        flows = flows.resize(new_size, resample=Image.BICUBIC)

    fname_out = os.path.splitext(filepath[-1])[0] + "_outlines.png"
    outpix.save(fname_out)

    if len(filepath) > 1:
        b1 = gr.DownloadButton(visible=True, value=zip_path)
    else:
        b1 = gr.DownloadButton(visible=True, value=fname_masks)
    b2 = gr.DownloadButton(visible=True, value=fname_out)
    b3 = gr.DownloadButton(visible=True, value=json_path)
    b4 = gr.DownloadButton(visible=True, value=hist_txt_path, label="Download Histogram Data (TXT)")

    return outpix, flows, b1, b2, b3, b4, hist_eccentricity, hist_squares


def norm_path(filepath):
    img = imread(filepath)
    img = normalize99(img)
    img = np.clip(img, 0, 1)
    fpath, fext = os.path.splitext(filepath)
    filepath = fpath + '.png'
    pil_image = Image.fromarray((255. * img).astype(np.uint8))
    pil_image.save(filepath)
    return filepath


def update_button(filepath):
    filepath_show = norm_path(filepath)
    return filepath_show, [filepath], fp0, fp0


def update_image(filepath):
    filepath_show = norm_path(filepath[-1])
    return filepath_show, filepath, fp0, fp0


# Gradio интерфейс
with gr.Blocks(title="Cellpose-SAM", css=".gradio-container {background:purple;}") as demo:
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
                        max_iter = gr.Number(label='max iterations', value=250)
                        flow_threshold = gr.Number(label='flow threshold', value=0.4)
                        cellprob_threshold = gr.Number(label='cellprob threshold', value=0)
                    bins_number = gr.Number(label='Number of bins in histograms', value=10, minimum=5, maximum=50)
                    up_btn = gr.UploadButton("Multi-file upload (png, jpg, tif etc)", visible=True,
                                             file_count="multiple")
                with gr.Column(scale=1):
                    send_btn = gr.Button("Run Cellpose-SAM")
                    down_btn = gr.DownloadButton("Download masks (TIF)", visible=False)
                    down_btn2 = gr.DownloadButton("Download outlines (PNG)", visible=False)
                    down_btn3 = gr.DownloadButton("Download contours (JSON)", visible=False)
                    down_btn4 = gr.DownloadButton("Download Histogram Data (TXT)", visible=False)

        with gr.Column(scale=2):
            outlines = gr.Image(label="Outlines", type="pil", format='png', value=fp0)
            flows = gr.Image(label="Cellpose flows", type="pil", format='png', value=fp0)
            hist_eccentricity = gr.Image(label="Eccentricity Distribution", type="pil", format='png', value=fp0)
            hist_squares = gr.Image(label="Squares Distribution", type="pil", format='png', value=fp0)

    sample_list = glob.glob("samples/*.png") if os.path.exists("samples") else []

    if sample_list:
        gr.Examples(sample_list, fn=update_button, inputs=input_image, outputs=[input_image, up_btn, outlines, flows],
                    examples_per_page=50, label="Click on an example to try it")

    input_image.upload(update_button, input_image, [input_image, up_btn, outlines, flows])
    up_btn.upload(update_image, up_btn, [input_image, up_btn, outlines, flows])

    send_btn.click(cellpose_segment, [up_btn, max_iter, flow_threshold, cellprob_threshold, bins_number],
                   [outlines, flows, down_btn, down_btn2, down_btn3, down_btn4, hist_eccentricity, hist_squares])

    gr.HTML("""<h4 style="color:white;"> Notes:<br> 
                    <li>you can load and process 2D, multi-channel tifs.
                    <li>the smallest dimension of a tif --> channels
                    <li>you can upload multiple files and download a zip of the segmentations
                    <li>install Cellpose-SAM locally for full functionality.
                    </h4>""")

demo.launch()
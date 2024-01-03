import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.signal import butter, lfilter, correlate
from skimage import io, color, filters, measure, morphology
from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.transform import ProjectiveTransform
from skimage.measure import ransac


def plot_signal(signal, time, title='Signal', xlabel='Time', ylabel='Amplitude'):
    plt.figure(figsize=(20, 5))
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.5)
    plt.show()

def auto_correlation(df_signal, lags, type):
    """
    Plots the auto-correlation of a signal with improved aesthetics alongside the original waveform.
    """
    fig, (ax_wave, ax_acf) = plt.subplots(1, 2, figsize=(24, 6))

    # Plot the original waveform
    ax_wave.plot(df_signal.index, df_signal['Amplitude'])
    ax_wave.set_title("Originale Waveform ({})".format(type))
    ax_wave.set_xlabel("Zeit [s]")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.grid(True)
    
    # Plot the auto-correlation
    sm.graphics.tsa.plot_acf(df_signal, lags=lags, ax=ax_acf, title="Auto-Korrelogramm des Signals mit {} Lags".format(lags))
    ax_acf.set_xlabel("Lags")
    ax_acf.set_ylabel("Korrelationsgrad")
    
    # Improve line and marker aesthetics for auto-correlation
    ax_acf.lines[0].set_color('blue')  # Set the color of the auto-correlation line
    ax_acf.lines[0].set_linewidth(2)   # Set the line width
    
    # Enhance markers for each point on the auto-correlation line
    for i in range(1, len(ax_acf.lines)):
        ax_acf.lines[i].set_marker('o')    # Set marker shape
        ax_acf.lines[i].set_markersize(5)  # Set marker size
        ax_acf.lines[i].set_markeredgecolor('black')  # Set marker edge color
        ax_acf.lines[i].set_markeredgewidth(0.5)      # Set marker edge width
        ax_acf.lines[i].set_markerfacecolor('blue')   # Set marker face color
    
    ax_acf.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_cross_correlation(cross_corr, lags, title='Kreuzkorrelation'):
    """
    Plots the cross-correlation of two signals with improved aesthetics.
    """
    plt.figure(figsize=(20, 5))
    plt.plot(lags, cross_corr)
    plt.title(title)
    plt.xlabel('Lags')
    plt.ylabel('Kreuzkorrelationskoeffizient')
    plt.grid(True, alpha=0.5)
    plt.show()


def plot_original_modified_signal(original_signal, modified_signal, title='Originales und modifiziertes Signal-Segment'):
    """
    Plots the original and modified signal with improved aesthetics.
    """
    plt.figure(figsize=(20, 5))
    plt.plot(original_signal, label='Original', alpha=0.8)
    plt.plot(modified_signal, label='Modifiziert', alpha=0.7)
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.show()

def butter_lowpass(cutoff, fs, order=5):
    """
    Design a Butterworth lowpass filter.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def read_images(text):
    """
    Reads all images from the folder "Bilder" and converts them to RGB, grey and HSV.
    """
    images_rgb = []
    images_grey = []
    images_hsv = []

    for filename in os.listdir("Bilder"):
        if text in filename:
            image = cv2.imread("Bilder/" + filename)

            # convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_rgb.append(image_rgb)

            # convert to grey
            image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images_grey.append(image_grey)

            # convert to HSV
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            images_hsv.append(image_hsv)

    return images_rgb, images_grey, images_hsv


def plot_image_list(img_list, titles=None, cmap=None):
    fig, axes = plt.subplots(1, len(img_list), figsize=(16, 10))

    for i, image in enumerate(img_list):
        axes[i].imshow(image, cmap=cmap)
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.show()


def plot_image(image, title=None, cmap=None):
    """
    Plots single image.
    """
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def get_keypoints_and_descriptors(images, orb_detector):
    """
    Detects keypoints and extracts descriptors from all images.
    """
    keypoints_list = []
    descriptors_list = []

    for image in images:
        keypoints, descriptors = detect_features(image, orb_detector)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    return keypoints_list, descriptors_list


def detect_features(image, orb_detector):
    """
    Detects keypoints and extracts descriptors from an image.
    """
    orb_detector.detect_and_extract(image)
    keypoints = orb_detector.keypoints
    descriptors = orb_detector.descriptors
    return keypoints, descriptors


def plot_keypoints(images, keypoints_list):
    """
    Plots the image with the keypoints.
    """

    num_images = len(images)

    for i in range(num_images):
        fig, ax = plt.subplots()
    
        ax.imshow(images[i], cmap='gray')
        ax.scatter(keypoints_list[i][:, 1], keypoints_list[i][:, 0], facecolors='none', edgecolors='r', s=2)
        ax.set_title('Keypoints in Bild {}'.format(i + 1))
        plt.axis('off')
        plt.show()


def match_and_visualize(images, keypoints_list, descriptors_list, n, plot=True):
    """
    Matches the keypoints of the images and visualizes the matches.
    """
    num_images = len(images)

    sum_accuracy = 0

    for i in range(num_images):
        for j in range(i + 1, num_images):
            matches = match_descriptors(descriptors_list[i], descriptors_list[j])   

            accuracy = round(len(matches) / n, 4)

            if plot:

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

                plt.gray()

                plot_matches(ax, images[i], images[j], keypoints_list[i], keypoints_list[j], matches, only_matches=True)
                ax.axis('off')
                ax.set_title('Bild {} - Bild {} ({} Matches für {} Keypoints) - ({} Genauigkeit)'.format(i + 1, j + 1, len(matches), n, accuracy))

                plt.show()

            else:
                print('Bild {} - Bild {} ({} Matches für {} Keypoints) - ({} Genauigkeit)'.format(i + 1, j + 1, len(matches), n, accuracy))

            sum_accuracy += accuracy

    if not plot:
        average_accuracy = round(sum_accuracy / (num_images * (num_images - 1) / 2), 4)
        print('Durchschnittliche Genauigkeit: {}'.format(average_accuracy))

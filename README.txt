Sticker Search: Sketch and Text-based Image Search on Mobile

Environment: 
    Python 3.8, PyTorch 1.11
    An environment.yml for setting Miniconda environment is also included.

OneDrive link for dataset, checkpoints and third-party models:
    -snip-

Training: 
    training.py
    Dataset is available at OneDrive, to be placed in dataset/ at root.
    For training, only batch_1/ and batch_3/ are used.

Inference: 
    inference.py
    Dataset is available at OneDrive dataset/, to be placed in dataset/ at root.
    For evaluation, only batch_2/ are used.
    Checkpoints are available at OneDrive checkpoints/, to be placed in outputs/checkpoints/.
    Only a selected number of checkpoints are included due to large size.

Data generation: 
    utils/gen_data.py
    Third-party models used for sketch imitation are available at OneDrive models/, to be placed in models/at root.
    test_img_process.py, test_photosketch.py and test_hed.py are testing script for sketch imitation models and augmentations.

OCR:
    test_easyocr.py
    Prints OCR results with EasyOCR.

Acknowledgments:
    EasyOCR is available in https://pypi.org/project/easyocr/.
    PhotoSketch is available in https://github.com/mtli/PhotoSketch; model for inference is exported from modified PhotoSketch source code.
    HED is available in https://github.com/sniklaus/pytorch-hed; model for inference is exported from modified pytorch-hed source code.
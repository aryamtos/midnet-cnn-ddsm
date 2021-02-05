
<h1> Image Processing - CBIS DDSM</h1>
<hr />

<h2>CBIS DDSM</h2>
<p>This CBIS-DDSM (Curated Breast Imaging Subset of DDSM) is an updated and standardized version of the  Digital Database for Screening Mammography (DDSM) .The DDSM is a database of 2,620 scanned film mammography studies. It contains normal, benign, and malignant cases with verified pathology information.</p>
<a href="https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM">Cancer Imaging Archive Public Access</a>

<h2> Image Processing</h2>
  
<p>Tasks include displays, basic manipulations like cropping, flipping, rotating, segmentation etc.</p>
<p><a href="https://github.com/aryamtos/augmentation-processing-ddsm/blob/master/DDSMProcessing.ipynb">Source - DDSMProcessing</a></p>

<ol>
  <li>Convert DICOM to PNG (DICOM is the standard for the communication of medical imaging information)</li>
  <li>Extract image from directory</li>
  <li>Segmentation and supress artifacts</li>
  <li>Extraction pectoral Muscle</li>
  <li>Extraction ROI </li>
</ol>

<h3>Python Image Manipulation Tools</h3>
<ul>
  <li><a href="https://numpy.org/">Numpy- Provides support for arrays </a></li>
  <li><a href="https://pillow.readthedocs.io/en/stable/">PIL- Support for opening, manipulating and saving different image file formats. </a></li>
  <li><a href="https://docs.opencv.org/master/d6/d00/tutorial_py_root.html">OpenCV-Python - Is one of the most widely used libraries for computer vision</a></li>
</ul>

<h2>Data Augmentation</h2>

<p><a href="https://www.tensorflow.org/tutorials/images/data_augmentation?hl=en">Data Augmentation Tensorflow</a></p>

<p>Technique used to expand or enlarge your dataset by using the existing data of the dataset.</p>

<ol>
  <li>Rotation</li>
  <li>Width Shifting</li>
  <li>Height Shifting</li>
  <li>Brightness</li>
  <li>Shear Intensity</li>
   <li>Rotation</li>
  <li>Zoom</li>
  <li>Channel Shift</li>
  <li>Horizontal Flip</li>
  <li>Vertical Flip</li>

</ol>

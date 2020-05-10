
# Session 14 - Dense Depth Model

###	Objective
Create a custom dataset for monocular depth estimation and segmentation simultaneously.

1.  **Background (bg):** Select "scene" images. Like the front of shops, schools, playgrounds etc. Download 100 such backgrounds. <br> 
*Sol* - I have created the dataset of schools backgrounds. The scene school images including from australia, india, us and uk. I hav downloaded 174 images of school backgrounds. I will choose randomly 100 images of school.  
2.  **Foreground (fg):** Make 100 images of objects with transparent background.<br> 
*Sol* - I have done this by power point remove background tool. The foreground images are not limited to the humans, so i have included the images of fruits, vegetables, balls, dices, robots etc.
3.  **Foreground mask (fg_mask):** Create 100 masks, one per foreground.<br>
*Sol* - I directly created Foreground overlayed on background mask.
4.  **Foreground overlayed on background (fg_bg):** Overlay the foreground on top of background randomly. Flip foreground as well. We call this fg_bg.<br>
*Sol* - This step done by using the open cv package to extract the image mask of both foreground and background. The numpy random randint provides the random locations on background for overlaying foreground images. 
5.  **Foreground overlayed on background mask (fg_bg_mask):**. Create equivalent masks for fg_bg images.<br>
*Sol* - Created empty array of size background. Then merge with the mask of foreground image. The locations are same as the previous step provided by numpy.
6. **Foreground overlayed on background depth maps (fg_bg_depth):** Create equivalent depth maps for fg_bg images.<br>
*Sol* - For creating the depth map, used the code [DenseDepth](https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb) by using **NYU Depth V2** model weights. The model Weights can be downloaded from this [link](https://s3-eu-west-1.amazonaws.com/densedepth/nyu.h5). 

# Dataset Creation

#### Background (bg)
 - I have created the dataset of school images.
 - 174 images of school were downloaded from the internet.
 - Each image was resized to 224 x 224
 - Number of images after resizing: 100
 - Image dimensions: ( 3)
 - Directory size: 
 - Mean: []
 - Std: []

<img src="images/bg.png">

#### Foreground (fg)
 - 100 images of different objects were downloaded from the internet.
 - Using Power Point, the foreground was cutout. and the background was made transparent by adding an alpha layer.
 - Each image was rescaled to keep height 100 and resizing 100.
 - Number of images: 100
 - Image dimensions: (100, 100, 4)
 - Directory size: 1.2M

<img src="images/fg.png">

#### Foreground Overlayed on Background (fg_bg)
 - For each background
	 - Overlay each foreground randomly 20 times on the background
	 - Flip the foreground and again overlay it randomly 20 times on the background
 - Number of images: 100\*100\*2\*20 = 400,000
 - Image dimensions: (224, 224, 3)
 - Directory size: 1.1 G
 - Mean: []
 - Std: []

<img src="images/fg_bg.png">

#### Foreground Overlayed on Background Mask (fg_bg_mask)
 - For every foreground overlayed on background, its corresponding mask was created.
 - The mask was created by opencv bitwise operator and paste on a black image of same dimension of bg image at the same position the foreground was overlayed.
 -  Image was stored as a grayscale image.
 - Number of images: 400,000
 - Image dimensions: (224, 224)
 - Compressed Directory size: 241 M
 - Mean: []
 - Std: []

<img src="images/fg_bg_mask.png">

#### Foreground Overlayed on Background Depth Map (fg_bg_depth)
 - For every foreground overlayed on background, its corresponding depth map was generated.
 - A pre-trained monocular depth estimation model [DenseDepth](https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb) was used to generate the depth maps.
 - Image was stored as a grayscale image.
 - Number of images: 400,000
 - Image dimensions: (224, 224)
 - Directory size: 1.6G
 - Mean: [0.4334]
 - Std: [0.2715]

<img src="images/fg_bg_depth.png">

# Dataset Statistics

|  | # | dim | mean | std | size | img |
|---|---|---|---|---|---|---|
| **bg** | 100 | (224,224,3) | (0.5039, 0.5001, 0.4849) | (0.2465, 0.2463, 0.2582) | 2.5M | <img src="images/bg_sample.jpg"> |
| **fg** | 100 | (105,w,4) |  |  | 1.2M | <img src="images/fg_sample.png"> |
| **fg_mask** | 100 | (105,w) |  |  | 404K | <img src="images/fg_mask_sample.jpg"> |
| **fg_bg** | 400k | (224,224,3) | (0.5056, 0.4969, 0.4817) | (0.2486, 0.2490, 0.2604) | 4.2G |  <img src="images/fg_bg_sample.jpg"> |
| **fg_bg_mask** | 400k | (224,224) | (0.0454) | (0.2038) | 1.6G | <img src="images/fg_bg_mask_sample.jpg"> |
| **fg_bg_depth** | 400k | (224,224) | (0.4334) | (0.2715) | 1.6G | <img src="images/fg_bg_depth_sample.jpg"> |

### Dataset Link

 - Link: 
 - Size:
	 - Zip: 2G
	 - Unzip: 12G 
	 
# 7z to zip the data
7zip is the best compression to zip the data and load the data. The commands to zip and extract the data. 

- To zip the data<br> 
`!7z`
- To extract the data<br>
`!7z`

# Dataset Visualization
<img src="images/dataset.png">

# Resources

 - Code to overlay foreground on background and corresponding masks: 
	 - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
 - Code to generate depth maps for foreground overlayed on background: 
	 - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
 - Code to compute the combine the dataset and analyse the statistics:
	 - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

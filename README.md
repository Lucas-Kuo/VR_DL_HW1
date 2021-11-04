# VR_DL_HW1
> **It's strongly recommended to open up the .ipynb file and execute the cells one by one**\

[Link](https://drive.google.com/file/d/1-wnA207-0fuqKZMPL1r5lT_w9KqK2oKl/view?usp=sharing) (.h5 file) to my final model if you can handle the rest of the work.\
To reproduce the final submission, the following steps will show how you can do it.
1. [Cloning repo](#cloning-this-repo)
2. [Downloading files](#downloading-files)
3. [Build directories](#build-directories)
4. [Inference result](#inference-result)

### Cloning this repo
You can clone this repo in many ways, but the easiest is\
`git clone https://github.com/Lucas-Kuo/VR_DL_HW1.git`

### Downloading files
It's a must to download the dataset.
I'm providing a link to my Google drive [here](https://drive.google.com/file/d/1dYt4iLy0euxVXordHq4RRHgWojgUjgIf/view?usp=sharing).
If you have [gdown](https://github.com/wkentaro/gdown) in your environment, it's also fine to run\
`gdown https://drive.google.com/u/0/uc?id=1dYt4iLy0euxVXordHq4RRHgWojgUjgIf&export=download`\
on command line

It should be a zip file called *2021VRDL_HW1_datasets.zip*. Unzip it and you will see another zip file called *testing_images.zip*.\
Now make a folder called *dataset* and in that directory, build a subdirectory called *evaluation*. And then unzip *testing_images.zip* into *dataset/evaluation*.
On command line, you can run\
`unzip testing_images.zip -d dataset/evaluation`.

Now, your *VR_DL_HW1/dataset/evaluation* should contain 3033 images.\
**Make sure we're still on the same page so far.**

### Build directories
To build a convenient directory architecture, simply run config.py and build_dataset2.py by\
`python3 config.py` \
`python3 build_dataset2.py`

### Inference result
The final inference result can be produced by simply running inference.py\
`python3 inference.py`\
It should create a text file called *answer.txt* and this is how you can get it.

> Again, it's quite annoying and error-prone to follow the steps above. For simplicity, just open the .ipynb file and execute the cells.

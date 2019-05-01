
# On-the-fly training data generation
This repository contains a data-generation layer for Caffe (v1).
The layer takes a texture database, and produces opticalflow training data (images and groundtruth) live at training time.
It was used in the IJCV 2018 paper "What Makes Good Synthetic Training Data for Learning Disparity and Optical Flow Estimation?" (http://dx.doi.org/10.1007/s11263-018-1082-6), where it proved to produce better data (benchmarked on Sintel) than the original FlyingChairs dataset.
You could (if you were fond of terrible puns) call this one **On-the-Fly-ingChairs**.

Predefined options include various object shapes and motion types.


![example1-a](img/image_00002.png)
![example1-b](img/image_00003.png)
![example1-f](img/flow2.png)
![example2-a](img/image_00006.png)
![example2-b](img/image_00007.png)
![example2-f](img/flow6.png)



# Author
Nikolaus Mayer (mayern@cs.uni-freiburg.de)

# License
This code is provided for research purposes only and without any warranty. Any commercial use is prohibited. If you use the code or parts of it in your research, you should cite the aforementioned paper: 
```
@Article{MIFDB18,
  author       = "N. Mayer and E. Ilg and P. Fischer and C. Hazirbas and D. Cremers and A. Dosovitskiy and T. Brox",
  title        = "What Makes Good Synthetic Training Data for Learning Disparity and Optical Flow Estimation?",
  journal      = "International Journal of Computer Vision",
  number       = "9",
  volume       = "126",
  pages        = "942--960",
  month        = "Sep",
  year         = "2018",
  note         = "https://arxiv.org/abs/1801.06397",
  url          = "http://lmb.informatik.uni-freiburg.de/Publications/2018/MIFDB18"
}
```

The codebase utilizes the wonderful libraries [CImg](http://cimg.eu/) (download it and put `CImg.h` into `include/thirdparty`) and [Anti-Grain Geometry](https://github.com/tomhughes/agg).

# Usage
See `example/train.prototxt`. 
The layer includes a number of hardcoded data modes, but is trivial to extend. 
Textures are provided as a list of image files (loaded once at startup).

The data dimensions are hardcoded in `include/caffe/data_generation/DataGenerator.h`, lines 55 et seq.

**Note** that this layer uses a *lot* of CPU power.


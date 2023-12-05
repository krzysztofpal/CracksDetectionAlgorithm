# CracksDetectionAlgorithm
This repository contains code for cracks detection. This program is described in detail in the paper titled: "An image processing approach for fatigue crack identification in cellulose acetate replicas"

# Environment installation

This repository uses *conda* environments to run the algorithm. In order to install *conda* click [here](https://anaconda.org/anaconda/conda) and follow the instructions.

After *conda* installation the new *conda environment* must be created. To do so run open the repository's main folder in terminal/command prompt. Then, run the following line of code

```
conda env create -f environment.yml
```

This line of code creates *conda environment* called *cracks* with all required libraries to run the algorithm. Next, you have to activate this environment by running the command:

```
conda activate cracks
```

At this point you have installed and activated the *cracks* environment. You can now proceed to run the algorithm.

# Algorithm

The algorithm is stored in *algorithm.py* file. To use the algorithm, call function *compute*. The function *compute* takes four input parameters:

- img: this is image object from opencv library
- width: the width of the reduced image used for computations. In order to reduce amount of comuptations the algorithm shrinks input image to the specified width, performs computations, and restores original width. The smaller the *width*, the quicker the computations happend, but the algorithm is less precise. We recommend using it with value equal to **256**
- size_limit: the minimal amount of pixels on the reduced image required for forming a crack. This parameter governs filtering of noise through setting a threshold. We recommend using it with value equal to **8**
- output_plots_path: this parameter contains the path to folder, where computation results visualizations are saved. If the value is **None**, then visualizations will not be generated.

# Running the example

The file *run_algorithm_example.py* provides quick demo of the algorithm. It takes the image from *data* folder, process it and saves the results visualizations in *demo_results* folder.

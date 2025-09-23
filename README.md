# ML Audio Classification System

## How to use the repo
Here are the **core essentials** to get the code running:

### Install requirements
You can install dependencies with either **bash** or **python** (depending on your environment)
```bash
pip install -r requirements.txt
```

### Create dataset
You have multiple options for how you want to create your dataset, either 
- Provide **YouTube links**
- Upload a **Google Drive Folder ID** (I have a public speech folder for publicly available)
```bash
python -m src.download_from_youtube
python -m src.download_speech_from_drive
```

You can then convert raw audio into *.npz* spectrograms and split into *train.csv*, *val.csv* and *test.csv*
```bash
python -m src.preprocess
```

If you ever need to reset the processed files(including _.csv_ files and _.json files_). You can just run:
```bash
python -m src.reset_processed
```

### Train the model
Now you have all the necessary *.csv*,*.json* and *.npz* files to begin training. 
```bash
python -m src.train
```
This will return to you `best_model.pt` and within the `artifacts/` directory you have
- `best_model.pt`
- `pred_labels.npy`
- `true_labels.npy`
- `val_confusion_matrix.png`
- `val_report.txt` 
Training logs are *also* shown on the CLI.

### Sanity and Debug checks
**Important:** to run this you need to do this from the **repo root**, this is due to `sys.path.append()` I utilized
Although there `train.py` and other scripts include internal sanity checks, but I’ve also added extra ones…: 
```bash
python notebooks/00_sanity_check.py
python notebooks/01_iterable_dataloader_check.py
```

## Overview
Classify drone, car and speech sounds into their respective classes. However, due to the processing nature of the repository any users can upload the sounds of their choice and therefore, classify realistically **any amount** of sound classes, from any origin.

#### Repo workflow diagram
```
Raw audio → Preprocess → `.npz` spectrograms → Train/Val/Test CSVs → Model → Artifacts
```

#### Processing files and creating the dataset
The system first process files into suitable `npz` files which are then broken down into 2 second chunks using the function `choppinator()`, which I thought sounded funny. We turn the audio to mel spectrograms(mels resembling human hearing better than dB) which is the most common approach when working with audio data. Although spectrogram is an image, during code its actually rather long arrays. 

Afterwards the spectrograms are split into multiple `.csv` files which are train/va/test. This is standard ML stratifying and works effectively, using the train.csv as a file to train then using the validation file to validate if model is truly effective and is not overfitting. Finally test is left as a global truth, sort of like a final test to see if the model truly works. In addition metadata on the stratifying information is stored in `.json` file which can help create judgement on if your data is nicely balanced. 

You can reset your dataset using the `reset_processed.py` file which brings everything back to its original state. 

#### Understanding the workflow from dataset to trained results
Then we have `data.py` which uses pytorch's `torch.utils.data` which utilizes the `Dataset` and `Dataloader` classes which create custom instruction examples in `Dataset` and then `Dataloader` runs through them to create iterable datasets that can then be turned into tensors and used by the model to train itself.

In `model.py` we create a simple standard CNN which goes through three convolution layers, two pooling layers and then one global pool(reducing each feature map into just one value), taking the highest value in the map. The `ReLU()` class has the purpose of preventing the convolutions, to just become a stack of linear transformations; zeroing out we features and enhancing the more potent ones - in short makes a region more jagged and different. 

Combining the `data.py` utility and the `model.py` setup we are able to train the dataset. As a sanity check, we first just train a mini-batch over one epoch this allows us to ensure that the model **can actually overfit**, ensuring that the model, chunking and the rest of the process are working properly. We then train the model across X amount of epochs and display all both train and validation sets accuracy and loss. 

#### Understanding how the model works
To conceptually understand what the model does, I'll attempt to explain it in a clear manner. We begin at a single channel, which inversely relates to resolution, it is how much input from the tensors are we allowing into our model. As we pool the data we shrink our map and zoom out, making our resolution decrease but our channels increase, therefore we capture coarses features rather than finer ones: going from **shallower layers**(capture fine details) -> **deeper layers**(capture complex/abstract features). The model adjusts its weights - these can be thought of as cogs that decide how the model predicts - and after every layer it adapts improving on itself after each new feature. After the final pool we flatten the data, this means breaking down our X size tensors into [B, C] from in our case [B, C, 1 , 1]. Conceptually, think about a long array of "tables" with size[1,1] which can therefore just be transformed into a normal array or in the case of pytorch a 2D tensor.

Although I used lots of documentation, an essential document that helped me train my classifier was "https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"  

#### Forward pass and backward pass
Furthermore, a neural network you got forward pass, which in short just predicts + measures how wrong we are. Then there is the backward pass which is - figure out how to nudge each weight to make predictions less wrong. The forward pass does the math and compares the outputs using the **loss function** while the backward pass calculates how much each parameter contributed to each error, computes gradients(partial derivates wrt each weight) and then stores these gradients inside each parameter tensor.

#### Understanding backwards pass and gradients
The way I like to imagine training for a backwards pass on a neural network: Training a neural net is like balancing a beam on pivots (weights). The loss measures how unbalanced the beam is. The gradients tell you how "wrong" each pivot is and in which direction. A postive gradient -> leaning one way you need to decrease the weight to rebalance, while negative gradient -> need to add more weight to rebalance. The **optimizer** then nudges the weights slowly till the beam is in equilibrium(minimum loss).

#### What is CrossEntropyLoss
**CrossEntropyLoss** in simple terms finds out how much different the predicted labels are to the true ones. It takes raw scores(logits) and turns them into a probability distribution, this in other terms is called **softmax**. The package works the following way: if the model is perfect -> loss is 0, if confused(uniform prediction) -> loss is aprox. 1.1, if confident but wrong -> loss is huge 5+. In short, **CrossEntropyLoss** punishes confident mistakes hard, while rewarding confident correct answers.

#### Seeding
To allow for the reproducibility of the code I ensured that both the `preprocess.py` would create seeded train/val/test sets and also in `train.py` the model would give the same results if the input parameters were the same. This allows for debugging to become simpler - its easier to spot errors since there is no random chance at play and also allows for full reproducibility, comparing of accuracies between model parameters. However, seeding ≠ full determinism, this is since I have not enforced deterministic kernels, since for most ML training bit for bit identical implmentation is not necessary, it also hurts performance speed and just general unnecessary strictness.

#### Adam optimizer
An optimizer updates the model's parameters using information from the **loss function** and its gradients. The Adam optimizer is just one of many optimizers available on PyTorch. The Adam optimizer:
- fast, low long term convergence -> basically first few epochs it reduces loss fast and then slows down a lot
- robust to noisy small batch gradients (bs=32)
- works well when data is limited, don't have to tune learning rates per layer(Adam figures out learning rates by layer)

#### Augmentations
To prevent overfitting I included augmentation which is a form of **regularization**(technique used to prevent model from overfitting to training data). In the `data.py`, the two techniques I decided was gaussian noise which just includes random noise at a magnitude which helps distort samples. I used `torch.randn_like` it has a mean=0 and var=1 so within 99% of cases the noise would stay within (-3,3), therefore I decided not to include other safety measures if for example 12 popped up since it is so unlikely. The other augmentation I included was **time masking** specifically I took inspiraton from a paper called **SpecAugment** which found effective results by completely silencing there timemasked segments instead of damping, therefore I implemented there method.

#### Sanity and Debug explanations
For each **sanity or debug check** the function of each file is the following: `00_sanity_check.py` -> loads metadata and the training csv then opens one `.npz` file and plots its mel-spectrogram with label, `01_iterable-dataloader-check.py` -> builds train/val/test Dataloaders and then fetches batches and prints out input/target shapes. 

#### Understanding `config.yaml`
Holds all hyperparameters that the user can change(batch size, learning rate, paths, seeding) so you can tweak experiments without ever having to touch the code. Its a convenient form of trying different inputs to test outputs, without going into the extensive code.

### Results
The two folders `pred_labels.npy` and `true_labels.npy` give you the raw values if ever needed for deeper analysis, while the *val_confusion_matrix.png* and *val_report.txt* will give you the precision, recall and F1 score. The F1 score being a harmonic between precision and recall, therefore it punishes imbalances greatly, it wants both variables to be good if one is bad the F1 score drops with it. The **confusion matrix** is just simple visual representation of the **validation report**.

After training with the default configuration you can expect for a `best_epoch_004_acc_0.916_seed_69` something like this as your `val_report.txt`

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Drone    | 0.9592    | 0.9724 | 0.9658   | 290     |
| Car      | 0.8168    | 0.9784 | 0.8903   | 278     |
| Speech   | 0.9891    | 0.8144 | 0.8933   | 334     |
| **Accuracy** |           |        | **0.9157** | 902     |
| **Macro Avg** | 0.9217    | 0.9217 | 0.9165   | 902     |
| **Weighted Avg** | 0.9264    | 0.9157 | 0.9157   | 902     |

**Interpretation**:
- Drone → strong performance overall (F1 ≈ 0.97).  
- Car → high recall (0.98), but lower precision (0.82).  
- Speech → very precise (0.99) but lower recall (0.81).


### Comments on the code
Here I will comment on the code, just general comments I picked up along the way that more interested readers can read over. The comments are rather extensive, look through documentation if necessary.

- Used `MaxPool2d()` since it could capture the strongest activation, then using `AdaptiveAvgPool2d` since it would get the average of all these activations and keep the result balanced
- One can implement the class torchvision.transforms.v2.GaussianNoise() to help implement Gaussian noise, however I feel like the "barebones" method I implemented does just a good a job and is clearer to understand. Furthermore, extra overhead is not necessary, and it is not a standard bringing in an image into the class then applying all the augmentations to it, like  a standalone augmentation class.
- Since numpy sometimes stores its own values as `numpy.dtype.str` it might look like a string but its actually a numpy string, therefore if I use `str()` to convert what looks like a string into a string its probably due to this
- It is a common convention in pytorch to refer to final dimension with [-1] instead of [...] cause you might add more dimensions later on or even remove them
- `.item()` only works for 0D tensors while `.float` can turn boolean expressions into numbers and works for B size tensors.
- When using `torch.argmax()` if you use dim=1 you will compare columns for every row while dim=0 compares rows per column instead, you would then be classifying the classes’ strongest sample rather than the strongest class per sample
- Placing model in `.train()` mode is important it allows for Dropout, BatchNorm and other features which would not occur if not in training mode. Dropout layers randomly drop neurons during training. Batch normalization, basically stabilizes every layer by keeping the distribution roughly centered and scaled, this is regardless of the batch's raw activations
- I originally had the same variables for the sanity check and the true training, it was contaminating my future epochs so I separated them
- I added sanity checks so that `non_blocking` and `pin_memory` so that even in `config.yaml` the users incorrect parameters with respect to their hardware would raise no warnings, keeping the code cleaner. I basically gave the parameters in `config.yaml` with respect to this scenario a "false freedom".
- When I used `np.random.randint(low, high)`, you need to consider that it is high exclusive so if you do T-8 you will actually get T-8 -1 which means you'll never truly get the final batch making you miss data. Therefore, I explicitly added +1, so its clear that we are using 8 as our range.
- train/val/test split is seeded, you’ll get repeatable experiments. The exact NPZ filenames don’t affect model behavior.
- Back in older versions of `matploblib` you had the Old style `(set_xticks + set_xticklabels)` required two calls: one to set tick positions, another to set their labels. But now you can just join it in `.set_xticks(ticks, labels=...)`, so in the name of innovation I picked that.
- When using `Dataloader` there is no need to `import pickle` because `DataLoader` already calls pickle internally by itself, when spawning extra workers for multiprocessing. That also means you cant have lambda function in your `Dataset` due it not having a consistent name and it only existing in the memory during that runtime.
- When using DataLoader I decided not to use the argument `drop_last` only because the personal dataset that I was using was large enough and compared to the batch size of 32 it would not change the gradients nor the loss function of the program. However if you want reproducibility or are using a mutli-GPU system where batch size across devices matters then you probably would want to turn it on.
- When trying to add gaussian noise, using `randn` like so `x = x + 0.01*torch.randn(x.shape)` works because you get the shape of the tensor so you can create a 2D tensor that has random values. However the result from `.randn()` is automatically going to have as its internal `.device()` as CPU and if x tensor is on the GPU which it _probably_ will be there will an error. From the documentation directly “It is important to know that in order to do computation involving two or more tensors, all of the tensors must be on the same device.”
- When creating from the numpy arrays to tensors one can use `tensor.from_numpy()` instead of `tensor()` I decided against that since I did not want both the tensor and the numpy array to have a shared storage since in almost all use cases with these spectrograms, no changes will be made after they have been processed so its safer to just use `tensor()`
- I use snake_case throughout almost the entire code but I will have times that I use camelCase, if I did not fix it it is cause I find it alright or I missed it.
- I used PEP-8 official Python style guide, not to the tea but in the holistic sense.
- Used `tdqm` over Python’s built-in progress bar for cleaner, more informative and aesthetic appeal.

## Documentation I used
I tracked the documentation I used to complete this. 
**Disclaimer**: the order of the documentation is roughly the order I used it in but not completely. Also I did not include every single page of documentation, this may because I did not think it was important enough or I simply forgot, it more likely the latter than the former.

<details>
<summary>Details</summary>

- https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html#librosa-feature-melspectrogram 
- https://librosa.org/doc/latest/generated/librosa.util.fix_length.html#librosa.util.fix_length 
- https://librosa.org/doc/latest/generated/librosa.power_to_db.html#librosa-power-to-db 
- https://numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html 
- https://docs.python.org/3/library/uuid.html 
- https://numpy.org/devdocs/reference/generated/numpy.savez_compressed.html#numpy-savez-compressed 
- https://stackoverflow.com/questions/19907442/explain-dict-attribute 
- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html 
- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html 
- https://docs.python.org/3/library/pathlib.html 
- https://medium.com/@niranjanky14/mastering-gitignore-10eca727a264 
- https://stackoverflow.com/questions/57296168/pathlib-path-write-text-in-append-mode 
- https://docs.python.org/3/library/functions.html#open 
- https://www.geeksforgeeks.org/python/json-dumps-in-python/ 
- https://www.codecademy.com/resources/docs/python/json-module/dump 
- https://docs.python.org/3/howto/argparse.html 
- https://www.geeksforgeeks.org/python/json-load-in-python/ 
- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html 
- https://numpy.org/doc/2.0/reference/generated/numpy.load.html 
- https://matplotlib.org/stable/plot_types/arrays/imshow.html#sphx-glr-plot-types-arrays-imshow-py 
- https://matplotlib.org/stable/api/pyplot_summary.html 
- https://docs.python.org/3/library/shutil.html 
- https://docs.pytorch.org/docs/stable/index.html 
- https://docs.pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor 
- https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html#torch.from_numpy 
- https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html 
- https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html 
- https://stackoverflow.com/questions/73045968/pytorch-why-conv2d-results-are-all-different-their-data-type-is-all-integer 
- https://docs.pytorch.org/docs/stable/generated/torch.set_default_dtype.html 
- https://numpy.org/doc/2.0/reference/generated/numpy.ndarray.item.html 
- https://docs.python.org/3/library/functions.html#hasattr 
- https://numpy.org/doc/2.1/reference/generated/numpy.dtype.str.html 
- https://discuss.pytorch.org/t/what-is-the-difference-between-none-and-unsqueeze/28451 
- https://wandb.ai/mostafaibrahim17/ml-articles/reports/A-Deep-Dive-Into-Time-Masking-Using-PyTorch--Vmlldzo1Njg1Nzc5 
- https://numpy.org/doc/2.1/reference/random/generated/numpy.random.randint.html 
- https://www.isca-archive.org/interspeech_2019/park19e_interspeech.pdf 
- https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-epoch-in-machine-learning 
- https://stackoverflow.com/questions/59090533/how-do-i-add-some-gaussian-noise-to-a-tensor-in-pytorch 
- https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.GaussianNoise.html 
- https://docs.pytorch.org/docs/stable/generated/torch.randn.html 
- https://stackoverflow.com/questions/71278607/pytorch-expected-all-tensors-on-same-device 
- https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html 
- https://docs.pytorch.org/docs/stable/generated/torch.randn_like.html 
- https://en.wikipedia.org/wiki/Hessian_matrix 
- https://machinelearningmastery.com/gradient-in-machine-learning/ 
- https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html 
- https://docs.pytorch.org/docs/stable/data.html 
- https://peps.python.org/pep-0008/ 
- https://docs.python.org/3/library/sys.html 
- https://stackoverflow.com/questions/56639952/what-does-pathlib-path-cwd-return 
- https://www.w3schools.com/python/ref_func_next.asp 
- https://www.w3schools.com/python/ref_func_super.asp 
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential 
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d 
- https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html 
- https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html 
- https://medium.com/@benjybo7/7-pytorch-pool-methods-you-should-be-using-495eb00325d6 
- https://stackoverflow.com/questions/65993494/difference-between-torch-flatten-and-nn-flatten 
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Flatten.html 
- https://discuss.pytorch.org/t/what-is-forward-and-when-must-you-define-it/136969 
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html 
- https://realpython.com/python-modules-packages/
- https://docs.pytorch.org/docs/stable/optim.html 
- https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/ 
- https://github.com/tqdm/tqdm 
- https://tqdm.github.io/ 
- https://tqdm.github.io/docs/tqdm/#set_description  
- https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam 
- https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180/6 
- https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html 
- https://docs.pytorch.org/docs/stable/generated/torch.mean.html 
- https://www.w3schools.com/python/ref_func_float.asp 
- https://docs.pytorch.org/docs/stable/generated/torch.argmax.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout  
- https://www.geeksforgeeks.org/computer-vision/what-is-batch-normalization-in-cnn/ 
- https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html 
- https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 
- https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html 
- https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html 
- https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html 
- https://docs.pytorch.org/docs/stable/generated/torch.save.html 
- https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html 
- https://stackoverflow.com/questions/76451315/difference-between-pathlib-path-resolve-and-pathlib-path-parent 
- https://spacelift.io/blog/yaml 
- https://pyyaml.org/wiki/PyYAMLDocumentation 
- https://www.w3schools.com/python/ref_dictionary_get.asp 
- https://www.w3schools.com/python/ref_keyword_del.asp 
- https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/4 
- https://docs.python.org/3/library/os.html#os.environ 
- https://www.w3schools.com/python/ref_random_seed.asp 
- https://numpy.org/doc/2.1/reference/random/generator.html 
- https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html 
- https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator 
- https://docs.pytorch.org/docs/stable/generated/torch.cuda.manual_seed_all.html#torch.cuda.manual_seed_all 
- https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader 
- https://docs.pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed 
- https://docs.python.org/3/library/functools.html#functools.partial 
- https://docs.pytorch.org/docs/stable/generated/torch.get_rng_state.html 
- https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_rng_state_all.html 
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html 
- https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report 
- https://www.w3schools.com/python/python_file_open.asp 
- https://docs.python.org/3/library/os.path.html#os.path.join 
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html 
- https://matplotlib.org/stable/users/explain/quick_start.html#quick-start 
- https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size 
- https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.subplot.html 
- https://matplotlib.org/stable/tutorials/pyplot.html 
- https://matplotlib.org/3.5.3/gallery/images_contours_and_fields/interpolation_methods.html 
- https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots 
- https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html 
- https://matplotlib.org/stable/api/axes_api.html 
- https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticks.html#matplotlib.axes.Axes.set_xticks 
- https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots 
- https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html 
- https://matplotlib.org/stable/users/explain/axes/tight_layout_guide.html 
- https://stackoverflow.com/questions/16032389/pad-inches-0-and-bbox-inches-tight-makes-the-plot-smaller-than-declared-figsiz 
- https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.close.html 
- https://stackoverflow.com/questions/72504734/what-is-the-purpose-of-with-torch-no-grad 
- https://docs.pytorch.org/docs/stable/generated/torch.cat.html 
- https://stackoverflow.com/questions/66833911/how-can-torch-cat-only-have-one-tensor 
- https://numpy.org/doc/stable/user/basics.interoperability.html 
- https://www.w3schools.com/python/ref_dictionary_items.asp 
- https://www.w3schools.com/python/python_lambda.asp 
- https://www.w3schools.com/python/ref_func_sorted.asp 
- https://numpy.org/devdocs/reference/generated/numpy.save.html 
- https://spec.commonmark.org/0.31.2/ 
- https://c3.ai/glossary/data-science/f1-score/ 

</details>

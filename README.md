# Project_Semantics


# Code basis
Code practical 1.
## Preparation
In order to run this code, you need to have the following packages installed:
* PyTorch
* tensorboardX
* scipy

Additionally, SentEval needs to be installed to run evaluations with SentEval. You might need to adjust the path to SentEval at the top of file `sent_eval.py`. Make sure that all datasets for the downstream tasks are downloaded.

The data needed to train the model on SNLI are provided [here](https://drive.google.com/open?id=173ZBT5AZwLHMX5bs-gB-h2j5f8TBsOfA). Please store them under the main directory (SMNLS_Practical1/snli_1.0/...). In the same folder, you find files storing the glove embedding. Please download `small_glove_embeds.npy` and `small_glove_embeds.txt` and store them in the code folder. 


## Functionality

### Overview

This training framework is splitted into multiple files. Each of these files represents a different module that is more or less independent (e.g. data processing and model implementation). Furthermore, the implementations are, in general, independent of the specific task of _Natural Language Inference_, so that this framework can easily be extended to new tasks. A short description for each of files/modules is given here:

* ```train.py```: This file summarizes all functionality required for training. Please read the section _Training_ below for more details.
* ```eval.py```: The implementation of the evaluation of the models is done in this file. Read the section _Evaluation_ for more information.
* ```infer.py```: To perform inference and test the model's prediction on new sentences, this file can be used. The section _Inference_ describes the usage in detail.
* ```sent_eval.py```: The SentEval evaluation is implemented in a separate file. This makes the framework more modular in the sense that new evaluation methods can be added if needed. Furthermore, if only SentEval should be executed on a spefic model, this file can be run as ```python infer.py --checkpoint_path VAL``` where VAL should be replaced by the path to the model's checkpoint. Note that the results are only printed and not saved in a file. For that, please refer to ```eval.py```.
* ```model.py```: The network models are implemented in the file _model.py_. Currently, only networks for the task of _Natural Language Inference_ are implemented. Those are all based on the class ```NLIModel```, which separates the network into a encoder and a classifier. Both can be separately configured and implemented. In this practical, we are mainly concerned with the design of the encoder. For extending the framework by a new encoder, please refer to the section _Extending the framework_ below. 
* ```data.py```: All operations based on the raw data and handling the dataset is summarized in this file. This also includes the processing of the word embeddings. Please refer to the next section _Data pre-processing_ for more information.
* ```mutils.py```: This file summarizes all basic functionality that is shared among many files. This includes for example the loading of a model, and the parameter configuration. 
* ```visualizing_max_pooling.py```: All functionality to visualize and analyse the functionality of the max pooling operation in the Bi-LSTM model is summarized in this file.

### Data pre-processing

The data pre-processing contains four main steps:
* **GloVe word embeddings**: First, the token embeddings are loaded from the pre-trained GloVe vectors. Note that s smaller version of GloVe is here used after removing all words that do not occur in either SNLI or any downstream task in SentEval. The functionality for creating such a small embedding is also included in the `data.py` file, but not required if the network is only trained on SNLI and evaluated on SentEval (see downloaded files in the beginning).
* **Dataset handling with  ```DatasetTemplate```**: The general functionality of a dataset is summarized in the class `DatasetTemplate`. Neither torchtext nor torchnlp is used, as a own dataset function provides the most traceable pre-processing.
* **Token processing**: To convert a file into a sequence of tokens, the text is splitted along spaces. For the end token, we also consider dots, question marks etc. as separate token. In contrast to the original paper, we also remove any other non-alphabetic letter and split words connected by "-" if they cannot be found in the GloVe word embedding.
* **Start and end sentence token**: Similar to the original paper, a start of sentence token and end of sentence token is added to every sentence.

### Training

The training file provides many options to configure the training process.

1. **Model parameters** 

	The model is configured by an encoder and a classifier.

	1. *Encoder*

		Currently, four different encoder architectures are implemented. They can be chosen by the option ```--model```:

		* ```--model 0```: Bow Baseline
		* ```--model 1```: Uni-directional LSTM
		* ```--model 2```: Bi-directional LSTM
		* ```--model 3```: Bi-directional LSTM with max pooling

		Further options for the encoder are:

		* ```--embed_dim VAL```: The embedding dimensionality of the sentence (output of the encoder). Will be automatically set to 300 in the case of the BoW model (equals to word dimensionality). Default: 2048

	2. *Classifier*

		The default classifier consists of a MLP with two linear layers. The hidden dimensionality is 512, and the output are 3 classes. 

		* ```--fc_dim VAL```: The number of hidden units in the classifier. Default: 512
		* ```--fc_dropout VAL```: Dropout applied on the hidden layer. Default: 0.0, no dropout
		* ```--fc_nonlinear```: If selected, a non-linear tanh activation is applied between the two linear layers.

2. **Optimizer parameters**

	Two different optimizers are supported:

	* **SGD** (```--optimizer 0```). The vanilla SGD optimizer with additional parameters:

		* ```--weight_decay VAL```: Applies weight decay by adding a l2 loss over weights. Default: 1e-2 (value in the paper)
		* ```--momentum VAL```: Applies momentum. Default: 0.0 (no momentum)

	* **Adam** (```--optimizer 1```). Adam optimizer with default parameter values as specified by PyTorch optimizer package.

	Additional parameters valid for every optimizer:

	* ```--learning_rate VAL```: Learning rate. Default: 0.1
	* ```--lr_decay VAL```: The learning rate is reduced after every epoch if the evaluation accuracy is worse than the mean of the two previous. If the option ```--intermediate_evals```is activated, then the average of the intermediate evaluations is compared to the pre-last epoch accuracy. The reduction factor can be specified by this option. Default: 0.2
	* ```--lr_max_red_steps VAL```: After how many learning rate reduction steps the training automatically stops. Default: 5
	* ```--batch_size VAL```: Batch size used during training. Default: 64

3. **Training configuration**
	
	Parameters specifying training configuration:

	* ```--checkpoint_path VAL```: The path where the checkpoints should be saved. Default: a new folder in "checkpoints/" will be created based on the date and time.
	* ```--restart```: If checkpoint folder is not empty, the previously stored checkpoints etc. are deleted. 
	* ```--load_config```: If checkpoint folder is not empty, the configuration saved in this folder (from the previous experiment) will be loaded. Recommended if experiment should be continued.
	* ```--tensorboard```: If selected, a tensorboard is created during training. Stored in the same folder as the checkpoints.
	* ```--intermediate_evals```: If selected, a evaluation will be performed every 2000 iterations. Gives a less noisy estimate of the evaluation accuracy. 
	* ```--seed VAL```: What seed to set to guarantee reproducability of experiment. Default: 42. 
	* ```--cluster```: Strongly recommended if the experiments are executed on the clusters. Reduces the output on the terminal.
	* ```--debug```: If selected, the evaluation set will be used for training. _ONLY_ use this option for debugging of the model.


### Evaluation

Models can be evaluated with the file ```eval.py```. Thereby, the best checkpoint is tested on the train, validation and test dataset, and the results will be exported to the file _evaluation.txt_ in the same checkpoint folder. Additionally, the model is tested separately on an easy and hard subset of the test dataset as specified on the [SNLI dataset website](https://nlp.stanford.edu/projects/snli/). These scores are stored in a separate file called _extra\_evaluation.txt_. 

To visualize the embeddings learned by the model, the two subsets of the test dataset and some SentEval downstream tasks can be exported to a tensorboard. In that, tSNE and PCA can be performed to find certain patterns in the data. This evaluation can be selected with the option ```--visualize_embeddings```. The tensorboard file is exported in the same checkpoint folder.

Furthermore, the model is tested on a bunch of [SentEval](https://github.com/facebookresearch/SentEval) tasks as specified in the [InferSent]() paper: 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICKEntailment', 'SICKRelatedness', 'STS14'. The results are saved in a dictionary as pickle file called _sent\_eval.pik_. 

The options for configuring the evaluation are:
* ```--checkpoint_path VAL```: The path to the checkpoint folder which should be evaluated. This option also accepts regular expressions like "checkpoints/LSTM\*". If such is specified (note that those have to be in quotation marks to be parsed as string), all folders that correspond to this regex are evaluated.
* ```--overwrite```: Usually, an evaluation is skipped if the files _evaluation.txt_, _extra\_evaluation.txt_ and _sent\_eval.pik_ exist. If this option is selected, the existing files are ignored and overwritten. 
* ```--visualize_embeddings```: As described above, this option exports the embeddings of the test subsets for visualization.

### Inference

The file ```infer.py``` provides an interface for running inference on arbitrary sentence pairs. For this, it takes as input a file with a certain format: every line contains the premise and hypothesis separated by the sequence "#SEP#". A example file is the following:

```
Two kids at a ballgame wash their hands. #SEP# Two kids in jackets walk to school. 
``` 

The output are the labels predicted by the network. To specify the inference, the file takes the following parameters:
* ```--checkpoint_path VAL```: Path to the checkpoint which should be loaded. If the specified path is a folder, the newest checkpoint is used. Otherwise, in case it is the file, the certain file is loaded.
* ```--input_file VAL```: Path to the input file which should be processed. Please make sure this file follows the format as specified above.
* ```--output_file VAL```: Path to the output file to which the results should be exported. Default: _infer\_out.txt_

### Examples/Experiments

* Training Baseline
```
python train.py --model 0 --optimizer 0 --learning_rate 0.1 --weight_decay 1e-5 --lr_max_red_steps 5 --intermediate_evals --checkpoint_path checkpoints/Baseline
```

* Training uni-directional LSTM 
```
python train.py --model 1 --optimizer 0 --learning_rate 0.1 --weight_decay 1e-5 --lr_max_red_steps 5 --intermediate_evals --checkpoint_path checkpoints/LSTM
```

* Training bidirectional LSTM 
```
python train.py --model 2 --optimizer 0 --learning_rate 0.1 --weight_decay 1e-5 --lr_max_red_steps 5 --intermediate_evals --checkpoint_path checkpoints/BiLSTM
```

* Training Bi-LSTM with max pooling
```
python train.py --model 3 --optimizer 0 --learning_rate 0.1 --weight_decay 1e-5 --lr_max_red_steps 5 --intermediate_evals --checkpoint_path checkpoints/BiLSTM_Max
```

## Extending the framework

The framework is created very modular, so that it can be easily extended. A short description how, and where to add new aspects.

### Adding new dataset

The datasets are summarized in the file `data.py`. In order to add a new dataset, create a new class in this file which inherits from the class `DatasetTemplate`. If the data simply consists of a single sentence and a label, you can also use the class `DatasetTemplate` directly and create a new instance of it. The single data objects are from type `SentData`. For an example, please have a look in the file `mutils.py`, function `def get_transfer_datasets()`.

### Adding new encoder

New models can be implemented in the file `model.py`. Specifically, a new encoder for the NLI task has to inherit the class `EncoderModule`. In there, define a forward pass and a initialization which has access to the model parameters (see `mutils.py` function `def args_to_params(args)` for details). Also, add a new constant in the class `NLIModel` similar to `AVERAGE_WORD_VECS, LSTM, BILSTM, BILSTM_MAX`, and add a corresonding line in the function `_choose_encoder` (replacing the constant `NEW_MODEL` and the class name `NewEncoder`):

```python
def _choose_encoder(self, model_type, model_params):
	...
	elif model_type == NLIModel.NEW_MODEL:
		self.encoder = NewEncoder(model_params)
	...
```

From the train function, you can now select the new encoder by setting the option `--model VAL` to the value specified by the new constant in `NLIModel` (constant `NEW_MODEL` in the previous example).

### Adding new classifier

A new classifier can be similarly added as a new encoder. However, if the classifier only slightly differs from the MLP (or is just a specific parameterization of it), you can extend the class `NLIClassifier` for that. 
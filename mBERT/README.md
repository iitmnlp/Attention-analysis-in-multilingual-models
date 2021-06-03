Introduction
---------------
This repository contains the code to perform statistical analysis of the roles played by attention heads in mBERT.  
We consider 7 functional roles:  
a)CLS  
b)SEP  
c)Local   
d)nsubj  
e)amod  
f)advmod  
g)obj

We carry out the analysis for both pretrained and fine-tuned mBERT as detailed in the paper.

The process mainly consists of two steps:
1. Run the statistical tests using the datasets to compute sieve bias scores and persist it.
2. Do data analysis using the sieve scores to draw various insights.

Requirements
---------------
Python 3.7.10  
PyTorch 1.8.1+cu101 (on Colab)  
Numpy, Pickle, Scipy, Pandas  
Visualization libraries: Matplotlib, Seaborn and Plotly 

Code files and their usage
---------------
1. PreTrainedDelimAndLocalTests.ipynb - This is used to run the statistical tests for CLS, SEP and Local roles in pretrained mBERT.   
Only the 'languages' variable has to be changed to run the statistical tests for the appropriate language.

2. PreTrainedSyntacticTests.ipynb - This is used to run the statistical tests for nsubj, amod, advmod and obj roles in pretrained mBERT.  
Only the 'languages' and 'functions' variable has to be changed to run the statistical tests for the appropriate language and role.

3. FineTunedDelimAndLocalTests.ipynb - This is used to run the statistical tests for CLS, SEP and Local roles in fine tuned mBERT.   
'fineTunedPath' variable has to be changed to point to the fine tuned model.  
'languages' and 'functions' variable has to be changed to run the statistical tests for the appropriate language and role.

4. FineTunedSyntacticTests.ipynb - This is used to run the statistical tests for nsubj, amod, advmod and obj roles in fine tuned mBERT.  
'fineTunedPath' variable has to be changed to point to the fine tuned model.  
'languages' and 'functions' variable has to be changed to run the statistical tests for the appropriate language and role.

5. DataAnalysis.ipynb - This is used to run data analysis to reproduce the plots/numbers mentioned in the paper.  

6. FineTuning.ipynb - This is used to fine tune mBERT, using the code from XGLUE on 5 tasks: XNLI, NC, PAWSX, QADSM and QAM.  

7. GenerateHeatMaps.ipynb - This is used to plot the attention heat maps for the specified language.    

These notebooks can be imported to Google Colaboratory and directly run from there, using the datasets.

Dependencies
---------------
transformers == 4.5.1  
XGLUE Unicoder Code - https://github.com/microsoft/Unicoder

Computing infrastructure
---------------

All experiments are run on Google Colaboratory Pro.

For fine tuning on NC,PAWSX,QADSM,QAM - Tesla P100 GPUs are used, since dataset is relatively small.

For fine tuning on XNLI in each of the six languages - Tesla V100 GPUs are used, since XNLI is a relativly big dataset.

Average runtime for fine tuning
--------------------------------------------------------------------------------

Fine tuning on Tesla P100 GPU - approx 45 mins per epoch  
Fine tuning on Tesla V100 GPU - approx 1.5 hours per epoch

Number of parameters
--------------------------------------------------------------------------------
'bert-base-multilingual-cased' from HuggingFace is used for all experiments. It has 179M parameters.

Test and validation accuracy after fine tuning
--------------------------------------------------------------------------------

|Task			       |Test accuracy |	Validation accuracy  |
| ------------- |:-------------:| --------------------:|
|NC	    		     |82.2          |82.2                  |
|PAWSX   		     |87				    |86.1  |
|QADSM 			     |63.4			    |63.6  |
|QAM 			       |65.8			    |65.6  | 
|XNLI English	   |66.7			    |66.8  |
|XNLI German		 |68.2			    |68.1  |
|XNLI French		 |67.8			    |67.7  |
|XNLI Spanish	   |67.9			    |67.6  |
|XNLI Hindi		   |66.1			    |65.8  | 
|XNLI Urdu		   |63.0			    |62.8  |

Hyperparameters
--------------------------------------------------------------------------------

The hyperparameters are the same as mentioned in XGLUE for the respective tasks and is mentioned in the main paper.

Datasets
--------------------------------------------------------------------------------
For each language, the following files are provided, all of which is extracted from Universal Dependencies dataset.

1. \<language\>-sentences-1000.txt - 1000 sentences used to perform the statistical tests for the positional functional roles. (CLS,SEP,Local)  
2. \<language\>-sentences-\<role\>-1000.txt - 1000 sentences used to perform the statistical test for the given syntactic functional role.  
3. \<language\>-\<role\>-1000-new.txt - Two lines per sentence in the above file. The first line contains the list of indices of dependent and head. Second line contains the tokens in the sentence, as specified in the CONLLU file from Universal Dependencies. 

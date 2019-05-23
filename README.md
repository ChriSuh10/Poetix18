# Poetix18


## Important files:

### Models:
  #### comebined_forward: Trained on all the corpuses combined. Text was NOT reversed.
  #### combined_back: Trained on all the corpuses combined. Text was reversed.
  #### model_back.py: Specified the model backwards. Methods:  compute_fx, score_a_list
  #### model_forward.py: Specified the model forward. Methods:  compute_fx, score_a_list
  
### Functions:
  #### generate.py: Specified the class Generate and different methods to generate lines.
  #### Transversal_Glove.py: Clas MetaPoetry and the methods to generate the meta poems.
  #### functions.py: All auxiliar functions.
  
### Notebooks
#### Limericks and sonnets: Examples of limmerics and sonnets.
#### generate_line: Examples of sonnets.
#### Transversal: examples of meta poetry.


### Files:

#### postag_dict_all: Sets of POS:[words] and WORD:[POS]
#### cmu-dict: metric

### Rhetorical Analysis:

#### 0. recursively download submodule
#### git submodule update --init --recursive

#### 1. install mono
#### https://www.mono-project.com/docs/compiling-mono/linux/

#### 2. restore packages
#### mono nuget.exe restore Rhetorica.sln 

#### 3. compile exe
#### xbuild Rhetorica.sln /p:Configuration=Release /p:Platform=x64

#### 4. run exe

## Freesound Audio Tagging - COMP47650

COMP47650 - Deep Learning Group Project     

by Shane Kennedy, Thu Trang Nguyen and Daniel Hand      

## Dataset
The contents of data/ can be found here:        

https://ucd-my.sharepoint.com/:u:/g/personal/daniel_hand_ucdconnect_ie/EZVF_TnCRnpOtuqch0sGDv8B_fPaLB0vkQffHID8LUiWSQ?e=T5g7de      

The audio_test, audio_train and submission files can be downloaded via:     

wget -O FSDKaggle2018.audio_train.zip https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_train.zip?download=1          
wget -O FSDKaggle2018.audio_test.zip https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_test.zip?download=1        
wget -O FSDKaggle2018.meta.zip https://zenodo.org/record/2552860/files/FSDKaggle2018.meta.zip?download=1        

or by executing main.py which executes the commands in dataset.sh.

        
            
## Repository Structure
The file structure of this repository is the following:

├── FreesoundAudioTagging                       
│   ├── checkpoints         
│   │   ├── cnn-weights.05-0.66.hdf5            
│   │   ├── cnn-weights.05-0.68.hdf5            
│   │   ├── cnn-weights.05-0.69.hdf5        
│   │   ├── cnn-weights.10-0.67.hdf5        
│   │   ├── vnn-weights.05-0.80.hdf5        
│   │   ├── vnn-weights.05-0.82.hdf5        
│   │   ├── vnn-weights.05-0.85.hdf5        
│   │   ├── weights.05-0.80.hdf5        
│   │   ├── weights.05-0.81.hdf5        
│   │   ├── weights.05-0.82.hdf5        
│   │   ├── weights.05-0.83.hdf5        
│   │   ├── weights.05-0.84.hdf5        
│   │   ├── weights.05-0.85.hdf5        
│   │   ├── weights.10-0.73.hdf5        
│   │   ├── weights.10-0.74.hdf5        
│   │   ├── weights.10-0.75.hdf5        
│   │   ├── weights.10-0.77.hdf5        
│   │   ├── weights.15-0.71.hdf5        
│   │   ├── weights.15-0.72.hdf5        
│   │   ├── weights.15-0.73.hdf5        
│   │   ├── weights.15-0.74.hdf5        
│   │   ├── weights.15-0.75.hdf5        
│   │   ├── weights.20-0.70.hdf5        
│   │   ├── weights.20-0.71.hdf5        
│   │   ├── weights.20-0.72.hdf5        
│   │   ├── weights.20-0.73.hdf5        
│   │   ├── weights.25-0.69.hdf5        
│   │   ├── weights.25-0.70.hdf5               
│   │   ├── weights.25-0.71.hdf5            
│   │   ├── weights.25-0.72.hdf5            
│   │   ├── weights.30-0.69.hdf5        
│   │   ├── weights.30-0.72.hdf5        
│   │   ├── weights.35-0.68.hdf5        
│   │   ├── weights.35-0.71.hdf5        
│   │   ├── weights.35-0.72.hdf5        
│   │   ├── weights.40-0.69.hdf5        
│   │   ├── weights.40-0.70.hdf5        
│   │   ├── weights.40-0.72.hdf5        
│   │   ├── weights.45-0.72.hdf5        
│   │   └── weights.55-0.72.hdf5        
│   ├── data        
│   │   ├── audio_test_trim     
│   │   ├── audio_train_trim        
│   │   ├── mel_spec_test       
│   │   ├── test        
│   │   ├── test_tab_feats.pkl      
│   │   ├── train       
│   │   └── train_tab_feats.pkl     
│   ├── dataset.sh      
│   ├── download_dataset.py             
│   ├── figures     
│   │   ├── cnn-Accuracy_vs_Epochs.png          
│   │   ├── cnn-Loss_vs_Epochs.png          
│   │   ├── vnn-Accuracy_vs_Epochs.png          
│   │   └── vnn-Loss_vs_Epochs.png          
│   ├── freesound_audio_tagging.ipynb           
│   ├── log             
│   │   └── log1.txt            
│   ├── main.py         
│   ├── models          
│   │   ├── cnn.py      
│   │   ├── sgd.py      
│   │   ├── svm.py      
│   │   └── vnn.py      
│   ├── preprocess.py       
│   ├── requirements.txt        
│   ├── submission          
│   │   ├── test_post_competition_scoring_clips.csv     
│   │   └── train_post_competition.csv      
│   ├── summary_feats_funcs.py      
│   ├── tree.txt            
│   └── visualisation.py        
└── tree.txt        
                                
14 directories, 52232 files     


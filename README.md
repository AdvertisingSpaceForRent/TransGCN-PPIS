# MVGNN-PPIS
MVGNN is a state-of-the-art graph neural network designed for predicting protein-protein binding sites with high speed and accuracy. <br>The MVGNN web server is available for free access [here](http://www.isme.ln.cn:5001/protein_site_prediction1).
# System requirement
python 3.9.19<br>
torch 1.13.1<br>
cuda 11.7<br>
transformers 4.41.1<br>
Bio 1.83
# Feature Generation
## Sequence feature generation
Please visit [ProtTrans](https://github.com/agemagician/ProtTrans) on GitHub to download the model. Store the downloaded model in `./process_feature/pretrained_model/Rostlab/prot_t5_xl_uniref50.`
## Predicted structural feature generation
**1.** Our protein structure predictions are generated using AlphaFold3. For detailed installation instructions, please refer to [AlphaFold3](https://github.com/google-deepmind/alphafold3) on GitHub.<br>
**2.** We use the DSSP tool to extract secondary structure information. For detailed instructions, please refer to [DSSP](https://github.com/PDB-REDO/dssp) on GitHub. Before using the software, please grant execution permissions with:
```
chmod +x ./process_feature/Software/dssp-2.0.4/mkdssp
```
## Generate Features
After completing the steps above, run the following command to generate features and save them in the `./feature` directory: 
```
python ./process_feature/process_feature.py
```
Alternatively, you can download the features directly from our cloud drive: [Google Drive Link](https://drive.google.com/drive/folders/1sHd-2MmzdhmxvMjrNIugHw0RrBU8QYnR?usp=drive_link).
# Run MVGNN for prediction
To predict datasets located in the ./datasets directory, run the following command:
```
python ./main.py
```
Ensure all dependencies are installed and the data is correctly formatted before executing the command.

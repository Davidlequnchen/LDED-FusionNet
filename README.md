# LDED-FusionNet
 LDED-FusionNet: Audio-visual fusion for defect detection in laser-directed energy deposition (LDED) Additive Manufacturing Process.

 ## Installation

 - create a new conda environment with Python version 3.8.10
    ```
    conda create --name torch python=3.8.10
    ```  
- Activate your torch environment: ```conda activate torch```
- Check CUDA Version: Find out the version of CUDA toolkit installed on your machine by running ```nvcc --version```.   
Check the driver API version: ` nvidia-smi`
- For example, if CUDA version 11.0, then
    ```
    conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
    ```
- Check Installation: After installation, you can verify that PyTorch is using the GPU with the following Python command:
    ```
    import torch
    print(torch.cuda.is_available())
    ```

- install the required libraries:
```
pip install -r requirements.txt
```
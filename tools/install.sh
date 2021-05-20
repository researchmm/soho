# conda create -n soho_test python=3.7 -y
# conda activate soho_test

#install pytorch
conda install pytorch torchvision  cudatoolkit=10.2 -c pytorch -y

pip uninstall -y apex || :
cd /tmp
git clone https://github.com/NVIDIA/apex.git
cd /tmp/apex/
python setup.py install --cuda_ext --cpp_ext
rm -rf /tmp/apex*

python setup.py develop
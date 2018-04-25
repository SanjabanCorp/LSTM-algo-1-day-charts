# Initial packages set-up
apt-get update && sudo apt-get -y upgrade
apt-get install python3-pip

# Virtual Env setup
pip3 install virtualenv
virtualenv -p python3 venv
. venv/bin/activate
pip install --upgrade pip
apt-get install python3-tk

pip install pandas scipy image matplotlib keras tensorflow lxml beautifulsoup4 h5py

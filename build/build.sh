# create env
sudo apt install python3-pip
pip install virtualenv
virtualenv -p /usr/bin/python3 .env_uq

# activate environment
source .env_up/bin/activate

# update packages
pip install -r requirements.txt


#########
# Intro #
#########
echo "This script will install DeepClean into a virtual environment. Make
sure that you are not already sourced in another environment. The VE will
be created and installed to $HOME/deepclean_ve."
echo ""

echo -n "Do you wish to proceed? (y/n): "
read proceed

if [ "$proceed" = "y" ]; then
    echo "starting install..."
    basedir=$(pwd)
else
    echo "exiting..."
    exit
fi

##################################
# Create the virtual environment #
##################################
cd $HOME

# Make sure virtualenv exists
if hash virtualenv 2> /dev/null; then
    virtualenv --python=$(which python) deepclean_ve
else
    echo "Need to install virtualenv"
    echo -n "Install it now? (y/n): "
    read install

    if [ "$install" = "n" ]; then
        echo "exiting..."
        exit
    else
        echo "installing virtualenv..."
        pip install virtualenv
        virtualenv --python=$(which python) deepclean_ve
    fi
fi

# source the VE
source $HOME/deepclean_ve/bin/activate
pip install numpy

#####################################
# Install the repo and dependencies #
#####################################
# Install DeepClean
cd $basedir
pip install --upgrade pip
pip install -r requirements.txt && pip install -e .

# Install nds2
cd $HOME/deepclean_ve
wget http://www.lsc-group.phys.uwm.edu/daswg/download/software/source/nds2-client-0.15.2.tar.gz
tar -xzvf nds2-client-0.15.2.tar.gz
mv nds2-client-0.15.2.tar.gz nds2-client-0.15.2/
cd $HOME/deepclean_ve/nds2-client-0.15.2/
mkdir obj
cd obj
cmake -DCMAKE_INSTALL_PREFIX=$HOME/deepclean_ve/ -DCMAKE_C_COMPILER=$(which cc) -DCMAKE_CXX_COMPILER=$(which c++) ..
cmake --build .
cmake --build . -- install

#############
# Finish up #
#############
echo "
Your config file is in $HOME/deepclean_ve/configs and needs to be edited before
running (namely, change 'albert.einstein'). Remember to source the virtual
environment before running. i.e.,

$ source $HOME/deepclean_ve/bin/activate

Check the docs for further usage instructions. Email rich.ormiston@ligo.org
with further questions, bug reporting, or functionality requests.
Done"

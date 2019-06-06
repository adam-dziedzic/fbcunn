# https://www.learnopencv.com/install-opencv3-on-ubuntu/
# Remove any previous installations of x264</h3>
sudo apt-get remove x264 libx264-dev

# We will Install dependencies now

sudo apt-get install build-essential checkinstall cmake pkg-config yasm
sudo apt-get install git gfortran
sudo apt-get install libjpeg8-dev libjasper-dev libpng12-dev

# If you are using Ubuntu 14.04
sudo apt-get install libtiff4-dev
# If you are using Ubuntu 16.04
# sudo apt-get install libtiff5-dev

sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev --yes
sudo apt-get install libxine2-dev libv4l-dev --yes
sudo apt-get install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev --yes

sudo apt-get install qt5-default libgtk2.0-dev libtbb-dev --yes
sudo apt-get install libatlas-base-dev --yes
sudo apt-get install libfaac-dev libmp3lame-dev libtheora-dev --yes
sudo apt-get install libvorbis-dev libxvidcore-dev --yes
sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev --yes
sudo apt-get install x264 v4l-utils --yes
sudo apt-get install libprotobuf-dev protobuf-compiler --yes
sudo apt-get install libgoogle-glog-dev libgflags-dev --yes
sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen --yes

sudo apt-get install python-dev python-pip python3-dev python3-pip --yes
sudo -H pip2 install -U pip numpy
sudo -H pip3 install -U pip numpy

sudo pip3 install virtualenv virtualenvwrapper
echo "# Virtual Environment Wrapper"  >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc

############ For Python 3 ############
# create virtual environment
mkvirtualenv facecourse-py3 -p python3
workon facecourse-py3
  
# now install python libraries within this virtual environment
pip install numpy scipy matplotlib scikit-image scikit-learn ipython
  
# quit virtual environment
deactivate
######################################

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON ..

# find out number of CPU cores in your machine
nproc
# substitute 4 by output of nproc
make -j4
sudo make install
sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
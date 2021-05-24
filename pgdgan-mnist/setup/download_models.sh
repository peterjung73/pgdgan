mkdir models
mkdir optimization
mkdir mnist_vae/models

#wget https://www.cs.utexas.edu/~ashishb/csgm/csgm_pretrained.zip
wget  https://github.com/AshishBora/csgm/blob/master/csgm_pretrained.zip?raw=true -O csgm_pretrained.zip
unzip csgm_pretrained.zip
mv csgm_pretrained/celebA_64_64/ models/

rm -r csgm_pretrained
rm csgm_pretrained.zip

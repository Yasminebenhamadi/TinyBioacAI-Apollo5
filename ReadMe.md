
# TinyBioacAI Benchmarking on Apoll5

This is a draft repository for testing different audio frontends on Apollo5. It includes code for:

1. Running STFT and computing mel spectograms
2. Running .tflite models


## Compile and Run on Apollo5

This repo is a module within the neuralspot library. To compile it run from the neuralspot directory:

* Get `neuralSPOT` if you don't already have it:
```
git clone git@github.com:AmbiqAI/neuralSPOT.git
make clean
make -j
```

* Add this repository as a neuralSPOT module. From the `neuralSPOT` root run:
```
cd apps/
git clone this repo
```

* To compile, deploy and view the code you can you the `neuralspot` `make` system. Go back the the `neuralSPOT` root directory and run:
```
make EXAMPLE=yh_app PLATFORM=apollo510_evb mcpu=cortex-m55
make deploy
make view
```

* If you're using `VScode` you can also set up a `launch.json` under the `.vscode` folder for debugging. 
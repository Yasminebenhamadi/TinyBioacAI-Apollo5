# TinyBioacAI Benchmarking on Apollo5

## Overview

This repository contains benchmarking code for evaluating log-mel spectrograms, convolutional neural networks (CNNs), and bit-sliced Gabor frontends proposed in the [associated paper (HAL)](https://hal.science/hal-05758524v2).

It currently includes implementations for:

1. **Audio feature extraction:** Computing Short-Time Fourier Transforms (STFTs) and mel spectrograms.
2. **TFLite inference:** Running TensorFlow Lite models on the Ambiq Apollo5 platform.
3. **Bit-sliced gabor frontend** Running a real Gabor layer on MSB and LSB of an audio signal. 

## Compile and Run on Apollo5

This repository is designed to be used as an application module within the [neuralSPOT](https://github.com/AmbiqAI/neuralSPOT) library.

### 1. Clone and Build neuralSPOT

If you have not already cloned neuralSPOT, run:

```bash
git clone git@github.com:AmbiqAI/neuralSPOT.git
cd neuralSPOT
make clean
make -j
```

### 2. Add This Repository as a neuralSPOT Module

From the neuralSPOT root directory, navigate to `apps/` and clone this repository:

```bash
cd apps/
git clone git@github.com:Yasminebenhamadi/TinyBioacAI-Apollo5.git
```

The repository will now be available as an application within the neuralSPOT project.

### 3. Connect the Apollo5 Board and Open the Serial Monitor

Connect the Apollo510 evaluation board to your computer via USB.

Open a terminal in the neuralSPOT root directory and run:

```bash
make view
```

This opens the serial monitor, allowing you to view benchmark output from the board.

### 4. Compile and Deploy

Open a second terminal, navigate to the neuralSPOT root directory, and run:

```bash
make EXAMPLE=TinyBioacAI-Apollo5 PLATFORM=apollo510_evb mcpu=cortex-m55 deploy
```

This command compiles the application and deploys it to the Apollo510 evaluation board.

To view the output, run:

```bash
make view
```

### 5. Debugging with VS Code (Optional)

If you are using Visual Studio Code, you can configure debugging by creating a `launch.json` file inside the `.vscode/` directory.

This allows you to set breakpoints and debug the application directly from VS Code.



## Structure

`conv1D_inference.cc` contains the Bit-sliced Gabor inference implementation, while `mel_inference.cc` contains the log-mel spectrogram inference implementation.

To switch between the two modes, change `#define MEL_MAIN 1` in `audio.h`:

- `#define MEL_MAIN 1` → Log-mel inference
- `#define MEL_MAIN 0` → Bit-sliced Gabor inference

**Important:** Run `make clean` before rebuilding whenever you switch between the two modes, to ensure that all source files are recompiled with the correct configuration.

```bash
make clean
make -j
```
# Reading Notes

---

## Predictive Coding in the Visual Cortex: a Functional Interpretation of Some Extra-classical Receptive-field Effects — [Rajesh P. N. Rao, 1999]

**Date read: 09.06.2026**

**Link / DOI:10.1038/4580**

**Core idea in 2-3 sentences:**
 1. The approach postulates that neural networks learn the statistical regularities of the natural world, signaling deviations 
 from such regularities to higher processing centers.
 2. This reduces redundancy by removing the predictable, and hence redundant, components of the input signal.
 3. Using a hierarchical model of predictive coding, Rao et al. show that visual cortical neurons with extra-classical(unusual, different) RF properties can be interpreted as residual error detectors, signaling the difference between an input signal and its statistical predictionbased on an efficient internal model of natural images.
 4. Paradox of end-stopping: Neurons show more activity when shown a short line occupying the center of its RF, than when it's shown a longer line going out of its bounds. This is opposite of the prevailing theory of activity being proportional to the stimulus in a strictly feedforward sense.
 5. Top-down connections carry predictions of expected neural activity from a higher hierarchical level to a lower one, while Bottom-up connections carry only the error between the predictions and the actual neural activity from lower to higher levels.
 6. Lower levels operate on smaller spatial and possibly temporal scales.
 7. Given an input image, the initial predictions at any given level are based on an arbitrary random combination of the basis vec-tors, giving large error signals. To minimize this error, the network converges to the responses that best predict the current input by subtracting the prediction from the input (via inhibition) and propagating the residual error signal to the neurons at the next level, which integrate this error and generate a better prediction (Methods)
 8. Input image example : Shorter bar in the center of the RF --> Large error signals, as short bars seldom occur in natural images, on which the model was trained. Longer bar extending to into the neighbouring RF --> necessary context for predicting bar in the center --> accurate prediction --> Lower magnitude of error.


**How it connects to the project:**
Theoretical background for Predictive Coding models.


## [Paper Title] — [Author(s), Year]

**Date read:**

**Link / DOI:**

**Core idea in 2-3 sentences:**


## An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity — [Whittington, 2017]

**Date read: 10.06.2026**

**Link / DOI: 10.1162/NECO_a_00949**

**Core idea in 2-3 sentences:**
1. Brain neurons cannot perform backpropogation due to their being no mechanism with which to store downstream weights to update upstream ones(weight transport problem) and a lack of global error signal.
2. Brains are highly local and learning happens via Hebbian Plasticity
3. Paper demonstrates mathematically that Predictive Coding architecture based on two populations of neurons, namely value nodes(that carry predictions downstream) and error nodes(that carry errors upstream)
4. These error signals flow backwards until the total network error reduces to a minimum and neural activities settle to a minimum energy state.
5. Weights obtained at this minima correspond exactly to weights obtained by backpropogation but much more efficiently and using local Hebbian plasticity where weight change is proportional to pre-synaptic and post-synaptic activity.
**Key equations / mechanisms:**

**How it connects to the project:**

---

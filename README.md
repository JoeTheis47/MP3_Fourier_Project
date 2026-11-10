# MP3_Fourier_Project
This is a project attempting to determine a good way of identifying a well plucked guitar's timbre from its recorded sound signals.

To begin using the file, look at harmonic_functions.py, which holds most of the tonal analysis functions. However, before changing any of the preset perameters, run deep_model_train with the changes implemented, otherwise youll get very wrong results
also, just run deep_model_train before anything anyway, it will store the file path which holds the model weights
One final thing about this, you'll need a large quantity of audio data. It can be recorded by you or found on the web, but it needs to be of the form "namethatiswhateveryouwantittobe_(insert note name here)", or whatever note the file is denoting. You can see naming procedure in the data_loader script.
